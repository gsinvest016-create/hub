#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日選股掃描（精選 23 個訊號）
- 讀 data/index.json + data/chunks/*.json + data/indices.json
- 個股套用 14 個趨勢突破訊號（台股／美股），指數套用 9 個指數訊號
- 依勝率庫的平均 R 加權評分，寫 data/picks.json
- 若設定 TELEGRAM_TOKEN / TELEGRAM_CHAT_ID，推播當日摘要
"""
import base64, gzip, json, os, math, datetime as dt, urllib.request, urllib.parse
import numpy as np, warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")

# 勝率庫（台股十年）平均 R，當作評分權重
W_STOCK = {"hi52": .594, "vcp": .512, "gap_up": .449, "cup": .414, "kd_high": .414, "donch55": .380, "flag": .378,
           "box": .336, "donch20": .318, "asc_tri": .306, "macd_gc_above": .306, "bb_upper": .305, "adx": .292, "vol_break": .276}
# 流動性門檻（可用環境變數調整）：台股 20 日均量 ≥ MIN_TW_LOTS 張 且 均成交金額 ≥ MIN_TW_TURNOVER 元；美股 均成交金額 ≥ MIN_US_TURNOVER 美元
MIN_TW_LOTS = int(os.environ.get("MIN_TW_LOTS", "1000"))
MIN_TW_TURNOVER = float(os.environ.get("MIN_TW_TURNOVER", "50000000"))
MIN_US_TURNOVER = float(os.environ.get("MIN_US_TURNOVER", "20000000"))
MIN_TW_CAPITAL = float(os.environ.get("MIN_TW_CAPITAL", "10"))      # 台股實收資本額門檻（億元），低於剔除
MIN_US_MCAP_B = float(os.environ.get("MIN_US_MCAP_B", "1"))         # 美股市值門檻（十億美元），低於剔除


def fetch_json(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8-sig"))


def load_tw_capital():
    """台股實收資本額（億元）：TWSE / TPEx 開放資料，快取 30 天在 data/capital_tw.json"""
    path = os.path.join(DATA, "capital_tw.json")
    try:
        c = json.load(open(path, encoding="utf-8"))
        if (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(c["at"])).days < 30 and len(c["cap"]) > 500:
            return c["cap"]
    except Exception:
        pass
    cap = {}
    for url in ["https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
                "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O",
                "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O.json"]:
        try:
            rows = fetch_json(url)
        except Exception as e:
            print("股本來源失敗", url, e); continue
        for row in rows:
            code = next((str(v).strip() for k, v in row.items() if ("代號" in k or "Code" in k) and str(v).strip().isdigit()), None)
            val = next((v for k, v in row.items() if "資本額" in k or "Capital" in k), None)
            if not code or val is None: continue
            try:
                n = float(str(val).replace(",", ""))
            except ValueError:
                continue
            if n > 0: cap[code] = round(n / 1e8, 2)
    if cap:
        json.dump({"at": dt.datetime.now(dt.timezone.utc).isoformat(), "cap": cap}, open(path, "w", encoding="utf-8"))
    return cap


def load_us_mcap(symbols):
    """美股市值（十億美元）：只查有訊號的候選股，yfinance fast_info，快取 30 天在 data/mcap_us.json"""
    path = os.path.join(DATA, "mcap_us.json")
    try:
        c = json.load(open(path, encoding="utf-8"))
    except Exception:
        c = {}
    now = dt.datetime.now(dt.timezone.utc)
    need = [s for s in symbols if s not in c or (now - dt.datetime.fromisoformat(c[s]["at"])).days >= 30]
    if need:
        try:
            import yfinance as yf
            for sym in need[:300]:
                try:
                    mc = yf.Ticker(sym.replace(".", "-")).fast_info.get("marketCap")
                    c[sym] = {"at": now.isoformat(), "b": round(mc / 1e9, 2) if mc else None}
                except Exception:
                    c[sym] = {"at": now.isoformat(), "b": None}
        except Exception as e:
            print("市值查詢失敗", e)
        json.dump(c, open(path, "w", encoding="utf-8"))
    return {k: v.get("b") for k, v in c.items()}

ZH = {"hi52": "52 週新高", "vcp": "VCP 突破", "gap_up": "跳空帶量長紅", "cup": "杯柄突破", "kd_high": "KD 高檔鈍化", "donch55": "55 日新高",
      "flag": "旗形突破", "box": "箱型突破", "donch20": "20 日新高", "asc_tri": "上升三角突破", "macd_gc_above": "MACD 零軸上金叉",
      "bb_upper": "站上布林上軌", "adx": "ADX 趨勢啟動", "vol_break": "爆量長紅突破",
      "ichimoku": "一目均衡翻多", "supertrend": "SuperTrend 翻多", "ma20_up": "站上 MA20", "nr7": "NR7 突破", "td9": "九轉買 9",
      "rsi2": "RSI2 超賣", "down3": "連 3 黑（MA200 上）", "bb_lower": "跌破布林下軌", "c_harami": "多頭母子"}
MAIN = {"hi52", "vcp", "gap_up", "cup", "kd_high", "donch55", "flag"}


def load_z(path):
    with open(path, "r", encoding="utf-8") as f:
        o = json.load(f)
    if isinstance(o, dict) and "z" in o and len(o) == 1:
        return json.loads(gzip.decompress(base64.b64decode(o["z"])).decode("utf-8"))
    return o


# ---------- 指標 ----------
def sma(a, n):
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        c = np.cumsum(np.insert(a, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out

def ema(a, n):
    out = np.full(len(a), np.nan); k = 2 / (n + 1)
    if len(a) == 0: return out
    out[0] = a[0]
    for i in range(1, len(a)): out[i] = a[i] * k + out[i - 1] * (1 - k)
    return out

def atr(h, l, c, n=14):
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    tr = np.insert(tr, 0, h[0] - l[0])
    out = np.full(len(c), np.nan)
    if len(c) < n: return out
    out[n - 1] = tr[:n].mean()
    for i in range(n, len(c)): out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out

def rsi(c, n):
    d = np.diff(c); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    out = np.full(len(c), np.nan)
    if len(c) <= n: return out
    au, ad = up[:n].mean(), dn[:n].mean()
    out[n] = 100 - 100 / (1 + au / ad) if ad > 0 else 100
    for i in range(n + 1, len(c)):
        au = (au * (n - 1) + up[i - 1]) / n; ad = (ad * (n - 1) + dn[i - 1]) / n
        out[i] = 100 - 100 / (1 + au / ad) if ad > 0 else 100
    return out

def kd(h, l, c, n=9):
    k = np.full(len(c), 50.0); d = np.full(len(c), 50.0)
    for i in range(n - 1, len(c)):
        hh, ll = h[i - n + 1:i + 1].max(), l[i - n + 1:i + 1].min()
        rsv = 50 if hh == ll else (c[i] - ll) / (hh - ll) * 100
        k[i] = k[i - 1] * 2 / 3 + rsv / 3; d[i] = d[i - 1] * 2 / 3 + k[i] / 3
    return k, d

def macd(c):
    dif = ema(c, 12) - ema(c, 26); dea = ema(dif, 9); return dif, dea

def adx(h, l, c, n=14):
    up = h[1:] - h[:-1]; dn = l[:-1] - l[1:]
    pdm = np.where((up > dn) & (up > 0), up, 0.0); ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    def wild(x):
        o = np.full(len(x), np.nan)
        if len(x) < n: return o
        o[n - 1] = x[:n].sum()
        for i in range(n, len(x)): o[i] = o[i - 1] - o[i - 1] / n + x[i]
        return o
    trn, pn, nn = wild(tr), wild(pdm), wild(ndm)
    pdi = 100 * pn / trn; ndi = 100 * nn / trn
    dx = 100 * abs(pdi - ndi) / (pdi + ndi)
    ax = np.full(len(dx), np.nan)
    if len(dx) >= 2 * n:
        ax[2 * n - 2] = np.nanmean(dx[n - 1:2 * n - 1])
        for i in range(2 * n - 1, len(dx)): ax[i] = (ax[i - 1] * (n - 1) + dx[i]) / n
    pad = lambda x: np.insert(x, 0, np.nan)
    return pad(ax), pad(pdi), pad(ndi)

def supertrend(h, l, c, n=10, m=3.0):
    a = atr(h, l, c, n); hl2 = (h + l) / 2
    ub, lb = hl2 + m * a, hl2 - m * a
    fu, fl = ub.copy(), lb.copy(); dirn = np.ones(len(c))
    for i in range(1, len(c)):
        if np.isnan(a[i]): continue
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if dirn[i - 1] == 1: dirn[i] = -1 if c[i] < fl[i] else 1
        else: dirn[i] = 1 if c[i] > fu[i] else -1
    return dirn

def td_buy_count(c):
    cnt = 0
    for i in range(len(c) - 1, 3, -1):
        if c[i] < c[i - 4]: cnt += 1
        else: break
    return cnt


# ---------- 個股訊號 ----------
def stock_signals(o, h, l, c, v):
    n = len(c); sig = {}
    if n < 60: return sig, {}
    i = n - 1
    ma20, ma50, ma200, ma150 = sma(c, 20), sma(c, 50), sma(c, 200), sma(c, 150)
    vol20 = sma(v.astype(float), 20)
    a14 = atr(h, l, c)
    hi252 = c[max(0, i - 252):i].max()
    near52 = (hi252 - c[i]) / hi252 * 100 if hi252 > 0 else 99
    above200 = (not np.isnan(ma200[i])) and c[i] > ma200[i]
    volr = v[i] / vol20[i] if vol20[i] and vol20[i] > 0 else 0
    avgvol = float(vol20[i]) if vol20[i] == vol20[i] else 0.0          # 20 日均量（股）
    turnover = float(np.mean(v[i - 19:i + 1] * c[i - 19:i + 1]))         # 20 日均成交金額
    chg = c[i] / c[i - 1] - 1
    # 52 週新高（20 日來首次）
    if n > 252 and c[i] > hi252 and not any(c[j] > c[max(0, j - 252):j].max() for j in range(i - 20, i) if j > 252):
        sig["hi52"] = True
    # 55 / 20 日新高
    if c[i] > c[i - 55:i].max(): sig["donch55"] = True
    if c[i] > c[i - 20:i].max(): sig["donch20"] = True
    # 跳空帶量長紅
    if o[i] >= c[i - 1] * 1.04 and volr >= 2 and c[i] > o[i]: sig["gap_up"] = True
    # 爆量長紅突破 20 日高
    if chg >= 0.03 and volr >= 2 and c[i] > c[i - 20:i].max(): sig["vol_break"] = True
    # KD 高檔鈍化
    k, _ = kd(h, l, c)
    if k[i] > 80 and k[i - 1] > 80 and k[i - 2] > 80 and not (k[i - 3] > 80 and k[i - 4] > 80 and k[i - 5] > 80): sig["kd_high"] = True
    # MACD 零軸上金叉
    dif, dea = macd(c)
    if dif[i] > 0 and dif[i] > dea[i] and dif[i - 1] <= dea[i - 1]: sig["macd_gc_above"] = True
    # 布林上軌
    sd = np.std(c[i - 19:i + 1]); up = ma20[i] + 2 * sd
    if c[i] > up and c[i - 1] <= (ma20[i - 1] + 2 * np.std(c[i - 20:i])): sig["bb_upper"] = True
    # ADX
    ax, pdi, ndi = adx(h, l, c)
    if not np.isnan(ax[i]) and ax[i] > 25 >= ax[i - 1] and pdi[i] > ndi[i]: sig["adx"] = True
    # 箱型突破：前 40 日區間 5–15%，今收盤 > 前 40 日最高
    box = c[i - 40:i]; rng = (box.max() - box.min()) / box.min()
    if 0.05 <= rng <= 0.15 and c[i] > box.max(): sig["box"] = True
    # 旗形：25 日內有 ≤15 日漲 ≥20% 的旗桿，之後 5–15 日整理回檔 ≤ 旗桿一半，今收盤 > 整理高
    for pole_len in range(5, 16):
        for cons in range(5, 16):
            s0 = i - cons - pole_len
            if s0 < 0: continue
            pole = c[s0 + pole_len] / c[s0] - 1
            if pole < 0.20: continue
            seg = c[s0 + pole_len:i]
            if seg.min() < c[s0 + pole_len] - (c[s0 + pole_len] - c[s0]) * 0.5: continue
            if c[i] > seg.max() and c[i - 1] <= seg.max(): sig["flag"] = True
            break
        if "flag" in sig: break
    # 上升三角：60 日內壓力水平（≥2 次高點在 ±2%）、低點墊高，今收盤突破壓力
    win = 60 if i >= 60 else i
    hh = h[i - win:i]; ll = l[i - win:i]; res = hh.max()
    touches = int((hh >= res * 0.98).sum())
    lows_idx = np.argsort(ll)[:3]
    x = np.arange(win); slope = np.polyfit(x, ll, 1)[0]
    if touches >= 2 and slope > 0 and c[i] > res and c[i - 1] <= res and (res - ll.min()) / res < 0.25: sig["asc_tri"] = True
    # 杯柄：杯 35–160 日、深 12–33%、底在中段、右緣回到左緣 5% 內；柄 5–20 日、回檔 ≤ 杯深一半且 ≤12%；今收盤首次突破柄高
    for hl_ in range(5, 21):
        hs = i - hl_
        handle = c[hs:i]; hmax = handle.max()
        if not (c[i] > hmax >= c[i - 1]): continue
        for cl in range(35, 161, 5):
            cs = hs - cl
            if cs < 0: break
            cup = c[cs:hs]; rim = max(3, cl // 6)
            left = cup[:rim].max(); right = cup[-rim:].max(); bottom = cup.min(); bi = int(cup.argmin())
            depth = (left - bottom) / left
            if not (0.12 <= depth <= 0.33): continue
            if right < left * 0.95 or right > left * 1.08: continue
            if not (cl * 0.2 <= bi <= cl * 0.8): continue
            if (right - handle.min()) / right > min(0.12, depth / 2): continue
            if hmax < right * 0.97: continue
            sig["cup"] = True; break
        if "cup" in sig: break
    # VCP：趨勢模板 + 60 日內三段回檔逐段縮小且最後 <10% + 量縮 + 今收盤突破 20 日高且量放大
    if above200 and not np.isnan(ma150[i]) and c[i] > ma50[i] > ma150[i] > ma200[i] and ma200[i] > ma200[i - 20] \
            and near52 <= 25 and c[i] >= c[i - 252:i].min() * 1.3:
        segs = [c[i - 60:i - 40], c[i - 40:i - 20], c[i - 20:i]]
        dds = [(s.max() - s.min()) / s.max() for s in segs]
        if dds[0] > dds[1] > dds[2] and dds[2] < 0.10 and vol20[i - 1] < sma(v.astype(float), 50)[i - 1] \
                and c[i] > c[i - 20:i].max() and volr >= 1.3:
            sig["vcp"] = True
    meta = dict(near52=round(near52, 1), above200=bool(above200), volr=round(volr, 2), chg=round(chg * 100, 2),
                atr=round(float(a14[i]), 3) if not np.isnan(a14[i]) else None, close=round(float(c[i]), 2),
                hi52=round(float(hi252), 2), ma200=round(float(ma200[i]), 2) if not np.isnan(ma200[i]) else None,
                avgvol=round(avgvol), turnover=round(turnover))
    return sig, meta


# ---------- 指數訊號 ----------
def index_signals(o, h, l, c, v):
    n = len(c); sig = {}
    if n < 260: return sig, {}
    i = n - 1
    ma5, ma20, ma200 = sma(c, 5), sma(c, 20), sma(c, 200)
    # 一目
    conv = (h[i - 8:i + 1].max() + l[i - 8:i + 1].min()) / 2; base = (h[i - 25:i + 1].max() + l[i - 25:i + 1].min()) / 2
    sa = (conv + base) / 2; sb = (h[i - 51:i + 1].max() + l[i - 51:i + 1].min()) / 2
    pconv = (h[i - 9:i].max() + l[i - 9:i].min()) / 2; pbase = (h[i - 26:i].max() + l[i - 26:i].min()) / 2
    # 雲層用 26 日前的值（前移）
    j = i - 26
    cl_a = ((h[j - 8:j + 1].max() + l[j - 8:j + 1].min()) / 2 + (h[j - 25:j + 1].max() + l[j - 25:j + 1].min()) / 2) / 2
    cl_b = (h[j - 51:j + 1].max() + l[j - 51:j + 1].min()) / 2
    bull = c[i] > max(cl_a, cl_b) and conv > base
    pbull = c[i - 1] > max(cl_a, cl_b) and pconv > pbase
    if bull and not pbull: sig["ichimoku"] = True
    st = supertrend(h, l, c)
    if st[i] == 1 and st[i - 1] == -1: sig["supertrend"] = True
    if c[i] > ma20[i] and c[i - 1] <= ma20[i - 1]: sig["ma20_up"] = True
    rng = h - l
    if rng[i - 1] == rng[i - 7:i].min() and c[i] > h[i - 1]: sig["nr7"] = True
    if td_buy_count(c) == 9: sig["td9"] = True
    r2 = rsi(c, 2)
    if r2[i] < 10 and c[i] > ma200[i]: sig["rsi2"] = True
    if c[i] < o[i] and c[i - 1] < o[i - 1] and c[i - 2] < o[i - 2] and c[i] > ma200[i]: sig["down3"] = True
    lb = ma20[i] - 2 * np.std(c[i - 19:i + 1])
    if c[i] < lb: sig["bb_lower"] = True
    if c[i - 1] < o[i - 1] and (o[i - 1] - c[i - 1]) / o[i - 1] > 0.01 and max(o[i], c[i]) < o[i - 1] and min(o[i], c[i]) > c[i - 1]:
        sig["c_harami"] = True
    meta = dict(close=round(float(c[i]), 2), ma20=round(float(ma20[i]), 2), ma50=round(float(sma(c, 50)[i]), 2),
                ma200=round(float(ma200[i]), 2), above50=bool(c[i] > sma(c, 50)[i]))
    return sig, meta


def arr(bars):
    b = np.array([[float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5] or 0)] for x in bars if x[4] is not None])
    return b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4]


def main():
    index = load_z(os.path.join(DATA, "index.json"))
    cap_tw = load_tw_capital(); print(f"台股股本資料 {len(cap_tw)} 檔")
    picks = []; date_tw = ""; date_us = ""
    by_chunk = {}
    for s in index["stocks"]: by_chunk.setdefault(s["k"], []).append(s)
    for k, group in sorted(by_chunk.items()):
        path = os.path.join(DATA, "chunks", k + ".json")
        if not os.path.exists(path): continue
        chunk = load_z(path)
        for s in group:
            bars = chunk.get(s["c"])
            if not bars or len(bars) < 60: continue
            try:
                o, h, l, c, v = arr(bars)
                sig, meta = stock_signals(o, h, l, c, v)
            except Exception:
                continue
            d = bars[-1][0]
            if s["m"] == "TW": date_tw = max(date_tw, d)
            else: date_us = max(date_us, d)
            if not sig: continue
            if s["m"] == "TW" and (not s["c"].isdigit() or s["c"].startswith("00")): continue   # 排除 ETF、特別股
            if meta.get("close", 0) < (10 if s["m"] == "TW" else 5): continue   # 太低價不要
            if not meta["above200"] or meta["near52"] >= 20: continue           # 選股濾網：MA200 之上、距 52 週高 <20%
            if s["m"] == "TW" and (meta["avgvol"] < MIN_TW_LOTS * 1000 or meta["turnover"] < MIN_TW_TURNOVER): continue   # 量太小
            if s["m"] == "US" and meta["turnover"] < MIN_US_TURNOVER: continue
            if s["m"] == "TW":
                cp = cap_tw.get(s["c"])
                if cp is not None and cp < MIN_TW_CAPITAL: continue                 # 股本太小
                meta["capital"] = cp
            score = sum(W_STOCK[x] for x in sig) + (0.3 if meta["near52"] < 10 else (0.1 if meta["near52"] < 20 else -0.15)) + (0.1 if meta["above200"] else -0.3)
            stop = round(meta["close"] - 2 * meta["atr"], 2) if meta.get("atr") else None
            picks.append(dict(c=s["c"], n=s["n"], m=s["m"], date=d, signals=list(sig), main=any(x in MAIN for x in sig), score=round(score, 3), stop=stop, **meta))
    # 美股市值（只查候選）
    us_syms = [p["c"] for p in picks if p["m"] == "US"]
    mcap = load_us_mcap(us_syms) if us_syms else {}
    kept = []
    for p in picks:
        if p["m"] == "US":
            b = mcap.get(p["c"]); p["mcap"] = b
            if b is not None and b < MIN_US_MCAP_B: continue
        kept.append(p)
    picks = kept
    picks.sort(key=lambda p: -p["score"])
    # 指數
    ind = load_z(os.path.join(DATA, "indices.json")); idx_out = []
    for key, bars in ind["bars"].items():
        if key == "TNX" or len(bars) < 260: continue
        try:
            o, h, l, c, v = arr(bars); sig, meta = index_signals(o, h, l, c, v)
        except Exception:
            continue
        idx_out.append(dict(key=key, name=ind.get("names", {}).get(key, key), date=bars[-1][0], signals=list(sig), **meta))
    twii = next((x for x in idx_out if x["key"] == "TWII"), None)
    spx = next((x for x in idx_out if x["key"] == "GSPC"), None)
    out = dict(generated=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), date_tw=date_tw, date_us=date_us,
               market=dict(TW=twii["above50"] if twii else None, US=spx["above50"] if spx else None),
               zh=ZH, weights=W_STOCK, picks=picks, indices=idx_out)
    with open(os.path.join(DATA, "picks.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    tw = [p for p in picks if p["m"] == "TW"]; us = [p for p in picks if p["m"] == "US"]
    print(f"台股 {len(tw)} 檔、美股 {len(us)} 檔符合；指數訊號 {sum(len(x['signals']) for x in idx_out)} 個")
    # Telegram 推播
    tok, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if tok and chat:
        def fmt(p): return f"{p['c']} {p['n']}  {'、'.join(ZH[x] for x in p['signals'])}  收 {p['close']} 停損 {p['stop']} 距高 {p['near52']}% 均量 {round(p['avgvol']/1000) if p['m']=='TW' else round(p['avgvol']/1e6,1)}{'張' if p['m']=='TW' else 'M'}{(' 股本 '+str(p['capital'])+'億') if p.get('capital') else ''}{(' 市值 '+str(p['mcap'])+'B') if p.get('mcap') else ''}"
        lines = [f"📡 每日選股 {date_tw}（台股）/ {date_us}（美股）", f"大盤：加權{'在' if out['market']['TW'] else '低於'} MA50、標普{'在' if out['market']['US'] else '低於'} MA50", ""]
        lines.append(f"【台股 前 15／共 {len(tw)}】"); lines += [fmt(p) for p in tw[:15]]
        lines.append(""); lines.append(f"【美股 前 10／共 {len(us)}】"); lines += [fmt(p) for p in us[:10]]
        isig = [f"{x['name']}：{'、'.join(ZH[s] for s in x['signals'])}" for x in idx_out if x["signals"]]
        if isig: lines.append(""); lines.append("【指數訊號】"); lines += isig
        text = "\n".join(lines)[:3900]
        try:
            req = urllib.request.Request(f"https://api.telegram.org/bot{tok}/sendMessage", data=urllib.parse.urlencode(dict(chat_id=chat, text=text)).encode())
            urllib.request.urlopen(req, timeout=20); print("Telegram 已推播")
        except Exception as e:
            print("Telegram 推播失敗", e)


if __name__ == "__main__":
    main()
