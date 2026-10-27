#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投研總台 資料更新程式（GitHub Actions / 本機皆可執行）
- 以 data/index.json 內既有的股票清單為範圍（TW 2,354 檔 + US 1,520 檔）
- 透過 yfinance 抓最近 N 日 K 線，增量合併進 data/chunks/cNNN.json
- 更新 data/indices.json（加權、櫃買、S&P 500、那斯達克、費半、美債 10 年）
- 更新 data/index.json 的 date / generated
格式：chunks 與 indices 為 {"z": base64(gzip(json))}；index.json 為明文 JSON。
"""
import base64, gzip, json, os, sys, time, datetime as dt
from concurrent.futures import ThreadPoolExecutor

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    sys.exit("請先安裝套件：pip install yfinance pandas")

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
CHUNKS = os.path.join(DATA, "chunks")
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "20"))   # 每次抓最近幾個日曆日
KEEP_BARS = int(os.environ.get("KEEP_BARS", "560"))          # 每檔保留幾根 K
BATCH = 150
INDEX_MAP = {"TWII": "^TWII", "TWOII": "^TWOII", "GSPC": "^GSPC", "IXIC": "^IXIC", "SOX": "^SOX", "TNX": "^TNX"}


def load_z(path):
    with open(path, "r", encoding="utf-8") as f:
        o = json.load(f)
    if isinstance(o, dict) and "z" in o and len(o) == 1:
        return json.loads(gzip.decompress(base64.b64decode(o["z"])).decode("utf-8"))
    return o


def save_z(path, obj):
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    z = base64.b64encode(gzip.compress(raw, 9)).decode("ascii")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"z": z}, f)


def r4(x):
    return None if x is None or x != x else round(float(x), 4)


def df_to_bars(df):
    bars = []
    if df is None or df.empty:
        return bars
    for idx, row in df.iterrows():
        try:
            o, h, l, c, v = row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]
        except KeyError:
            continue
        if pd.isna(c) or pd.isna(o):
            continue
        bars.append([idx.strftime("%Y%m%d"), r4(o), r4(h), r4(l), r4(c), int(0 if pd.isna(v) else v)])
    return bars


def merge(old, new):
    if not new:
        return old
    by = {b[0]: b for b in (old or [])}
    for b in new:
        by[b[0]] = b
    out = [by[k] for k in sorted(by)]
    return out[-KEEP_BARS:]


def merge_all(old, new):
    by = {b[0]: b for b in (old or [])}
    for b in new:
        by[b[0]] = b
    return [by[k] for k in sorted(by)]


def download(tickers, start):
    """回傳 {ticker: bars}；失敗的 ticker 不在結果裡"""
    out = {}
    if not tickers:
        return out
    for attempt in range(3):
        try:
            df = yf.download(tickers, start=start, group_by="ticker", threads=True,
                             auto_adjust=False, progress=False, timeout=60)
            break
        except Exception as e:
            print("  下載失敗，重試", attempt + 1, e)
            time.sleep(5 * (attempt + 1))
    else:
        return out
    if df is None or df.empty:
        return out
    if len(tickers) == 1:
        out[tickers[0]] = df_to_bars(df)
        return out
    for t in tickers:
        try:
            sub = df[t].dropna(how="all")
            b = df_to_bars(sub)
            if b:
                out[t] = b
        except KeyError:
            pass
    return out


def yahoo_symbol(s, suffix_pref):
    if s["m"] == "US":
        return s["c"].replace(".", "-")
    return s["c"] + suffix_pref.get(s["c"], ".TW")


def main():
    t0 = time.time()
    index = load_z(os.path.join(DATA, "index.json"))
    stocks = index["stocks"]
    start = (dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
    print(f"股票 {len(stocks)} 檔，抓取起日 {start}")

    # 記住台股哪些代號在櫃買（.TWO），省掉每次重試
    pref_path = os.path.join(DATA, "tw_suffix.json")
    suffix_pref = {}
    if os.path.exists(pref_path):
        suffix_pref = json.load(open(pref_path, encoding="utf-8"))

    # 依 chunk 分組
    by_chunk = {}
    for s in stocks:
        by_chunk.setdefault(s["k"], []).append(s)

    latest_tw, latest_us, ok, fail = "", "", 0, 0
    for k in sorted(by_chunk):
        path = os.path.join(CHUNKS, k + ".json")
        chunk = load_z(path) if os.path.exists(path) else {}
        group = by_chunk[k]
        sym2stock = {yahoo_symbol(s, suffix_pref): s for s in group}
        got = {}
        syms = list(sym2stock)
        for i in range(0, len(syms), BATCH):
            got.update(download(syms[i:i + BATCH], start))
        # 台股抓不到的改試 .TWO
        retry = [s for sym, s in sym2stock.items() if sym not in got and s["m"] == "TW" and sym.endswith(".TW")]
        if retry:
            two = {s["c"] + ".TWO": s for s in retry}
            got2 = download(list(two), start)
            for sym, bars in got2.items():
                got[sym] = bars
                sym2stock[sym] = two[sym]
                suffix_pref[two[sym]["c"]] = ".TWO"
        for sym, s in sym2stock.items():
            bars = got.get(sym)
            if bars:
                chunk[s["c"]] = merge(chunk.get(s["c"], []), bars)
                ok += 1
                d = chunk[s["c"]][-1][0]
                if s["m"] == "TW":
                    latest_tw = max(latest_tw, d)
                else:
                    latest_us = max(latest_us, d)
            else:
                fail += 1
        save_z(path, chunk)
        print(f"  {k}: {len(group)} 檔完成（累計成功 {ok}、失敗 {fail}）")
    json.dump(suffix_pref, open(pref_path, "w", encoding="utf-8"))

    # 指數
    ind_path = os.path.join(DATA, "indices.json")
    ind = load_z(ind_path) if os.path.exists(ind_path) else {"names": {}, "bars": {}}
    got = download(list(INDEX_MAP.values()), (dt.date.today() - dt.timedelta(days=max(LOOKBACK_DAYS, 40))).isoformat())
    for key, sym in INDEX_MAP.items():
        if got.get(sym):
            ind["bars"][key] = merge_all(ind["bars"].get(key, []), got[sym])
            print(f"  指數 {key}: 最新 {ind['bars'][key][-1][0]}")
    save_z(ind_path, ind)

    # index.json
    d = latest_tw or latest_us or index.get("date", "").replace("-", "")
    index["date"] = f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else index.get("date")
    index["generated"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    index["src"] = "yfinance"
    with open(os.path.join(DATA, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, separators=(",", ":"))
    print(f"完成：資料日 {index['date']}，成功 {ok}、失敗 {fail}，耗時 {time.time() - t0:.0f} 秒")
    if ok == 0:
        sys.exit("沒有任何資料更新成功，請檢查網路或 Yahoo 限流")


if __name__ == "__main__":
    main()
