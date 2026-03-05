dashboard_tick = ""

import argparse
import csv
import os
import datetime as dt
import hashlib
import json

try:
    import pandas as pd
except Exception as e:
    raise SystemExit(f"[FATAL] pandas required for metrics_update.py: {e}")

LOGIC_VERSION = "v1.1"

THRESH_LEAN = 60.0
THRESH_BET = 68.0
THRESH_STRONG = 72.0

ROW_STATE_PATH = os.path.join("data", "row_state.csv")
SIGNAL_LEDGER_PATH = os.path.join("data", "signal_ledger.csv")

ROW_STATE_COLS = [
    "sport","game_id","market","side",
    "logic_version",
    "last_score","last_ts","last_bucket","last_bucket_ts",
    "peak_score","peak_ts",
    "last_net_edge","last_net_edge_ts",
    "last_seen_tick"
]

LEDGER_COLS = [
    "ts_utc","logic_version","event","from_bucket","to_bucket",
    "sport","game_id","market","side",
    "game","current_line","current_odds","bets_pct","money_pct","score","net_edge","timing_bucket","tick"
]

RANK = {"": 0, "NO BET": 0, "NO_BET": 0, "LEAN": 1, "BET": 2, "STRONG_BET": 3, "STRONG": 3}

def _now_iso_utc() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _dashboard_content_tick(dash):
    cols = sorted(dash.columns.tolist())
    payload = dash[cols].to_dict(orient="records")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha256(raw).hexdigest()[:16]
    return f"dash_{h}"

def _blank(x) -> str:
    if x is None:
        return ""
    s = str(x)
    return "" if s.strip() == "" or s.strip().lower() in ("nan","none") else s.strip()

def _f(x, default=None):
    try:
        s = _blank(x)
        if s == "":
            return default
        return float(s)
    except Exception:
        return default

def _ensure_csv(path: str, cols: list[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(cols)

def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, keep_default_na=False, dtype=str)

def _append_rows(path: str, cols: list[str], rows: list[dict]):
    if not rows:
        return
    _ensure_csv(path, cols)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        for r in rows:
            w.writerow({c: _blank(r.get(c, "")) for c in cols})

def _score_bucket_from_score(score: float) -> str:
    if score is None:
        return "NO BET"
    if score >= THRESH_STRONG:
        return "STRONG_BET"
    if score >= THRESH_BET:
        return "BET"
    if score >= THRESH_LEAN:
        return "LEAN"
    return "NO BET"

def _bucket_from_row(row: dict, m: str, score: float) -> str:
    # Prefer the dashboard’s own decision/bucket fields if present (do NOT recompute if available)
    for key in (f"{m}_decision", f"{m}_bucket", f"{m}_score_bucket", f"{m}_label", "decision", "bucket"):
        v = _blank(row.get(key, ""))
        if v != "":
            v2 = v.upper().replace("_", " ")
            # normalize common forms
            if v2 in ("NO BET","NO_BET"): return "NO BET"
            if v2 in ("LEAN",): return "LEAN"
            if v2 in ("BET",): return "BET"
            if v2 in ("STRONG BET","STRONG_BET","STRONG"): return "STRONG_BET"
    # Fallback: compute from score only (used only if dashboard doesn't provide decision)
    return _score_bucket_from_score(score)

def _tick_from_row(row: dict) -> str:
    # Prefer stable run/snapshot identifiers if present
    for k in ("snapshot_id","run_id","run_ts","snapshot_ts","ts","generated_ts"):
        v = _blank(row.get(k, ""))
        if v != "":
            return v
    return ""  # defer to dashboard-level tick

def _extract_market_rows(dash: pd.DataFrame) -> list[dict]:
    # We expect a game-level dashboard with per-market columns like:
    # SPREAD_model_score, SPREAD_side, SPREAD_current_line, SPREAD_current_odds, SPREAD_bets_pct, SPREAD_money_pct, SPREAD_net_edge, etc.
    markets = []
    for m in ("SPREAD","TOTAL","MONEYLINE"):
        if f"{m}_model_score" in dash.columns or f"{m}_score" in dash.columns:
            markets.append(m)

    out = []
    for _, rr in dash.iterrows():
        row = rr.to_dict()
        sport = _blank(row.get("sport","")).upper()
        game_id = _blank(row.get("game_id",""))
        game = _blank(row.get("game",""))
        timing_bucket = _blank(row.get("timing_bucket","")).upper()
        tick = dashboard_tick

        # If sport/game_id absent, skip — metrics keys must be stable
        if sport == "" or game_id == "":
            continue

        for m in markets:
            score = _f(row.get(f"{m}_model_score", row.get(f"{m}_score","")), default=None)
            side = _blank(row.get(f"{m}_side",""))
            if side == "":
                # if this dashboard is game-level and doesn’t include side, we cannot key state reliably
                continue

            current_line = _blank(row.get(f"{m}_current_line", ""))
            current_odds = _blank(row.get(f"{m}_current_odds", ""))
            bets_pct = _blank(row.get(f"{m}_bets_pct", ""))
            money_pct = _blank(row.get(f"{m}_money_pct", ""))

            # net_edge: prefer per-market, else fall back to global net_edge
            net_edge = _f(row.get(f"{m}_net_edge", row.get("net_edge","")), default=0.0)

            bucket = _bucket_from_row(row, m, score)
            out.append({
                "sport": sport,
                "game_id": game_id,
                "market": m,
                "side": side,
                "game": game,
                "timing_bucket": timing_bucket,
                "tick": tick,
                "score": "" if score is None else f"{score:.2f}",
                "bucket": bucket,
                "net_edge": f"{(net_edge or 0.0):.2f}",
                "current_line": current_line,
                "current_odds": current_odds,
                "bets_pct": bets_pct,
                "money_pct": money_pct,
            })
    return out

def _metrics_key(r: dict) -> tuple[str,str,str,str]:
    return (r["sport"], r["game_id"], r["market"], r["side"])

def run(dashboard_csv: str, debug: bool = False) -> int:
    global dashboard_tick
    _ensure_csv(ROW_STATE_PATH, ROW_STATE_COLS)
    _ensure_csv(SIGNAL_LEDGER_PATH, LEDGER_COLS)

    dash = pd.read_csv(dashboard_csv, keep_default_na=False, dtype=str)
    dashboard_tick = _dashboard_content_tick(dash)
    if dash.empty:
        print("[metrics] dashboard empty; nothing to do")
        return 0

    rows = _extract_market_rows(dash)
    if debug:
        print(f"[metrics] extracted per-market rows: {len(rows)}")

    if not rows:
        print("[metrics] no per-market rows extracted (missing *_side or *_model_score cols?)")
        return 0

    state_df = _read_csv(ROW_STATE_PATH)
    if state_df.empty:
        state_df = pd.DataFrame(columns=ROW_STATE_COLS)

    # index existing state by stable key
    state_idx = {}
    for _, r in state_df.iterrows():
        key = (_blank(r.get("sport","")).upper(), _blank(r.get("game_id","")), _blank(r.get("market","")), _blank(r.get("side","")))
        if key[0] and key[1] and key[2] and key[3]:
            state_idx[key] = r.to_dict()

    now_ts = _now_iso_utc()
    new_state_rows = []
    ledger_rows = []

    for r in rows:
        key = _metrics_key(r)
        prev = state_idx.get(key, {})

        tick = _blank(r.get("tick",""))
        prev_tick = _blank(prev.get("last_seen_tick",""))
        if tick != "" and prev_tick != "" and tick == prev_tick:
            # idempotent on repeated runs for same snapshot tick
            continue

        score = _f(r.get("score",""), default=None)
        net_edge = _f(r.get("net_edge",""), default=0.0)
        bucket = _blank(r.get("bucket","")) or "NO BET"

        prev_bucket = _blank(prev.get("last_bucket","")) or "NO BET"
        prev_rank = RANK.get(prev_bucket.replace(" ", "_"), RANK.get(prev_bucket, 0))
        cur_rank = RANK.get(bucket.replace(" ", "_"), RANK.get(bucket, 0))

        prev_peak = _f(prev.get("peak_score",""), default=None)
        is_new_peak = (score is not None) and (prev_peak is None or score > prev_peak)

        # crossing events: only upward into LEAN/BET/STRONG_BET
        if cur_rank > prev_rank and bucket in ("LEAN","BET","STRONG_BET"):
            ledger_rows.append({
                "ts_utc": now_ts,
                "logic_version": LOGIC_VERSION,
                "event": "BUCKET_UP",
                "from_bucket": prev_bucket,
                "to_bucket": bucket,
                "sport": r["sport"],
                "game_id": r["game_id"],
                "market": r["market"],
                "side": r["side"],
                "game": r.get("game",""),
                "current_line": r.get("current_line",""),
                "current_odds": r.get("current_odds",""),
                "bets_pct": r.get("bets_pct",""),
                "money_pct": r.get("money_pct",""),
                "score": r.get("score",""),
                "net_edge": r.get("net_edge",""),
                "timing_bucket": r.get("timing_bucket",""),
                "tick": tick
            })

        if is_new_peak:
            ledger_rows.append({
                "ts_utc": now_ts,
                "logic_version": LOGIC_VERSION,
                "event": "NEW_PEAK",
                "from_bucket": "",
                "to_bucket": bucket,
                "sport": r["sport"],
                "game_id": r["game_id"],
                "market": r["market"],
                "side": r["side"],
                "game": r.get("game",""),
                "current_line": r.get("current_line",""),
                "current_odds": r.get("current_odds",""),
                "bets_pct": r.get("bets_pct",""),
                "money_pct": r.get("money_pct",""),
                "score": r.get("score",""),
                "net_edge": r.get("net_edge",""),
                "timing_bucket": r.get("timing_bucket",""),
                "tick": tick
            })

        out = dict(prev) if prev else {c:"" for c in ROW_STATE_COLS}
        out.update({
            "sport": r["sport"],
            "game_id": r["game_id"],
            "market": r["market"],
            "side": r["side"],
            "logic_version": LOGIC_VERSION,
            "last_score": "" if score is None else f"{score:.2f}",
            "last_ts": now_ts,
            "last_bucket": bucket,
            "last_bucket_ts": now_ts,
            "last_net_edge": f"{(net_edge or 0.0):.2f}",
            "last_net_edge_ts": now_ts,
            "last_seen_tick": tick or now_ts
        })

        if is_new_peak:
            out["peak_score"] = f"{score:.2f}"
            out["peak_ts"] = now_ts

        new_state_rows.append(out)
        state_idx[key] = out

    # Write outputs (append-only)
    _append_rows(ROW_STATE_PATH, ROW_STATE_COLS, new_state_rows)
    _append_rows(SIGNAL_LEDGER_PATH, LEDGER_COLS, ledger_rows)

    print(f"[metrics] row_state appended={len(new_state_rows)} ledger_appended={len(ledger_rows)}")
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dashboard", default="data/dashboard.csv")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    try:
        raise SystemExit(run(args.dashboard, debug=args.debug))
    except Exception as e:
        # non-blocking: never crash the engine; emit and exit 0
        print("[metrics] EXCEPTION (suppressed):", repr(e))
        raise SystemExit(0)

if __name__ == "__main__":
    main()
