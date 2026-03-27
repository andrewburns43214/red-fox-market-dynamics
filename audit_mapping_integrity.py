import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


def infer_market(side: str, current_line: str) -> str:
    s = f"{side} {current_line}".upper()
    if "OVER" in s or "UNDER" in s:
        return "TOTAL"
    if re.search(r"\s[+-]\d+(?:\.\d+)?\s*@", current_line or ""):
        return "SPREAD"
    if re.search(r"\s[+-]\d+(?:\.\d+)?\s*$", side or ""):
        return "SPREAD"
    return "MONEYLINE"


def normalize_side(market: str, side: str) -> str:
    side = (side or "").strip()
    if market == "TOTAL":
        if side.lower().startswith("over"):
            return "TOTAL_OVER"
        if side.lower().startswith("under"):
            return "TOTAL_UNDER"
        return "TOTAL_UNKNOWN"
    return re.sub(r"\s[+-]\d+(?:\.\d+)?\s*$", "", side).strip().lower()


def _to_float(value: str) -> float | None:
    try:
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except Exception:
        return None


def audit_snapshots(path: Path) -> list[str]:
    issues: list[str] = []
    if not path.exists():
        return [f"missing snapshots file: {path}"]

    groups = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            market = infer_market(row.get("side", ""), row.get("current_line", ""))
            key = (row.get("sport", ""), row.get("game_id", ""), market)
            groups[key].append(row)

    for key, rows in groups.items():
        latest_ts = max((str(r.get("timestamp", "")).strip() for r in rows), default="")
        current_rows = [r for r in rows if str(r.get("timestamp", "")).strip() == latest_ts]
        if not current_rows:
            current_rows = rows

        side_groups = defaultdict(list)
        for row in current_rows:
            side_groups[normalize_side(key[2], row.get("side", ""))].append(row)

        if key[2] in {"SPREAD", "MONEYLINE", "TOTAL"} and len(side_groups) != 2:
            issues.append(f"{key}: expected 2 normalized sides, found {len(side_groups)}")

        for norm_side, norm_rows in side_groups.items():
            raw_sides = sorted({(r.get('side') or '').strip() for r in norm_rows})
            if len(raw_sides) > 1:
                issues.append(f"{key}: normalized side {norm_side!r} has raw variants {raw_sides}")

        latest_per_side = []
        for norm_rows in side_groups.values():
            latest_per_side.append(norm_rows[-1])

        if len(latest_per_side) == 2:
            bets = [_to_float(r.get("bets_pct", "")) for r in latest_per_side]
            money = [_to_float(r.get("money_pct", "")) for r in latest_per_side]

            if all(v is not None for v in bets):
                total_bets = sum(v for v in bets if v is not None)
                if abs(total_bets - 100.0) > 2.0:
                    issues.append(f"{key}: latest paired bets_pct sums to {total_bets:.1f}, not ~100")

            if all(v is not None for v in money):
                total_money = sum(v for v in money if v is not None)
                if abs(total_money - 100.0) > 2.0:
                    issues.append(f"{key}: latest paired money_pct sums to {total_money:.1f}, not ~100")

    return issues


def audit_dashboard(path: Path) -> list[str]:
    issues: list[str] = []
    if not path.exists():
        return [f"missing dashboard file: {path}"]

    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            state = (row.get("pattern_primary") or "").strip().upper()
            decision = (row.get("game_decision") or row.get("Decision") or "").strip().upper()
            game = row.get("game", "")
            market = row.get("market_display", "")
            side = row.get("favored_side", "")
            if state == "FADE" and decision in {"BET", "LEAN", "STRONG_BET"}:
                issues.append(f"{game} | {market} | {side}: FADE state paired with {decision}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit side mapping and semantic consistency.")
    parser.add_argument("--snapshots", default="data/snapshots.csv")
    parser.add_argument("--dashboard", default="data/dashboard.csv")
    args = parser.parse_args()

    issues = []
    issues.extend(audit_snapshots(Path(args.snapshots)))
    issues.extend(audit_dashboard(Path(args.dashboard)))

    if not issues:
        print("audit passed: no mapping or semantic inconsistencies found")
        return 0

    print("audit found issues:")
    for issue in issues[:200]:
        print(f"- {issue}")
    if len(issues) > 200:
        print(f"... and {len(issues) - 200} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
