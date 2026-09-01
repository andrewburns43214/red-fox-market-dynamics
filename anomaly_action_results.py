import re
from pathlib import Path

import pandas as pd


RESULT_COLUMNS = [
    "action_id", "decision_date_et", "sport", "game_id", "game", "market_display", "reaction",
    "observed_side", "action_side", "action_line", "action_type", "first_anomaly_seen",
    "team1", "team1_score", "team2", "team2_score", "outcome",
]


def rebuild_action_results(data_dir):
    """Grade locked anomaly actions against final scores without changing the decision line."""
    data_dir = Path(data_dir)
    ledger_path = data_dir / "anomaly_action_ledger.csv"
    scores_path = data_dir / "final_scores_history.csv"
    if not ledger_path.exists():
        return 0

    actions = pd.read_csv(ledger_path, dtype=str, keep_default_na=False)
    scores = pd.read_csv(scores_path, dtype=str, keep_default_na=False) if scores_path.exists() else pd.DataFrame()
    if not scores.empty:
        scores = scores.drop_duplicates(subset=["game_id"], keep="last")
        actions = actions.merge(
            scores[["game_id", "team1", "team1_score", "team2", "team2_score"]],
            on="game_id", how="left",
        )
    else:
        for column in ("team1", "team1_score", "team2", "team2_score"):
            actions[column] = ""

    actions["outcome"] = actions.apply(_grade_action, axis=1)
    timestamps = pd.to_datetime(actions.get("first_anomaly_seen"), errors="coerce", utc=True)
    actions["decision_date_et"] = timestamps.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d").fillna("")
    for column in RESULT_COLUMNS:
        if column not in actions.columns:
            actions[column] = ""
    actions = actions[RESULT_COLUMNS]
    _atomic_csv(actions, data_dir / "anomaly_action_results.csv")

    decided = actions[actions["outcome"].isin(["WIN", "LOSS", "PUSH"])].copy()
    if decided.empty:
        summary = pd.DataFrame(columns=["decision_date_et", "action_type", "market_display", "n", "wins", "losses", "pushes", "win_rate_ex_push"])
    else:
        summary = decided.groupby(["decision_date_et", "action_type", "market_display"], as_index=False).agg(
            n=("outcome", "size"),
            wins=("outcome", lambda values: (values == "WIN").sum()),
            losses=("outcome", lambda values: (values == "LOSS").sum()),
            pushes=("outcome", lambda values: (values == "PUSH").sum()),
        )
        summary["win_rate_ex_push"] = summary.apply(
            lambda row: round(row.wins / (row.wins + row.losses), 4) if row.wins + row.losses else "",
            axis=1,
        )
    _atomic_csv(summary, data_dir / "anomaly_action_kpi_daily.csv")
    return len(decided)


def _grade_action(row):
    try:
        score1 = float(row.get("team1_score", ""))
        score2 = float(row.get("team2_score", ""))
    except (TypeError, ValueError):
        return "UNRESOLVED"
    market = str(row.get("market_display", "")).upper()
    side = str(row.get("action_side", "")).strip()
    if market == "TOTAL":
        number = _last_number(side)
        if number is None:
            return "UNRESOLVED"
        total = score1 + score2
        if total == number:
            return "PUSH"
        return "WIN" if ("OVER" in side.upper()) == (total > number) else "LOSS"
    if market == "SPREAD":
        match = re.match(r"^(.*)\s+([+-]\d+(?:\.\d+)?)$", side)
        if not match:
            return "UNRESOLVED"
        team, spread = match.group(1), float(match.group(2))
        pick_score, opponent_score = _team_scores(team, row, score1, score2)
        if pick_score is None:
            return "UNRESOLVED"
        adjusted = pick_score + spread
        return "PUSH" if adjusted == opponent_score else ("WIN" if adjusted > opponent_score else "LOSS")
    if market == "MONEYLINE":
        pick_score, opponent_score = _team_scores(side, row, score1, score2)
        if pick_score is None:
            return "UNRESOLVED"
        return "PUSH" if pick_score == opponent_score else ("WIN" if pick_score > opponent_score else "LOSS")
    return "UNRESOLVED"


def _team_scores(team, row, score1, score2):
    target = _normalize_team(team)
    if target == _normalize_team(row.get("team1", "")):
        return score1, score2
    if target == _normalize_team(row.get("team2", "")):
        return score2, score1
    return None, None


def _normalize_team(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower().replace(" state", " st"))


def _last_number(value):
    matches = re.findall(r"\d+(?:\.\d+)?", str(value))
    return float(matches[-1]) if matches else None


def _atomic_csv(frame, path):
    temporary = path.with_name("." + path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
