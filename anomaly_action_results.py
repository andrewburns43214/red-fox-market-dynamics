import hashlib
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd


RESULT_COLUMNS = [
    "action_id", "decision_date_et", "sport", "game_id", "game", "market_display", "reaction",
    "observed_side", "action_side", "action_line", "action_type", "first_anomaly_seen",
    "official_decision", "official_decision_reason",
    "team1", "team1_score", "team2", "team2_score", "final_score",
    "score_source", "score_provider_event_id", "score_resolved_at",
    "graded_at_utc", "grade_evidence_hash", "outcome",
]

OFFICIAL_KEY = ["sport", "game_id", "market_display"]


def rebuild_action_results(data_dir):
    """Grade actions and publish one official result per event-market.

    The observation ledger remains append-only. Performance KPIs use only the
    first locked action for each event-market, while the complete result file
    retains every observation for lifecycle research.
    """
    data_dir = Path(data_dir)
    ledger_path = data_dir / "anomaly_action_ledger.csv"
    scores_path = data_dir / "final_scores_history.csv"
    if not ledger_path.exists():
        return 0

    actions = pd.read_csv(ledger_path, dtype=str, keep_default_na=False)
    prior_path = data_dir / "anomaly_action_results.csv"
    prior = pd.read_csv(prior_path, dtype=str, keep_default_na=False) if prior_path.exists() else pd.DataFrame()
    scores = pd.read_csv(scores_path, dtype=str, keep_default_na=False) if scores_path.exists() else pd.DataFrame()
    if not scores.empty:
        scores = scores.drop_duplicates(subset=["game_id"], keep="last")
        score_columns = ["game_id", "team1", "team1_score", "team2", "team2_score"]
        score_columns.extend(column for column in (
            "score_source", "score_provider", "score_provider_event_id", "resolved_at_utc"
        ) if column in scores.columns)
        actions = actions.merge(scores[score_columns], on="game_id", how="left")
    else:
        for column in ("team1", "team1_score", "team2", "team2_score"):
            actions[column] = ""

    actions = _mark_official_actions(actions)
    actions["outcome"] = actions.apply(_grade_action, axis=1)
    actions["final_score"] = actions.apply(_final_score, axis=1)
    if "score_source" not in actions:
        actions["score_source"] = ""
    if "score_provider" in actions:
        missing_source = actions["score_source"].fillna("").astype(str).str.strip().eq("")
        actions.loc[missing_source, "score_source"] = actions.loc[missing_source, "score_provider"]
    actions["score_source"] = actions["score_source"].fillna("").replace("", "final_scores_history.csv")
    actions["score_provider_event_id"] = actions.get(
        "score_provider_event_id", pd.Series("", index=actions.index)
    ).fillna("")
    actions["score_resolved_at"] = actions.get(
        "resolved_at_utc", pd.Series("", index=actions.index)
    ).fillna("")
    incomplete_score = ~actions.apply(_complete_score, axis=1)
    actions.loc[incomplete_score, ["score_source", "score_provider_event_id", "score_resolved_at"]] = ""
    actions["grade_evidence_hash"] = actions.apply(_grade_evidence_hash, axis=1)
    prior_grade_times = {}
    if not prior.empty and {"action_id", "grade_evidence_hash", "graded_at_utc"}.issubset(prior.columns):
        prior_grade_times = {
            (str(row["action_id"]), str(row["grade_evidence_hash"])): str(row["graded_at_utc"])
            for _, row in prior.iterrows() if str(row["grade_evidence_hash"]).strip()
        }
    graded_now = datetime.now(timezone.utc).isoformat()
    actions["graded_at_utc"] = actions.apply(
        lambda row: prior_grade_times.get(
            (str(row.get("action_id", "")), str(row.get("grade_evidence_hash", ""))), graded_now
        ) if str(row.get("grade_evidence_hash", "")).strip() else "",
        axis=1,
    )
    timestamps = pd.to_datetime(actions.get("first_anomaly_seen"), errors="coerce", utc=True)
    actions["decision_date_et"] = timestamps.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d").fillna("")
    for column in RESULT_COLUMNS:
        if column not in actions.columns:
            actions[column] = ""
    actions = actions[RESULT_COLUMNS]
    _atomic_csv(actions, data_dir / "anomaly_action_results.csv")
    official = actions[actions["official_decision"].eq("yes")].copy()
    _atomic_csv(official, data_dir / "anomaly_action_official_results.csv")

    decided = official[official["outcome"].isin(["WIN", "LOSS", "PUSH"])].copy()
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
    if not math.isfinite(score1) or not math.isfinite(score2):
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


def _mark_official_actions(actions):
    result = actions.copy()
    for column in OFFICIAL_KEY:
        if column not in result:
            result[column] = ""
    result["_first_at"] = pd.to_datetime(result.get("first_anomaly_seen"), errors="coerce", utc=True)
    result["_captured_at"] = pd.to_datetime(result.get("captured_at_utc"), errors="coerce", utc=True)
    ordered = result.sort_values(
        OFFICIAL_KEY + ["_first_at", "_captured_at", "action_id"],
        kind="mergesort", na_position="last",
    )
    official_indices = ordered.drop_duplicates(OFFICIAL_KEY, keep="first").index
    result["official_decision"] = "no"
    result["official_decision_reason"] = "Retained observation; another action was locked first for this event-market."
    result.loc[official_indices, "official_decision"] = "yes"
    result.loc[official_indices, "official_decision_reason"] = "First KPI-eligible action locked for this event-market."
    return result.drop(columns=["_first_at", "_captured_at"])


def _complete_score(row):
    try:
        values = (float(row.get("team1_score", "")), float(row.get("team2_score", "")))
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(value) for value in values)


def _final_score(row):
    if not _complete_score(row):
        return ""
    return f"{row.get('team1', '')} {row.get('team1_score', '')} - {row.get('team2', '')} {row.get('team2_score', '')}"


def _grade_evidence_hash(row):
    if str(row.get("outcome", "")) not in {"WIN", "LOSS", "PUSH"} or not _complete_score(row):
        return ""
    payload = "|".join(str(row.get(column, "")) for column in (
        "action_id", "game_id", "market_display", "action_side", "action_line",
        "team1", "team1_score", "team2", "team2_score", "score_source",
        "score_provider_event_id", "score_resolved_at", "outcome",
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_csv(frame, path):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
