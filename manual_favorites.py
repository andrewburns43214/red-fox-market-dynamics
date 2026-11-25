"""Durable, event-scoped owner decisions and their pregame/settlement projections."""

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from contextlib import closing

import pandas as pd
from mlb_fixtures import match_fixture, read_context, score_payload
import ufc_bout_integrity as ufc

VERSION = "red_fox_favorite_owner_v1"
HANDOFF = "owner_verified_pregame_handoff"
DATABASE = "manual_favorites.sqlite3"
PRODUCTION_DATA = Path("/opt/red-fox-market-dynamics/data")
PRODUCTION_STORE = Path("/var/lib/redfox-manual-favorites/decisions.sqlite3")
MARKETS = ("SPREAD", "MONEYLINE", "TOTAL")
MAX_QUOTE_AGE = timedelta(minutes=10)


def now_utc():
    return datetime.now(timezone.utc)


def stamp(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except ValueError:
        return None


def scope(row):
    return str(row.get("sport", "")), str(row.get("game_id", ""))


def sides(row):
    try:
        parsed = json.loads(row.get("market_sides") or "[]")
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []


def selected_side(market, name):
    if market == "TOTAL":
        match = re.fullmatch(r"(over|under|o|u)\s*\d+(?:\.\d+)?", name.strip(), re.I)
        return ("Over" if match.group(1).lower() in {"over", "o"} else "Under") if match else ""
    if market == "SPREAD":
        return re.sub(r"\s[+-]\d+(?:\.\d+)?$", "", name.strip())
    return name.strip()


def quote(market, name, value, observed):
    identity = selected_side(market, name)
    if not identity or not observed:
        return None
    price = re.search(r"(?:@\s*|\()([+-]\d{3,7})\)?\s*$", value) or re.fullmatch(r"([+-]\d{3,7})", value)
    if not price or not 100 <= abs(int(price.group(1))) <= 1_000_000:
        return None
    quoted_side = value[:price.start()].strip()
    if quoted_side and quoted_side != name:
        return None
    number = ""
    if market != "MONEYLINE":
        line = re.search(r"([+-]\d+(?:\.\d+)?)$", name) if market == "SPREAD" else re.search(r"(\d+(?:\.\d+)?)$", name)
        if not line:
            return None
        number = line.group(1)
    label = f"{identity} {number}" if number else identity
    canonical_price = "+100" if abs(int(price.group(1))) == 100 else price.group(1)
    token = hashlib.sha256(json.dumps([market, identity, number, canonical_price]).encode()).hexdigest()
    return dict(market=market, side=identity, label=label, line=number, price=price.group(1),
                observed=observed.isoformat(), token=token)


def csv_rows(path):
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            yield from csv.DictReader(stream)
    except FileNotFoundError:
        return


def latest_event_capture(path, event_rows, now):
    sport, event_id = scope(event_rows[0])
    game = event_rows[0]["game"]
    start = stamp(event_rows[0].get("kickoff_iso"))
    columns = ["timestamp", "sport", "game_id", "game", "side", "current_line",
               "dk_start_iso", "bets_pct", "money_pct"]
    latest, raw = None, []
    try:
        # Parse the CSV in bounded batches and allocate row objects only for the
        # requested game in a recent capture. Keep CSV quoting and record order.
        with pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False, chunksize=32768) as captures:
            for chunk in captures:
                sport_rows = chunk.loc[chunk["sport"].eq(sport)]
                for captured_at, batch in sport_rows.groupby("timestamp", sort=False):
                    captured = stamp(captured_at)
                    if not captured or not timedelta(0) <= now - captured <= MAX_QUOTE_AGE or (latest and captured < latest):
                        continue
                    if captured != latest:
                        latest, raw = captured, []
                    matches = batch.loc[batch["game_id"].eq(event_id) & batch["game"].eq(game)]
                    raw.extend(row for row in matches.to_dict("records") if stamp(row["dk_start_iso"]) == start)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None, []
    return latest, raw


def latest_quotes(data_dir, event_rows, now):
    """Read the newest capture for this sport. A missing/incomplete pair is unavailable."""
    offer = {}
    if not event_rows:
        return offer
    latest, raw = latest_event_capture(Path(data_dir) / "snapshots.csv", event_rows, now)
    for board in event_rows:
        market = board["market_display"]
        if board.get("data_badge", "").lower() != "clean" or yes(board.get("cross_market_mismatch")) or yes(board.get("cross_market_opener_mismatch_verified")):
            continue
        expected = {selected_side(market, s.get("flagged_side", "")) for s in sides(board)}
        if len(expected) != 2 or "" in expected:
            continue
        found = {}
        conflicts = False
        for row in raw:
            name = row["side"]
            kind = ("TOTAL" if re.fullmatch(r"(over|under|o|u)\s*\d+(?:\.\d+)?", name, re.I)
                    else "SPREAD" if re.search(r"\s[+-]\d+(?:\.\d+)?$", name) else "MONEYLINE")
            if kind != market:
                continue
            item = quote(market, name, row["current_line"], latest)
            if not item or item["side"] not in expected:
                conflicts = True
                continue
            previous = found.get(item["side"])
            item.update(bets_pct=row["bets_pct"], money_pct=row["money_pct"])
            if previous and previous != item:
                conflicts = True
            found[item["side"]] = item
        if not conflicts and found.keys() == expected:
            if market == "TOTAL" and len({q["line"] for q in found.values()}) != 1:
                continue
            if market == "SPREAD" and sum(float(q["line"]) for q in found.values()) != 0:
                continue
            offer.update({q["token"]: (q, board) for q in found.values()})
    return offer


def yes(value):
    return str(value).lower() in {"true", "yes", "1"}


def db_path(data_dir):
    data_dir = Path(data_dir).resolve()
    return PRODUCTION_STORE if data_dir == PRODUCTION_DATA else data_dir / DATABASE


def connect(data_dir):
    conn = sqlite3.connect(db_path(data_dir), timeout=4)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY,
            sport TEXT NOT NULL,
            event TEXT NOT NULL,
            saved_at TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            owner_email TEXT NOT NULL,
            action TEXT NOT NULL,
            reason TEXT NOT NULL,
            payload TEXT NOT NULL,
            previous TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS event_decisions ON decisions(sport, event, id)")
    return conn


def decisions(data_dir, event=None):
    path = db_path(data_dir)
    if not path.exists():
        return {}
    sql = "SELECT * FROM decisions WHERE (sport,event,id) IN (SELECT sport,event,max(id) FROM decisions GROUP BY sport,event)"
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql + (" AND sport=? AND event=?" if event else ""), event or ()).fetchall()
    return {(row["sport"], row["event"]): {**json.loads(row["payload"]), "revision": row["id"],
                                          "action": row["action"], "saved_at": row["saved_at"]}
            for row in rows}


def snapshot_revision(data_dir):
    state = (Path(data_dir) / "snapshots.csv").stat()
    return state.st_ino, state.st_size, state.st_mtime_ns


def save(data_dir, event, revision, owner, action, reason, previous, payload, capture=None):
    with closing(connect(data_dir)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        old = conn.execute("SELECT id FROM decisions WHERE sport=? AND event=? ORDER BY id DESC LIMIT 1", event).fetchone()
        if (old["id"] if old else 0) != revision:
            return None
        at = now_utc()
        status_at = stamp(payload["fixture"]["observed"])
        start = stamp(payload["fixture"].get("start"))
        entry = payload.get("quote")
        if (not status_at or not timedelta(0) <= at - status_at <= timedelta(seconds=30)
                or (start and at >= start)
                or (capture is not None and capture != snapshot_revision(data_dir))
                or (entry and not timedelta(0) <= at - stamp(entry["observed"]) <= MAX_QUOTE_AGE)):
            return None
        record = conn.execute("INSERT INTO decisions(sport,event,saved_at,owner_id,owner_email,action,reason,payload,previous) VALUES (?,?,?,?,?,?,?,?,?)",
                              (*event, at.isoformat(), owner["id"], owner["email"], action, reason,
                               json.dumps(payload), json.dumps(previous)))
        return record.lastrowid


def event_rows(data_dir, event):
    return [r for r in csv_rows(Path(data_dir) / "anomaly_board.csv")
            if scope(r) == event and r.get("market_display") in MARKETS]


def official_board_choice(rows):
    return next(({"market": row["market_display"], "side": row["favorite_side"],
                  "origin": "Manual" if row.get("favorite_origin") == "manual" else "Automatic"}
                 for row in rows if yes(row.get("red_fox_favorite"))), None)


def mark_selected(row, decision):
    q = decision["quote"]
    return dict(row, red_fox_favorite="true", favorite_origin="manual", favorite_side=q["label"],
                favorite_rule_version=VERSION, favorite_pathway="owner_manual",
                favorite_state="qualified", favorite_reason="Manual Red Fox Favorite.",
                favorite_late_invalidated="false", favorite_late_invalidated_at="",
                favorite_late_invalidated_reason="", favorite_review_state="", favorite_review_reason="",
                favorite_review_at="", favorite_first_qualified_at=decision["saved_at"],
                favorite_final_qualified_at=decision["saved_at"], manual_entry_price=q["price"],
                manual_entry_line=q["line"], manual_quote_at=q["observed"],
                manual_revision=str(decision["revision"]))


def same_fixture(row, decision):
    return row.get("game") == decision["game"] and stamp(row.get("kickoff_iso")) == stamp(decision["board_start"])


def project_board(board, data_dir, now, ufc_proofs=None):
    active = {key: value for key, value in decisions(data_dir).items() if value["action"] != "restore"}
    if not active or board.empty:
        return board
    result = board.copy()
    for index, row in board.iterrows():
        decision = active.get(scope(row))
        if not decision or row.get("game") != decision["game"]:
            continue
        result.at[index, "manual_revision"] = str(decision["revision"])
        if (decision["action"] == "remove" or not same_fixture(row, decision)
                or row.get("market_display") != decision["quote"]["market"]):
            result.at[index, "red_fox_favorite"] = "false"
            result.at[index, "favorite_side"] = ""
            continue
        q = decision["quote"]
        start = stamp(decision["fixture"].get("start"))
        names = [selected_side(q["market"], side.get("flagged_side", "")) for side in sides(row)]
        current = stamp(row.get("state_as_of_utc"))
        proof = (ufc_proofs or {}).get(scope(row)[1])
        wrong_bout = scope(row)[0] == "ufc" and (
            not ufc.allowed_pre(proof) or proof["bout_id"] != decision["fixture"]["id"]
            or proof["athlete_ids"] != decision["fixture"]["proof"]["athlete_ids"]
        )
        if ((start and now >= start) or names.count(q["side"]) != 1
                or not current or not timedelta(0) <= now - current <= timedelta(minutes=30)
                or str(row.get("data_badge", "")).lower() in {"retained", "unverified"}
                or yes(row.get("cross_market_mismatch")) or yes(row.get("cross_market_opener_mismatch_verified"))
                or wrong_bout):
            result.at[index, "red_fox_favorite"] = "false"
            result.at[index, "favorite_side"] = ""
            continue
        for field, value in mark_selected({}, decision).items():
            result.at[index, field] = value
    return result


def ufc_handoff(decision, raw, bouts, notices, now):
    pinned = decision["fixture"]["proof"]
    bout = bouts.get(pinned["bout_id"])
    if (not bout or bout["phase"] not in {"live", "final", "void"}
            or bout["pair"] != pinned["pair"] or bout["athlete_ids"] != pinned["athlete_ids"]
            or bout["card_id"] != pinned["card_id"]):
        return None
    lo, hi = ufc.cutoff_bounds(bout)
    saved = stamp(decision["saved_at"])
    if bout["phase"] == "void":
        return max(saved + timedelta(microseconds=1), stamp(decision["board_start"])), ufc.score_payload(bout), pinned
    if pd.isna(lo) or lo != hi or not saved < lo <= now or ufc.risks_for(bout, notices, lo):
        return None
    proof = ufc.read_market_proof(raw) if raw is not None else {}
    final_source, checked = stamp(proof.get("source_at")), stamp(proof.get("verified_pre_at"))
    if not (proof.get("bout_id") == bout["id"] and proof.get("athlete_ids") == bout["athlete_ids"]
            and proof.get("pair") == bout["pair"] and proof.get("card_id") == bout["card_id"]
            and ufc.allowed_pre(proof) and final_source and checked
            and final_source <= checked < lo and lo - final_source <= timedelta(minutes=30)):
        entry = stamp(decision["quote"]["observed"])
        pre = stamp(decision["fixture"]["pre_observed"])
        if not entry <= pre < lo or lo - entry > timedelta(minutes=30):
            return None
        proof = pinned
    return lo.to_pydatetime(), ufc.score_payload(bout), proof


def freeze_overrides(live, market_candidates, data_dir, bouts, notices, now):
    """Only the last saved decision for each exact fixture can enter settlement."""
    active = {key: value for key, value in decisions(data_dir).items() if value["action"] != "restore"}
    if not active:
        return live
    result = live.copy()
    already = set()
    for index, row in result.iterrows():
        decision = active.get(scope(row))
        if not decision:
            continue
        if str(row.get("manual_revision", "")) == str(decision["revision"]) and row.get("freeze_method") in {HANDOFF, ufc.HANDOFF}:
            already.add(scope(row))
        elif row.get("game") == decision["game"] or row.get("score_provider_event_id") == decision["fixture"]["id"]:
            result.at[index, "red_fox_favorite"] = "false"
            result.at[index, "favorite_side"] = ""
    frozen = []
    for event, decision in active.items():
        if event in already or decision["action"] == "remove":
            continue
        saved, fixture, q = stamp(decision["saved_at"]), decision["fixture"], decision["quote"]
        start = stamp(fixture.get("start"))
        method, result_fields, proof = HANDOFF, {}, {}
        if fixture["source"] == "ufc":
            raw = next((r for r in market_candidates.to_dict("records") if scope(r) == event and r.get("market_display") == "MONEYLINE"), None)
            evidence = ufc_handoff(decision, raw, bouts, notices, now)
            if not evidence:
                continue
            start, result_fields, proof = evidence
            method = ufc.HANDOFF
        if not start or not saved < start <= now or now - start > timedelta(hours=10):
            continue
        row = mark_selected(decision["row"], decision)
        # Freeze the exact saved number and price. Never reuse an automatic entry
        # or substitute a later quote for a number the owner explicitly saved.
        captured_side = next((s for s in sides(row) if selected_side(q["market"], s.get("flagged_side", "")) == q["side"]), {})
        entry_display = q["price"] if q["market"] == "MONEYLINE" else f'{q["line"]} ({q["price"]})'
        evidence = {ufc.PROOF: proof} if proof else {}
        row.update(result_fields, market_sides=json.dumps([dict(captured_side, flagged_side=q["label"],
                   current_line=entry_display, source_latest_at=q["observed"],
                   bets_pct=q.get("bets_pct"), money_pct=q.get("money_pct"))]),
                   favorite_supporting_evidence=json.dumps(evidence), flagged_side=q["label"],
                   state_as_of_utc=q["observed"], final_pregame_state_at_utc=decision["saved_at"],
                   kickoff_iso=start.isoformat(), frozen_at_utc=now.isoformat(), freeze_method=method,
                   manual_provider=fixture["source"], manual_provider_event_id=fixture["id"],
                   manual_provider_start=fixture.get("provider_start") or decision["board_start"])
        frozen.append(row)
    return pd.concat([result, pd.DataFrame(frozen)], ignore_index=True, sort=False) if frozen else result


def manual_score(row, scores, data_dir, now):
    """Grade only against the provider event pinned before the owner saved."""
    identity = str(row.get("manual_provider_event_id", ""))
    start = stamp(row.get("manual_provider_start"))
    if row.get("manual_provider") == "mlb":
        fixture = (scores or {}).get(identity)
        if not fixture:
            return None
        if (stamp(fixture["scheduled_start"]) == start and "cancel" in fixture.get("status", "").lower()):
            return dict(score_state="post", score_status="Canceled", score_away="-", score_home="-",
                        score_status_code="STATUS_CANCELED", score_provider="mlb", score_provider_event_id=identity)
        verified = match_fixture(scores, row.get("game"), start, game_pk=identity)
        return score_payload(verified)
    from build_live_recent import exact_scores

    for item in exact_scores(scores, row):
        if str(item.get("score_provider_event_id")) == identity and stamp(item.get("event_time")) == start:
            return item
    return None
