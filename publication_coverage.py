"""Durable internal discovery/exclusion journal. No scoring or public row writes."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid

import pandas as pd
from dk_discovery import inventory_html

KEYS = ["sport", "game_id", "market_display"]


def active_sports(now):
    # Match the runner's host-local `date` season switches. The football
    # publication horizon itself remains explicitly America/New_York.
    mmdd = int(pd.Timestamp(now).to_pydatetime().astimezone().strftime("%m%d"))
    ranges = {"nfl": (801, 225), "ncaaf": (801, 201), "mlb": (301, 1115),
              "nba": (1001, 715), "nhl": (920, 715), "ncaab": (1001, 415)}
    return {"ufc"} | {sport for sport, (start, end) in ranges.items()
                       if (start <= mmdd <= end if start <= end else mmdd >= start or mmdd <= end)}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def keys(frame):
    return set(frame[KEYS].fillna("").astype(str).itertuples(index=False, name=None)) if len(frame) else set()


class CoverageStore:
    def __init__(self, data_dir):
        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "publication_coverage.sqlite3"
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS markets (sport TEXT, game_id TEXT, market_display TEXT,
                    payload TEXT NOT NULL, PRIMARY KEY(sport, game_id, market_display));
                CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, kind TEXT, sport TEXT,
                    started_at TEXT, finished_at TEXT, state TEXT, detail TEXT);
                CREATE TABLE IF NOT EXISTS pages (run_id TEXT, page INTEGER, sha256 TEXT,
                    url TEXT, captured_at TEXT, html_gzip BLOB, PRIMARY KEY(run_id,page,sha256));
                CREATE TABLE IF NOT EXISTS transitions (run_id TEXT, sport TEXT, game_id TEXT,
                    market_display TEXT, recorded_at TEXT, state TEXT, payload TEXT);
                CREATE INDEX IF NOT EXISTS transitions_market ON transitions(sport,game_id,market_display);
            ''')

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def begin(self, kind, sport="", now=None):
        run_id = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?)", (run_id, kind, sport, now or utcnow(), "", "RUNNING", ""))
        return run_id

    def finish(self, run_id, state, detail=""):
        with self.connect() as db:
            db.execute("UPDATE runs SET finished_at=?, state=?, detail=? WHERE run_id=?", (utcnow(), state, detail, run_id))

    def inventory(self):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT payload FROM markets")]

    def update(self, records, run_id, state):
        now = utcnow()
        with self.connect() as db:
            for record in records:
                key = tuple(str(record.get(k, "")) for k in KEYS)
                old = db.execute("SELECT payload FROM markets WHERE sport=? AND game_id=? AND market_display=?", key).fetchone()
                previous = json.loads(old[0]) if old else {}
                payload = dict(previous)
                payload.update(record)
                payload.setdefault("first_discovered_at", now)
                encoded = json.dumps(payload, default=str, sort_keys=True)
                db.execute("INSERT OR REPLACE INTO markets VALUES (?,?,?,?)", (*key, encoded))
                # Retain actual state changes, not another full historical slate
                # solely because the publication clock/run ID advanced.
                state_fields = ("state", "validation_state", "capture_exclusion_reason", "published", "publication_eligible", "in_window_scope")
                changed = any(previous.get(field) != payload.get(field) for field in state_fields)
                if state != "PUBLICATION_ACCOUNTED" or not old or changed:
                    db.execute("INSERT INTO transitions VALUES (?,?,?,?,?,?,?)", (run_id, *key, now, state, encoded))


class ScrapeCoverage:
    def __init__(self, data_dir, sport):
        self.store = CoverageStore(data_dir)
        self.sport = sport
        self.run_id = self.store.begin("SCRAPE", sport)
        self.capture_run_id = self.store.begin("CAPTURE", sport)
        self.discovered = {}

    def page(self, number, url, html):
        now = utcnow()
        raw = html.encode("utf-8")
        with self.store.connect() as db:
            db.execute("INSERT OR IGNORE INTO pages VALUES (?,?,?,?,?,?)", (
                self.run_id, number, hashlib.sha256(raw).hexdigest(), url, now, gzip.compress(raw)))
        inventory = inventory_html(html, self.sport, now, source=url)
        records = inventory.to_dict("records")
        for record in records:
            record.update(last_discovered_at=now, discovery_run_id=self.run_id,
                          capture_exclusion_reason="RAW_MARKET_PARSE_FAILED" if record["identity_verified"] else "UNRESOLVED_EVENT_IDENTITY",
                          validation_state="NOT_VALIDATED")
            self.discovered[tuple(str(record[k]) for k in KEYS)] = record
        self.store.update(records, self.run_id, "DISCOVERED")

    def validation(self, rows):
        from main import infer_market_type
        updates = {}
        for row in rows:
            market = infer_market_type(row.get("side", ""), row.get("current") or row.get("current_line", ""))
            key = (self.sport, str(row.get("game_id", "")), market)
            if key not in self.discovered:
                continue
            record = dict(self.discovered[key])
            record.update(validation_state=row.get("_validation_state", "VALIDATED"),
                          capture_exclusion_reason=row.get("_capture_exclusion_reason", ""))
            # A rejection on either side wins; partial validity cannot erase it.
            if key not in updates or record["capture_exclusion_reason"]:
                updates[key] = record
        self.store.update(list(updates.values()), self.run_id, "VALIDATED")

    def finish(self, state):
        if state == "COMPLETE" and (not self.discovered or any(
            not r.get("league_identified") or not r.get("identity_verified") or r.get("market_display") == "UNKNOWN"
            for r in self.discovered.values())):
            state = "DISCOVERY_INCOMPLETE"
        self.store.finish(self.run_id, state)

    def captured(self, state="COMPLETE", detail=""):
        self.store.finish(self.capture_run_id, state, detail)


class PublicationCoverage:
    def __init__(self, data_dir, now):
        self.store = CoverageStore(data_dir)
        self.now = pd.Timestamp(now)
        self.run_id = self.store.begin("PUBLICATION", now=self.now.isoformat())
        self.reasons = {}
        self.last_captures = {}
        self.last_pairs = {}
        self.gate_ready = set()

    def seed(self, snapshots):
        """Bootstrap retained captures without claiming a complete DK census."""
        records = snapshots.fillna("").copy()
        for key, group in records.groupby(KEYS, dropna=False):
            self.last_captures[key] = str(group.timestamp.max())
        known = {tuple(r.get(k, "") for k in KEYS) for r in self.store.inventory()}
        latest = records.sort_values("timestamp").drop_duplicates(KEYS, keep="last")
        seed = []
        for row in latest.to_dict("records"):
            key = tuple(str(row.get(k, "")) for k in KEYS)
            if key not in known:
                seed.append({k: row.get(k, "") for k in KEYS + ["game", "dk_start_iso"]} |
                            {"discovery_basis": "CAPTURED_ONLY_NOT_SOURCE_CENSUS", "identity_verified": bool(key[1]),
                             "capture_exclusion_reason": "", "validation_state": "LEGACY_CAPTURE"})
        self.store.update(seed, self.run_id, "BOOTSTRAPPED_CAPTURE")

    def stage(self, before, after, reason):
        for key in keys(before) - keys(after):
            self.reasons.setdefault(key, reason)

    def pairs(self, paired):
        for key, group in paired.groupby(KEYS, dropna=False):
            self.last_pairs[key] = str(group.timestamp.max())

    def validated(self, frame):
        blocked = {tuple(str(r.get(k, "")) for k in KEYS): r["capture_exclusion_reason"]
                   for r in self.store.inventory() if r.get("capture_exclusion_reason")}
        for key, reason in blocked.items():
            self.reasons[key] = reason
        return frame.loc[[tuple(str(row[k]) for k in KEYS) not in blocked for row in frame.to_dict("records")]].copy()

    def publish(self, board, board_path, window_filter):
        published = keys(board)
        records = []
        for item in self.store.inventory():
            key = tuple(str(item.get(k, "")) for k in KEYS)
            kickoff = pd.to_datetime(item.get("dk_start_iso", ""), utc=True, errors="coerce")
            supported = key[0] in {"nfl", "ncaaf", "nba", "ncaab", "mlb", "nhl", "ufc"} and key[2] in {"MONEYLINE", "SPREAD", "TOTAL"}
            probe = pd.DataFrame([dict(sport=key[0], dk_start_iso=kickoff)])
            in_window = bool(pd.notna(kickoff) and len(window_filter(probe, now=self.now)))
            pregame = bool(pd.notna(kickoff) and kickoff > self.now - pd.Timedelta(minutes=5))
            enabled = key[0] in active_sports(self.now)
            league_verified = item.get("discovery_basis") != "RAW_DK_HEADER" or item.get("league_identified") is True
            target = supported and enabled and league_verified and in_window and pregame
            if not league_verified:
                state = "SPORT_LEAGUE_NOT_VERIFIED"
            elif not supported:
                state = "NORMALIZATION_FAILED"
            elif pd.isna(kickoff):
                state = "KICKOFF_UNKNOWN"
            elif not in_window:
                state = "OUTSIDE_PUBLICATION_WINDOW"
            elif not pregame:
                state = "STARTED"
            elif not enabled:
                state = "SPORT_DISABLED_BY_SEASON"
            elif item.get("capture_exclusion_reason"):
                state = item["capture_exclusion_reason"]
            elif key in published:
                state = "PUBLISHED"
            else:
                state = self.reasons.get(key, "STALE_CAPTURE" if key in self.last_captures else "CAPTURE_UNAVAILABLE")
                if key in self.gate_ready:
                    state = "UNEXPLAINED_PUBLICATION_GAP"
            record = {**item, "state": state, "in_window_scope": target,
                      "publication_eligible": key in self.gate_ready, "published": key in published,
                      "last_capture_at": self.last_captures.get(key, item.get("last_capture_at", "")),
                      "last_complete_pair_at": self.last_pairs.get(key, item.get("last_complete_pair_at", "")),
                      "publication_run_id": self.run_id, "as_of": self.now.isoformat()}
            capture = pd.to_datetime(record["last_capture_at"], utc=True, errors="coerce")
            record["capture_age_minutes"] = (self.now - capture).total_seconds() / 60 if pd.notna(capture) else None
            pair_capture = pd.to_datetime(record["last_complete_pair_at"], utc=True, errors="coerce")
            record["complete_pair_age_minutes"] = (self.now - pair_capture).total_seconds() / 60 if pd.notna(pair_capture) else None
            records.append(record)
        self.store.update(records, self.run_id, "PUBLICATION_ACCOUNTED")
        frame = pd.DataFrame(records)
        summary = {"run_id": self.run_id, "as_of": self.now.isoformat(), "board_sha256": hashlib.sha256(Path(board_path).read_bytes()).hexdigest(), "sports": {}}
        for sport in ["nfl", "ncaaf", "ALL"]:
            subset = [r for r in records if sport == "ALL" or r["sport"] == sport]
            scoped = [r for r in subset if r["in_window_scope"]]
            reasons = {}
            for row in scoped:
                if row["state"] != "PUBLISHED":
                    reasons[row["state"]] = reasons.get(row["state"], 0) + 1
            summary["sports"][sport] = dict(discovered=len(subset), outside_publication_window=sum(r["state"] == "OUTSIDE_PUBLICATION_WINDOW" for r in subset),
                eligible_in_window=len(scoped), gate_eligible=sum(r["publication_eligible"] for r in scoped),
                published=sum(r["state"] == "PUBLISHED" for r in scoped), excluded_by_reason=reasons)
        with self.store.connect() as db:
            latest = db.execute("SELECT sport,state,started_at FROM runs WHERE kind='SCRAPE' ORDER BY started_at").fetchall()
        scrape_states = {s: {"state": state, "started_at": started} for s, state, started in latest}
        summary["latest_scrapes"] = scrape_states
        expected = active_sports(self.now)
        summary["expected_active_sports"] = sorted(expected)
        summary["source_census_complete"] = expected.issubset(scrape_states) and all(
            scrape_states[s]["state"] in {"COMPLETE", "EMPTY_COMPLETE"}
            and -1 <= (self.now - pd.Timestamp(scrape_states[s]["started_at"])).total_seconds() / 60 <= 25
            for s in expected)
        summary["unexplained_gaps"] = sum(r["state"] == "UNEXPLAINED_PUBLICATION_GAP" for r in records)
        summary["publication_conflicts"] = sum(r["published"] and r["state"] != "PUBLISHED" for r in records)
        for filename, content in [("publication_coverage.csv", frame.to_csv(index=False)), ("publication_coverage.json", json.dumps(summary, indent=2))]:
            temp = self.store.root / ("." + filename + ".tmp")
            temp.write_text(content, encoding="utf-8")
            temp.replace(self.store.root / filename)
        self.store.finish(self.run_id, "COMPLETE", json.dumps(summary, sort_keys=True))
        return summary
