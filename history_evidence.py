"""Complete, versioned ledgers for the protected comparison popup."""

from datetime import datetime
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

from history_comparisons import implied


COHORTS = ("closest", "broader", "cross")
LEDGER_VERSION = 1
_QUOTE_FIELDS = (
    "side", "open_line", "open_price", "line", "price", "tickets", "money", "opening_tickets",
    "opening_money", "line_toward", "price_toward_implied", "trajectory", "path_turns",
    "path_retrace", "path_direction_changes", "captured_at", "sustained_toward",
)


def market_quote(source):
    return {key: source[key] for key in _QUOTE_FIELDS if source.get(key) is not None}


def row_evidence(row, side, selection, grade, final_score, factors, distance):
    peers = side.get("game_context") or {}
    opposite = next((q for q in row["sides"] if q["side"] != side["side"]), {})
    context = side.get("context") or {}
    known = {key: context[key] for key in ("data_badge", "reaction", "evidence_role", "context_chips")
             if context.get(key)}
    return {
        "sport": row["sport"], "id": row["event_id"], "game": row["matchup"],
        "date": row["scheduled_at"], "entry": row["entry_at"], "decision": row.get("decision_at"),
        "actual_start": row.get("verified_start"), "selection": selection, "grade": grade, "score": final_score,
        "side": market_quote(side), "opposite": market_quote(opposite),
        "peers": {market: market_quote(value) for market, value in peers.items()},
        "reference": side.get("context_reference"), "context": known, "distance": round(distance, 4),
        "matched": [label for _, label in sorted((d, label) for label, d, _ in factors if d <= .75)[:5]],
        "differences": [label for _, label in sorted(((d, label) for label, d, _ in factors if d > 1),
                                                    reverse=True)[:5]],
    }


class Ledger:
    def __init__(self):
        self.rows = []
        self.identities = {}
        self.groups = {group: [] for group in COHORTS}

    def add(self, group, row, side, selection, grade, final_score, factors, distance):
        key = (row["sport"], row["event_id"], side["side"], row["entry_at"])
        if key not in self.identities:
            self.identities[key] = len(self.rows)
            self.rows.append(row_evidence(row, side, selection, grade, final_score, factors, distance))
        self.groups[group].append(self.identities[key])

    def publish(self, store, query, sport, event, market, setup, reports):
        if not self.rows:
            return None
        for group, members in self.groups.items():
            report = reports.get(group)
            if len(members) != (report or {}).get("graded", 0):
                raise ValueError("Historical ledger does not reconcile to its published cohort")
        body = {"version": LEDGER_VERSION, "game": sport+"--"+event, "market": market, "side": query["side"],
                "observed": setup["observed"], "rows": self.rows, "groups": self.groups}
        return store.write(body)


class EvidenceStore:
    def __init__(self, path, now):
        self.path = Path(path)
        self.now = now

    def write(self, body):
        raw = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        name = "evidence--" + sha256(raw).hexdigest()[:32]
        path = self.path / (name + ".json.gz")
        if not path.exists():
            with tempfile.NamedTemporaryFile(dir=self.path, prefix=".ledger-", delete=False) as handle:
                temp = Path(handle.name)
                try:
                    handle.write(gzip.compress(raw, compresslevel=1, mtime=0))
                    handle.close()
                    temp.chmod(0o644)
                    temp.replace(path)
                finally:
                    temp.unlink(missing_ok=True)
        self.touch(name)
        return name

    def touch(self, name):
        path = self.path / (name + ".json.gz")
        try:
            os.utime(path, (self.now.timestamp(), self.now.timestamp()))
        except FileNotFoundError:
            return False
        return True

    def pin(self, payload):
        # Keep the exact supporting ledger available for redisplayed frozen
        # snapshots. A hard link keeps one copy on disk.
        target = self.path / ".frozen-ledgers"
        for name in ledger_names(payload):
            source = self.path / (name + ".json.gz")
            target.mkdir(exist_ok=True, mode=0o700)
            try:
                (target / source.name).hardlink_to(source)
            except (FileExistsError, FileNotFoundError):
                pass

    def retain(self, payload):
        for name in ledger_names(payload):
            if not self.touch(name):
                file = name + ".json.gz"
                original = self.path / ".frozen-ledgers" / file
                if original.exists():
                    (self.path / file).hardlink_to(original)
                    self.touch(name)

    def retire(self):
        # Browser manifests expire after 40 min. A user may still be viewing
        # an earlier popup. Frozen ledgers retain a separate immutable link.
        cutoff = self.now.timestamp() - 90 * 60
        for path in self.path.glob("evidence--*.json.gz"):
            if path.stat().st_mtime < cutoff:
                path.unlink()

def ledger_names(payload):
    return {side["comparison"]["ledger"] for setup in payload.get("markets", {}).values()
            for side in setup["sides"] if side.get("comparison", {}).get("ledger")}


def fair_probability(a, b):
    left, right = implied(a.get("price")), implied(b.get("price"))
    return left / (left + right) if left is not None and right is not None else None


def lead_minutes(entry, scheduled):
    return (datetime.fromisoformat(scheduled) - datetime.fromisoformat(entry)).total_seconds() / 60
