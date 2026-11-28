"""Debounced Red Fox storage and archival notifications via the existing email route."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import urllib.error
import urllib.request

from history_store import atomic_text, packed


GIB = 1024 ** 3
RECIPIENT = "andrewburns43214@gmail.com"
LEVELS = (20, 17, 15)


def command(*args):
    return subprocess.run(args, capture_output=True, check=False, text=True, timeout=10).stdout


def json_messages(unit):
    raw = command("journalctl", "-u", unit, "--since", "4 hours ago", "-o", "json", "-n", "100", "--no-pager")
    for line in raw.splitlines():
        try:
            wrapper = json.loads(line)
            message = json.loads(wrapper.get("MESSAGE", ""))
            if isinstance(message, dict):
                yield int(wrapper["__REALTIME_TIMESTAMP"]) / 1e6, message
        except (TypeError, ValueError, KeyError):
            continue


def failed(name):
    return command("systemctl", "show", "-p", "ActiveState", "--value", name).strip() == "failed"


def age(path, now):
    return now - path.stat().st_mtime if path.is_file() else float("inf")


def collection_issue(root, now):
    status_path = root / "data" / "freshness.json"
    if age(status_path, now) > 65 * 60:
        return "The last successful board publication is more than 65 minutes old."
    log = Path("/var/log/redfox_update.log")
    if not log.is_file():
        return None
    with log.open("rb") as handle:
        handle.seek(max(0, log.stat().st_size - 512 * 1024))
        events = re.findall(rb"snapshot (DONE|ERROR) --sport (\w+)", handle.read())
    failures = defaultdict(int)
    for state, sport in events:
        failures[sport] = failures[sport] + 1 if state == b"ERROR" else 0
    broken = sorted(sport.decode("ascii") for sport, count in failures.items() if count >= 3)
    if broken:
        return f"At least three consecutive capture errors were recorded for {', '.join(broken)}."
    return None


def forecast(daily, current):
    data = sorted((int(day), value) for day, value in daily.items())
    if len(data) < 21 or data[-1][0] - data[0][0] < 20:
        return None
    slopes = [(a[1] - b[1]) / (b[0] - a[0]) for i, a in enumerate(data)
              for b in data[i + 1:] if b[0] - a[0] >= 7]
    if not slopes:
        return None
    burn = statistics.median(slopes)
    return (current - 15 * GIB) / burn if burn > 0 and current > 15 * GIB else None


def alerts(root, private, state, now, free):
    messages = {}
    instant = {"archive_integrity"}
    thresholds = state.setdefault("disk_levels", {})
    for level in LEVELS:
        name = f"disk_{level}"
        if free <= GIB * level:
            if name not in thresholds:
                messages[name] = f"Available disk is {free / GIB:.2f} GiB. Red Fox crossed the {level} GiB threshold."
                instant.add(name)
        elif free > GIB * (level + .5):
            thresholds.pop(name, None)
    daily = state.setdefault("daily", {})
    day = int(now // 86400)
    daily[str(day)] = free
    for key in list(daily):
        if int(key) < day - 44:
            del daily[key]
    if (days := forecast(daily, free)) is not None and days <= 365:
        messages["capacity_forecast"] = f"The recent storage trend projects reaching the 15 GiB reserve in about {round(days)} days."
    if issue := collection_issue(root, now):
        messages["collection"] = issue
    if failed("redfox-history-health.service"):
        messages["archive_integrity"] = "The automated archive integrity check failed. Inspect redfox-history-health.service."
    if failed("redfox-history-details.service") or failed("redfox-history-maintenance.service"):
        messages["archive_jobs"] = "A historical archive or maintenance service reported a failure."
    maintenance = list(json_messages("redfox-history-maintenance.service"))
    outcome_runs = [(clock, msg["outcome_reconciliation"]) for clock, msg in maintenance
                    if isinstance(msg.get("outcome_reconciliation"), dict)]
    if len(outcome_runs) >= 2 and all(
        result.get("unavailable", 0) > 0 and not result.get("verified")
        for _, result in outcome_runs[-2:]
    ):
        messages["outcomes"] = "The last two outcome reconciliation runs encountered unavailable sources without verifying a result."
    elif not outcome_runs and not failed("redfox-history-maintenance.service"):
        started = state.setdefault("monitor_started", now)
        if now - started >= 4 * 3600:
            messages["outcomes"] = "No outcome reconciliation completion marker was found in the past four hours."
    manifest = private.parent / "public" / "comparisons" / "index.json"
    if not Path("/etc/redfox-history-comparisons.disabled").exists() and (
        failed("redfox-history-public.service") or age(manifest, now) > 45 * 60
    ):
        messages["comparisons"] = "Historical comparison processing is failing or has stopped updating its publication index."
    if command("systemctl", "is-enabled", "redfox-team-intelligence.timer").strip() == "enabled":
        team_status = Path("/var/lib/redfox-team-intelligence/public/status.json")
        try:
            coverage = json.loads(team_status.read_text(encoding="utf-8"))["sports"]
        except (OSError, ValueError, KeyError):
            coverage = {}
        stale = []
        for sport in ("nfl", "ncaaf", "nba", "ncaab", "mlb", "nhl", "ufc"):
            status = coverage.get(sport, {})
            if age(team_status.with_name(sport + ".json"), now) > 4 * 3600 or not status.get("fixtures") or not status.get("completed"):
                stale.append(sport.upper())
            if status.get("active_unmapped", 0) >= max(3, status.get("active", 0) // 5):
                messages["team_intelligence_mapping"] = "Team Intelligence reported an unusual number of identity conflicts. Inspect its private health manifest."
        if stale or failed("redfox-team-intelligence.service"):
            messages["team_intelligence"] = "Team Intelligence worker failed or snapshots are missing or stale for " + (", ".join(stale) or "a worker run") + "."
        failed_sources = []
        for status in coverage.values():
            for name, source in status.get("sporting", {}).items():
                if source.get("error") and (not source.get("at") or
                        now - datetime.fromisoformat(source["at"]).timestamp() > 72 * 3600):
                    failed_sources.append(name.replace("_"," "))
        if failed_sources:
            messages["team_intelligence_supplemental_sources"] = (
                "Optional Team Intelligence sporting data sources require attention: " + ", ".join(sorted(failed_sources)) + "."
            )
        nfl = coverage.get("nfl", {})
        if nfl.get("source_error") and nfl.get("source_at"):
            if now - datetime.fromisoformat(nfl["source_at"]).timestamp() > 48 * 3600:
                messages["team_intelligence_source"] = "The Team Intelligence NFL season source has not been verified in 48 hours."
        elif nfl.get("checked") and not nfl.get("source_at"):
            messages["team_intelligence_source"] = "The Team Intelligence NFL season source could not be collected."
    backup_enabled = command("systemctl", "is-enabled", "redfox-history-backup.timer").strip() == "enabled"
    if backup_enabled:
        if (not Path("/etc/redfox-history-backup.env").is_file() or failed("redfox-history-backup.service") or
                not any(now - t < 48 * 3600 for t, _ in backup_success())):
            messages["backup_attention"] = "The enabled off-host backup is missing configuration, failed, or has no successful run in 48 hours."
    else:
        last_reminder = state.setdefault("backup_reminder", now)
        if now - last_reminder >= 90 * 86400:
            instant.add("backup_reminder")
            messages["backup_reminder"] = "Quarterly reminder: off-host backups are disabled. A same-host restore does not protect against loss of the Droplet."
    return messages, instant


def backup_success():
    value = command("systemctl", "show", "redfox-history-backup.service",
                    "-p", "ExecMainExitTimestamp", "-p", "ExecMainStatus", "-p", "Result")
    props = dict(line.split("=", 1) for line in value.splitlines() if "=" in line)
    if props.get("ExecMainStatus") == "0" and props.get("Result") == "success":
        raw = props.get("ExecMainExitTimestamp", "")
        epoch = command("date", "-d", raw, "+%s").strip() if raw else ""
        if epoch.isdigit():
            yield int(epoch), True


def configured(name):
    if value := os.environ.get(name):
        return value
    # The existing Red Fox deployment also uses shell-style declarations in
    # /etc/environment. Read the two assignments literally without sourcing it.
    env = Path("/etc/environment")
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^\s*(?:export\s+)?" + re.escape(name) + r"\s*=\s*(.*)$", line)
            if match:
                return " ".join(shlex.split(match[1], comments=True))
    return None


def deliver(subject, text, key):
    token = configured("RESEND_API_KEY")
    if not token:
        raise RuntimeError("Missing existing Resend credential in the monitoring environment")
    request = urllib.request.Request("https://api.resend.com/emails", data=packed({
        "from": configured("NOTIFY_FROM") or "Red Fox <alerts@redfoxmi.com>",
        "to": [RECIPIENT], "subject": subject, "text": text,
    }).encode("utf-8"), headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        "Idempotency-Key": key, "User-Agent": "RedFox-History-Monitor/1",
    })
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            return json.load(response)["id"]
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Resend rejected the request with HTTP {error.code}") from None


def verify(message_id):
    if not re.fullmatch(r"[a-fA-F0-9-]{36}", message_id):
        raise ValueError("Expected a Resend message ID")
    token = configured("RESEND_API_KEY")
    if not token:
        raise RuntimeError("Missing existing Resend credential")
    request = urllib.request.Request(f"https://api.resend.com/emails/{message_id}",
                                     headers={"Authorization": f"Bearer {token}", "User-Agent": "RedFox-History-Monitor/1"})
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            return json.load(response)["last_event"]
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Resend delivery lookup returned HTTP {error.code}") from None


def monitor(root, private, now):
    path = private / "monitor.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    messages, instant = alerts(root, private, state, now, shutil.disk_usage(root).free)
    pending = state.setdefault("pending", {})
    for key in set(pending) - set(messages):
        del pending[key]
    ready = {}
    for key, message in messages.items():
        entry = pending.setdefault(key, {"seen": 0, "first": int(now), "sent": False, "message": message})
        entry["seen"] += 1
        if not entry["sent"] and (key in instant or entry["seen"] >= 2):
            ready[key] = entry.get("message", message)
    try:
        if ready:
            key = hashlib.sha256(packed([(k, pending[k]["first"]) for k in sorted(ready)]).encode()).hexdigest()
            email_id = deliver("Red Fox historical infrastructure alert", "\n\n".join(ready.values()), key)
            for name in ready:
                pending[name]["sent"] = True
                if name.startswith("disk_"):
                    state["disk_levels"][name] = now
                if name == "backup_reminder":
                    state["backup_reminder"] = now
            print(packed({"delivered_to_provider": email_id, "alerts": list(ready)}))
        else:
            print(packed({"alerts": [], "watching": sorted(messages)}))
    finally:
        with atomic_text(path, packed(state)):
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/opt/red-fox-market-dynamics"))
    parser.add_argument("--state", type=Path, default=Path("/var/lib/redfox-history/private"))
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--verify")
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    if args.verify:
        print(packed({"delivery": verify(args.verify)}))
    elif args.test:
        message_id = deliver("Red Fox infrastructure monitoring delivery test",
                             "This verifies the Red Fox production email route for historical infrastructure alerts. No action is needed.",
                             f"redfox-history-delivery-test/{now:%Y-%m-%d}")
        print(packed({"provider_id": message_id}))
    else:
        monitor(args.root, args.state, now.timestamp())


if __name__ == "__main__":
    main()
