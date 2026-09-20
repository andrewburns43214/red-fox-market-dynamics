import json
import os
from datetime import datetime, timedelta, timezone

import pytest

import prop_projection_service as service


NOW = datetime(2026, 9, 19, 16, 0, tzinfo=timezone.utc)


def team(display, abbreviation, name, location):
    return {
        "displayName": display, "shortDisplayName": name, "name": name,
        "location": location, "slug": display.lower().replace(" ", "-"),
        "abbreviation": abbreviation,
    }


def test_team_aliases_match_complete_tokens_not_embedded_abbreviations():
    miami = team("Miami Dolphins", "MIA", "Dolphins", "Miami")
    philadelphia = team("Philadelphia Eagles", "PHI", "Eagles", "Philadelphia")
    arizona = team("Arizona Cardinals", "ARI", "Cardinals", "Arizona")
    carolina = team("Carolina Panthers", "CAR", "Panthers", "Carolina")
    kansas_city = team("Kansas City Chiefs", "KC", "Chiefs", "Kansas City")
    chicago = team("Chicago Bears", "CHI", "Bears", "Chicago")

    assert service.espn_team_matches("MIA Dolphins", miami)
    assert not service.espn_team_matches("MIA Dolphins", philadelphia)
    assert service.espn_team_matches("ARI Cardinals", arizona)
    assert not service.espn_team_matches("ARI Cardinals", carolina)
    assert service.espn_team_matches("KC Chiefs", kansas_city)
    assert not service.espn_team_matches("KC Chiefs", chicago)


class Response:
    def __init__(self, payload, headers=None, status=200):
        self.payload, self.headers, self.status_code = payload, headers or {}, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def json(self):
        return self.payload


class Session:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response(self.payload, {"X-Daily-Remaining": "999"})


def test_api_key_is_header_only_and_never_in_url(tmp_path):
    session = Session([])
    budget = service.RequestBudget(tmp_path / "quota.json")
    client = service.PropLineClient(api_key="secret-value", session=session, budget=budget)
    client.bulk_odds("football_nfl", ("player_pass_yds",))
    url, kwargs = session.calls[0]
    assert "secret-value" not in url
    assert "secret-value" not in json.dumps(kwargs.get("params", {}))
    assert kwargs["headers"] == {"X-API-Key": "secret-value"}


def test_quota_guard_stops_before_request(tmp_path):
    session = Session([])
    budget = service.RequestBudget(tmp_path / "quota.json", cap=1)
    client = service.PropLineClient(api_key="x", session=session, budget=budget)
    client.get("/sports")
    with pytest.raises(RuntimeError, match="local_daily_request_cap"):
        client.get("/sports")
    assert len(session.calls) == 1


def test_missing_key_is_clean_failure(tmp_path):
    client = service.PropLineClient(api_key="", session=Session([]), budget=service.RequestBudget(tmp_path / "quota.json"))
    with pytest.raises(RuntimeError, match="not_configured"):
        client.get("/sports")


def test_poll_cadence_changes_by_lead_time():
    def events(hours):
        return [{"commence_time": (NOW + timedelta(hours=hours)).isoformat()}]
    assert service._poll_interval_seconds(events(12), NOW) == 3600
    assert service._poll_interval_seconds(events(3), NOW) == 1800
    assert service._poll_interval_seconds(events(0.5), NOW) == 720


def test_change_hash_avoids_duplicate_observation(tmp_path):
    path = tmp_path / "observations.jsonl"
    assert service._append_changed(path, {"x": 1}, "same", None)
    assert not service._append_changed(path, {"x": 1}, "same", "same")
    assert len(path.read_text().splitlines()) == 1


def test_change_audit_is_compact_and_does_not_embed_raw_event():
    event = {"id": "1", "away_team": "A", "home_team": "B", "commence_time": NOW.isoformat(), "bookmakers": [{"key": "draftkings", "markets": [{"key": "x", "outcomes": [{}, {}]}]}]}
    summary = service._event_change_summary(event, "nfl", "abc", NOW)
    assert "event" not in summary
    assert summary["book_count"] == 1
    assert summary["market_blocks"] == 1
    assert summary["outcome_count"] == 2


def test_provider_outage_uses_cache_and_never_raises(monkeypatch, tmp_path):
    data = tmp_path / "private"
    public = tmp_path / "public.json"
    monkeypatch.setattr(service, "DATA_ROOT", data)
    monkeypatch.setattr(service, "PUBLIC_PATH", public)
    event = {"id": "1", "sport_key": "football_nfl", "away_team": "A", "home_team": "B", "commence_time": (NOW + timedelta(hours=2)).isoformat(), "bookmakers": []}
    (data / "cache").mkdir(parents=True)
    (data / "cache" / "nfl.json").write_text(json.dumps([event]))
    class Client:
        def bulk_odds(self, *_): raise RuntimeError("down")
    class Resolver:
        def prefetch(self, *_): return {"1": {}}
    payload = service.run_collection(client=Client(), resolver=Resolver(), force=True, now=NOW)
    assert payload["projections"][0]["status"] == "NOT_OPEN"
    assert payload["projections"][0]["display_status"] == "Props not open yet"
    assert "shadow" not in json.loads(public.read_text())["projections"][0]
    assert (data / "projection_v2_shadow_ledger.jsonl").exists()
    assert public.exists()
    state = json.loads((data / "state.json").read_text())
    assert state["sports"]["nfl"]["last_poll"] == NOW.isoformat()
    assert state["sports"]["nfl"]["last_error"] == "RuntimeError"


@pytest.mark.skipif(os.name == "nt", reason="Windows does not expose POSIX file modes")
def test_public_projection_is_world_readable_but_private_state_is_not_forced_public(monkeypatch, tmp_path):
    data = tmp_path / "private"
    public = tmp_path / "public.json"
    monkeypatch.setattr(service, "DATA_ROOT", data)
    monkeypatch.setattr(service, "PUBLIC_PATH", public)

    class Client:
        def bulk_odds(self, *_): return []
    class Resolver:
        def prefetch(self, *_): return {}

    service.run_collection(client=Client(), resolver=Resolver(), force=True, now=NOW)

    assert public.stat().st_mode & 0o777 == 0o644


def test_process_lock_prevents_overlap(tmp_path):
    lock = tmp_path / "collector.lock"
    with service.process_lock(lock) as first:
        assert first
        with service.process_lock(lock) as second:
            assert not second


def test_disabled_sports_are_architected_but_not_enabled():
    assert service.SPORTS["nba"]["enabled"] is False
    assert service.SPORTS["ncaab"]["enabled"] is False
    assert service.SPORTS["nhl"]["enabled"] is False
