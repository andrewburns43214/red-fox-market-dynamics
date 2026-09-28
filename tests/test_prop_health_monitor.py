import json
from datetime import datetime, timedelta, timezone

import prop_health_monitor as monitor


NOW = datetime(2026, 9, 28, 21, 0, tzinfo=timezone.utc)


def projection(now=NOW, *, source_age=10, model="prop_projection_nfl_v3_best_1"):
    return {
        "sport": "nfl",
        "event_id": "phi-chi",
        "commence_time": (now + timedelta(hours=3)).isoformat(),
        "model_version": model,
        "status": "AVAILABLE",
        "away_score": 24,
        "home_score": 17,
        "oldest_observation_age_minutes": source_age,
    }


def write_payload(path, now=NOW, **overrides):
    payload = {
        "schema_version": 2,
        "generated_at": now.isoformat(),
        "collection_status": "OK",
        "failed_sports": [],
        "projections": [projection(now)],
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_healthy_current_payload_passes(tmp_path):
    path = tmp_path / "props.json"
    write_payload(path)
    report = monitor.inspect_prop_health(path, now=NOW)
    assert report["ok"] is True
    assert report["upcoming_available_checked"] == 1


def test_stale_near_game_source_and_payload_fail(tmp_path):
    path = tmp_path / "props.json"
    item = projection(NOW, source_age=84)
    write_payload(
        path,
        now=NOW - timedelta(minutes=25),
        projections=[item],
    )
    report = monitor.inspect_prop_health(path, now=NOW)
    assert report["ok"] is False
    assert any("payload stale" in issue for issue in report["issues"])
    assert any("source stale" in issue for issue in report["issues"])


def test_degraded_upstream_and_old_model_fail(tmp_path):
    path = tmp_path / "props.json"
    write_payload(
        path,
        collection_status="DEGRADED",
        failed_sports=[{"sport": "nfl", "error": "Timeout"}],
        projections=[projection(model="prop_projection_nfl_v1")],
    )
    report = monitor.inspect_prop_health(path, now=NOW)
    assert report["ok"] is False
    assert "prop upstream degraded:nfl" in report["issues"]
    assert any("wrong model" in issue for issue in report["issues"])


def test_coverage_age_is_accepted_for_recovered_v3_projection(tmp_path):
    path = tmp_path / "props.json"
    item = projection()
    item.pop("oldest_observation_age_minutes")
    item["coverage"] = {"age_minutes": 12}
    write_payload(path, projections=[item])
    assert monitor.inspect_prop_health(path, now=NOW)["ok"] is True


def test_missing_payload_fails_closed(tmp_path):
    report = monitor.inspect_prop_health(tmp_path / "missing.json", now=NOW)
    assert report == {"ok": False, "issues": ["prop payload missing"], "projection_count": 0}


def test_healthcheck_contract_includes_props():
    script = (monitor.Path(__file__).resolve().parents[1] / "ops" / "redfox-healthcheck.sh").read_text(encoding="utf-8")
    assert "prop_health_monitor.py" in script
    assert "prop collection DONE" in script
    assert "prop collection UNAVAILABLE" in script
    assert "expected production revision is not recorded" in script


def test_prop_runner_propagates_failure_exit_code():
    script = (monitor.Path(__file__).resolve().parents[1] / "ops" / "run_prop_collection.sh").read_text(encoding="utf-8")
    assert 'exit "$status"' in script
    assert 'REDFOX_PROP_FORCE' in script
    assert 'REDFOX_PROP_TIMEOUT_SECONDS:-300' in script
    assert 'prop collection SKIPPED active-lock' in script
    assert '[[ "$status" == "75" ]]' in script


def test_production_deploy_is_fast_forward_verified_and_health_gated():
    script = (monitor.Path(__file__).resolve().parents[1] / "ops" / "deploy_production.sh").read_text(encoding="utf-8")
    assert 'git merge --ff-only "origin/$BRANCH"' in script
    assert '"$PY" -m pytest -q' in script
    assert '"$PY" -m compileall -q' in script
    assert "import main, refresh_anomaly_board, prop_projection_service, prop_health_monitor" in script
    assert 'nginx -t' in script
    assert '"$ROOT/ops/run_prop_collection.sh"' in script
    assert "REDFOX_PROP_FORCE=1" in script
    assert 'REDFOX_PROP_TIMEOUT_SECONDS:-420' in script
    assert '"$ROOT/run_all_sports.sh"' in script
    assert '"$ROOT/ops/redfox-healthcheck.sh"' in script
