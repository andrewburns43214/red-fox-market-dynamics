from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOARD = (ROOT / "site" / "board.html").read_text(encoding="utf-8")
RUNNER = (ROOT / "run_all_sports.sh").read_text(encoding="utf-8")
NGINX = (ROOT / "deploy" / "redfox-board-locations.conf").read_text(encoding="utf-8")


def test_board_has_no_prop_projection_widget_or_request():
    assert "prop-score" not in BOARD
    assert "prop_projections.json" not in BOARD


def test_prop_collection_remains_independent_of_board_refresh():
    assert "prop_projection_service.py collect" not in RUNNER
    prop_runner = (ROOT / "ops" / "run_prop_collection.sh").read_text(encoding="utf-8")
    assert 'prop_projection_service.py "${PROP_ARGS[@]}"' in prop_runner
    assert "timeout" in prop_runner


def test_public_cache_is_explicitly_allowlisted():
    assert "location = /data/prop_projections.json" in NGINX
    assert "no-cache, no-store, must-revalidate" in NGINX


def test_no_secret_or_raw_prop_dump_is_shipped_to_browser():
    assert "PROPLINE_API_KEY" not in BOARD
    assert "canonical_lines" not in BOARD
    assert "_private_lines" not in BOARD

