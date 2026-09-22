from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOARD = (ROOT / "site" / "board.html").read_text(encoding="utf-8")
RUNNER = (ROOT / "run_all_sports.sh").read_text(encoding="utf-8")
NGINX = (ROOT / "deploy" / "redfox-board-locations.conf").read_text(encoding="utf-8")


def test_prop_score_is_cache_only_and_display_only():
    assert "const PROP_PROJECTION_URL = '/data/prop_projections.json';" in BOARD
    assert "propScoreHtml(r)" in BOARD
    assert "Broad prop coverage is collected" in BOARD
    assert "No spread, total, moneyline, Market Read, rank, or Favorite input" in BOARD
    assert "prop_projection_service.py collect" not in RUNNER
    prop_runner = (ROOT / "ops" / "run_prop_collection.sh").read_text(encoding="utf-8")
    assert "prop_projection_service.py collect" in prop_runner
    assert "timeout" in prop_runner


def test_desktop_and_mobile_header_placement_contract():
    assert ".detail-game-header .prop-score { order:3; flex:1 1 100%;" in BOARD
    assert "<div class=\"detail-game-header\">" in BOARD
    assert "<aside class=\"prop-score\"" in BOARD
    assert "Props not open yet" in BOARD
    assert "Insufficient coverage" in BOARD


def test_public_cache_is_explicitly_allowlisted():
    assert "location = /data/prop_projections.json" in NGINX
    assert "no-cache, no-store, must-revalidate" in NGINX


def test_all_expected_scores_show_two_decimals_and_source_age():
    assert "Number(prop.away_mean).toFixed(2)" in BOARD
    assert "Number(prop.home_mean).toFixed(2)" in BOARD
    assert "source line up to " in BOARD
    assert "ageAtPublication+elapsedMinutes" in BOARD
    assert "rushing TDs estimated from yards" in BOARD
    assert "Waiting for verified props for " in BOARD


def test_no_secret_or_raw_prop_dump_is_shipped_to_browser():
    assert "PROPLINE_API_KEY" not in BOARD
    assert "canonical_lines" not in BOARD
    assert "_private_lines" not in BOARD

