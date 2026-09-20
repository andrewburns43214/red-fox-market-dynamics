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
    assert "prop_projection_service.py collect" in RUNNER


def test_desktop_and_mobile_header_placement_contract():
    assert ".detail-game-header .prop-score { order:3; flex:1 1 100%;" in BOARD
    assert "<div class=\"detail-game-header\">" in BOARD
    assert "<aside class=\"prop-score\"" in BOARD
    assert "Props not open yet" in BOARD
    assert "Insufficient coverage" in BOARD


def test_public_cache_is_explicitly_allowlisted():
    assert "location = /data/prop_projections.json" in NGINX
    assert "no-cache, no-store, must-revalidate" in NGINX


def test_no_secret_or_raw_prop_dump_is_shipped_to_browser():
    assert "PROPLINE_API_KEY" not in BOARD
    assert "canonical_lines" not in BOARD
    assert "_private_lines" not in BOARD

