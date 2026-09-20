from pathlib import Path


BOARD = (Path(__file__).resolve().parents[1] / "site" / "board.html").read_text(encoding="utf-8")


def test_prop_projection_uses_the_four_customer_statuses():
    assert "Props-Only · " in BOARD
    assert "Props not open yet" in BOARD
    assert "Insufficient coverage" in BOARD
    assert "Prop Score unavailable" not in BOARD


def test_prop_projection_explains_only_the_strongest_anchors():
    assert "5–7 strongest anchors" in BOARD
    assert "anchors.slice(0,7)" in BOARD
    assert "Broad prop coverage is collected" in BOARD
    assert "never independently double-counted" in BOARD
