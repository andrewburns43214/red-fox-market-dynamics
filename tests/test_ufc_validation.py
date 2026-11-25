import unittest

from main import validate_snapshot_rows


class UfcSnapshotValidationTests(unittest.TestCase):
    def test_accepts_moneyline_when_scraper_uses_current_field(self):
        rows = [
            {
                "game": "Regression Fighter A vs Regression Fighter B",
                "side": "Regression Fighter A",
                "current": "Regression Fighter A @ -150",
                "current_line": "",
            },
            {
                "game": "Regression Fighter A vs Regression Fighter B",
                "side": "Regression Fighter B",
                "current": "Regression Fighter B @ +125",
                "current_line": "",
            },
        ]

        accepted, note = validate_snapshot_rows(rows, "ufc")

        self.assertEqual(accepted, rows)
        self.assertEqual(note, "")


def test_bout_timing_and_replacement_history_are_scoped_to_exact_fighters():
    import pandas as pd
    import ufc_bout_integrity as ufc
    from collect_ufc_bouts import build_bouts

    now = pd.Timestamp("2026-12-05T23:40:00Z")
    scheduled = "2026-12-05T22:00:00Z"

    def contest(identifier, fighters, state="pre", label="STATUS_SCHEDULED"):
        return {"id": identifier, "status": {"type": {"state": state, "name": label}},
                "competitors": [{"id": athlete_id, "athlete": {"displayName": name}} for athlete_id, name in fighters]}

    original = contest("100", [("1", "Fighter Alpha"), ("2", "Fighter Charlie")])
    replacement = contest("101", [("1", "Fighter Alpha"), ("3", "Fighter Beta")])
    earlier_bout = contest("102", [("4", "Fighter Delta"), ("5", "Fighter Echo")], "in", "STATUS_IN_PROGRESS")

    def card(*bouts):
        return [{"id": "600", "date": scheduled, "name": "UFC 340", "competitions": bouts}]

    old = build_bouts(card(original, earlier_bout), {}, now - pd.Timedelta(minutes=10))
    unknown = build_bouts(card(earlier_bout), old, now - pd.Timedelta(minutes=5))
    assert unknown["100"]["phase"] == "unknown"
    current = build_bouts(card(replacement, earlier_bout), unknown, now)
    assert current["100"]["phase"] == "void"
    assert current["100"]["risks"] == ["Opponent Change"]

    def quotes(game_id, a, b, timestamp):
        return [dict(sport="ufc", game_id=game_id, game=f"{a} vs {b}", market_display="MONEYLINE",
                     side=name, dk_start_iso=scheduled, timestamp=timestamp) for name in (a, b)]

    latest = quotes("vendor", "Fighter Alpha", "Fighter Beta", now)
    frame = pd.DataFrame(latest + quotes("early", "Fighter Delta", "Fighter Echo", now)
                         + quotes("unknown", "Fighter Iota", "Fighter Omega", now))
    filtered, proof = ufc.filter_markets(frame, current, {}, now)
    assert filtered.game_id.tolist() == ["vendor", "vendor"]
    history = pd.DataFrame(quotes("vendor", "Fighter Alpha", "Fighter Charlie", now - pd.Timedelta(hours=2)) + latest)
    assert ufc.scoped_history(history, proof, current).game.unique().tolist() == ["Fighter Alpha vs Fighter Beta"]
    assert ufc.filter_markets(frame, current, {}, now + pd.Timedelta(minutes=4))[0].empty


def test_official_weigh_in_evidence_marks_exact_pair_and_weight_miss():
    import pandas as pd
    from collect_ufc_bouts import notices_from_article

    article = """
        <meta property="article:published_time" content="2026-12-04T18:00:00Z">
        <main><h4>Lightweight: Fighter Alpha (157) vs Fighter Beta (155)</h4>
        <h4>Lightweight: Fighter Delta (155) vs Fighter Echo (156)</h4>
        <p>Fighter Alpha missed weight.</p></main>
    """
    date = "2026-12-05T23:00:00Z"
    bouts = {
        "one": dict(id="one", card_id="card", card_at=date, names=["Fighter Alpha", "Fighter Beta"],
                    pair=["fighteralpha", "fighterbeta"]),
        "two": dict(id="two", card_id="card", card_at=date, names=["Fighter Delta", "Fighter Echo"],
                    pair=["fighterdelta", "fighterecho"]),
    }
    evidence = notices_from_article("https://www.ufc.com/news/test", article, bouts, pd.Timestamp("2026-12-04T19:00:00Z"))
    assert evidence["one"]["risks"] == ["Weight Miss"]
    assert evidence["two"]["weigh_in"] == "confirmed"
    assert evidence["two"]["risks"] == []
