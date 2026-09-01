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

