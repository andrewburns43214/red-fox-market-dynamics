import importlib.util
import unittest
from pathlib import Path


def _load_scoring_fixtures():
    fixture_path = Path("tests") / "fixtures" / "scoring_fixtures.py"
    spec = importlib.util.spec_from_file_location("scoring_fixtures", fixture_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.SCORING_FIXTURES


def _load_market_classifier():
    import scoring_reaction

    candidates = [
        "classify_reaction_market",
        "classify_market_reaction",
        "classify_scoring_fixture",
    ]

    for name in candidates:
        fn = getattr(scoring_reaction, name, None)
        if callable(fn):
            return fn

    return None


class TestScoringSpecFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = _load_scoring_fixtures()
        cls.classifier_fn = _load_market_classifier()

    def test_market_level_classifier_exists(self):
        self.assertIsNotNone(
            self.classifier_fn,
            (
                "scoring_reaction.py does not yet expose a market-level classifier for "
                "spec fixtures. Expected one of: "
                "classify_reaction_market, classify_market_reaction, "
                "or classify_scoring_fixture. This is an explicit failure because "
                "the spec is ahead of the code."
            ),
        )

    def test_all_scoring_fixtures_match_spec(self):
        if self.classifier_fn is None:
            self.fail(
                "Cannot evaluate scoring spec fixtures because no market-level "
                "classifier exists yet in scoring_reaction.py."
            )

        required_output_keys = {
            "reaction_state",
            "signal_class",
            "owning_side",
            "decision",
        }

        for fixture in self.fixtures:
            with self.subTest(fixture=fixture["name"]):
                result = type(self).classifier_fn(
                    market_rows=fixture["market_rows"],
                    evaluated_side=fixture["evaluated_side"],
                    pressure_side=fixture["pressure_side"],
                )

                self.assertIsInstance(
                    result,
                    dict,
                    f"{fixture['name']}: classifier must return a dict.",
                )

                missing_keys = required_output_keys - set(result.keys())
                self.assertFalse(
                    missing_keys,
                    (
                        f"{fixture['name']}: classifier output is missing required keys: "
                        f"{sorted(missing_keys)}. This is an explicit failure because "
                        "the scoring spec requires reaction_state, signal_class, "
                        "owning_side, and decision."
                    ),
                )

                for key, expected_value in fixture["expected"].items():
                    self.assertEqual(
                        result[key],
                        expected_value,
                        (
                            f"{fixture['name']}: expected {key}={expected_value!r}, "
                            f"got {result[key]!r}. "
                            f"Evaluated side: {fixture['evaluated_side']}. "
                            f"Pressure side: {fixture['pressure_side']}."
                        ),
                    )

    def test_opposite_ownership_fixtures_require_market_context(self):
        opposite_fixtures = [
            f for f in self.fixtures if f["expected"]["owning_side"] == "opposite"
        ]

        self.assertTrue(
            opposite_fixtures,
            "Expected at least one opposite-ownership fixture.",
        )

        for fixture in opposite_fixtures:
            with self.subTest(fixture=fixture["name"]):
                self.assertGreaterEqual(
                    len(fixture["market_rows"]),
                    2,
                    (
                        f"{fixture['name']}: opposite-side ownership fixtures must include "
                        "both sides of the market."
                    ),
                )
                self.assertIsNotNone(
                    fixture["pressure_side"],
                    (
                        f"{fixture['name']}: opposite-side ownership fixtures must declare "
                        "pressure_side explicitly."
                    ),
                )
                self.assertNotEqual(
                    fixture["evaluated_side"],
                    fixture["pressure_side"],
                    (
                        f"{fixture['name']}: evaluated_side should be the non-pressure side "
                        "for opposite-ownership validation."
                    ),
                )

    def test_stale_fixture_declares_external_validation_input(self):
        stale_fixtures = [
            f for f in self.fixtures if f["expected"]["reaction_state"] == "STALE"
        ]

        self.assertTrue(
            stale_fixtures,
            "Expected at least one STALE fixture.",
        )

        for fixture in stale_fixtures:
            with self.subTest(fixture=fixture["name"]):
                row = fixture["market_rows"][0]
                self.assertIn(
                    "consensus_current_odds",
                    row,
                    (
                        f"{fixture['name']}: STALE fixture must include explicit external "
                        "cross-book validation input such as consensus_current_odds."
                    ),
                )
                self.assertTrue(
                    row.get("market_stale", False),
                    (
                        f"{fixture['name']}: STALE fixture must explicitly declare "
                        "market_stale=True."
                    ),
                )


if __name__ == "__main__":
    unittest.main()
