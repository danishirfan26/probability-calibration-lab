import math
import unittest

from calibration import evaluate, fit_temperature, loss, sigmoid, synthetic_batches


class CalibrationTests(unittest.TestCase):
    def test_loss_is_finite_for_large_logits(self):
        self.assertEqual(loss(1000, 0), 1000)
        self.assertEqual(loss(-1000, 1), 1000)
        self.assertTrue(math.isfinite(loss(1000, 1)))

    def test_reliability_buckets_cover_probability_extremes(self):
        result = evaluate([(-1000, 0), (0, 0), (1000, 1)], bins=2)
        self.assertEqual([row["count"] for row in result["reliability"]], [1, 2])
        self.assertAlmostEqual(result["brier_score"], 0.0833)
        self.assertEqual(sigmoid(1000), 1)

    def test_temperature_fits_validation_and_improves_heldout_loss(self):
        validation, test = synthetic_batches()
        temperature = fit_temperature(validation)
        before, after = evaluate(test), evaluate(test, temperature)
        self.assertGreater(temperature, 1)
        self.assertLess(after["log_loss"], before["log_loss"])
        self.assertEqual(after["accuracy"], before["accuracy"])

    def test_empty_data_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_temperature([])


if __name__ == "__main__":
    unittest.main()
