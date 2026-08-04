import unittest

from scripts.analysis.extract_steady_decode import extract


class ExtractSteadyDecodeTest(unittest.TestCase):

    def test_drops_mixed_window_and_cross_checks_device(self):
        log = "\n".join([
            "old Avg generation throughput: 9.9 tokens/s, Running: 1 reqs",
            "[GLM_JAX_TRACE] traced 20 decode steps -> /tmp/trace",
            "Avg generation throughput: 0.2 tokens/s, Running: 1 reqs",
            "Avg generation throughput: 2.6 tokens/s, Running: 1 reqs",
            "Avg generation throughput: 2.7 tokens/s, Running: 1 reqs",
            "Avg generation throughput: 2.6 tokens/s, Running: 1 reqs",
            "Avg generation throughput: 2.7 tokens/s, Running: 1 reqs",
        ])
        got = extract(log, {"device_step_ms": 369.201660}, 20)
        self.assertEqual(got["dropped_mixed_window_tok_s"], 0.2)
        self.assertEqual(got["steady_samples_tok_s"], [2.6, 2.7, 2.6, 2.7])
        self.assertAlmostEqual(got["steady_median_tok_s"], 2.65)
        self.assertGreater(got["wall_to_device_ratio"], 0.9)

    def test_refuses_missing_or_mismatched_evidence(self):
        analysis = {"device_step_ms": 369.201660}
        with self.assertRaisesRegex(ValueError, "no completed"):
            extract("", analysis, 20)
        with self.assertRaisesRegex(ValueError, "unexpected trace"):
            extract("[GLM_JAX_TRACE] traced 19 decode steps", analysis, 20)
        short = "\n".join([
            "[GLM_JAX_TRACE] traced 20 decode steps",
            "Avg generation throughput: 2.7 tokens/s, Running: 1 reqs",
        ])
        with self.assertRaisesRegex(ValueError, "need >=2"):
            extract(short, analysis, 20)

    def test_refuses_wall_device_disagreement(self):
        log = "\n".join([
            "[GLM_JAX_TRACE] traced 20 decode steps",
            *["Avg generation throughput: 0.5 tokens/s, Running: 1 reqs"] * 5,
        ])
        with self.assertRaisesRegex(ValueError, "wall/device cadence mismatch"):
            extract(log, {"device_step_ms": 369.201660}, 20)


if __name__ == "__main__":
    unittest.main()
