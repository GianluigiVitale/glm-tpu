#!/usr/bin/env python
"""CPU-only integrity tests for fleet XPlane aggregation."""
import copy
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import parse_xplane


DSA_CAT = "pallas: dsa_sparse_decode (sparse MLA attend)"


def fake_core(host, plane, *, steps=20, sparse=True):
    selected = []
    for i in range(steps):
        categories = {"collectives": 100_000_000_000}
        ops = {
            "all-reduce": {
                "self_ps": 100_000_000_000,
                "count": 232,
                "category": "collectives",
                "hlo_category": "all-reduce",
            }
        }
        if sparse:
            categories[DSA_CAT] = 10_000_000_000
            ops["dsa_sparse_decode"] = {
                "self_ps": 10_000_000_000,
                "count": 78,
                "category": DSA_CAT,
            }
        selected.append({
            "offset_ps": i * 210_000_000_000,
            "end_ps": i * 210_000_000_000 + 200_000_000_000,
            "duration_ps": 200_000_000_000,
            "busy_ps": sum(categories.values()),
            "per_category_ps": categories,
            "per_op": ops,
        })
    return {
        "host": host,
        "plane": f"/device:TPU:{plane}",
        "steps": selected,
        # Whole-window poison values prove fleet math uses selected steps.
        "per_category": {"whole-trace-only": {"self_ps": 10**18, "count": 1}},
        "per_op": {"whole-trace-only": {
            "self_ps": 10**18, "count": 1, "category": "whole-trace-only"}},
        "busy_ps": 10**18,
        "window_ps": 10**18,
        "outside_step_self_ps": 5_000_000_000,
    }


class FleetIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.by_file = {}
        for host_id in range(8):
            path = self.root / f"host{host_id}.xplane.pb"
            path.touch()
            host = f"worker-{host_id}"
            self.by_file[str(path)] = [
                fake_core(host, plane) for plane in range(8)
            ]

    def tearDown(self):
        self.tmp.cleanup()

    def aggregate(self):
        def load(path, _step_re):
            return copy.deepcopy(self.by_file[path])

        with mock.patch.object(parse_xplane, "aggregate_host", side_effect=load):
            return parse_xplane.aggregate_fleet(self.root)

    def test_selected_windows_and_exact_e0_gate(self):
        summary = self.aggregate()
        self.assertNotIn("whole-trace-only", summary["categories"])
        self.assertAlmostEqual(summary["device_step_ms"], 200.0)
        self.assertAlmostEqual(summary["busy_ms_per_step"], 110.0)
        self.assertAlmostEqual(summary["step_cycle_ms"], 210.0)
        self.assertEqual(summary["sparse_dsa_cores"], 64)
        parse_xplane.validate_fleet_expectations(
            summary, n_files=8, n_cores=64, n_hosts=8,
            cores_per_host=8, steps_per_core=20, arm="sparse",
            dsa_invocations_per_step=78,
            all_reduce_invocations_per_step=232,
            hlo_all_reduce_invocations_per_step=232)

    def test_hlo_all_reduce_signature_rejects_renamed_launches(self):
        summary = self.aggregate()
        with self.assertRaisesRegex(ValueError, "HLO all-reduce"):
            parse_xplane.validate_fleet_expectations(
                summary, n_files=8, n_cores=64, n_hosts=8,
                cores_per_host=8, steps_per_core=20, arm="sparse",
                hlo_all_reduce_invocations_per_step=157)

    def test_duplicate_host_rejected(self):
        last = str(self.root / "host7.xplane.pb")
        for core in self.by_file[last]:
            core["host"] = "worker-0"
        with self.assertRaisesRegex(ValueError, "duplicate host"):
            self.aggregate()

    def test_inconsistent_plane_identity_rejected(self):
        last = str(self.root / "host7.xplane.pb")
        self.by_file[last][-1]["plane"] = "/device:TPU:99"
        with self.assertRaisesRegex(ValueError, "plane identities"):
            self.aggregate()

    def test_mixed_step_count_rejected(self):
        first = str(self.root / "host0.xplane.pb")
        self.by_file[first][0]["steps"].pop()
        with self.assertRaisesRegex(ValueError, "step counts"):
            self.aggregate()

    def test_partial_fleet_fails_explicit_campaign_gate(self):
        summary = self.aggregate()
        summary["n_files"] = 1
        with self.assertRaisesRegex(ValueError, "topology mismatch"):
            parse_xplane.validate_fleet_expectations(
                summary, n_files=8, n_cores=64, n_hosts=8,
                cores_per_host=8, steps_per_core=20, arm="sparse")

    def test_arm_mixture_rejected(self):
        first = str(self.root / "host0.xplane.pb")
        self.by_file[first] = [
            fake_core("worker-0", plane, sparse=False) for plane in range(8)
        ]
        summary = self.aggregate()
        with self.assertRaisesRegex(ValueError, "DSA step coverage"):
            parse_xplane.validate_fleet_expectations(
                summary, n_files=8, n_cores=64, n_hosts=8,
                cores_per_host=8, steps_per_core=20, arm="sparse")

    def test_half_sparse_steps_rejected(self):
        first = str(self.root / "host0.xplane.pb")
        for core in self.by_file[first]:
            for step in core["steps"][10:]:
                step["per_category_ps"].pop(DSA_CAT)
                step["per_op"].pop("dsa_sparse_decode")
                step["busy_ps"] -= 10_000_000_000
        summary = self.aggregate()
        with self.assertRaisesRegex(ValueError, "DSA step coverage"):
            parse_xplane.validate_fleet_expectations(
                summary, n_files=8, n_cores=64, n_hosts=8,
                cores_per_host=8, steps_per_core=20, arm="sparse")

    def test_noncanonical_plane_set_rejected_by_campaign_gate(self):
        for cores in self.by_file.values():
            cores[-1]["plane"] = "/device:TPU:99"
        summary = self.aggregate()
        with self.assertRaisesRegex(ValueError, "device-plane topology"):
            parse_xplane.validate_fleet_expectations(
                summary, n_files=8, n_cores=64, n_hosts=8,
                cores_per_host=8, steps_per_core=20, arm="sparse")

    def test_missing_xspace_hostname_rejected(self):
        fake_xspace = mock.Mock(hostnames=[], planes=[])
        with mock.patch.object(parse_xplane, "load_xspace", return_value=fake_xspace):
            with self.assertRaisesRegex(ValueError, "one nonempty XSpace hostname"):
                parse_xplane.aggregate_host("missing-host.xplane.pb")


if __name__ == "__main__":
    unittest.main()
