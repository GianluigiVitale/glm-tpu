"""Release CLI must remain safe on a controller with active TPU work."""

from contextlib import redirect_stdout
from importlib import metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest

from glm_tpu.cli import environment_manifest, environment_report, main


class CliTests(unittest.TestCase):
    def test_complete_metadata(self):
        pins = {
            name: value
            for group in environment_manifest()["profiles"].values()
            for name, value in group.items()
        }
        result = environment_report(
            "benchmark", version=pins.__getitem__, python_version=(3, 12)
        )
        self.assertTrue(result["passed"])
        self.assertTrue(
            any(row["expected"] == "2.10.0+cpu" for row in result["packages"])
        )

    def test_missing_and_wrong_versions_refuse(self):
        def lookup(name):
            if name == "jax":
                raise metadata.PackageNotFoundError(name)
            return "wrong"

        result = environment_report("core", version=lookup, python_version=(3, 12))
        self.assertFalse(result["passed"])
        self.assertEqual(
            {row["status"] for row in result["packages"]}, {"missing", "mismatch"}
        )

    def test_wrong_python_refuses(self):
        pins = environment_manifest()["profiles"]["core"]
        self.assertFalse(
            environment_report(
                "core", version=pins.__getitem__, python_version=(3, 13)
            )["passed"]
        )

    def test_unknown_profile_refuses(self):
        with self.assertRaises(ValueError):
            environment_report("unknown")

    def test_info_is_honest(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["info"]), 0)
        info = json.loads(output.getvalue())
        self.assertIn("not established", info["quality"])
        self.assertEqual(info["concurrent_requests"], 4)
        self.assertIn("one output-capped", info["concurrent_validation"])

    def test_no_model_import_in_fresh_process(self):
        code = """
import sys
from glm_tpu.cli import main
main(["doctor", "--profile", "tpu"])
for name in ("jax", "jaxlib", "libtpu", "torch", "transformers"):
    assert name not in sys.modules, name
"""
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_manifest_matches_declared_dependencies(self):
        repo = Path(__file__).resolve().parents[2]
        project = tomllib.loads((repo / "pyproject.toml").read_text())["project"]
        profiles = environment_manifest()["profiles"]
        for group, pins in profiles.items():
            declared = (
                project["dependencies"]
                if group == "core"
                else project["optional-dependencies"][group]
            )
            expected = {
                f"{name}=={pin.split('+')[0] if name == 'torch' else pin}"
                for name, pin in pins.items()
            }
            self.assertEqual(set(declared), expected)


if __name__ == "__main__":
    unittest.main()
