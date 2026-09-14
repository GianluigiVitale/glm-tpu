"""Small, stdlib-only tests: no model import, network or TPU initialization."""

from pathlib import Path
import subprocess
import tempfile
import unittest

from tools.release_inventory import closure, imports, module_name, scan


class InventoryTests(unittest.TestCase):
    def test_relative_imports(self):
        names, strings, sites = imports(
            "pkg/runtime/run.py", "from ..kernels import op\nfrom . import helper\n"
        )
        self.assertEqual(
            names,
            {"pkg.kernels", "pkg.kernels.op", "pkg.runtime", "pkg.runtime.helper"},
        )
        self.assertEqual((strings, sites), (set(), []))

    def test_package_init(self):
        self.assertEqual(module_name("pkg/runtime/__init__.py"), "pkg.runtime")
        self.assertIn(
            "pkg.runtime.helper",
            imports("pkg/runtime/__init__.py", "from . import helper")[0],
        )

    def test_cycle(self):
        self.assertEqual(
            closure(["a"], {"a": {"b"}, "b": {"a"}, "unused": set()}), {"a", "b"}
        )

    def test_missing_edge_does_not_hide_parse_report(self):
        self.assertEqual(closure(["broken"], {}), {"broken"})

    def test_dynamic_is_review_not_execution(self):
        names, strings, sites = imports(
            "pkg/run.py", 'importlib.import_module("danger")\nsubprocess.run(["x"])'
        )
        self.assertEqual(sites, [1, 2])
        self.assertEqual(names, set())
        self.assertIn("danger", strings)

    def test_syntax_error_not_silently_ignored(self):
        with self.assertRaises(SyntaxError):
            imports("broken.py", "def broken(")

    def test_scan_does_not_import_or_follow_symlink(self):
        with tempfile.TemporaryDirectory(prefix="glm-release-inventory-") as td:
            repo = Path(td)
            subprocess.run(["git", "init", "-q", td], check=True)
            (repo / "entry.py").write_text(
                'import helper\nraise RuntimeError("never import")\n'
            )
            (repo / "helper.py").write_text('asset = "manifest.json"\n')
            (repo / "manifest.json").write_text("{}")
            (repo / "link").symlink_to("/nonexistent/glm-release-test")
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                cwd=repo,
                check=True,
            )
            report = scan(repo, ("entry.py",))
            self.assertEqual(report["static_import_closure"], ["entry.py", "helper.py"])
            self.assertEqual(
                report["literal_tracked_assets"], {"helper.py": ["manifest.json"]}
            )
            self.assertEqual(
                report["findings"],
                [{"path": "link", "kind": "symlink_requires_review"}],
            )
            self.assertEqual(report["errors"], [])
            self.assertFalse(report["deletion_authorized"])

    def test_secrets_report_location_not_value(self):
        from tools.release_inventory import SECRET_PATTERNS

        token = "ghp_" + "a" * 36
        self.assertIsNotNone(SECRET_PATTERNS["github_token"].search(token.encode()))
        self.assertIsNone(SECRET_PATTERNS["github_token"].search(b"ghp_placeholder"))

    def test_named_lazy_export_keeps_its_real_module_dependency(self):
        with tempfile.TemporaryDirectory(prefix="glm-lazy-inventory-") as td:
            repo = Path(td)
            subprocess.run(["git", "init", "-q", td], check=True)
            (repo / "pkg").mkdir()
            (repo / "pkg/__init__.py").write_text(
                '_EXPORTS = {"wanted": "used", "other": "unused"}\n'
                "def __getattr__(name):\n    return _import_module(_EXPORTS[name])\n"
            )
            (repo / "pkg/used.py").write_text("wanted = 1\n")
            (repo / "pkg/unused.py").write_text("other = 2\n")
            (repo / "entry.py").write_text("from pkg import wanted\n")
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                cwd=repo,
                check=True,
            )
            report = scan(repo, ("entry.py",))
            self.assertEqual(
                report["static_import_closure"],
                ["entry.py", "pkg/__init__.py", "pkg/used.py"],
            )
            self.assertIn("pkg/__init__.py", report["dynamic_dispatch_review"])
            (repo / "entry.py").write_text("from pkg import *\n")
            self.assertIn(
                "pkg/unused.py", scan(repo, ("entry.py",))["static_import_closure"]
            )
            (repo / "entry.py").write_text("import pkg\n")
            self.assertEqual(
                scan(repo, ("entry.py",))["static_import_closure"],
                ["entry.py", "pkg/__init__.py"],
            )


if __name__ == "__main__":
    unittest.main()
