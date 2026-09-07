"""Execute both deployed guards: absence is a snapshot, never a census."""
import ast
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
MISSING = 'Specified filename /tmp/libtpu_lockfile does not exist.'
PRESENT = SimpleNamespace(st_dev=1, st_ino=2, st_mode=0o100644)
SYMLINK = SimpleNamespace(st_dev=1, st_ino=3, st_mode=0o120777)


@pytest.fixture(params=['watch_ws32_run', 'collect_ws32_worker_evidence'])
def guard(request):
    path = ROOT / 'scripts/greenfield' / (request.param + '.py')
    spec = importlib.util.spec_from_file_location(request.param, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module.REMOTE if hasattr(module, 'REMOTE') else path.read_text()
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'libtpu_holders')
    namespace = dict(Path=Path, pathlib=SimpleNamespace(Path=Path), subprocess=subprocess)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['libtpu_holders']


def setup(monkeypatch, states, rc, stdout='', stderr=''):
    sequence = iter(states)
    def lstat(path):
        assert str(path) == '/tmp/libtpu_lockfile'
        value = next(sequence)
        if isinstance(value, Exception):
            raise value
        return value
    def run(argv, **kwargs):
        assert argv == ['sudo', '-n', 'env', 'LC_ALL=C', 'fuser', '/tmp/libtpu_lockfile']
        assert kwargs == dict(capture_output=True, text=True, timeout=15)
        return SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)
    monkeypatch.setattr(Path, 'lstat', lstat)
    monkeypatch.setattr(subprocess, 'run', run)


@pytest.mark.parametrize('states,rc,stdout,stderr,expected', [
    ([FileNotFoundError(), FileNotFoundError()], 1, '', MISSING + '\n', []),
    ([PRESENT, PRESENT], 1, '', '', []),
    ([PRESENT, PRESENT], 0, ' 123 456\n', '/tmp/libtpu_lockfile:\n', [123, 456]),
])
def test_known_observations(guard, monkeypatch, states, rc, stdout, stderr, expected):
    setup(monkeypatch, states, rc, stdout, stderr)
    assert guard() == expected


@pytest.mark.parametrize('states,rc,stdout,stderr', [
    ([PermissionError()], 1, '', MISSING),
    ([FileNotFoundError(), PermissionError()], 1, '', MISSING),
    ([FileNotFoundError(), PRESENT], 1, '', MISSING),
    ([PRESENT, FileNotFoundError()], 1, '', MISSING),
    ([SYMLINK, SYMLINK], 1, '', MISSING),
    ([FileNotFoundError(), FileNotFoundError()], 1, '', MISSING + '\npermission denied'),
    ([FileNotFoundError(), FileNotFoundError()], 1, '123', MISSING),
    ([FileNotFoundError(), FileNotFoundError()], 0, '', MISSING),
    ([FileNotFoundError(), FileNotFoundError()], 1, '', ''),
    ([PRESENT, PRESENT], 2, '', ''),
    ([PRESENT, PRESENT], 1, '', 'sudo: a password is required'),
    ([PRESENT, PRESENT], 0, '', '/tmp/libtpu_lockfile:'),
    ([PRESENT, PRESENT], 0, '123', 'unexpected warning'),
    ([PRESENT, PRESENT], 1, '123', ''),
    ([PRESENT, PRESENT], 0, '-1', '/tmp/libtpu_lockfile:'),
])
def test_unknown_is_not_idle(guard, monkeypatch, states, rc, stdout, stderr):
    setup(monkeypatch, states, rc, stdout, stderr)
    with pytest.raises((OSError, RuntimeError, ValueError)):
        guard()
