"""Site configuration: the one place a deployment's private values live (DESIGN 5.6).

A site file is untracked TOML (``$GLM_TPU_SITE_CONFIG``, else ``$GLM_TPU_CONFIG_ROOT/site.toml``,
default ``~/.config/glm-tpu/site.toml``); ``examples/site.example.toml`` documents every key.
Loading is fail-closed: the file must be a regular, owner-only file (mode 0600 or 0400) owned by
the caller, the key sets are exact (an unknown key or a missing required key refuses and names the
key), paths are absolute after ``~`` expansion and never contain ``..``, checkpoint children lie
strictly inside their namespaces, pins are 64-hex digests and the fleet shape is the one this
engine runs (8 hosts x 4 chips, coordinator port 8476).

Precedence (vLLM layering): command-line flag > ``GLM_TPU_*`` environment variable > site file >
built-in default. Only generic, non-private values have defaults. Fleet values and the launch
policy exist only in the site file.

The controller stages the *resolved* configuration to every host as canonical JSON ``site.json``
(controller-only keys -- ``fleet.known_hosts``, ``paths.repo`` and ``[launch]`` -- omitted) and
passes its SHA-256 as ``--site-sha256``; workers never read ``$HOME``: they re-hash the staged file
and parse it with :meth:`SiteConfig.from_resolved_json`, which runs the same validation.

A process that packs, verifies or loads a checkpoint, or answers a request, installs its site with
:func:`set_current_site` (the executor, worker and pack worker entry points do); code that needs a
site value but receives no site argument (the approved source-bucket prefix of the checkpoint
format, the tokenizer location of the worker's answer writer) reads it with
:func:`get_current_site`, which refuses when none is installed.

Standard library only; importing this module never imports JAX.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

SCHEMA = "glm_tpu_site_v1"
RESOLVED_SCHEMA = "glm_tpu_site_resolved_v1"
NUM_HOSTS = 8
CHIPS_PER_HOST = 4
COORDINATOR_PORT = "8476"
DEFAULT_HOST_RANK_REGEX = r"-w-(\d+)$"
DEFAULT_HLO_DUMP_ROOT = "/dev/shm/glm-tpu-hlo"
DEFAULT_KNOWN_HOSTS = "~/.ssh/google_compute_known_hosts"
SITE_FILE_CAP = 64 << 10

_HEX64 = re.compile(r"[0-9a-f]{64}")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_PROJECT = re.compile(r"[a-z0-9][a-z0-9:._-]*")
_COMMAND = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_PLAIN_PATH = re.compile(r"/[A-Za-z0-9._+/@=,-]*")
_GS_PREFIX = re.compile(r"gs://[a-z0-9][a-z0-9._-]*/(?:[A-Za-z0-9._-]+/)*")


class SiteConfigError(ValueError):
    """The site configuration is missing, unsafe or invalid (the message names the key)."""


def host_rank(hostname: str, regex: str = DEFAULT_HOST_RANK_REGEX) -> int | None:
    """The rank a TPU-VM hostname encodes (the regex's one group), or None."""
    match = re.search(regex, hostname) if isinstance(hostname, str) else None
    return int(match.group(1)) if match else None


def rank_matches(hostname: str, rank: int, regex: str = DEFAULT_HOST_RANK_REGEX) -> bool:
    """Whether ``hostname`` names exactly ``rank`` (the digits equal ``str(rank)``: no leading zeros)."""
    match = re.search(regex, hostname) if isinstance(hostname, str) else None
    return match is not None and match.group(1) == str(rank)


# ----------------------------------------------------------------------------- sections
@dataclass(frozen=True)
class FleetConfig:
    """One TPU v4-64 slice: ``num_hosts`` TPU VMs with ``chips_per_host`` chips each."""

    tpu_name: str
    zone: str
    project: str
    num_hosts: int
    chips_per_host: int
    host_rank_regex: str
    coordinator_address: str
    worker_python: str
    worker_pythonpath: tuple[str, ...]
    helper_python: str
    known_hosts: Path | None  # controller only

    def host_rank(self, hostname: str) -> int | None:
        """The rank a TPU-VM hostname encodes (``host_rank_regex``), or None."""
        return host_rank(hostname, self.host_rank_regex)

    def rank_matches(self, hostname: str, rank: int) -> bool:
        return rank_matches(hostname, rank, self.host_rank_regex)


@dataclass(frozen=True)
class PathsConfig:
    repo: Path | None  # controller only; None: the checkout that contains this package
    run_root: Path
    model_path: Path
    hlo_dump_root: Path


@dataclass(frozen=True)
class CheckpointConfig:
    namespace: Path
    root: Path
    inventory_namespace: Path
    source_inventory: Path
    source_inventory_sha256: str
    manifest_sha256: str
    success_sha256: str
    source_complete_sha256: str


@dataclass(frozen=True)
class TopologyConfig:
    binding_dir: Path
    binding_sha256: str
    capture_root: Path
    topology_sha256: str
    topology_fleet_sha256: str
    mesh_sha256: str
    slice_name: str


@dataclass(frozen=True)
class StorageConfig:
    source_uri: str
    allowed_source_uri_prefixes: tuple[str, ...]

    def approved(self, uri: Any) -> bool:
        return isinstance(uri, str) and any(uri.startswith(p) for p in self.allowed_source_uri_prefixes)


@dataclass(frozen=True)
class LocksConfig:
    workload: tuple[Path, ...]  # LOCK_EX | LOCK_NB: a live model owner refuses the launch
    sync: tuple[Path, ...]      # LOCK_EX blocking: a repository backup delays staging; released after it


@dataclass(frozen=True)
class LaunchPolicy:
    """Which checkouts may launch (built only from the site file's ``[launch]`` table)."""

    allowed_branches: tuple[str, ...] = ("main", "release/*")
    expected_origin: str | None = None
    require_clean: bool = True
    require_pushed: bool = True


@dataclass(frozen=True)
class SiteConfig:
    fleet: FleetConfig
    paths: PathsConfig
    checkpoint: CheckpointConfig
    topology: TopologyConfig
    storage: StorageConfig
    locks: LocksConfig
    launch: LaunchPolicy | None = field(default=None)  # controller only

    # ------------------------------------------------------------------ constructors
    @classmethod
    def load(cls, path: Path | str | None = None, *, environ: bool = True) -> SiteConfig:
        """Load and validate the site file (``path``, else ``$GLM_TPU_SITE_CONFIG``, else
        ``$GLM_TPU_CONFIG_ROOT/site.toml``). ``environ`` applies the controller-side overrides
        ``GLM_TPU_RUN_ROOT``, ``GLM_TPU_MODEL_PATH`` and ``GLM_TPU_HLO_DUMP_ROOT``."""
        import tomllib

        from glm_tpu import envs

        location = Path(os.path.expanduser(str(path))) if path is not None else envs.GLM_TPU_SITE_CONFIG
        raw = _read_private(location, what="site file")
        try:
            value = tomllib.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise SiteConfigError(f"site file {location} is not valid TOML: {exc}") from None
        site = cls.from_mapping(value)
        if environ:
            site = site.with_overrides(run_root=envs.GLM_TPU_RUN_ROOT, model_path=envs.GLM_TPU_MODEL_PATH,
                                       hlo_dump_root=envs.GLM_TPU_HLO_DUMP_ROOT)
        return site

    @classmethod
    def from_mapping(cls, value: Any) -> SiteConfig:
        """Validate a parsed site file (the TOML tables as a mapping)."""
        return _parse(value, resolved=False)

    @classmethod
    def from_resolved_json(cls, raw: bytes) -> SiteConfig:
        """Validate the staged ``site.json`` a controller resolved (no controller-only keys)."""
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, ValueError):
            raise SiteConfigError("staged site.json is not valid JSON") from None
        site = _parse(value, resolved=True)
        if site.resolved_json() != raw:
            raise SiteConfigError("staged site.json is not in canonical resolved form")
        return site

    @classmethod
    def from_staged(cls, path: Path, expected_sha256: str) -> SiteConfig:
        """The staged ``site.json`` of a run directory: owner-only, its SHA-256 equal to the one
        the controller passed (``--site-sha256``), canonical and valid."""
        raw = _read_private(path, what="staged site.json")
        if not isinstance(expected_sha256, str) or sha256(raw).hexdigest() != expected_sha256:
            raise SiteConfigError("staged site.json digest differs")
        return cls.from_resolved_json(raw)

    # ------------------------------------------------------------------ views
    def with_overrides(self, *, run_root: Path | str | None = None, model_path: Path | str | None = None,
                       hlo_dump_root: Path | str | None = None) -> SiteConfig:
        """Controller-side path overrides (CLI flag or environment), validated like the file."""
        changes = {name: _path(value, f"paths.{name}") for name, value in
                   dict(run_root=run_root, model_path=model_path, hlo_dump_root=hlo_dump_root).items()
                   if value not in (None, "")}
        return replace(self, paths=replace(self.paths, **changes)) if changes else self

    def resolved(self) -> dict[str, Any]:
        """The configuration a worker receives (controller-only keys omitted), JSON-ready."""
        fleet = {k: v for k, v in _asdict(self.fleet).items() if k != "known_hosts"}
        paths = {k: v for k, v in _asdict(self.paths).items() if k != "repo"}
        return dict(schema=RESOLVED_SCHEMA, fleet=fleet, paths=paths, checkpoint=_asdict(self.checkpoint),
                    topology=_asdict(self.topology), storage=_asdict(self.storage), locks=_asdict(self.locks))

    def resolved_json(self) -> bytes:
        """Canonical bytes of :meth:`resolved` (hash contract: sorted, compact, ASCII) + newline."""
        return json.dumps(self.resolved(), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False).encode() + b"\n"

    def resolved_sha256(self) -> str:
        return sha256(self.resolved_json()).hexdigest()


# ----------------------------------------------------------------------------- current site
_CURRENT: SiteConfig | None = None


def set_current_site(site: SiteConfig | None) -> SiteConfig | None:
    """Install the process's site (entry points; tests restore the returned previous value)."""
    global _CURRENT
    if site is not None and not isinstance(site, SiteConfig):
        raise TypeError("set_current_site expects a SiteConfig")
    previous, _CURRENT = _CURRENT, site
    return previous


def get_current_site() -> SiteConfig:
    if _CURRENT is None:
        raise SiteConfigError("no site configuration is installed in this process (the executor, worker and "
                              "pack worker install the validated site; tests install an example site)")
    return _CURRENT


def approved_source_uri(uri: Any) -> bool:
    """Whether ``uri`` lies under one of the current site's ``storage.allowed_source_uri_prefixes``."""
    return get_current_site().storage.approved(uri)


# ----------------------------------------------------------------------------- TOML text
def to_toml(value: dict[str, Any]) -> str:
    """Serialize a site mapping (string/int/bool scalars and string lists in named tables) as TOML."""
    def scalar(item: Any, key: str) -> str:
        if isinstance(item, bool):
            return "true" if item else "false"
        if isinstance(item, int):
            return str(item)
        if isinstance(item, (str, Path)):
            return json.dumps(str(item), ensure_ascii=True)
        if isinstance(item, (list, tuple)):
            return "[" + ", ".join(scalar(x, key) for x in item) + "]"
        raise SiteConfigError(f"{key}: cannot write {type(item).__name__} to a site file")

    lines = [f"{k} = {scalar(v, k)}" for k, v in value.items() if not isinstance(v, dict)]
    for table, entries in value.items():
        if isinstance(entries, dict):
            lines += ["", f"[{table}]"] + [f"{k} = {scalar(v, f'{table}.{k}')}" for k, v in entries.items()]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- validation
_REQUIRED, _CONTROLLER = object(), object()
_TABLES: dict[str, dict[str, Any]] = {
    "fleet": dict(tpu_name=_REQUIRED, zone=_REQUIRED, project="", num_hosts=NUM_HOSTS, chips_per_host=CHIPS_PER_HOST,
                  host_rank_regex=DEFAULT_HOST_RANK_REGEX, coordinator_address=_REQUIRED, worker_python=_REQUIRED,
                  worker_pythonpath=_REQUIRED, helper_python="python3", known_hosts=DEFAULT_KNOWN_HOSTS),
    "paths": dict(repo="", run_root=_REQUIRED, model_path=_REQUIRED, hlo_dump_root=DEFAULT_HLO_DUMP_ROOT),
    "checkpoint": dict(namespace=_REQUIRED, root=_REQUIRED, inventory_namespace=_REQUIRED, source_inventory=_REQUIRED,
                       source_inventory_sha256=_REQUIRED, manifest_sha256=_REQUIRED, success_sha256=_REQUIRED,
                       source_complete_sha256=_REQUIRED),
    "topology": dict(binding_dir=_REQUIRED, binding_sha256=_REQUIRED, capture_root=_REQUIRED,
                     topology_sha256=_REQUIRED, topology_fleet_sha256=_REQUIRED, mesh_sha256=_REQUIRED,
                     slice_name=_REQUIRED),
    "storage": dict(source_uri=_REQUIRED, allowed_source_uri_prefixes=_REQUIRED),
    "locks": dict(workload=_REQUIRED, sync=_REQUIRED),
    "launch": dict(allowed_branches=["main", "release/*"], expected_origin="", require_clean=True,
                   require_pushed=True),
}
_CONTROLLER_ONLY = {"fleet": ("known_hosts",), "paths": ("repo",)}


def _parse(value: Any, *, resolved: bool) -> SiteConfig:
    if not isinstance(value, dict):
        raise SiteConfigError("site configuration must be a table")
    schema = RESOLVED_SCHEMA if resolved else SCHEMA
    if value.get("schema") != schema:
        raise SiteConfigError(f"schema must be {schema!r}")
    tables = {name: dict(spec) for name, spec in _TABLES.items() if not (resolved and name == "launch")}
    if resolved:
        for name, keys in _CONTROLLER_ONLY.items():
            for key in keys:
                del tables[name][key]
    unknown = sorted(set(value) - {"schema", *tables})
    if unknown:
        raise SiteConfigError(f"unknown site key {unknown[0]!r}")
    read: dict[str, dict[str, Any]] = {}
    for name, spec in tables.items():
        given = value.get(name, {})
        if not isinstance(given, dict):
            raise SiteConfigError(f"[{name}] must be a table")
        extra = sorted(set(given) - set(spec))
        if extra:
            raise SiteConfigError(f"unknown site key {name}.{extra[0]}")
        entries = {}
        for key, default in spec.items():
            if key in given:
                entries[key] = given[key]
            elif default is _REQUIRED or resolved:
                raise SiteConfigError(f"missing site key {name}.{key}")
            else:
                entries[key] = default
        read[name] = entries
    fleet = _fleet(read["fleet"], resolved=resolved)
    paths = _paths(read["paths"], resolved=resolved)
    checkpoint = _checkpoint(read["checkpoint"])
    topology = _topology(read["topology"])
    storage = _storage(read["storage"])
    locks = _locks(read["locks"])
    launch = None if resolved else _launch(read["launch"])
    return SiteConfig(fleet=fleet, paths=paths, checkpoint=checkpoint, topology=topology, storage=storage, locks=locks,
                      launch=launch)


def _string(value: Any, key: str, pattern: re.Pattern[str] | None = None, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not empty):
        raise SiteConfigError(f"{key} must be a non-empty string")
    if value and pattern is not None and pattern.fullmatch(value) is None:
        raise SiteConfigError(f"{key} has an invalid form")
    return value


def _integer(value: Any, key: str, expected: int) -> int:
    if type(value) is not int or value != expected:
        raise SiteConfigError(f"{key} must be {expected} (the fleet shape this engine runs)")
    return value


def _flag(value: Any, key: str) -> bool:
    if type(value) is not bool:
        raise SiteConfigError(f"{key} must be true or false")
    return value


def _strings(value: Any, key: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty) or not all(isinstance(x, str) for x in value):
        raise SiteConfigError(f"{key} must be a {'' if allow_empty else 'non-empty '}list of strings")
    if len(set(value)) != len(value):
        raise SiteConfigError(f"{key} lists a value twice")
    return value


def _path(value: Any, key: str) -> Path:
    text = _string(value if not isinstance(value, Path) else str(value), key)
    path = Path(os.path.expanduser(text))
    if not path.is_absolute() or ".." in path.parts or "~" in str(path):
        raise SiteConfigError(f"{key} must be an absolute path without '..' (after '~' expansion)")
    return Path(os.path.normpath(path))


def _hex(value: Any, key: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise SiteConfigError(f"{key} must be a lower-case 64-hex SHA-256")
    return value


def _inside(child: Path, parent: Path, key: str, parent_key: str) -> None:
    if child == parent or not child.is_relative_to(parent):
        raise SiteConfigError(f"{key} must lie strictly inside {parent_key}")


def _fleet(v: dict[str, Any], *, resolved: bool) -> FleetConfig:
    regex = _string(v["host_rank_regex"], "fleet.host_rank_regex")
    try:
        groups = re.compile(regex).groups
    except re.error:
        raise SiteConfigError("fleet.host_rank_regex is not a regular expression") from None
    if groups != 1:
        raise SiteConfigError("fleet.host_rank_regex must have exactly one group (the rank digits)")
    coordinator = _string(v["coordinator_address"], "fleet.coordinator_address")
    host, _, port = coordinator.rpartition(":")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        raise SiteConfigError("fleet.coordinator_address must be <IP address>:8476 (the worker admits an IP "
                              "literal only)") from None
    if port != COORDINATOR_PORT:
        raise SiteConfigError("fleet.coordinator_address port must be 8476")
    worker_python = str(_path(v["worker_python"], "fleet.worker_python"))
    if _PLAIN_PATH.fullmatch(worker_python) is None:
        raise SiteConfigError("fleet.worker_python must be a plain absolute path")
    pythonpath = tuple(str(_path(p, "fleet.worker_pythonpath"))
                       for p in _strings(v["worker_pythonpath"], "fleet.worker_pythonpath", allow_empty=True))
    if any(":" in p for p in pythonpath):
        raise SiteConfigError("fleet.worker_pythonpath entries must not contain ':'")
    helper = _string(v["helper_python"], "fleet.helper_python")
    if _COMMAND.fullmatch(helper) is None and (_PLAIN_PATH.fullmatch(helper) is None or ".." in helper.split("/")):
        raise SiteConfigError("fleet.helper_python must be a bare command or a plain absolute path")
    return FleetConfig(
        tpu_name=_string(v["tpu_name"], "fleet.tpu_name", _NAME), zone=_string(v["zone"], "fleet.zone", _NAME),
        project=_string(v["project"], "fleet.project", _PROJECT, empty=True),
        num_hosts=_integer(v["num_hosts"], "fleet.num_hosts", NUM_HOSTS),
        chips_per_host=_integer(v["chips_per_host"], "fleet.chips_per_host", CHIPS_PER_HOST),
        host_rank_regex=regex, coordinator_address=coordinator, worker_python=worker_python,
        worker_pythonpath=pythonpath, helper_python=helper,
        known_hosts=None if resolved else _path(v["known_hosts"], "fleet.known_hosts"))


def _paths(v: dict[str, Any], *, resolved: bool) -> PathsConfig:
    repo = None
    if not resolved:
        text = _string(v["repo"], "paths.repo", empty=True)
        repo = _path(text, "paths.repo") if text else None
    return PathsConfig(repo=repo, run_root=_path(v["run_root"], "paths.run_root"),
                       model_path=_path(v["model_path"], "paths.model_path"),
                       hlo_dump_root=_path(v["hlo_dump_root"], "paths.hlo_dump_root"))


def _checkpoint(v: dict[str, Any]) -> CheckpointConfig:
    namespace, root = _path(v["namespace"], "checkpoint.namespace"), _path(v["root"], "checkpoint.root")
    inventories = _path(v["inventory_namespace"], "checkpoint.inventory_namespace")
    inventory = _path(v["source_inventory"], "checkpoint.source_inventory")
    _inside(root, namespace, "checkpoint.root", "checkpoint.namespace")
    _inside(inventory, inventories, "checkpoint.source_inventory", "checkpoint.inventory_namespace")
    return CheckpointConfig(
        namespace=namespace, root=root, inventory_namespace=inventories, source_inventory=inventory,
        **{k: _hex(v[k], "checkpoint." + k) for k in ("source_inventory_sha256", "manifest_sha256", "success_sha256",
                                                      "source_complete_sha256")})


def _topology(v: dict[str, Any]) -> TopologyConfig:
    return TopologyConfig(
        binding_dir=_path(v["binding_dir"], "topology.binding_dir"),
        capture_root=_path(v["capture_root"], "topology.capture_root"),
        slice_name=_string(v["slice_name"], "topology.slice_name", _NAME),
        **{k: _hex(v[k], "topology." + k) for k in ("binding_sha256", "topology_sha256", "topology_fleet_sha256",
                                                    "mesh_sha256")})


def _storage(v: dict[str, Any]) -> StorageConfig:
    prefixes = tuple(_strings(v["allowed_source_uri_prefixes"], "storage.allowed_source_uri_prefixes"))
    if any(_GS_PREFIX.fullmatch(p) is None for p in prefixes):
        raise SiteConfigError("storage.allowed_source_uri_prefixes entries must be gs://<bucket>/[<path>/]")
    storage = StorageConfig(source_uri=_string(v["source_uri"], "storage.source_uri"),
                            allowed_source_uri_prefixes=prefixes)
    if not storage.approved(storage.source_uri) or storage.source_uri in prefixes:
        raise SiteConfigError("storage.source_uri must lie under storage.allowed_source_uri_prefixes")
    return storage


def _locks(v: dict[str, Any]) -> LocksConfig:
    workload = tuple(_path(p, "locks.workload") for p in _strings(v["workload"], "locks.workload"))
    sync = tuple(_path(p, "locks.sync") for p in _strings(v["sync"], "locks.sync"))
    if set(workload) & set(sync):
        raise SiteConfigError("a lock is listed as both workload and sync")
    return LocksConfig(workload=workload, sync=sync)


def _launch(v: dict[str, Any]) -> LaunchPolicy:
    branches = _strings(v["allowed_branches"], "launch.allowed_branches")
    if any(not b or b != b.strip() or any(c in b for c in " \t\n") for b in branches):
        raise SiteConfigError("launch.allowed_branches entries must be branch names or fnmatch patterns")
    origin = _string(v["expected_origin"], "launch.expected_origin", empty=True)
    return LaunchPolicy(allowed_branches=tuple(branches), expected_origin=origin or None,
                        require_clean=_flag(v["require_clean"], "launch.require_clean"),
                        require_pushed=_flag(v["require_pushed"], "launch.require_pushed"))


def _asdict(section: Any) -> dict[str, Any]:
    out = {}
    for name in section.__dataclass_fields__:
        item = getattr(section, name)
        if isinstance(item, Path):
            item = str(item)
        elif isinstance(item, tuple):
            item = [str(x) for x in item]
        out[name] = item
    return out


def _read_private(path: Path, *, what: str) -> bytes:
    """A bounded read of a regular owner-only file (not a symlink, mode 0600 or 0400)."""
    path = Path(path)
    try:
        facts = os.lstat(path)
    except FileNotFoundError:
        raise SiteConfigError(f"no {what} at {path} (copy examples/site.example.toml, chmod 600, fill in every "
                              "<...> value)") from None
    if stat.S_ISLNK(facts.st_mode) or not stat.S_ISREG(facts.st_mode):
        raise SiteConfigError(f"{what} {path} must be a regular file, not a symlink")
    if facts.st_uid != os.geteuid() or stat.S_IMODE(facts.st_mode) not in (0o600, 0o400):
        raise SiteConfigError(f"{what} {path} must be owned by you with mode 0600 or 0400")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read(SITE_FILE_CAP + 1)
    if not 0 < len(raw) <= SITE_FILE_CAP:
        raise SiteConfigError(f"{what} {path} is empty or larger than {SITE_FILE_CAP} bytes")
    return raw
