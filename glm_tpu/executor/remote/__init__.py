"""Remote helper programs (DESIGN 6.5, D6, D7; H12).

Every module here is a complete, standard-library-only Python program that the controller sends to
the hosts as the text of ``<interpreter> -c <program text> <one JSON argument>``
(``glm_tpu.executor.fleet.remote_command``). A helper is never imported on a host and never
templated: it reads everything from its JSON argument, parses under Python 3.10 (the hosts' system
``python3``), exposes ``main(argv) -> int`` and runs it under ``if __name__ == "__main__"``.

After ``idle_before`` a helper receives the eight authenticated hostnames (``hosts``) and derives
its rank as ``hosts.index(socket.gethostname())``; the ``-w-N`` naming check stays with the
controller (``fleet.host_rank_regex``).

| helper | interpreter | purpose |
|---|---|---|
| ``idle_probe`` | ``fleet.helper_python`` | no libtpu holder, no live worker of this run; prints ``IDLE <host>`` |
| ``stage_bundle`` | ``fleet.worker_python`` | verify the bundle digest (stdin) and extract it with ``filter="data"`` |
| ``start_worker`` | ``fleet.helper_python`` | write the start marker, then ``execv`` the worker module |
| ``cleanup`` | ``fleet.helper_python`` | ``SIGKILL`` only the authenticated worker of this run |
| ``fetch`` | ``fleet.helper_python`` | print named records of a directory as base64 JSON |
"""

HELPERS = ("idle_probe", "stage_bundle", "start_worker", "cleanup", "fetch")
