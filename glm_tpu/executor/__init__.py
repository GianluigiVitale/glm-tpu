"""The rank-0 controller and its launch machinery (DESIGN 6.5, 6.6).

``multihost_executor`` is the controller process (``python -m glm_tpu.executor.multihost_executor``);
``launch_policy`` decides which checkout may launch and which commit it stages; ``fleet`` builds
the exact remote command strings; ``remote`` holds the stdlib-only helper programs those commands
carry to the hosts. Standard library only: importing this package never imports JAX.
"""
