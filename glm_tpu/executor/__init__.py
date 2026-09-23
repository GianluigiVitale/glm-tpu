"""The rank-0 controller's launch machinery (DESIGN 6.5, 6.6).

``launch_policy`` decides which checkout may launch and which commit it stages; ``fleet`` builds
the exact remote command strings; ``remote`` holds the stdlib-only helper programs those commands
carry to the hosts. Standard library only: importing this package never imports JAX.
"""
