"""The request path shared by the controller, the worker, the CLI and serving (DESIGN 6.4): request
schemas (``request``), the per-request host loop (``request_session``) and the resident protocol
(``resident_protocol``).

Importing this package imports nothing (the submodules import JAX where they need it).
"""
