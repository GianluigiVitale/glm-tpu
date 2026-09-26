"""OpenAI-compatible protocol constants and errors: model ids and aliases, reasoning efforts, request
caps and ``ApiError`` (defined in :mod:`glm_tpu.exceptions`, so that the engine's resident client raises it
without importing the serving layer; the same class).
"""

from glm_tpu.exceptions import ApiError as ApiError

MODEL_ID = "glm-5.3"
# A client picks reasoning effort by model id when it cannot send the field
# itself, so cheap side calls never think at full effort.
ALIASES = {MODEL_ID: None, MODEL_ID + "-low": "low", MODEL_ID + "-high": "high"}
EFFORTS = ("low", "high", "max")
MESSAGES_CAP = 1 << 20
TOOLS_CAP = 1 << 18
