"""OpenAI-compatible protocol constants and errors: model ids and aliases, reasoning efforts, request
caps and ``ApiError``.
"""


MODEL_ID = 'glm-5.3'
# A client picks reasoning effort by model id when it cannot send the field
# itself, so cheap side calls never think at full effort.
ALIASES = {MODEL_ID: None, MODEL_ID + '-low': 'low', MODEL_ID + '-high': 'high'}
EFFORTS = ('low', 'high', 'max')
MESSAGES_CAP = 1 << 20
TOOLS_CAP = 1 << 18


class ApiError(ValueError):
    def __init__(self, message, *, status=400, kind='invalid_request_error'):
        super().__init__(message)
        self.status = status
        self.kind = kind

    def body(self):
        return dict(error=dict(message=str(self), type=self.kind, param=None, code=None))
