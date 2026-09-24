"""The local API key file, created owner-only on first use."""

import os
import secrets


def api_token(path):
    """One local key, created 0600 on first use and never written to a log."""
    if path.exists():
        if path.is_symlink() or path.stat().st_mode & 0o077:
            raise ValueError('The API key file must be private (mode 0600).')
        token = path.read_text().strip()
        if not token:
            raise ValueError('The API key file is empty.')
        return token
    token = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(token + '\n')
    return token
