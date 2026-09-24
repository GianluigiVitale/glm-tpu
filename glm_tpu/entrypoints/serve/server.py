"""Loopback chat UI attached to an existing resident GLM controller.

This process never initializes devices, starts workers, or sends a stop command.
Private chat history and requests live outside the checkout.
"""
import argparse
from http.server import ThreadingHTTPServer
import os
from pathlib import Path
import threading

from glm_tpu.entrypoints.openai import serving_chat as api
from glm_tpu.engine.resident_client import Resident
from glm_tpu.entrypoints.serve.http_handler import handler
from glm_tpu.entrypoints.serve.job_queue import Chats
from glm_tpu.entrypoints.serve.security import api_token


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dispatch', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8011)
    parser.add_argument('--api-key-file', type=Path,
                        help='default: <state>/api-key, created 0600 on first start')
    parser.add_argument('--no-api', action='store_true',
                        help='serve only the browser workspace, without /v1')
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--site', type=Path,
                        help='site file for the tokenizer location (default: $GLM_TPU_SITE_CONFIG, '
                             'else $GLM_TPU_CONFIG_ROOT/site.toml)')
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.state.resolve().is_relative_to(args.repo.resolve()):
        parser.error('Private chat state must be outside the source checkout.')
    from glm_tpu.config.site import SiteConfig
    site = SiteConfig.load(args.site)
    backend = Resident(args.run.resolve(), args.dispatch.resolve(), args.repo.resolve(), site.paths.model_path)
    chats = Chats(args.state, backend)
    service = token = key_path = None
    if not args.no_api:
        key_path = args.api_key_file or (args.state / 'api-key')
        token = api_token(key_path)
        service = api.Api(chats, backend, capacity=backend.capacity)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(chats, service, token))
    thread = threading.Thread(target=chats.work, daemon=True)
    thread.start()
    print(f'GLM-5.3 chat: http://127.0.0.1:{args.port} (existing model retained)', flush=True)
    if service is not None:
        print(f'OpenAI-compatible API: http://127.0.0.1:{args.port}/v1 '
              f'(key in {key_path})', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        chats.stopping.set()
        thread.join(timeout=5)
        server.server_close()
        # Closing this UI does not stop the resident controller or its workers.


if __name__ == '__main__':
    main()
