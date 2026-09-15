"""In-memory generation-checked GCS bucket/blob doubles shared by transport tests.

Verbatim from the retired dense-frontier transport test module; used by the
native collection/transport tests and the user archive test.
"""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace as NS

from google.api_core.exceptions import PreconditionFailed
import google_crc32c


class Blob:
    def __init__(self, bucket, name, generation=None):
        self.bucket, self.name = bucket, name
        self.requested = generation
        self.generation = generation
        if name in bucket.objects:
            self.reload(if_generation_match=generation)

    def reload(self, if_generation_match=None):
        generation, data = self.bucket.objects[self.name]
        if if_generation_match is not None and generation != if_generation_match:
            raise PreconditionFailed("generation changed")
        self.generation, self.size = generation, len(data)
        self.crc32c = base64.b64encode(google_crc32c.Checksum(data).digest()).decode()

    def upload_from_file(self, stream, *, if_generation_match, checksum):
        assert if_generation_match == 0 and checksum == "crc32c"
        if self.name in self.bucket.objects:
            raise PreconditionFailed("already exists")
        self.bucket.objects[self.name] = (
            9007199254740993 + len(self.bucket.objects),
            stream.read(),
        )
        self.bucket.events.append(("upload", self.name))

    def download_as_bytes(self, *, if_generation_match, checksum="crc32c"):
        assert checksum == "crc32c"
        self.reload(if_generation_match=if_generation_match)
        self.bucket.events.append(("download", self.name))
        return self.bucket.objects[self.name][1]

    def download_to_file(self, stream, **kwargs):
        stream.write(self.download_as_bytes(**kwargs))

    def download_to_filename(self, name, **kwargs):
        Path(name).write_bytes(self.download_as_bytes(**kwargs))


class Bucket:
    location = "US-CENTRAL2"

    def __init__(self):
        self.objects, self.events = {}, []

    def reload(self):
        self.events.append(("region", self.location))

    def blob(self, name, generation=None):
        return Blob(self, name, generation)

    def get_blob(self, name):
        return self.blob(name) if name in self.objects else None

    def client(self):
        def bucket(name):
            assert name == "driftbench-dsv4-uc"
            return self

        return NS(bucket=bucket)
