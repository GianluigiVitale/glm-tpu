"""GLM-5.3 identity, separate from the frozen GLM-5.2 sampled profile."""
from hashlib import sha256
from pathlib import Path

from ..user_request import read_bounded

MODEL_ID = 'zai-org/GLM-5.3'
REVISION = 'aca966e4e02791568aa6a4ced368624b3d897f42'
SOURCE_URI = 'gs://driftbench-dsv4-uc/models/GLM-5.3-FP8'
TOKENIZER_ROOT = Path('/home/gianl/gcs-models/models/GLM-5.3-FP8')
TEMPLATE_PATH = Path('reference/hf-glm53/chat_template.jinja')
TEMPLATE_SHA = '3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c'
CONFIG_SHA = '3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69'
INDEX_SHA = 'e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf'
GENERATION_SHA = 'ac76b43d8683d3b930126870fc8be73d8679308fe752fa1f381096d8354f6a55'
TOKENIZER_FILES = {
    'tokenizer.json': '19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d',
    'tokenizer_config.json': '98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc',
}


def verified_template(repo, tokenizer_root):
    """Verify local assets without downloads or importing any model/device code."""
    template = read_bounded(repo / TEMPLATE_PATH, 64 << 10)
    if sha256(template).hexdigest() != TEMPLATE_SHA:
        raise ValueError('chat template differs from pinned GLM-5.3')
    for name, digest in (('config.json', CONFIG_SHA), ('generation_config.json', GENERATION_SHA)):
        if sha256(read_bounded(repo / TEMPLATE_PATH.parent / name, 64 << 10)).hexdigest() != digest:
            raise ValueError('configuration differs from pinned GLM-5.3')
    for name, digest in TOKENIZER_FILES.items():
        if sha256(read_bounded(tokenizer_root / name, 32 << 20)).hexdigest() != digest:
            raise ValueError('tokenizer differs from pinned GLM-5.3')
    return template.decode()


def require_site(args):
    """Refuse inherited GLM-5.2 site bindings before any device initialization.

    The verified GLM-5.3 packing result must provide its own inventory digest.
    Source acquisition alone does not supply a usable runtime checkpoint.
    """
    if (getattr(args, 'model_id', None) != MODEL_ID
            or getattr(args, 'model_revision', None) != REVISION
            or not getattr(args, 'source_inventory_sha256', None)):
        raise ValueError('verified GLM-5.3 runtime checkpoint binding is required')


def require_inventory(inventory):
    """A valid historical inventory is insufficient for the new weights."""
    if (inventory.model_id != MODEL_ID or inventory.source_revision != REVISION
            or inventory.config_sha256 != CONFIG_SHA or inventory.index_sha256 != INDEX_SHA):
        raise ValueError('runtime inventory differs from pinned GLM-5.3 source')
