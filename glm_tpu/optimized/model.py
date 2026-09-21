"""GLM-5.3 identity, separate from the frozen GLM-5.2 sampled profile."""
from hashlib import sha256
import json
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


def geometry(repo=None):
    """Use the pinned GLM-5.3 dimensions with its own checkpoint identity.

    The retained parser names its historical GLM-5.2 target unconditionally.
    Keep that frozen parser intact and bind the new identity at this boundary.
    """
    from dataclasses import replace
    from ..greenfield.types import ModelGeometry
    repo = Path(__file__).resolve().parents[2] if repo is None else repo
    raw = read_bounded(repo / TEMPLATE_PATH.parent / 'config.json', 64 << 10)
    if sha256(raw).hexdigest() != CONFIG_SHA:
        raise ValueError('geometry configuration differs from pinned GLM-5.3')
    return replace(ModelGeometry.from_hf_config(json.loads(raw)), model_id=MODEL_ID)


def site_args(args, *, repo=None):
    """Load the source-bound GLM-5.3 packing result before device initialization.

    The packing workflow writes this small configuration only after all owners
    and their terminal seals are verified. Missing configuration is incomplete
    migration, never permission to fall back to retired GLM-5.2 weights.
    """
    repo = Path(__file__).resolve().parents[2] if repo is None else repo
    value = json.loads(read_bounded(repo / 'configs/glm53-site.json', 64 << 10))
    pins = ('source_inventory_sha256', 'checkpoint_manifest_sha256',
            'checkpoint_success_sha256', 'source_complete_sha256')
    expected = {'schema', 'model_id', 'model_revision', 'source_inventory',
                'checkpoint_root', *pins}
    if (type(value) is not dict or set(value) != expected
            or value['schema'] != 'glm_ws32_glm53_site_v1'
            or value['model_id'] != MODEL_ID or value['model_revision'] != REVISION):
        raise ValueError('GLM-5.3 site identity differs')
    if any(type(value[k]) is not str or len(value[k]) != 64
           or any(c not in '0123456789abcdef' for c in value[k]) for k in pins):
        raise ValueError('GLM-5.3 site digests must be SHA256')
    for field, parent in (
        ('source_inventory', Path('/home/gianl/gcs-models/checkpoints/greenfield/glm53')),
        ('checkpoint_root', Path('/dev/shm/glm-ws32-runtime')),
    ):
        path = Path(value[field])
        if '..' in path.parts or not path.is_relative_to(parent) or path == parent:
            raise ValueError('GLM-5.3 site asset namespace differs')
    raw = read_bounded(TOKENIZER_ROOT / 'SOURCE_COMPLETE.json', 1 << 20)
    if sha256(raw).hexdigest() != value['source_complete_sha256']:
        raise ValueError('GLM-5.3 source completion identity differs')
    complete = json.loads(raw)
    if (complete.get('passed') is not True or complete.get('repository') != MODEL_ID
            or complete.get('revision') != REVISION or complete.get('verified_shards') != 141
            or complete.get('verified_bytes') != 755632050320):
        raise ValueError('GLM-5.3 canonical source is incomplete')
    for name in ('model_id', 'model_revision', *pins):
        setattr(args, name, value[name])
    args.checkpoint_root = Path(value['checkpoint_root'])
    args.source_inventory = Path(value['source_inventory'])
    args.checkpoint_transport = 'shm'
    topology_args(args)
    require_site(args)
    return args


def topology_args(args):
    """Retained physical site identity shared by packing and inference admission."""
    args.topology_capture_root = Path('/home/gianl/gcs-models/results/greenfield_topology_20260826T194116460015528Z/host_records')
    args.topology_sha256 = '294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559'
    args.topology_fleet_sha256 = '4a0c9a338d55b8be37dab79396569aa10fc9e85b3c7210d72a70abfafe72c301'
    args.mesh_sha256 = 'de5f59cbadf2116745ee1dde921656424c9555c3ddc584dcdd66cb7845050a88'
    args.slice_name, args.num_processes = 'db-v4-64-od', 8
    return args
