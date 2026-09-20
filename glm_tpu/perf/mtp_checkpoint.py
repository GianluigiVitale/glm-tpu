"""Exact native-MTP source intervals and binding; no acquisition on import.

The source layer keeps its authenticated original name/offsets. Destination
body names use layer zero solely for the separate one-layer draft runtime.
Embedding/head are shared with the target, never packed or copied here.
"""
from dataclasses import replace

from ..greenfield.checkpoint.ws32_runtime import placements_for_ws32_source_tensor,_placement
from ..greenfield.runtime.ws32_decoder import bind_ws32_decoder_weights
from .bf16_resident import bf16_resident_weights
from .mtp_draft import MtpWeights,mtp_config
from .mtp_projection import MtpProjectionWeights


PROJECTION_NAMES=('mtp.enorm.weight','mtp.hnorm.weight','mtp.eh_proj.weight')


def mtp_source_placements(source,target_config):
    """Map one audited source tensor to exact WS32 owner intervals.

    Source identity, generation-bound header/payload integrity and complete
    coverage must be established by the acquisition controller before loading.
    This function checks role/dtype/shape; it reads no model payload.
    """
    prefix=f'model.layers.{target_config.geometry.num_layers}.'
    if not source.name.startswith(prefix):
        raise ValueError('MTP pack accepts only the target checkpoint next-token layer')
    suffix=source.name[len(prefix):]
    config=mtp_config(target_config)
    if suffix=='eh_proj.weight':
        if source.dtype!='BF16' or source.shape!=(config.geometry.hidden_size,2*config.geometry.hidden_size):
            raise ValueError('MTP eh_proj must be BF16 [H,2H]')
        return tuple(_placement(source,expert=e,feature=f,source_partitions=(1,None),
            destination_name='mtp.eh_proj.weight') for e in range(8) for f in range(4))
    if suffix in ('enorm.weight','hnorm.weight','shared_head.norm.weight'):
        alias='model.norm.weight'
        destination='model.norm.weight' if suffix=='shared_head.norm.weight' else 'mtp.'+suffix
    else:
        alias='model.layers.0.'+suffix
        destination=None
    placements=placements_for_ws32_source_tensor(replace(source,name=alias),config.geometry)
    return tuple(replace(p,source_name=source.name,
        destination_name=p.destination_name if destination is None else destination) for p in placements)


def bind_mtp_arrays(arrays,target_weights,mesh,target_config):
    """Bind verified final-owner native arrays and shared target BF16 I/O.

    Caller owns array/payload and memory admission. The strict ordinary name
    binder checks the complete body/norm map; projection geometry is checked
    when building/executing the native input projection.
    """
    config=mtp_config(target_config)
    if not set(PROJECTION_NAMES)<=set(arrays) or any(
            name in arrays for name in ('model.embed_tokens.weight','lm_head.weight')):
        raise ValueError('native arrays must contain projection tables and exclude shared target I/O')
    body={name:array for name,array in arrays.items() if name not in PROJECTION_NAMES}
    body.update({'model.embed_tokens.weight':target_weights.embedding_local,
                 'lm_head.weight':target_weights.lm_head_local})
    raw=bind_ws32_decoder_weights(body,config)
    resident=bf16_resident_weights(mesh,config,raw)
    return MtpWeights(resident,MtpProjectionWeights(*(arrays[name] for name in PROJECTION_NAMES)))
