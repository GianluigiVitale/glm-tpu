"""On-demand checkpoint format/loader exports.

Importing the WS32 runtime loader must not eagerly load unrelated formats.
Retained names keep their original module targets. The PP8/PP16 complete,
streaming, runtime, Gate C and one-layer-Pallas formats were retired with
their packers, loaders and tests (recoverable at b667f00f). No loader body,
manifest schema or checkpoint identity changes here."""

from importlib import import_module as _import_module

_EXPORTS = {
    "LoadedWs32OneLayerGlobal": "ws32_one_layer",
    "LoadedWs32OneLayerSlot": "ws32_one_layer",
    "LoadedWs32RuntimeCheckpoint": "ws32_runtime_checkpoint",
    "LoadedWs32StrategyNdDenseOverlay": "ws32_strategy_nd_dense",
    "OneLayerPackConfig": "one_layer",
    "VerifiedWs32RuntimeCheckpoint": "ws32_runtime_checkpoint",
    "WS32_RUNTIME_ARTIFACT_KIND": "ws32_runtime_checkpoint",
    "WS32_RUNTIME_FORMAT_VERSION": "ws32_runtime_checkpoint",
    "WS32_RUNTIME_SLOT_RECORD_KIND": "ws32_runtime_checkpoint",
    "Ws32OneLayerPackConfig": "ws32_one_layer",
    "Ws32RuntimeFilePlan": "ws32_runtime_checkpoint",
    "Ws32RuntimePackConfig": "ws32_runtime_checkpoint",
    "Ws32RuntimePlacementReport": "ws32_runtime",
    "Ws32RuntimeTensorPlan": "ws32_runtime_checkpoint",
    "Ws32SourcePlacement": "ws32_runtime",
    "Ws32StrategyNdDenseOverlay": "ws32_strategy_nd_dense",
    "build_ws32_runtime_file_plans": "ws32_runtime_checkpoint",
    "build_ws32_runtime_placement_report": "ws32_runtime",
    "finalize_ws32_runtime_checkpoint": "ws32_runtime_checkpoint",
    "inspect_one_layer_artifact": "one_layer",
    "inspect_ws32_one_layer": "ws32_one_layer",
    "iter_ws32_source_placements": "ws32_runtime",
    "load_ws32_one_layer_global": "ws32_one_layer",
    "load_ws32_one_layer_slot": "ws32_one_layer",
    "load_ws32_runtime_checkpoint": "ws32_runtime_checkpoint",
    "load_ws32_strategy_nd_dense_overlay": "ws32_strategy_nd_dense",
    "pack_one_layer_moe": "one_layer",
    "pack_ws32_one_layer": "ws32_one_layer",
    "pack_ws32_runtime_checkpoint": "ws32_runtime_checkpoint",
    "pack_ws32_runtime_slots": "ws32_runtime_checkpoint",
    "placements_for_ws32_source_tensor": "ws32_runtime",
    "strategy_nd_dense_tensor_names": "ws32_strategy_nd_dense",
    "verify_ws32_runtime_checkpoint": "ws32_runtime_checkpoint",
    "verify_ws32_strategy_nd_dense_overlay": "ws32_strategy_nd_dense",
}

__all__ = (
    "LoadedWs32OneLayerGlobal",
    "LoadedWs32OneLayerSlot",
    "OneLayerPackConfig",
    "Ws32OneLayerPackConfig",
    "Ws32RuntimePlacementReport",
    "Ws32RuntimeFilePlan",
    "Ws32RuntimePackConfig",
    "Ws32SourcePlacement",
    "Ws32RuntimeTensorPlan",
    "LoadedWs32RuntimeCheckpoint",
    "LoadedWs32StrategyNdDenseOverlay",
    "VerifiedWs32RuntimeCheckpoint",
    "Ws32StrategyNdDenseOverlay",
    "WS32_RUNTIME_ARTIFACT_KIND",
    "WS32_RUNTIME_FORMAT_VERSION",
    "WS32_RUNTIME_SLOT_RECORD_KIND",
    "build_ws32_runtime_placement_report",
    "build_ws32_runtime_file_plans",
    "finalize_ws32_runtime_checkpoint",
    "inspect_one_layer_artifact",
    "inspect_ws32_one_layer",
    "iter_ws32_source_placements",
    "load_ws32_one_layer_global",
    "load_ws32_one_layer_slot",
    "load_ws32_runtime_checkpoint",
    "load_ws32_strategy_nd_dense_overlay",
    "pack_one_layer_moe",
    "pack_ws32_one_layer",
    "pack_ws32_runtime_checkpoint",
    "pack_ws32_runtime_slots",
    "placements_for_ws32_source_tensor",
    "strategy_nd_dense_tensor_names",
    "verify_ws32_runtime_checkpoint",
    "verify_ws32_strategy_nd_dense_overlay",
)


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, _, attribute = target.partition(".")
    value = getattr(_import_module("." + module, __name__), attribute or name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
