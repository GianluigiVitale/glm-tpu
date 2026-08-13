from __future__ import annotations

from glm_tpu.greenfield.kernels.stage_local import (
    STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE,
)
from glm_tpu.greenfield.sharding.stablehlo_strategy_nd import (
    validate_strategy_nd_attention_stablehlo,
)


def _strategy_nd_stablehlo(*, layers: int = 1) -> str:
    lines = ["module {", "  func.func public @main() {"]
    outputs: list[str] = []

    def emit(value: str) -> None:
        lines.append(f"    {value}")

    def add_barrier(
        name: str, left: str, right: str, result_type: str
    ) -> str:
        emit(
            f"%{name}_add = stablehlo.add %{left}, %{right} : "
            f"{result_type}"
        )
        emit(
            f"%{name}_barrier = stablehlo.optimization_barrier "
            f"%{name}_add : {result_type}"
        )
        return f"{name}_barrier"

    def reduce_four(
        name: str,
        values: tuple[str, str, str, str],
        *,
        cross: bool,
        result_type: str,
    ) -> str:
        pairs = ((0, 3), (1, 2)) if cross else ((0, 1), (2, 3))
        first = add_barrier(
            f"{name}_first",
            values[pairs[0][0]],
            values[pairs[0][1]],
            result_type,
        )
        second = add_barrier(
            f"{name}_second",
            values[pairs[1][0]],
            values[pairs[1][1]],
            result_type,
        )
        return add_barrier(name, first, second, result_type)

    for layer in range(layers):
        prefix = f"l{layer}"
        emit(f"%{prefix}_zero = stablehlo.constant dense<0> : tensor<i32>")
        row_broadcasts: list[str] = []
        for shard in range(8):
            start = shard * 512
            scale_start = shard * 4
            emit(
                f"%{prefix}_lhs_{shard} = stablehlo.slice %{prefix}_lhs "
                f"[0:1, {start}:{start + 512}] : "
                "(tensor<1x4096xbf16>) -> tensor<1x512xbf16>"
            )
            emit(
                f"%{prefix}_lhs_pad_{shard} = func.call @_pad_lhs("
                f"%{prefix}_lhs_{shard}, %{prefix}_zero) : "
                "(tensor<1x512xbf16>, tensor<i32>) -> "
                "tensor<8x512xbf16>"
            )
            emit(
                f"%{prefix}_weight_{shard} = stablehlo.slice "
                f"%{prefix}_weight [0:6144, {start}:{start + 512}] : "
                "(tensor<6144x4096xui8>) -> tensor<6144x512xui8>"
            )
            emit(
                f"%{prefix}_scale_{shard} = stablehlo.slice "
                f"%{prefix}_scale [0:48, {scale_start}:{scale_start + 4}] : "
                "(tensor<48x32xf32>) -> tensor<48x4xf32>"
            )
            emit(
                f"%{prefix}_scale_transpose_{shard} = stablehlo.transpose "
                f"%{prefix}_scale_{shard}, dims = [1, 0] : "
                "(tensor<48x4xf32>) -> tensor<4x48xf32>"
            )
            emit(
                f"%{prefix}_scale_pad_{shard} = func.call @_pad_scale("
                f"%{prefix}_scale_transpose_{shard}, %{prefix}_zero) : "
                "(tensor<4x48xf32>, tensor<i32>) -> tensor<8x128xf32>"
            )
            emit(
                f"%{prefix}_kernel_{shard} = stablehlo.custom_call "
                f"@tpu_custom_call(%{prefix}_lhs_pad_{shard}, "
                f"%{prefix}_weight_{shard}, %{prefix}_scale_pad_{shard}) "
                '{kernel_name = "greenfield_fp8_strategy_nd_o_m8_k512_n6144"} '
                ": (tensor<8x512xbf16>, tensor<6144x512xui8>, "
                "tensor<8x128xf32>) -> tensor<8x6144xbf16>"
            )
            emit(
                f"%{prefix}_row_{shard} = stablehlo.slice "
                f"%{prefix}_kernel_{shard} [0:1, 0:6144] : "
                "(tensor<8x6144xbf16>) -> tensor<1x6144xbf16>"
            )
            emit(
                f"%{prefix}_row_broadcast_{shard} = "
                f"stablehlo.broadcast_in_dim %{prefix}_row_{shard}, "
                "dims = [1, 2] : (tensor<1x6144xbf16>) -> "
                "tensor<1x1x6144xbf16>"
            )
            row_broadcasts.append(f"%{prefix}_row_broadcast_{shard}")

        emit(
            f"%{prefix}_stack = stablehlo.concatenate "
            + ", ".join(row_broadcasts)
            + ", dim = 0 : ("
            + ", ".join("tensor<1x1x6144xbf16>" for _ in range(8))
            + ") -> tensor<8x1x6144xbf16>"
        )
        emit(
            f"%{prefix}_local = stablehlo.broadcast_in_dim %{prefix}_stack, "
            "dims = [1, 2, 3] : (tensor<8x1x6144xbf16>) -> "
            "tensor<1x8x1x6144xbf16>"
        )
        emit(
            f'%{prefix}_gather = "stablehlo.all_gather"(%{prefix}_local) '
            "<{all_gather_dim = 0 : i64, use_global_device_ids}> : "
            "(tensor<1x8x1x6144xbf16>) -> tensor<4x8x1x6144xbf16>"
        )
        emit(
            f"%{prefix}_flat = stablehlo.reshape %{prefix}_gather : "
            "(tensor<4x8x1x6144xbf16>) -> tensor<32x1x6144xbf16>"
        )

        physical_rows: list[str] = []
        for slot, physical_row in enumerate(
            STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE
        ):
            emit(
                f"%{prefix}_physical_slice_{slot} = stablehlo.slice "
                f"%{prefix}_flat [{physical_row}:{physical_row + 1}, "
                "0:1, 0:6144] : (tensor<32x1x6144xbf16>) -> "
                "tensor<1x1x6144xbf16>"
            )
            emit(
                f"%{prefix}_physical_reshape_{slot} = stablehlo.reshape "
                f"%{prefix}_physical_slice_{slot} : "
                "(tensor<1x1x6144xbf16>) -> tensor<1x6144xbf16>"
            )
            emit(
                f"%{prefix}_physical_broadcast_{slot} = "
                f"stablehlo.broadcast_in_dim %{prefix}_physical_reshape_{slot}, "
                "dims = [1, 2] : (tensor<1x6144xbf16>) -> "
                "tensor<1x1x6144xbf16>"
            )
            physical_rows.append(f"%{prefix}_physical_broadcast_{slot}")
        emit(
            f"%{prefix}_physical_stack = stablehlo.concatenate "
            + ", ".join(physical_rows)
            + ", dim = 0 : ("
            + ", ".join("tensor<1x1x6144xbf16>" for _ in range(32))
            + ") -> tensor<32x1x6144xbf16>"
        )
        emit(
            f"%{prefix}_physical_5d = stablehlo.reshape "
            f"%{prefix}_physical_stack : (tensor<32x1x6144xbf16>) -> "
            "tensor<4x4x2x1x6144xbf16>"
        )
        emit(
            f"%{prefix}_transposed = stablehlo.transpose "
            f"%{prefix}_physical_5d, dims = [1, 2, 0, 3, 4] : "
            "(tensor<4x4x2x1x6144xbf16>) -> "
            "tensor<4x2x4x1x6144xbf16>"
        )

        y_results: list[str] = []
        for chunk_index, (start, cross) in enumerate(
            ((0, False), (2048, True), (4096, False))
        ):
            emit(
                f"%{prefix}_y_chunk_{chunk_index} = stablehlo.slice "
                f"%{prefix}_transposed [0:4, 0:2, 0:4, 0:1, "
                f"{start}:{start + 2048}] : "
                "(tensor<4x2x4x1x6144xbf16>) -> "
                "tensor<4x2x4x1x2048xbf16>"
            )
            values: list[str] = []
            for row in range(4):
                emit(
                    f"%{prefix}_y_slice_{chunk_index}_{row} = "
                    f"stablehlo.slice %{prefix}_y_chunk_{chunk_index} "
                    f"[{row}:{row + 1}, 0:2, 0:4, 0:1, 0:2048] : "
                    "(tensor<4x2x4x1x2048xbf16>) -> "
                    "tensor<1x2x4x1x2048xbf16>"
                )
                emit(
                    f"%{prefix}_y_value_{chunk_index}_{row} = "
                    f"stablehlo.reshape %{prefix}_y_slice_{chunk_index}_{row} : "
                    "(tensor<1x2x4x1x2048xbf16>) -> "
                    "tensor<2x4x1x2048xbf16>"
                )
                values.append(f"{prefix}_y_value_{chunk_index}_{row}")
            y_results.append(
                reduce_four(
                    f"{prefix}_y_{chunk_index}",
                    tuple(values),  # type: ignore[arg-type]
                    cross=cross,
                    result_type="tensor<2x4x1x2048xbf16>",
                )
            )
        emit(
            f"%{prefix}_y = stablehlo.concatenate "
            + ", ".join(f"%{value}" for value in y_results)
            + ", dim = 3 : ("
            + ", ".join("tensor<2x4x1x2048xbf16>" for _ in range(3))
            + ") -> tensor<2x4x1x6144xbf16>"
        )
        x_values: list[str] = []
        for row in range(2):
            emit(
                f"%{prefix}_x_slice_{row} = stablehlo.slice %{prefix}_y "
                f"[{row}:{row + 1}, 0:4, 0:1, 0:6144] : "
                "(tensor<2x4x1x6144xbf16>) -> tensor<1x4x1x6144xbf16>"
            )
            emit(
                f"%{prefix}_x_value_{row} = stablehlo.reshape "
                f"%{prefix}_x_slice_{row} : (tensor<1x4x1x6144xbf16>) -> "
                "tensor<4x1x6144xbf16>"
            )
            x_values.append(f"{prefix}_x_value_{row}")
        x_reduced = add_barrier(
            f"{prefix}_x",
            x_values[0],
            x_values[1],
            "tensor<4x1x6144xbf16>",
        )

        z_results: list[str] = []
        for segment, start in enumerate(range(0, 6144, 256)):
            emit(
                f"%{prefix}_z_chunk_{segment} = stablehlo.slice "
                f"%{x_reduced} [0:4, 0:1, {start}:{start + 256}] : "
                "(tensor<4x1x6144xbf16>) -> tensor<4x1x256xbf16>"
            )
            values = []
            for row in range(4):
                emit(
                    f"%{prefix}_z_slice_{segment}_{row} = stablehlo.slice "
                    f"%{prefix}_z_chunk_{segment} "
                    f"[{row}:{row + 1}, 0:1, 0:256] : "
                    "(tensor<4x1x256xbf16>) -> tensor<1x1x256xbf16>"
                )
                emit(
                    f"%{prefix}_z_value_{segment}_{row} = stablehlo.reshape "
                    f"%{prefix}_z_slice_{segment}_{row} : "
                    "(tensor<1x1x256xbf16>) -> tensor<1x256xbf16>"
                )
                values.append(f"{prefix}_z_value_{segment}_{row}")
            z_results.append(
                reduce_four(
                    f"{prefix}_z_{segment}",
                    tuple(values),  # type: ignore[arg-type]
                    cross=bool(segment % 2),
                    result_type="tensor<1x256xbf16>",
                )
            )
        emit(
            f"%{prefix}_output = stablehlo.concatenate "
            + ", ".join(f"%{value}" for value in z_results)
            + ", dim = 1 : ("
            + ", ".join("tensor<1x256xbf16>" for _ in range(24))
            + ") -> tensor<1x6144xbf16>"
        )
        outputs.append(f"%{prefix}_output")

    emit("return " + ", ".join(outputs))
    lines.extend(
        (
            "  }",
            "  func.func private @_pad_lhs(%arg0: tensor<1x512xbf16>, "
            "%arg1: tensor<i32>) -> tensor<8x512xbf16> {",
            "    %0 = stablehlo.convert %arg1 : (tensor<i32>) -> tensor<bf16>",
            "    %1 = stablehlo.pad %arg0, %0, low = [0, 0], "
            "high = [7, 0], interior = [0, 0] : "
            "(tensor<1x512xbf16>, tensor<bf16>) -> tensor<8x512xbf16>",
            "    return %1 : tensor<8x512xbf16>",
            "  }",
            "  func.func private @_pad_scale(%arg0: tensor<4x48xf32>, "
            "%arg1: tensor<i32>) -> tensor<8x128xf32> {",
            "    %0 = stablehlo.convert %arg1 : (tensor<i32>) -> tensor<f32>",
            "    %1 = stablehlo.pad %arg0, %0, low = [0, 0], "
            "high = [4, 80], interior = [0, 0] : "
            "(tensor<4x48xf32>, tensor<f32>) -> tensor<8x128xf32>",
            "    return %1 : tensor<8x128xf32>",
            "  }",
            "}",
        )
    )
    return "\n".join(lines) + "\n"


def _case_scoped_strategy_nd_stablehlo() -> str:
    """Put two identical-SSA layer programs in sibling case regions."""

    source = _strategy_nd_stablehlo()
    main_prefix = "  func.func public @main() {\n"
    helper_marker = "\n  func.func private @_pad_lhs"
    main_start = source.index(main_prefix) + len(main_prefix)
    main_end = source.index("  }" + helper_marker, main_start)
    body = source[main_start:main_end]
    body_lines = body.splitlines()
    assert body_lines[-1].strip() == "return %l0_output"
    branch = "\n".join("      " + line.strip() for line in body_lines)
    helpers = source[source.index(helper_marker) :]
    return (
        "module {\n"
        "  func.func public @main(%selector: tensor<i32>) {\n"
        "    %case = \"stablehlo.case\"(%selector) ({\n"
        f"{branch}\n"
        "    }, {\n"
        f"{branch}\n"
        "    }) : (tensor<i32>) -> tensor<1x6144xbf16>\n"
        "    return %case\n"
        "  }"
        f"{helpers}"
    )


def test_exact_strategy_nd_stablehlo_contract_accepts_complete_tree() -> None:
    result = validate_strategy_nd_attention_stablehlo(
        _strategy_nd_stablehlo(), layers=1, enabled=True
    )
    assert result["passed"], result
    assert result["kernel_count"] == 8
    assert result["matched_tree_count"] == 1

    dense_gather = (
        "    %dense_seed = stablehlo.constant dense<0> : "
        "tensor<1x8x1x6144xbf16>\n"
        '    %dense_gather = "stablehlo.all_gather"(%dense_seed) '
        "<{all_gather_dim = 0 : i64, use_global_device_ids}> : "
        "(tensor<1x8x1x6144xbf16>) -> tensor<4x8x1x6144xbf16>\n"
    )
    coexisting = validate_strategy_nd_attention_stablehlo(
        _strategy_nd_stablehlo().replace(
            "    return %l0_output\n", dense_gather + "    return %l0_output\n", 1
        ),
        layers=1,
        enabled=True,
    )
    assert coexisting["passed"], coexisting
    assert coexisting["candidate_gather_count"] == 2
    assert coexisting["gather_count"] == 1

    missing = validate_strategy_nd_attention_stablehlo(
        None, layers=1, enabled=True
    )
    assert not missing["passed"]

    unexpected = validate_strategy_nd_attention_stablehlo(
        _strategy_nd_stablehlo(), layers=1, enabled=False
    )
    assert not unexpected["passed"]


def test_strategy_nd_stablehlo_tracks_sibling_case_ssa_scopes() -> None:
    stablehlo = _case_scoped_strategy_nd_stablehlo()
    result = validate_strategy_nd_attention_stablehlo(
        stablehlo, layers=2, enabled=True
    )
    assert result["passed"], result
    assert result["gather_count"] == 2
    assert result["kernel_count"] == 16
    assert result["matched_tree_count"] == 2

    unterminated = stablehlo.replace(
        "    }) : (tensor<i32>) -> tensor<1x6144xbf16>\n", "", 1
    )
    rejected = validate_strategy_nd_attention_stablehlo(
        unterminated, layers=2, enabled=True
    )
    assert not rejected["passed"]
    assert any("unterminated" in item for item in rejected["violations"])


def test_strategy_nd_stablehlo_rejects_swapped_partial_rows() -> None:
    stablehlo = _strategy_nd_stablehlo().replace(
        "%l0_row_broadcast_0, %l0_row_broadcast_1",
        "%l0_row_broadcast_1, %l0_row_broadcast_0",
        1,
    )
    result = validate_strategy_nd_attention_stablehlo(
        stablehlo, layers=1, enabled=True
    )
    assert not result["passed"]


def test_strategy_nd_stablehlo_rejects_cross_wired_layers() -> None:
    stablehlo = _strategy_nd_stablehlo(layers=2).replace(
        "%l1_row_broadcast_0, %l1_row_broadcast_1",
        "%l0_row_broadcast_0, %l1_row_broadcast_1",
        1,
    )
    result = validate_strategy_nd_attention_stablehlo(
        stablehlo, layers=2, enabled=True
    )
    assert not result["passed"]


def test_strategy_nd_stablehlo_rejects_wrong_pad_or_rogue_callee() -> None:
    stablehlo = _strategy_nd_stablehlo()
    wrong_pad = stablehlo.replace(
        "low = [0, 0], high = [7, 0]",
        "low = [7, 0], high = [0, 0]",
        1,
    )
    wrong_pad_result = validate_strategy_nd_attention_stablehlo(
        wrong_pad, layers=1, enabled=True
    )
    assert not wrong_pad_result["passed"]

    rogue_callee = stablehlo.replace(
        "func.call @_pad_lhs(", "func.call @_pad_rogue(", 1
    )
    rogue_callee_result = validate_strategy_nd_attention_stablehlo(
        rogue_callee, layers=1, enabled=True
    )
    assert not rogue_callee_result["passed"]


def test_strategy_nd_stablehlo_rejects_reassociated_or_bypassed_tree() -> None:
    stablehlo = _strategy_nd_stablehlo()
    reassociated = stablehlo.replace(
        "%l0_y_0_first_add = stablehlo.add "
        "%l0_y_value_0_0, %l0_y_value_0_1",
        "%l0_y_0_first_add = stablehlo.add "
        "%l0_y_value_0_0, %l0_y_value_0_3",
        1,
    )
    reassociated_result = validate_strategy_nd_attention_stablehlo(
        reassociated, layers=1, enabled=True
    )
    assert not reassociated_result["passed"]

    bypassed = stablehlo.replace(
        "return %l0_output", "return %l0_z_0_barrier", 1
    )
    bypassed_result = validate_strategy_nd_attention_stablehlo(
        bypassed, layers=1, enabled=True
    )
    assert not bypassed_result["passed"]

    branched = stablehlo.replace(
        "return %l0_output", "return %l0_output, %l0_flat", 1
    )
    branched_result = validate_strategy_nd_attention_stablehlo(
        branched, layers=1, enabled=True
    )
    assert not branched_result["passed"]
