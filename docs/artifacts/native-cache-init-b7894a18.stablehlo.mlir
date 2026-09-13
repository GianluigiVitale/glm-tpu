module @jit_initialize attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["expert"=8, "feature"=4]> {stablehlo.mesh = {axes = [{name = "expert", size = 8 : i64}, {name = "feature", size = 4 : i64}]}}
  func.func public @main(%arg0: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>}) -> (tensor<78x326x512x640xbf16> {jax.result_info = "result.decoder.kv_cache_local", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"expert"}, {}]>}, tensor<21x326x512x128xbf16> {jax.result_info = "result.decoder.index_cache_local", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"expert"}, {}]>}, tensor<1x2048xi32> {jax.result_info = "result.decoder.selected_positions", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<1xi32> {jax.result_info = "result.decoder.selected_valid_counts", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<1x2048xf32> {jax.result_info = "result.decoder.selected_scores", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<1xi32> {jax.result_info = "result.decoder.position", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<1x326xi32> {jax.result_info = "result.decoder.block_tables", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, tensor<1xi32> {jax.result_info = "result.decoder.context_lengths", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<1xi1> {jax.result_info = "result.decoder.contract_valid", sdy.sharding = #sdy.sharding<@mesh, [{}]>}, tensor<21x326x512x128xbf16> {jax.result_info = "result.repaired_index_local", sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"expert"}, {}]>}, tensor<i32> {jax.result_info = "result.prompt_length", sdy.sharding = #sdy.sharding<@mesh, []>}, tensor<i1> {jax.result_info = "result.finished", sdy.sharding = #sdy.sharding<@mesh, []>}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare GT, %arg0, %c, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<166912> : tensor<i32>
    %1 = stablehlo.compare LT, %arg0, %c_0, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2 = stablehlo.and %0, %1 : tensor<i1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<78x326x512x640xbf16>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %4 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<bf16>) -> tensor<21x326x512x128xbf16>
    %c_2 = stablehlo.constant dense<-1> : tensor<i32>
    %5 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1x2048xi32>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_4 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x2048xf32>
    %c_5 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %9 = stablehlo.iota dim = 0 : tensor<326xi32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<326xi32>) -> tensor<1x326xi32>
    %c_6 = stablehlo.constant dense<1> : tensor<i32>
    %11 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %12 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %13 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<bf16>) -> tensor<21x326x512x128xbf16>
    %c_8 = stablehlo.constant dense<false> : tensor<i1>
    return %3, %4, %5, %6, %7, %8, %10, %11, %12, %13, %arg0, %c_8 : tensor<78x326x512x640xbf16>, tensor<21x326x512x128xbf16>, tensor<1x2048xi32>, tensor<1xi32>, tensor<1x2048xf32>, tensor<1xi32>, tensor<1x326xi32>, tensor<1xi32>, tensor<1xi1>, tensor<21x326x512x128xbf16>, tensor<i32>, tensor<i1>
  }
}
