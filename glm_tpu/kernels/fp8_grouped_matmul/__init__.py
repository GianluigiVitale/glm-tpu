"""FP8 grouped (routed-expert) matmul kernels with 128x128 block scales: decode over scalar-prefetched
routes (``kernel``), the prefill panel kernel (``panel_kernel``) and expert-panel planning (``panels``)."""
