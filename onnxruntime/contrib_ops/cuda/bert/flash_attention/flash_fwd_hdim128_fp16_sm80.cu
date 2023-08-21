// Copyright (c) 2023, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.
#if USE_FLASH_ATTENTION

#include "contrib_ops/cuda/bert/flash_attention/flash_fwd_launch_template.h"

template <>
void run_mha_fwd_<cutlass::half_t, 128>(Flash_fwd_params& params, cudaStream_t stream) {
  run_mha_fwd_hdim128<cutlass::half_t>(params, stream);
}

#endif
