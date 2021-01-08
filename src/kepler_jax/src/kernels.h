#ifndef _KEPLER_JAX_KERNELS_H_
#define _KEPLER_JAX_KERNELS_H_

#include <cuda_runtime_api.h>

namespace kepler_jax {

void gpu_kepler(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);

}  // namespace kepler_jax

#endif