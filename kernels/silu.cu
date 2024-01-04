#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ float expg(float a) { return expf(a); }
__device__ __forceinline__ __half expg(__half a) { return hexp(a); }
__device__ __forceinline__ __nv_bfloat16 expg(__nv_bfloat16 a) { return hexp(a); }

template<typename scalar_t>
inline __device__ scalar_t silu(
  scalar_t __restrict__ x)
{
  return x / (static_cast<scalar_t>(1) + expg(-x));
}

template<typename scalar_t>
__global__ void silu_kernel(
  scalar_t* __restrict__ x_ptr,
  scalar_t* __restrict__ out_ptr,
  const int numel) {

  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
    out_ptr[i] = silu<scalar_t>(x_ptr[i]);
  }
}

#define CALL_SILU(T)                                                          \
  silu_kernel<T><<<grid, block, 0, stream>>>(                                 \
  reinterpret_cast<T*>(x),                                                    \
  reinterpret_cast<T*>(out),                                                  \
  numel);

extern "C" void silu(
  void *x,
  void *out,

  int32_t num_blocks,
  int32_t num_threads,
  int32_t numel,

  uint32_t dtype // 0 => f16; 1 => bf16; 2 => f32
  ) {
  dim3 grid(num_blocks);
  dim3 block(num_threads);
  const cudaStream_t stream = 0;

  if (dtype == 0){
    CALL_SILU(half);
  } else if (dtype == 1) {
    CALL_SILU(__nv_bfloat16);
  } else if (dtype == 2) {
    CALL_SILU(float);
  }
}
