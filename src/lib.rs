mod ffi;

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::cuda_backend::WrapErr;
use candle::{CpuStorage, DType, Layout, Result, Shape, Tensor};
use half::{bf16, f16};
use std::ffi::c_int;

struct Silu;

impl Silu {
    fn fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        x: &candle::CudaStorage,
        x_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        let dev = x.device();
        let dtype = x.dtype();

        let internal_type = match dtype {
            DType::F16 => 0,
            DType::BF16 => 1,
            DType::F32 => 2,
            dtype => candle::bail!("dtype {dtype:?} is not supported"),
        };

        if !x_l.is_contiguous() {
            candle::bail!("x must be contiguous");
        }

        // Get cuda slices for all tensors
        let x = x.as_cuda_slice::<T>()?;
        // Get cuda views for all tensors
        let x = x.slice(x_l.start_offset()..);

        let dst_shape = x_l.shape().clone();
        let elems = dst_shape.elem_count();
        let dst = unsafe { dev.alloc::<T>(elems) }.w()?;

        let x_ptr = *x.device_ptr() as *const core::ffi::c_void;
        let dst_ptr = *dst.device_ptr() as *const core::ffi::c_void;

        const NUM_THREADS: c_int = 1024;
        let num_blocks = (elems as c_int + NUM_THREADS - 1) / NUM_THREADS;

        unsafe {
            ffi::silu(
                x_ptr,
                dst_ptr,
                num_blocks,
                NUM_THREADS,
                elems as c_int,
                internal_type,
            )
        }

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());

        Ok((dst, dst_shape))
    }
}

impl candle::CustomOp1 for Silu {
    fn name(&self) -> &'static str {
        "silu"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for fused-layer-norm")
    }

    fn cuda_fwd(
        &self,
        x: &candle::CudaStorage,
        x_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match x.dtype() {
            DType::F16 => self.fwd::<f16>(x, x_l),
            DType::BF16 => self.fwd::<bf16>(x, x_l),
            DType::F32 => self.fwd::<f32>(x, x_l),
            dt => {
                candle::bail!("silu is only supported for f32, f16 and bf16 ({dt:?})")
            }
        }
    }
}

/// Apply Silu activation inplace
///
/// # Arguments
///
/// * `x` - Tensor
pub fn silu(x: &Tensor) -> Result<Tensor> {
    let op = Silu {};
    x.apply_op1(op)
}
