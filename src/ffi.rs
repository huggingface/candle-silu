use core::ffi::{c_int, c_void};

extern "C" {
    pub(crate) fn silu(
        x_ptr: *const c_void,
        dst_ptr: *const c_void,
        num_blocks: c_int,
        num_threads: c_int,
        numel: c_int,
        dtype: u32,
    );
}
