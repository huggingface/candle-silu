use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle_silu::silu;

fn to_vec2_round(t: Tensor, digits: i32) -> Result<Vec<Vec<f32>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec2::<f32>()?;
    let t = t
        .iter()
        .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
        .collect();
    Ok(t)
}

fn silu_truth(xs: &Tensor) -> candle::error::Result<Tensor> {
    xs / (xs.neg()?.exp()? + 1.0)?
}

#[test]
fn rotary() -> Result<()> {
    let device = Device::new_cuda(0)?;

    let m = 32;
    let n = 64;

    let h = Tensor::randn(0.0, 1.0, (m, n), &device)?.to_dtype(DType::F32)?;

    let expected = silu_truth(&h)?;
    let x = silu(&h)?;

    assert_eq!(to_vec2_round(expected, 3)?, to_vec2_round(x, 3)?);

    Ok(())
}
