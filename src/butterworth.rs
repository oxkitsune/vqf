use nalgebra::SVector;
use std::f32::consts::PI;

/// First-order Butterworth low-pass filter
#[derive(Copy, Clone, Debug)]
pub struct LowPassFilter<const N: usize> {
    alpha: f32,
    beta: f32,
    x: SVector<f32, N>,
    y: SVector<f32, N>,
}

impl<const N: usize> LowPassFilter<N> {
    #[must_use]
    pub fn new(omega: f32) -> Self {
        let alpha = omega / (1. + omega);
        let beta = (1. - omega) / (1. + omega);

        Self {
            alpha,
            beta,
            x: [0.; N].into(),
            y: [0.; N].into(),
        }
    }

    #[must_use]
    pub fn with_cutoff_freq(freq: f32, dt: f32) -> Self {
        Self::new((PI * freq * dt).tan())
    }

    pub fn update(&mut self, x: SVector<f32, N>) -> SVector<f32, N> {
        let y = self.alpha * (x + self.x) + self.beta * self.y;
        self.x = x;
        self.y = y;
        y
    }
}
