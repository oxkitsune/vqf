//! Module implementing low-pass filters.

use nalgebra::SVector;
use std::{f32::consts::PI, time::Duration};

/// An nth-order Butterworth low-pass filter for 3-dimensional signals, with a arithmetic mean initialization.
pub struct MeanInitializedLowPassFilter<const N: usize> {
    sample_count: usize,
    /// The denominator coefficients of the filter, assuming `a0 = 1.0`.
    a: [f32; 2],
    /// The numerator coefficients of the filter.
    b: [f32; 3],
    /// Time constant in seconds, used for the initialization step.
    tau: f32,
    /// The sampling rate of the filtered signal, in Hz.
    sampling_rate: f32,
    state: [SVector<f32, N>; 2],
}

impl<const N: usize> MeanInitializedLowPassFilter<N> {
    /// Create a new filter with the given time constant, sampling rate, and filter coefficients.
    #[must_use]
    pub fn new(tau: Duration, sampling_rate: f32, b: [f32; 3], a: [f32; 2]) -> Self {
        let tau = tau.as_secs_f32();
        Self {
            sample_count: 0,
            a,
            b,
            tau,
            sampling_rate,
            state: [SVector::zeros(), SVector::zeros()],
        }
    }

    /// Check if the filter has been initialized fully, i.e. if the number of samples is enough to
    /// have a valid output.
    #[inline]
    #[must_use]
    fn initialized(&self) -> bool {
        self.sample_count as f32 * self.sampling_rate >= self.tau
    }

    /// Filter the given signal.
    #[must_use]
    pub fn filter(&mut self, x: SVector<f32, N>) -> SVector<f32, N> {
        if !self.initialized() {
            self.sample_count += 1;
            self.state[1] += x;

            return self.state[1] / self.sample_count as f32;
        }

        let y = self.b[0] * x + self.state[0];
        self.state[0] = self.b[1] * x - self.a[0] * y + self.state[1];
        self.state[1] = self.b[2] * x - self.a[1] * y;

        y
    }
}

pub fn second_order_butterworth(tau: Duration, sampling_time: Duration) -> ([f32; 3], [f32; 2]) {
    let tau = tau.as_secs_f32();
    let sampling_time = sampling_time.as_secs_f32();

    let fc = 2_f32.sqrt() / (2.0 * PI * tau);
    let c = (PI * fc * sampling_time).tan();
    let d = c.powi(2) + 2_f32.sqrt() * c + 1.0;

    let b0 = c.powi(2) / d;
    let b1 = 2.0 * b0;

    let a1 = (2.0 * (c.powi(2) - 1.0)) / d;
    let a2 = (1.0 - 2_f32.sqrt() * c + c.powi(2)) / d;

    ([b0, b1, b0], [a1, a2])
}
