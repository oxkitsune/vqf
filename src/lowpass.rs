//! Module implementing low-pass filters.

use core::f32;
use nalgebra::SMatrix;
use std::{f32::consts::PI, time::Duration};

/// An second order Butterworth low-pass filter for NxM-dimensional signals, with a arithmetic mean initialization.
pub struct MeanInitializedLowPassFilter<const N: usize, const M: usize> {
    sample_count: usize,
    /// The denominator coefficients of the filter, assuming `a0 = 1.0`.
    a: [f32; 2],
    /// The numerator coefficients of the filter.
    b: [f32; 3],
    /// Time constant in seconds, used for the initialization step.
    tau: f32,
    /// The sampling rate of the filtered signal, in Hz.
    sampling_rate: f32,
    state: [SMatrix<f32, N, M>; 2],
    pub last_output: SMatrix<f32, N, M>,
}

impl<const N: usize, const M: usize> MeanInitializedLowPassFilter<N, M> {
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
            last_output: SMatrix::from_element(f32::NAN),
            state: [SMatrix::zeros(), SMatrix::zeros()],
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
    pub fn filter(&mut self, x: SMatrix<f32, N, M>) -> SMatrix<f32, N, M> {
        if !self.initialized() {
            self.sample_count += 1;
            self.state[1] += x;

            return self.state[1] / self.sample_count as f32;
        }

        let y = self.b[0] * x + self.state[0];
        self.state[0] = self.b[1] * x - self.a[0] * y + self.state[1];
        self.state[1] = self.b[2] * x - self.a[1] * y;
        self.last_output = y;
        y
    }
}

/// Create the coefficients for a second order Butterworth low-pass filter.
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
