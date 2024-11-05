//! Module implementing low-pass filters.

use core::f32;
use nalgebra::SMatrix;
use std::{f32::consts::PI, time::Duration};

/// An second order Butterworth low-pass filter for NxM-dimensional signals,
/// with a arithmetic mean initialization.
pub struct MeanInitializedLowPassFilter<const N: usize, const M: usize> {
    /// The number of samples processed by this filter.
    sample_count: u32,
    /// Whether this filter has been initialized.
    initialized: bool,
    /// The denominator coefficients of the filter, assuming `a0 = 1.0`.
    a: [f32; 2],
    /// The numerator coefficients of the filter.
    b: [f32; 3],
    /// Time constant in seconds, used for the initialization step.
    tau: Duration,
    /// The rate of new samples for the signal.
    sampling_rate: Duration,
    state: [SMatrix<f32, N, M>; 2],
    pub last_output: SMatrix<f32, N, M>,
}

impl<const N: usize, const M: usize> MeanInitializedLowPassFilter<N, M> {
    /// Create a new filter with the given time constant, sampling rate, and
    /// filter coefficients.
    #[must_use]
    pub fn new(tau: Duration, sampling_rate: Duration, b: [f32; 3], a: [f32; 2]) -> Self {
        Self {
            sample_count: 0,
            initialized: false,
            a,
            b,
            tau,
            sampling_rate,
            last_output: SMatrix::from_element(f32::NAN),
            state: [SMatrix::zeros(), SMatrix::zeros()],
        }
    }

    pub fn initialize_with_value(&mut self, value: SMatrix<f32, N, M>) {
        self.sample_count = 0;
        self.initialized = true;
        self.last_output = value;
        self.state = Self::filter_initial_state(value, self.b, self.a);
    }

    /// Filter the given signal.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn filter(&mut self, x: SMatrix<f32, N, M>) -> SMatrix<f32, N, M> {
        if !self.initialized {
            self.sample_count += 1;
            self.state[1] += x;

            let elapsed = self.sample_count * self.sampling_rate;
            if elapsed >= self.tau {
                let x0 = self.state[1] / self.sample_count as f32;

                // Set initial state based on the mean value
                self.state = Self::filter_initial_state(x0, self.b, self.a);
                self.initialized = true;
                self.last_output = x0;
                return x0;
            }

            // return the arithmetic mean of all samples up until this point.
            return self.state[1] / self.sample_count as f32;
        }

        let y = self.b[0] * x + self.state[0];
        self.state[0] = self.b[1] * x - self.a[0] * y + self.state[1];
        self.state[1] = self.b[2] * x - self.a[1] * y;
        self.last_output = y;
        y
    }

    fn filter_initial_state(
        initial_state: SMatrix<f32, N, M>,
        b: [f32; 3],
        a: [f32; 2],
    ) -> [SMatrix<f32, N, M>; 2] {
        let state0 = initial_state * (1.0 - b[0]);
        let state1 = initial_state * (b[2] - a[1]);
        [state0, state1]
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
    let b2 = b0;

    let a1 = (2.0 * (c.powi(2) - 1.0)) / d;
    let a2 = (1.0 - 2_f32.sqrt() * c + c.powi(2)) / d;

    ([b0, b1, b2], [a1, a2])
}
