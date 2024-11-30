//! A module implementing second-order Butterworth low-pass filters for
//! multi-dimensional signals.
//!
//! This module provides a low-pass filter structure,
//! [`MeanInitializedLowPassFilter`], which is initialized with an arithmetic
//! mean over the first few samples.

use nalgebra::SMatrix;
use std::{f32::consts::PI, time::Duration};

/// A second-order Butterworth low-pass filter for NxM-dimensional signals,
/// initialized with an arithmetic mean.
///
/// The [`MeanInitializedLowPassFilter`] applies a low-pass filter to
/// multi-dimensional signals, smoothing high-frequency components based on a
/// specified time constant and sampling rate.
///
/// The filter is initialized with an arithmetic mean of the first few samples
/// to ensure a stable starting state before applying the filtering operations.
///
/// # Example
///
/// ```rust
/// use nalgebra::Vector3;
/// use std::time::Duration;
/// use vqf::low_pass_filter::{second_order_butterworth, MeanInitializedLowPassFilter};
///
/// let tau = Duration::from_secs_f32(1. / 6.); // 6 Hz
/// let sampling_rate = Duration::from_secs_f32(0.1); // 10 Hz
///
/// // Create a filter with a 10 Hz sampling rate, and a 5 Hz cut-off frequency.
/// let (b, a) = second_order_butterworth(tau, sampling_rate);
/// let mut filter = MeanInitializedLowPassFilter::<3, 1>::new(tau, sampling_rate, b, a);
///
/// // Filter a signal with the filter.
/// let signal = Vector3::from_element(1.0);
/// let filtered_signal = filter.filter(signal);
///
/// // The filtered signal should be equal to the input signal, as the filter
/// // has not yet been initialized, and will return the arithmetic mean of
/// // all samples up until this point.
/// assert_eq!(filtered_signal, signal);
///
/// // Filter another signal, and the filter should still return the input signal.
/// // filter to the input signal, as we have not yet reached the initialization time.
/// let signal = Vector3::from_element(2.0);
///
/// let filtered_signal = filter.filter(signal);
/// assert_eq!(filtered_signal, Vector3::from_element(1.5));
///
/// // Filter the signal a third time, and the filter should now return a different signal.
/// let filtered_signal = filter.filter(signal);
/// assert_ne!(filtered_signal, Vector3::from_element(5. / 3.));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    /// The last output of the filter.
    pub last_output: SMatrix<f32, N, M>,
}

impl<const N: usize, const M: usize> MeanInitializedLowPassFilter<N, M> {
    /// Creates a new [`MeanInitializedLowPassFilter`].
    ///
    /// # Arguments
    ///
    /// * `tau` - Time constant for the filter, which determines the cut-off
    ///   frequency.
    /// * `sampling_rate` - The rate at which new samples are processed.
    /// * `b` - Numerator coefficients for the filter.
    /// * `a` - Denominator coefficients for the filter.
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

    /// Filters the provided signal `x` and returns the filtered output.
    ///
    /// The filter applies a second-order Butterworth low-pass filter to the
    /// signal.
    ///
    /// # Note
    ///
    /// If the filter is not yet initialized, this method will compute the
    /// arithmetic mean of incoming samples and apply it as an initial state
    /// for stability.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use vqf::low_pass_filter::MeanInitializedLowPassFilter;
    /// # use nalgebra::SMatrix;
    /// # use std::time::Duration;
    ///
    /// let tau = Duration::from_secs_f32(1.0);
    /// let sampling_rate = Duration::from_millis(100);
    /// let (b, a) = vqf::low_pass_filter::second_order_butterworth(tau, sampling_rate);
    /// let mut filter = MeanInitializedLowPassFilter::<3, 3>::new(tau, sampling_rate, b, a);
    ///
    /// let input_signal = SMatrix::<f32, 3, 3>::from_element(1.0);
    /// let output_signal = filter.filter(input_signal);
    /// ```
    #[inline]
    #[must_use]
    pub fn filter(&mut self, x: SMatrix<f32, N, M>) -> SMatrix<f32, N, M> {
        if !self.initialized {
            return self.filter_arithmetic_mean(x);
        }

        let y = self.b[0] * x + self.state[0];
        self.state[0] = self.b[1] * x - self.a[0] * y + self.state[1];
        self.state[1] = self.b[2] * x - self.a[1] * y;
        self.last_output = y;
        y
    }

    /// Computes the arithmetic mean of samples for filter initialization.
    ///
    /// # Note
    ///
    /// This function collects initial samples to compute a stable starting
    /// state. This method is only used during the initial sample collection
    /// phase and is bypassed once the filter has enough data to initialize.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    fn filter_arithmetic_mean(&mut self, x: SMatrix<f32, N, M>) -> SMatrix<f32, N, M> {
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
        self.state[1] / self.sample_count as f32
    }

    /// Initializes the filter state based on the mean value of initial samples.
    #[inline]
    #[must_use]
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

/// Return tuple containing the numerator (`b`) and denominator (`a`)
/// coefficients for a second-order Butterworth low-pass filter.
///
/// The time constant $\tau$ corresponds to the cutoff frequency $f_c$
/// of the second-order Butterworth low-pass filter as follows: $$f_c =
/// \frac{\sqrt(2)}{2 \pi \tau}$$
///
/// # Example
///
/// ```
/// use std::time::Duration;
/// use vqf::low_pass_filter::second_order_butterworth;
///
/// let tau = Duration::from_secs_f32(1.0);
/// let sampling_rate = Duration::from_millis(100);
/// let (b, a) = second_order_butterworth(tau, sampling_rate);
/// ```
#[must_use]
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
