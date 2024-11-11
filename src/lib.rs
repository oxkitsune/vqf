//! # Versatile Quaternion Filter (VQF) for Orientation Estimation.
//!
//! This crate implements the VQF based on [this paper](https://arxiv.org/pdf/2203.17024).
//!
//! ⚠️ Currently this crate does *not* implement the magnometer update.
//!
//! # Example
//! ```rust
//! use nalgebra::Vector3;
//! use std::time::Duration;
//! use vqf::{Vqf, VqfParameters};
//!
//! let gyro_rate = Duration::from_secs_f32(0.01); // 100Hz
//! let accel_rate = Duration::from_secs_f32(0.01);
//! let params = VqfParameters::default();
//!
//! let mut vqf = Vqf::new(gyro_rate, accel_rate, params);
//!
//! let gyro_data = Vector3::new(0.01, 0.02, -0.01); // rad/s
//! let accel_data = Vector3::new(0.0, 0.0, 9.81); // m/s^2
//! vqf.update(gyro_data, accel_data);
//!
//! let orientation = vqf.orientation();
//! println!("Current orientation: {:?}", orientation);
//! ```

pub mod low_pass_filter;

use core::f32;
use std::time::Duration;

use low_pass_filter::{second_order_butterworth, MeanInitializedLowPassFilter};
use nalgebra::{Matrix3, Quaternion, SVector, SimdPartialOrd, UnitQuaternion, Vector2, Vector3};

/// Scale factor for the estimated bias, used for numerical stability.
const BIAS_SCALE: f32 = 100.0;

/// The state of a [`Vqf`] filter.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VqfState {
    /// Angular velocity strapdown integration quaternion.
    pub gyroscope_quat: UnitQuaternion<f32>,
    /// Inclination correction quaternion.
    pub accelerometer_quat: UnitQuaternion<f32>,
    /// The current duration the system has been in `rest` state.
    ///
    /// # Important
    ///
    /// If this is not [`Option::None`], it does not guarantee the system is in
    /// rest. See [`Vqf::is_rest_phase()`] for more.
    rest: Option<Duration>,
    /// The low-pass filter state for the accelerometer readings.
    pub accelerometer_low_pass: MeanInitializedLowPassFilter<3, 1>,
    /// Low-pass filter state for gyroscope measurements, used for rest
    /// detection.
    pub rest_gyro_low_pass: MeanInitializedLowPassFilter<3, 1>,
    /// Low-pass filter state for accelerometer measurements, used for rest
    /// detection.
    pub rest_accel_low_pass: MeanInitializedLowPassFilter<3, 1>,
    /// Low-pass filter state for the rotation matrix coefficients used in the
    /// motion bias estimation.
    pub motion_bias_estimate_rotation_low_pass: MeanInitializedLowPassFilter<3, 3>,
    /// Low-pass filter state for the rotated bias estimate used in the motion
    /// bias estimation.
    pub motion_bias_estimate_low_pass: MeanInitializedLowPassFilter<2, 1>,
    /// The current gyroscope bias estimate in rad/s.
    pub bias: SVector<f32, 3>,
    /// The current covariance matrix of the gyroscope bias estimate.
    ///
    /// # Note
    ///
    /// For numeric reasons the internal unit used is 0.01 degrees/s, i.e. to
    /// get the standard deviation use: $$\sigma =
    /// \frac{\sqrt(p_{ii})}{100}$$
    pub bias_p: Matrix3<f32>,
}

impl VqfState {
    /// Initialize a new [`VqfState`], using the provided gyroscope sample rate
    /// in seconds, i.e. `100Hz = 0.01 seconds`.
    #[must_use]
    pub fn new(
        coefficients: &VqfCoefficients,
        params: &VqfParameters,
        gyro_rate: Duration,
        accel_rate: Duration,
    ) -> Self {
        Self {
            gyroscope_quat: UnitQuaternion::identity(),
            accelerometer_quat: UnitQuaternion::identity(),
            rest: None,
            accelerometer_low_pass: MeanInitializedLowPassFilter::new(
                params.tau_accelerometer,
                accel_rate,
                coefficients.accel_b,
                coefficients.accel_a,
            ),
            rest_gyro_low_pass: MeanInitializedLowPassFilter::new(
                params.rest_filter_tau,
                gyro_rate,
                coefficients.rest_gyro_b,
                coefficients.rest_gyro_a,
            ),
            rest_accel_low_pass: MeanInitializedLowPassFilter::new(
                params.rest_filter_tau,
                accel_rate,
                coefficients.rest_accel_b,
                coefficients.rest_accel_a,
            ),
            // TODO: is this correct?
            // we reuse the accelerometer coefficients for the motion bias
            motion_bias_estimate_rotation_low_pass: MeanInitializedLowPassFilter::new(
                params.tau_accelerometer,
                accel_rate,
                coefficients.accel_b,
                coefficients.accel_a,
            ),
            motion_bias_estimate_low_pass: MeanInitializedLowPassFilter::new(
                params.tau_accelerometer,
                accel_rate,
                coefficients.accel_b,
                coefficients.accel_a,
            ),
            bias: Vector3::zeros(),
            bias_p: coefficients.bias.p0 * Matrix3::identity(),
        }
    }
}

/// Parameters for configuring a [`Vqf`].
///
/// The [`Default`] implementation uses the default parameters as described in
/// the [`original implementation`].
///
/// # Example
///
/// ```rust
/// use std::time::Duration;
/// use vqf::VqfParameters;
///
/// // Initialize the default parameters, but set `tau_acc` to 10 seconds
/// // making it take longer to initialize the low pass filter state for the accelerometer filter.
/// let mut params = VqfParameters {
///     tau_accelerometer: Duration::from_secs(10),
///     ..Default::default()
/// };
/// ```
/// [original implementation]: `https://vqf.readthedocs.io/en/latest/ref_cpp_params.html`
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VqfParameters {
    /// Time constant $\tau_{acc}$ for accelerometer low-pass filtering.
    ///
    /// Small values for $\tau_{acc}$ imply trust on the accelerometer
    /// measurements, while large values of $\tau_{acc}$ imply trust on the
    /// gyroscope measurements.
    ///
    /// The time constant $\tau_{acc}$ corresponds to the cutoff frequency $f_c$
    /// of the second-order Butterworth low-pass filter as follows: $$f_c =
    /// \frac{\sqrt(2)}{2 \pi \tau_{acc}}$$
    pub tau_accelerometer: Duration,
    /// Enables gyroscope bias estimation during motion phases.
    ///
    /// # Note
    ///
    /// Gyroscope bias is estimated based on the inclination correction only!
    pub do_bias_estimation: bool,
    /// Enables gyroscope bias estimation during rest phases.
    ///
    /// # Note
    ///
    /// This enables "rest"-phase detection, phases in which the IMU is at rest.
    /// During rest-phases, the gyroscope bias is estimated from the
    /// low-pass filtered gyroscope readings.
    pub do_rest_bias_estimation: bool,
    /// Standard deviation of the initial bias estimation uncertainty, in
    /// degrees per second.
    pub bias_sigma_initial: f32,
    /// Time in which the bias estimation uncertainty increases from 0 °/s to
    /// 0.1 °/s. This value determines the system noise assumed by the
    /// Kalman filter.
    pub bias_forgetting_time: Duration,
    /// Maximum expected gyroscope bias, in degrees per second.
    ///
    /// This value is used to clip the bias estimate and the measurement error
    /// in the bias estimation update step.
    /// It is further used by the rest detection algorithm in order to not
    /// regard measurements with a large but constant angular rate as rest.
    pub bias_clip: f32,
    /// Standard deviation of the converged bias estimation uncertainty during
    /// motion, in degrees per second.
    pub bias_sigma_motion: f32,
    /// Forgetting factor for unobservable bias in vertical direction during
    /// motion.
    ///
    /// As magnetometer measurements are deliberately not used during motion
    /// bias estimation, gyroscope bias is not observable in vertical
    /// direction.
    ///
    /// This value is the relative weight of an artificial zero measurement that
    /// ensures that the bias estimate in the unobservable direction will
    /// eventually decay to zero.
    pub bias_vertical_forgetting_factor: f32,
    /// Standard deviation of the converged bias estimation uncertainty during a
    /// rest phase, in degrees per second.
    pub bias_sigma_rest: f32,
    /// Time threshold for rest detection.
    ///
    /// A rest phase is detected when the measurements have been close to the
    /// low-pass filtered reference for at least this duration.
    pub rest_min_duration: Duration,
    /// Time constant for the low-pass filter used in the rest detection.
    ///
    /// This time constant characterizes a second-order Butterworth low-pass
    /// filter used to obtain the reference for rest detection.
    pub rest_filter_tau: Duration,
    /// Angular velocity threshold for rest detection, in degrees per second.
    ///
    /// For a rest phase to be detected, the norm of the deviation between
    /// measurement and reference must be below the provided threshold.
    /// The absolute value of each component must also be below
    /// [`Self::bias_clip`].
    pub rest_threshold_gyro: f32,
    /// Acceleration threshold for rest phase detection in m/s^2.
    ///
    /// For a rest phase to be detected, the norm of the deviation between
    /// measurement and reference must be below the provided threshold.
    pub rest_threshold_accel: f32,
}

impl Default for VqfParameters {
    fn default() -> Self {
        Self {
            tau_accelerometer: Duration::from_secs(3),
            do_bias_estimation: true,
            do_rest_bias_estimation: true,
            bias_sigma_initial: 0.5,
            bias_forgetting_time: Duration::from_secs(100),
            bias_clip: 2.0,
            bias_sigma_motion: 0.1,
            bias_vertical_forgetting_factor: 0.0001,
            bias_sigma_rest: 0.03,
            rest_min_duration: Duration::from_secs_f32(1.5),
            rest_filter_tau: Duration::from_secs_f32(0.5),
            rest_threshold_gyro: 2.0,
            rest_threshold_accel: 0.5,
        }
    }
}

/// Coefficients for gyroscope bias estimation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VqfBiasCoefficients {
    /// Variance of the initial gyroscope bias estimate.
    p0: f32,
    /// System noise variance used in the gyroscope bias estimation.
    v: f32,
    /// Measurement noise variance for the motion gyroscope bias estimation
    /// update.
    motion_w: f32,
    /// Measurement noise variance for the motion gyroscope bias estimation
    /// update in vertical direction.
    vertical_w: f32,
    /// Measurement noise variance for the rest gyroscope bias estimation
    /// update.
    rest_w: f32,
}

impl VqfBiasCoefficients {
    /// Initialize the bias estimation coefficients using the given parameters.
    ///
    /// This function is roughly equivalent to the `InitializeKalmanFilter`
    /// procedure in Algorithm 2 of the original paper.
    #[must_use]
    fn new(accelerometer_rate: Duration, params: &VqfParameters) -> Self {
        // line 17 of Algorithm 2, the initial variance of the bias
        let p0 = (params.bias_sigma_initial * BIAS_SCALE).powi(2);

        // line 18 of Algorithm 2
        // System noise increases the variance from 0 to (0.1 °/s)^2 in
        // `bias_forgetting_time` duration
        let v = (0.1 * 100_f32).powi(2) * accelerometer_rate.as_secs_f32()
            / params.bias_forgetting_time.as_secs_f32();

        // line 19 of Algorithm 2
        let p_motion = (params.bias_sigma_motion * BIAS_SCALE).powi(2);
        let motion_w = p_motion.powi(2) / v + p_motion;
        let vertical_w = motion_w / params.bias_vertical_forgetting_factor.max(1e-10);

        // line 20 of Algorithm 2
        let p_rest = (params.bias_sigma_rest * BIAS_SCALE).powi(2);
        let rest_w = p_rest.powi(2) / v + p_rest;

        Self {
            p0,
            v,
            motion_w,
            vertical_w,
            rest_w,
        }
    }
}

/// Coefficients used for in a [`Vqf`] system.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VqfCoefficients {
    /// Numerator coefficients for the acceleration low-pass filter.
    accel_b: [f32; 3],
    /// Denominator coefficients for the acceleration low-pass filter.
    accel_a: [f32; 2],
    /// Numerator coefficients of the gyroscope measurement low-pass filter for
    /// rest phase detection.
    rest_gyro_b: [f32; 3],
    /// Denominator coefficients of the gyroscope measurement low-pass filter
    /// for rest phase detection.
    rest_gyro_a: [f32; 2],
    /// Numerator coefficients of the accelerometer measurement low-pass filter
    /// for rest phase detection.
    rest_accel_b: [f32; 3],
    /// Denominator coefficients of the accelerometer measurement low-pass
    /// filter for rest phase detection.
    rest_accel_a: [f32; 2],
    /// Coefficients for the gyroscope bias estimation.
    bias: VqfBiasCoefficients,
}

/// Filter for orientation estimation using accelerometer and gyroscope
/// measurements.
///
/// # Example
///
/// ```rust
/// use nalgebra::Vector3;
/// use std::time::Duration;
/// use vqf::{Vqf, VqfParameters};
///
/// let params = VqfParameters::default();
/// let imu_sample_rate = Duration::from_secs_f32(0.01); // 100 Hz
///
/// let mut vqf = Vqf::new(imu_sample_rate, imu_sample_rate, params);
///
/// vqf.update(Vector3::new(0.1, 0.1, 0.5), Vector3::new(0., 0., -9.81));
///
/// let orientation = vqf.orientation();
/// assert!(orientation.coords.iter().all(|&x| x.is_finite()));
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Vqf {
    /// The filter coefficients.
    pub coefficients: VqfCoefficients,
    /// The parameters of the filter.
    parameters: VqfParameters,
    /// The state of the filter.
    pub state: VqfState,
    /// The time between gyro samples.
    gyro_rate: Duration,
    /// The time between accelerometer samples.
    accel_rate: Duration,
}

impl Vqf {
    /// Creates a new VQF filter instance.
    ///
    /// # Parameters
    /// - `gyro_sampling_rate`: Duration between gyroscope samples.
    /// - `accel_sampling_rate`: Duration between accelerometer samples.
    /// - `params`: Filter parameters for controlling bias estimation and filter
    ///   behavior.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::time::Duration;
    /// use vqf::{Vqf, VqfParameters};
    ///
    /// let params = VqfParameters::default();
    /// let imu_sample_rate = Duration::from_millis(10); // 100 Hz
    ///
    /// let vqf = Vqf::new(imu_sample_rate, imu_sample_rate, params);
    /// ```
    #[must_use]
    pub fn new(
        gyro_sampling_rate: Duration,
        accel_sampling_rate: Duration,
        params: VqfParameters,
    ) -> Self {
        let (accel_coefficients_b, accel_coefficients_a) =
            second_order_butterworth(params.tau_accelerometer, accel_sampling_rate);

        let (rest_gyro_coefficients_b, rest_gyro_coefficients_a) =
            second_order_butterworth(params.rest_filter_tau, gyro_sampling_rate);

        let (rest_accel_coefficients_b, rest_accel_coefficients_a) =
            second_order_butterworth(params.rest_filter_tau, accel_sampling_rate);

        let bias = VqfBiasCoefficients::new(accel_sampling_rate, &params);

        let coefficients = VqfCoefficients {
            accel_b: accel_coefficients_b,
            accel_a: accel_coefficients_a,
            rest_gyro_b: rest_gyro_coefficients_b,
            rest_gyro_a: rest_gyro_coefficients_a,
            rest_accel_b: rest_accel_coefficients_b,
            rest_accel_a: rest_accel_coefficients_a,
            bias,
        };

        Self {
            state: VqfState::new(
                &coefficients,
                &params,
                accel_sampling_rate,
                gyro_sampling_rate,
            ),
            coefficients,
            parameters: params,
            gyro_rate: gyro_sampling_rate,
            accel_rate: accel_sampling_rate,
        }
    }

    /// Returns `true` if a rest has been detected.
    ///
    /// This will only return `true` if the duration is `>=`
    /// [`VqfParameters::rest_min_duration`].
    #[inline]
    #[must_use]
    pub fn is_rest_phase(&self) -> bool {
        self.state
            .rest
            .is_some_and(|rest_duration| rest_duration >= self.parameters.rest_min_duration)
    }

    /// Returns the 6D orientation quaternion.
    ///
    /// This is the orientation without the magnotometer correction.
    #[must_use]
    pub fn orientation(&self) -> UnitQuaternion<f32> {
        self.state.accelerometer_quat * self.state.gyroscope_quat
    }

    /// Updates the filter with new gyroscope and accelerometer readings.
    ///
    /// # Parameters
    /// - `gyro`: Gyroscope reading in radians per second.
    /// - `accel`: Accelerometer reading in meters/s^2.
    ///
    /// # Example
    ///
    /// ```rust
    /// use nalgebra::Vector3;
    /// use std::time::Duration;
    /// use vqf::{Vqf, VqfParameters};
    ///
    /// let params = VqfParameters::default();
    /// let imu_sample_rate = Duration::from_millis(10); // 100 Hz
    ///
    /// let mut vqf = Vqf::new(imu_sample_rate, imu_sample_rate, params);
    ///
    /// let gyro_reading = Vector3::new(0.4, 0.51, 0.66);
    /// let accel_reading = Vector3::new(0., 0., -9.81);
    ///
    /// vqf.update(gyro_reading, accel_reading);
    /// ```
    pub fn update(&mut self, gyro: Vector3<f32>, accel: Vector3<f32>) {
        self.gyro_update(gyro);
        self.accel_update(accel);
    }

    /// Perform the gyroscope update step, using the gyroscope readings in
    /// rad/s.
    fn gyro_update(&mut self, gyro: Vector3<f32>) {
        if self.parameters.do_rest_bias_estimation {
            self.gyro_rest_detection(gyro);
        }

        let unbiased_gyro = gyro - self.state.bias;
        let gyro_norm = unbiased_gyro.norm();

        // if the gyro norm is approaching zero, do not perform update
        if gyro_norm <= f32::EPSILON {
            return;
        }

        // predict the new orientation (eq. 3)
        let half_angle = (self.gyro_rate.as_secs_f32() * gyro_norm) / 2.;

        let cosine = (half_angle).cos();
        let sine = (half_angle).sin() / gyro_norm;
        let gyro_step = Quaternion::new(cosine, sine * gyro.x, sine * gyro.y, sine * gyro.z);

        self.state.gyroscope_quat =
            UnitQuaternion::from_quaternion(self.state.gyroscope_quat.quaternion() * gyro_step);
    }

    #[inline]
    fn gyro_rest_detection(&mut self, gyro: Vector3<f32>) {
        let gyro_lp = self.state.rest_gyro_low_pass.filter(gyro);
        let deviation = gyro - gyro_lp;
        let squared_deviation = deviation.dot(&deviation);

        let bias_clip = self.parameters.bias_clip.to_radians();
        if squared_deviation >= self.parameters.rest_threshold_gyro.powi(2)
            || gyro_lp.abs().max() > bias_clip
        {
            self.state.rest = None;
        }
    }

    /// Perform the accelerometer update step, using the accelerometer readings
    /// in m/s^2.
    fn accel_update(&mut self, accel: Vector3<f32>) {
        // ignore 0 acceleration
        if accel == Vector3::zeros() {
            return;
        }

        if self.parameters.do_rest_bias_estimation {
            self.accel_rest_detection(accel);
        }

        // first correct the accelerometer reading for gravity
        let acc_earth = self.state.gyroscope_quat * accel;
        let accel_low_pass = self.state.accelerometer_low_pass.filter(acc_earth);

        let acc_earth = (self.state.accelerometer_quat * accel_low_pass).normalize();

        // inclination correction
        let q_w = ((acc_earth.z + 1.0) / 2.0).sqrt(); // equation 4

        // equation 5
        let inclination_correction = if q_w > f32::EPSILON {
            let q_x = acc_earth.y / (2.0 * q_w);
            let q_y = -acc_earth.x / (2.0 * q_w);
            let q_z = 0.0;

            let inclination_correction = Quaternion::new(q_w, q_x, q_y, q_z);
            UnitQuaternion::from_quaternion(inclination_correction)
        } else {
            UnitQuaternion::from_quaternion(Quaternion::new(0., 1., 0., 0.))
        };

        self.state.accelerometer_quat = inclination_correction * self.state.accelerometer_quat;
        self.bias_estimation_step(acc_earth);
    }

    #[inline]
    fn accel_rest_detection(&mut self, acc: Vector3<f32>) {
        let accel_lp = self.state.rest_accel_low_pass.filter(acc);
        let deviation = acc - accel_lp;
        let squared_deviation = deviation.dot(&deviation);

        if squared_deviation >= self.parameters.rest_threshold_accel.powi(2) {
            self.state.rest = None;
        } else {
            self.state.rest = Some(self.state.rest.unwrap_or_default() + self.accel_rate);
        }
    }

    /// Perform the bias estimation step.
    ///
    /// This is roughly equal to the `BiasEstimationStep` procedure from
    /// Algorithm 2 in the paper.
    fn bias_estimation_step(&mut self, acc_earth: Vector3<f32>) {
        let bias_clip = self.parameters.bias_clip.to_radians();
        let mut bias = self.state.bias;

        let accel_gyro_quat = self.orientation();
        // R from line 23
        let r = accel_gyro_quat.to_rotation_matrix();

        // R b_hat from line 25, only x and y components are used
        // as the z component is the bias of the magnetometer
        let rb_hat = Vector2::new(
            r[(0, 0)] * bias[0] + r[(0, 1)] * bias[1] + r[(0, 2)] * bias[2],
            r[(1, 0)] * bias[0] + r[(1, 1)] * bias[1] + r[(1, 2)] * bias[2],
        );

        // line 24 from Algorithm 2
        let r = self
            .state
            .motion_bias_estimate_rotation_low_pass
            .filter(*r.matrix());

        // line 25 from Algorithm 2
        let bias_lp = self.state.motion_bias_estimate_low_pass.filter(rb_hat);

        // update the bias estimate for the respecive Kalman filter
        let (e, r, w) = if self.is_rest_phase() && self.parameters.do_rest_bias_estimation {
            (
                self.state.rest_gyro_low_pass.last_output - bias,
                Matrix3::identity(),
                Some(Vector3::from_element(self.coefficients.bias.rest_w)),
            )
        } else if self.parameters.do_bias_estimation {
            let acc_rate = self.accel_rate.as_secs_f32();
            (
                Vector3::new(
                    -acc_earth.y / acc_rate + bias_lp.x
                        - r[(0, 0)] * bias[0]
                        - r[(0, 1)] * bias[1]
                        - r[(0, 2)] * bias[2],
                    acc_earth.x / acc_rate + bias_lp.y
                        - r[(1, 0)] * bias[0]
                        - r[(1, 1)] * bias[1]
                        - r[(1, 2)] * bias[2],
                    0.0,
                ),
                r,
                Some(Vector3::new(
                    self.coefficients.bias.motion_w,
                    self.coefficients.bias.motion_w,
                    self.coefficients.bias.vertical_w,
                )),
            )
        } else {
            (Vector3::zeros(), r, None)
        };

        // Kalman filter update
        // step 1: P = P + V (also increase covariance if there is no measurement
        // update!)
        if self.state.bias_p[(0, 0)] < self.coefficients.bias.p0 {
            self.state.bias_p[(0, 0)] += self.coefficients.bias.v;
        }

        if self.state.bias_p[(1, 1)] < self.coefficients.bias.p0 {
            self.state.bias_p[(1, 1)] += self.coefficients.bias.v;
        }

        if self.state.bias_p[(2, 2)] < self.coefficients.bias.p0 {
            self.state.bias_p[(2, 2)] += self.coefficients.bias.v;
        }

        if let Some(w) = w {
            // clip disagreement to -2..2 degrees
            let bias_clip_vector = Vector3::repeat(bias_clip);
            let e = e.simd_clamp(-bias_clip_vector, bias_clip_vector);

            // step 2: K = P  R^T (W + R P R^T)^-1 (line 36)
            let w_diag = Matrix3::from_diagonal(&w);
            let w_r_p_r_t = w_diag + (r * self.state.bias_p * r.transpose());
            let w_r_p_r_t_inv = w_r_p_r_t
                .try_inverse()
                .expect("(W + R P R^T) isn't a square matrix");
            let k = self.state.bias_p * r.transpose() * w_r_p_r_t_inv;
            // step 3: b = b + k e (line 37)
            bias += k * e;

            // step 4: P = P - K R P (line 38)
            self.state.bias_p -= k * r * self.state.bias_p;

            // ensure that the new bias estimate is within the allowed range
            self.state.bias = bias.map(|x| x.clamp(-bias_clip, bias_clip));
        }
    }

    /// Resets the filter orientation to a specific quaternion.
    ///
    /// This will reset both the gyroscope and accelerometer quaternions,
    /// reinitialize the bias estimation, and reset all filter states.
    ///
    /// # Parameters
    /// - `quat`: The quaternion to reset the orientation to.
    pub fn reset_orientation(&mut self, quat: UnitQuaternion<f32>) {
        // Reset quaternions
        self.state.gyroscope_quat = quat;
        self.state.accelerometer_quat = UnitQuaternion::identity();

        // Reset rest detection
        self.state.rest = None;

        // Reset all filter states
        self.state.accelerometer_low_pass = MeanInitializedLowPassFilter::new(
            self.parameters.tau_accelerometer,
            self.accel_rate,
            self.coefficients.accel_b,
            self.coefficients.accel_a,
        );

        self.state.rest_gyro_low_pass = MeanInitializedLowPassFilter::new(
            self.parameters.rest_filter_tau,
            self.gyro_rate,
            self.coefficients.rest_gyro_b,
            self.coefficients.rest_gyro_a,
        );

        self.state.rest_accel_low_pass = MeanInitializedLowPassFilter::new(
            self.parameters.rest_filter_tau,
            self.accel_rate,
            self.coefficients.rest_accel_b,
            self.coefficients.rest_accel_a,
        );

        self.state.motion_bias_estimate_rotation_low_pass = MeanInitializedLowPassFilter::new(
            self.parameters.tau_accelerometer,
            self.accel_rate,
            self.coefficients.accel_b,
            self.coefficients.accel_a,
        );

        self.state.motion_bias_estimate_low_pass = MeanInitializedLowPassFilter::new(
            self.parameters.tau_accelerometer,
            self.accel_rate,
            self.coefficients.accel_b,
            self.coefficients.accel_a,
        );

        // Reset bias estimation
        self.state.bias = Vector3::zeros();
        self.state.bias_p = self.coefficients.bias.p0 * Matrix3::identity();
    }
}
