//! An implementation of the Vector-Quaternion Filter (VQF) for orientation estimation.

mod lowpass;

use core::f32;
use std::time::Duration;

use lowpass::{second_order_butterworth, MeanInitializedLowPassFilter};
use nalgebra::{
    Matrix3, Quaternion, Rotation3, SVector, SimdPartialOrd, UnitQuaternion, Vector2, Vector3,
};

/// The state of a [`Vqf`] filter.
pub struct VqfState {
    gyroscope_quat: UnitQuaternion<f32>,
    accelerometer_quat: UnitQuaternion<f32>,
    rest_duration: Option<Duration>,
    last_accelerometer: Vector3<f32>,
    accelerometer_low_pass: MeanInitializedLowPassFilter<3, 1>,
    rest_gyro_low_pass: MeanInitializedLowPassFilter<3, 1>,
    rest_accel_low_pass: MeanInitializedLowPassFilter<3, 1>,
    motion_bias_estimate_rotation_low_pass: MeanInitializedLowPassFilter<3, 3>,
    motion_bias_estimate_low_pass: MeanInitializedLowPassFilter<2, 1>,
    bias: SVector<f32, 3>,
    bias_p: Matrix3<f32>,

    /// The current estimated gyroscope bias.
    gyro_bias: Vector3<f32>,
}

impl VqfState {
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
            rest_duration: None,
            last_accelerometer: Vector3::zeros(),
            accelerometer_low_pass: MeanInitializedLowPassFilter::new(
                params.tau_accelerometer,
                1. / accel_rate.as_secs_f32(),
                coefficients.accel_b,
                coefficients.accel_a,
            ),
            rest_gyro_low_pass: MeanInitializedLowPassFilter::new(
                params.rest_filter_tau,
                1. / gyro_rate.as_secs_f32(),
                coefficients.rest_gyro_b,
                coefficients.rest_gyro_a,
            ),
            rest_accel_low_pass: MeanInitializedLowPassFilter::new(
                params.rest_filter_tau,
                1. / accel_rate.as_secs_f32(),
                coefficients.rest_accel_b,
                coefficients.rest_accel_a,
            ),
            // TODO: is this correct?
            // we reuse the accelerometer coefficients for the motion bias
            motion_bias_estimate_rotation_low_pass: MeanInitializedLowPassFilter::new(
                params.tau_accelerometer,
                1. / accel_rate.as_secs_f32(),
                coefficients.accel_b,
                coefficients.accel_a,
            ),
            motion_bias_estimate_low_pass: MeanInitializedLowPassFilter::new(
                params.tau_accelerometer,
                1. / accel_rate.as_secs_f32(),
                coefficients.accel_b,
                coefficients.accel_a,
            ),
            bias: Vector3::zeros(),
            bias_p: Matrix3::from_element(f32::NAN),
            gyro_bias: Vector3::zeros(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VqfParameters {
    pub tau_accelerometer: Duration,
    pub do_bias_estimation: bool,
    pub do_rest_bias_estimation: bool,
    pub bias_sigma_initial: f32,
    pub bias_forgetting_time: Duration,
    pub bias_clip: f32,
    pub bias_sigma_motion: f32,
    pub bias_vertical_forgetting_factor: f32,
    pub bias_sigma_rest: f32,
    pub rest_min_duration: Duration,
    pub rest_filter_tau: Duration,
    pub rest_threshold_gyro: f32,
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

#[derive(Debug, Clone, PartialEq)]
pub struct VqfBiasCoefficients {
    p0: f32,
    v: f32,
    motion_w: f32,
    vertical_w: f32,
    rest_w: f32,
}

impl VqfBiasCoefficients {
    /// Initialize the bias estimation coefficients using the given parameters.
    ///
    /// This function is roughly equivalent to the "InitializeKalmanFilter" procedure in
    /// Algorithm 2 of the original paper.
    #[must_use]
    fn new(accelerometer_rate: Duration, params: &VqfParameters) -> Self {
        // line 17 of Algorithm 2, the initial variance of the bias
        let p0 = (params.bias_sigma_initial * 100.0).powi(2);

        // line 18 of Algorithm 2
        // System noise increases the variance from 0 to (0.1 °/s)^2 in `bias_forgetting_time` duration
        let v = (0.1 * 100_f32).powi(2) * accelerometer_rate.as_secs_f32()
            / params.bias_forgetting_time.as_secs_f32();

        // line 19 of Algorithm 2
        let p_motion = (params.bias_sigma_motion * 100.0).powi(2);
        let motion_w = p_motion.powi(2) / v + p_motion;
        let vertical_w = motion_w / params.bias_vertical_forgetting_factor.max(1e-10);

        // line 20 of Algorithm 2
        let p_rest = (params.bias_sigma_rest * 100.0).powi(2);
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

#[derive(Debug, Clone, PartialEq)]
pub struct VqfCoefficients {
    accel_b: [f32; 3],
    accel_a: [f32; 2],
    rest_gyro_b: [f32; 3],
    rest_gyro_a: [f32; 2],
    rest_accel_b: [f32; 3],
    rest_accel_a: [f32; 2],
    bias: VqfBiasCoefficients,
}

/// The VQF filter.
pub struct Vqf {
    /// The filter coefficients.
    coefficients: VqfCoefficients,
    /// The parameters of the filter.
    parameters: VqfParameters,
    /// The state of the filter.
    state: VqfState,
    /// The time between gyro samples.
    gyro_rate: Duration,
    /// The time between accelerometer samples.
    accel_rate: Duration,
}

impl Vqf {
    // TODO: Initialization step
    #[must_use]
    pub fn new(
        gyro_rate: Duration,
        accelerometer_tau: Duration,
        accel_rate: Duration,
        params: VqfParameters,
    ) -> Self {
        let (accel_coefficients_b, accel_coefficients_a) =
            second_order_butterworth(accelerometer_tau, accel_rate);

        let (rest_gyro_coefficients_b, rest_gyro_coefficients_a) =
            second_order_butterworth(params.rest_filter_tau, gyro_rate);

        let (rest_accel_coefficients_b, rest_accel_coefficients_a) =
            second_order_butterworth(params.rest_filter_tau, accel_rate);

        let bias = VqfBiasCoefficients::new(accel_rate, &params);

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
            state: VqfState::new(&coefficients, &params, accel_rate, gyro_rate),
            coefficients,
            parameters: params,
            gyro_rate,
            accel_rate,
        }
    }

    /// Returns `true` if a rest has been detected.
    pub fn rest_detected(&self) -> bool {
        self.state
            .rest_duration
            .is_some_and(|rest_duration| rest_duration >= self.parameters.rest_min_duration)
    }

    /// Update step of the filter, using the gyroscope and accelerometer readings.
    pub fn update(&mut self, gyro: Vector3<f32>, accel: Vector3<f32>) {
        self.gyro_update(gyro);
        self.accel_update(accel);
    }

    /// Perform the gyroscope update step, using the gyroscope readings in rad/s.
    fn gyro_update(&mut self, gyro: Vector3<f32>) {
        if self.parameters.do_rest_bias_estimation {
            self.gyro_rest_detection(gyro);
        }

        let unbiased_gyro = gyro - self.state.gyro_bias;
        let gyro =
            UnitQuaternion::from_euler_angles(unbiased_gyro.x, unbiased_gyro.y, unbiased_gyro.z);

        // predict the new orientation (eq. 3)
        let angle = self.gyro_rate.as_secs_f32() * gyro.norm();

        let cosine = angle.cos();
        let sine = angle.sin();
        let gyro_step = Quaternion::new(cosine, sine * gyro.i, sine * gyro.j, sine * gyro.k);

        self.state.gyroscope_quat =
            UnitQuaternion::from_quaternion(self.state.gyroscope_quat.quaternion() * gyro_step);
    }

    fn gyro_rest_detection(&mut self, gyro: Vector3<f32>) {
        let gyro_lp = self.state.rest_gyro_low_pass.filter(gyro);
        let deviation = gyro - gyro_lp;
        let squared_deviation = deviation.dot(&deviation);

        let bias_clip = self.parameters.bias_clip.to_degrees();
        if squared_deviation >= self.parameters.rest_threshold_gyro.to_degrees().powi(2)
            || gyro_lp.abs().max() > bias_clip
        {
            self.state.rest_duration = None;
        }

        // TODO: store rest deviations
    }

    /// Perform the accelerometer update step, using the accelerometer readings in m/s^2.
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
        let q_w = (acc_earth.z + 1.0 / 2.0).sqrt(); // equation 4

        // equation 5
        let corrected_accel_quat = if q_w > f32::EPSILON {
            let q_x = -acc_earth.y / (2.0 * q_w);
            let q_y = acc_earth.x / (2.0 * q_w);
            let q_z = 0.0;

            let inclination_correction = Quaternion::new(q_w, q_x, q_y, q_z);
            UnitQuaternion::from_quaternion(
                (self.state.accelerometer_quat.quaternion() * inclination_correction).normalize(),
            )
        } else {
            UnitQuaternion::identity()
        };

        self.state.accelerometer_quat = corrected_accel_quat * self.state.accelerometer_quat;

        // TODO: bias estimation
        self.bias_estimation_step(acc_earth);
    }

    fn accel_rest_detection(&mut self, acc: Vector3<f32>) {
        let accel_lp = self.state.rest_accel_low_pass.filter(acc);
        let deviation = acc - accel_lp;
        let squared_deviation = deviation.dot(&deviation);

        if squared_deviation >= self.parameters.rest_threshold_accel.powi(2) {
            self.state.rest_duration = None;
        } else {
            self.state.rest_duration =
                Some(self.state.rest_duration.unwrap_or_default() + self.accel_rate);
        }

        // TODO: store rest deviations
    }

    /// Perform the bias estimation step.
    ///
    /// This is roughly equal to the "BiasEstimationStep" procedure from Algorithm 2
    /// in the paper.
    fn bias_estimation_step(&mut self, acc_earth: Vector3<f32>) {
        let bias_clip = self.parameters.bias_clip.to_degrees();
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
        let (e, r, w) = if self.rest_detected() && self.parameters.do_rest_bias_estimation {
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
        // step 1: P = P + V (also increase covariance if there is no measurement update!)
        if self.state.bias[(0, 0)] < self.coefficients.bias.p0 {
            self.state.bias[(0, 0)] += self.coefficients.bias.v;
        }

        if self.state.bias[(1, 1)] < self.coefficients.bias.p0 {
            self.state.bias[(1, 1)] += self.coefficients.bias.v;
        }

        if self.state.bias[(2, 2)] < self.coefficients.bias.p0 {
            self.state.bias[(2, 2)] += self.coefficients.bias.v;
        }

        if let Some(w) = w {
            // clip disagreement to -2..2 degrees
            let e = e.map(|x| x.clamp(-bias_clip, bias_clip));

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

    /// Returns the 6D orientation quaternion.
    ///
    /// This is the orientation without the magnotometer correction.
    fn orientation(&self) -> UnitQuaternion<f32> {
        self.state.accelerometer_quat * self.state.gyroscope_quat
    }
}
