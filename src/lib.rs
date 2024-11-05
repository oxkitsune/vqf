//! An implementation of the Vector-Quaternion Filter (VQF) for orientation estimation.

mod lowpass;

use core::f32;
use std::time::Duration;

use lowpass::{second_order_butterworth, MeanInitializedLowPassFilter};
use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};

/// The state of a [`Vqf`] filter.
pub struct VqfState {
    gyroscope_quat: UnitQuaternion<f32>,
    accelerometer_quat: UnitQuaternion<f32>,
    rest_detected: bool,
    last_accelerometer: Vector3<f32>,
    accelerometer_low_pass: MeanInitializedLowPassFilter<3>,
    bias: Vector3<f32>,
    bias_sigma: Matrix3<f32>,

    /// The current estimated gyroscope bias.
    gyro_bias: Vector3<f32>,
}

impl VqfState {
    #[must_use]
    pub fn new(coefficients: &VqfCoefficients, params: &VqfParameters, acc_rate: Duration) -> Self {
        Self {
            gyroscope_quat: UnitQuaternion::identity(),
            accelerometer_quat: UnitQuaternion::identity(),
            rest_detected: false,
            last_accelerometer: Vector3::zeros(),
            accelerometer_low_pass: MeanInitializedLowPassFilter::new(
                params.tau_accelerometer,
                1. / acc_rate.as_secs_f32(),
                coefficients.acc_b,
                coefficients.acc_a,
            ),
            bias: Vector3::zeros(),
            bias_sigma: Matrix3::from_element(f32::NAN),
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
    bias_p0: f32,
    bias_v: f32,
    bias_motion_w: f32,
    bias_vertical_w: f32,
    bias_rest_w: f32,
}

impl VqfBiasCoefficients {
    /// Initialize the bias estimation coefficients using the given parameters.
    ///
    /// This function is roughly equivalent to the "InitializeKalmanFilter" procedure in
    /// Algorithm 2 of the original paper.
    #[must_use]
    fn new(accelerometer_rate: Duration, params: &VqfParameters) -> Self {
        // line 17 of Algorithm 2, the initial variance of the bias
        let bias_p0 = (params.bias_sigma_initial * 100.0).powi(2);

        // line 18 of Algorithm 2
        // System noise increases the variance from 0 to (0.1 °/s)^2 in `bias_forgetting_time` duration
        let bias_v = (0.1 * 100_f32).powi(2) * accelerometer_rate.as_secs_f32()
            / params.bias_forgetting_time.as_secs_f32();

        // line 19 of Algorithm 2
        let p_motion = (params.bias_sigma_motion * 100.0).powi(2);
        let bias_motion_w = p_motion.powi(2) / bias_v + p_motion;
        let bias_vertical_w = bias_motion_w / params.bias_vertical_forgetting_factor.max(1e-10);

        // line 20 of Algorithm 2
        let p_rest = (params.bias_sigma_rest * 100.0).powi(2);
        let bias_rest_w = p_rest.powi(2) / bias_v + p_rest;

        Self {
            bias_p0,
            bias_v,
            bias_motion_w,
            bias_vertical_w,
            bias_rest_w,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VqfCoefficients {
    acc_b: [f32; 3],
    acc_a: [f32; 2],
    rest_gyro_b: [f32; 3],
    rest_gyro_a: [f32; 2],
    rest_acc_b: [f32; 3],
    rest_acc_a: [f32; 2],
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
    /// The sampling rate of the gyroscope, in seconds.
    gyro_rate: f32,
    /// The sampling rate of the accelerometer in seconds.
    accel_rate: f32,
}

impl Vqf {
    // TODO: Initialization step
    #[must_use]
    pub fn new(
        gyro_rate: Duration,
        accelerometer_tau: Duration,
        accelerometer_rate: Duration,
        params: VqfParameters,
    ) -> Self {
        let (acc_coefficients_b, acc_coefficients_a) =
            second_order_butterworth(accelerometer_tau, accelerometer_rate);

        let (rest_gyro_coefficients_b, rest_gyro_coefficients_a) =
            second_order_butterworth(params.rest_filter_tau, gyro_rate);

        let (rest_acc_coefficients_b, rest_acc_coefficients_a) =
            second_order_butterworth(params.rest_filter_tau, accelerometer_rate);

        let bias = VqfBiasCoefficients::new(accelerometer_rate, &params);

        let coefficients = VqfCoefficients {
            acc_b: acc_coefficients_b,
            acc_a: acc_coefficients_a,
            rest_gyro_b: rest_gyro_coefficients_b,
            rest_gyro_a: rest_gyro_coefficients_a,
            rest_acc_b: rest_acc_coefficients_b,
            rest_acc_a: rest_acc_coefficients_a,
            bias,
        };

        Self {
            state: VqfState::new(&coefficients, &params, accelerometer_rate),
            coefficients,
            parameters: params,
            gyro_rate: gyro_rate.as_secs_f32(),
            accel_rate: accelerometer_rate.as_secs_f32(),
        }
    }

    /// Update step of the filter, using the gyroscope and accelerometer readings.
    pub fn update(&mut self, gyro: Vector3<f32>, accel: Vector3<f32>) {
        self.gyro_update(gyro);
        self.accel_update(accel);
    }

    /// Perform the gyroscope update step, using the gyroscope readings in rad/s.
    fn gyro_update(&mut self, gyro: Vector3<f32>) {
        // todo: rest detection
        let unbiased_gyro = gyro - self.state.gyro_bias;
        let gyro =
            UnitQuaternion::from_euler_angles(unbiased_gyro.x, unbiased_gyro.y, unbiased_gyro.z);

        // predict the new orientation (eq. 3)
        let angle = self.gyro_rate * gyro.norm();

        let cosine = angle.cos();
        let sine = angle.sin();
        let gyro_step = Quaternion::new(cosine, sine * gyro.i, sine * gyro.j, sine * gyro.k);

        self.state.gyroscope_quat =
            UnitQuaternion::from_quaternion((self.state.gyroscope_quat.quaternion() * gyro_step));
    }

    /// Perform the accelerometer update step, using the accelerometer readings in m/s^2.
    fn accel_update(&mut self, accel: Vector3<f32>) {
        // ignore 0 acceleration
        if accel == Vector3::zeros() {
            return;
        }

        // todo: rest detection

        // first correct the accelerometer reading for gravity
        let earth_acc = self.state.gyroscope_quat * accel;
        let acc_low_pass = self.state.accelerometer_low_pass.filter(earth_acc);

        let earth_acc = (self.state.accelerometer_quat * acc_low_pass).normalize();

        // inclination correction
        let q_w = (earth_acc.z + 1.0 / 2.0).sqrt(); // equation 4

        // equation 5
        let corrected_acc_quat = if q_w > f32::EPSILON {
            let q_x = -earth_acc.y / (2.0 * q_w);
            let q_y = earth_acc.x / (2.0 * q_w);
            let q_z = 0.0;

            let inclination_correction = Quaternion::new(q_w, q_x, q_y, q_z);
            UnitQuaternion::from_quaternion(
                (self.state.accelerometer_quat.quaternion() * inclination_correction).normalize(),
            )
        } else {
            UnitQuaternion::identity()
        };

        self.state.accelerometer_quat = corrected_acc_quat * self.state.accelerometer_quat;

        // TODO: bias estimation
    }
}
