//! An implementation of the Vector-Quaternion Filter (VQF) for orientation estimation.

mod butterworth;

use std::time::Duration;

use nalgebra::{Quaternion, UnitQuaternion, Vector3};

/// The state of a [`Vqf`] filter.
pub struct VqfState {
    /// The current estimated orientation.
    orientation: Quaternion<f32>,

    /// The current estimated gyroscope bias.
    gyro_bias: Vector3<f32>,

    gyroscope_quat: Quaternion<f32>,
    accelerometer_quat: Quaternion<f32>,
}

/// The VQF filter.
pub struct Vqf {
    /// The state of the filter.
    state: VqfState,
    /// The sample rate of the gyroscope.
    gyro_rate: Duration,
    /// The sample rate of the accelerometer.
    accel_rate: Duration,
}

impl Vqf {
    // TODO: Initialization step
    //

    /// Perform the orientation prediction step, using the gyroscope readings in radians per second.
    fn gyro_update(&mut self, gyro: Vector3<f32>) {
        let biased_gyro = UnitQuaternion::from_euler_angles(gyro.x, gyro.y, gyro.z);

        let predicted_gyro = self.state.orientation
            * (self.gyro_rate.as_secs_f32() * biased_gyro.norm() * biased_gyro.quaternion());
    }
}
