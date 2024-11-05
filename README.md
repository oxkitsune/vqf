# VQF

[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](https://github.com/oxkitsune/vqf#license)
[![Crates.io](https://img.shields.io/crates/v/vqf.svg)](https://crates.io/crates/vqf)
[![Downloads](https://img.shields.io/crates/d/vqf.svg)](https://crates.io/crates/vqf)
[![Docs](https://docs.rs/vqf/badge.svg)](https://docs.rs/vqf/latest/vqf/)

A Rust implementation of the Versatile Quaternion-based Filter (VQF) algorithm, as described in [this paper](https://arxiv.org/pdf/2203.17024).

> [!NOTE]
> Currently this crate does *not* implement the magnometer update.

## Example

```rust
use nalgebra::Vector3;
use std::time::Duration;
use vqf::{Vqf, VqfParameters};

let gyro_rate = Duration::from_secs_f32(0.01); // 100Hz
let accel_rate = Duration::from_secs_f32(0.01);

let params = VqfParameters::default();
let mut vqf = Vqf::new(gyro_rate, accel_rate, params);

let gyro_data = Vector3::new(0.01, 0.02, -0.01); // rad/s
let accel_data = Vector3::new(0.0, 0.0, 9.81); // m/s^2

vqf.update(gyro_data, accel_data);

let orientation = vqf.orientation();
println!("Current orientation: {:?}", orientation);
```
