// TODO: Remove once feature hits stable:
// https://github.com/rust-lang/rust/issues/76560
#![deny(missing_docs)]
#![deny(clippy::undocumented_unsafe_blocks)]
#![doc = include_str!("../README.md")]
#![cfg_attr(feature = "const-realfft", allow(incomplete_features))]
#![cfg_attr(feature = "const-realfft", feature(generic_const_exprs))]
#![warn(clippy::pedantic)]
// We do many casts from usize to f64. This is what triggers this lint. Casting here is fine
// because the usize represents an index in an fft object. In practice these objects will NEVER be
// even close to having 32 bits represent the number of indices.
#![allow(clippy::cast_precision_loss)]
// The pattern `SIZE / 2 + 1` is common in this code. Removing the trailing `+ 1` is confusing.
#![allow(clippy::range_plus_one)]

use realfft::ComplexToReal;
use realfft::RealFftPlanner;
use realfft::RealToComplex;
use rustfft::FftPlanner;
use std::sync::Arc;

pub use rustfft::num_complex;
pub use rustfft::FftNum;

pub mod const_size;
pub mod dyn_size;
/// This module re-exports all the traits under a single namespace to be easily consumed.
///
/// I generally believe glob-imports are to be avoided. There are exceptions though and I believe
/// this is one of those situations. These traits are NOT named when used so having them named when
/// imported does not aid understanding. See [this blog post] for more information.
///
/// [this blog post]: https://drs.is/post/against-globs/
pub mod prelude {
    #[cfg(feature = "const-realfft")]
    pub use crate::const_size::realfft::RealDft;
    #[cfg(feature = "const-realfft")]
    pub use crate::const_size::realfft::RealFft;
    #[cfg(feature = "const-realfft")]
    pub use crate::const_size::realfft::RealIfft;

    pub use crate::const_size::Fft;
    pub use crate::const_size::FftMut;
    pub use crate::const_size::Ifft;
    pub use crate::const_size::IfftMut;

    pub use crate::dyn_size::realfft::DynRealFft;
    pub use crate::dyn_size::realfft::DynRealIfft;

    pub use crate::dyn_size::DynFft;
    pub use crate::dyn_size::DynFftMut;
    pub use crate::dyn_size::DynIfft;
    pub use crate::dyn_size::DynIfftMut;
}

fn with_fft_planner<T: FftNum>(with: impl FnMut(&mut FftPlanner<T>)) {
    generic_singleton::get_or_init_thread_local!(|| FftPlanner::new(), with);
}

fn with_fft_algorithm<T: FftNum>(size: usize, mut with: impl FnMut(Arc<dyn rustfft::Fft<T>>)) {
    with_fft_planner(|planner: &mut FftPlanner<T>| with(planner.plan_fft_forward(size)));
}

fn with_inverse_fft_algorithm<T: FftNum>(
    size: usize,
    mut with: impl FnMut(Arc<dyn rustfft::Fft<T>>),
) {
    let with = |planner: &mut FftPlanner<T>| with(planner.plan_fft_inverse(size));
    with_fft_planner(with);
}

fn with_real_fft_algorithm<T: FftNum>(
    size: usize,
    mut with: impl FnMut(Arc<dyn RealToComplex<T>>),
) {
    let with = |planner: &mut RealFftPlanner<T>| with(planner.plan_fft_forward(size));
    generic_singleton::get_or_init_thread_local!(|| RealFftPlanner::new(), with);
}

fn with_inverse_real_fft_algorithm<T: FftNum>(
    size: usize,
    mut with: impl FnMut(Arc<dyn ComplexToReal<T>>),
) {
    let with = |planner: &mut RealFftPlanner<T>| with(planner.plan_fft_inverse(size));
    generic_singleton::get_or_init_thread_local!(|| RealFftPlanner::new(), with);
}
