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

#[rustfmt::skip]
use ::realfft::ComplexToReal;
#[rustfmt::skip]
use ::realfft::RealFftPlanner;
#[rustfmt::skip]
use ::realfft::RealToComplex;
use rustfft::FftPlanner;
use std::cell::RefCell;
use std::sync::Arc;

pub use rustfft::num_complex::Complex;
pub use rustfft::FftNum;

pub mod const_size;
pub mod dyn_size;
/// This module re-exports all the traits under a single namespace to be easily consumed.
pub mod prelude {
    #[cfg(feature = "const-realfft")]
    pub use crate::const_size::realfft::*;

    pub use crate::const_size::*;
    pub use crate::dyn_size::realfft::*;
    pub use crate::dyn_size::*;
    pub use crate::*;
}

pub(crate) struct PrivateWrapper<T>(T);

// TODO: Consider using UnsafeCell to avoid runtime borrow-checking.
fn get_fft_algorithm<T: FftNum>(size: usize) -> Arc<dyn rustfft::Fft<T>> {
    generic_singleton::get_or_init(|| RefCell::new(PrivateWrapper(FftPlanner::new())))
        .borrow_mut()
        .0
        .plan_fft_forward(size)
}

// TODO: Consider using UnsafeCell to avoid runtime borrow-checking.
fn get_inverse_fft_algorithm<T: FftNum>(size: usize) -> Arc<dyn rustfft::Fft<T>> {
    generic_singleton::get_or_init(|| RefCell::new(PrivateWrapper(FftPlanner::new())))
        .borrow_mut()
        .0
        .plan_fft_inverse(size)
}

// TODO: Consider using UnsafeCell to avoid runtime borrow-checking.
fn get_real_fft_algorithm<T: FftNum>(size: usize) -> Arc<dyn RealToComplex<T>> {
    generic_singleton::get_or_init(|| RefCell::new(PrivateWrapper(RealFftPlanner::new())))
        .borrow_mut()
        .0
        .plan_fft_forward(size)
}

// TODO: Consider using UnsafeCell to avoid runtime borrow-checking.
fn get_inverse_real_fft_algorithm<T: FftNum>(size: usize) -> Arc<dyn ComplexToReal<T>> {
    generic_singleton::get_or_init(|| RefCell::new(PrivateWrapper(RealFftPlanner::new())))
        .borrow_mut()
        .0
        .plan_fft_inverse(size)
}
