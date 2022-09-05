// TODO: Remove once feature hits stable:
// https://github.com/rust-lang/rust/issues/76560
#![deny(missing_docs)]
#![deny(clippy::undocumented_unsafe_blocks)]
#![doc = include_str!("../README.md")]
#![cfg_attr(feature = "realfft", allow(incomplete_features))]
#![cfg_attr(feature = "realfft", feature(generic_const_exprs))]

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
