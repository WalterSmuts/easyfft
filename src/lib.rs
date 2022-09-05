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

pub mod dynfft;
#[cfg(feature = "realfft")]
pub mod realfft;

/// A trait for performing fast DFT's on structs representing complex signals with a size known at
/// compile time.
pub trait Fft<T, const SIZE: usize> {
    /// Perform a complex-valued FFT on a signal with input and outpiut size `SIZE`.
    fn fft(&self) -> [Complex<T>; SIZE];
}

/// A trait for performing fast IDFT's on structs representing complex signals with a size known at
/// compile time.
pub trait Ifft<T, const SIZE: usize> {
    /// Perform a complex-valued IFFT on a signal with input and outpiut size `SIZE`.
    fn ifft(&self) -> [Complex<T>; SIZE];
}

impl<T: FftNum + Default, const SIZE: usize> Fft<T, SIZE> for [T; SIZE] {
    fn fft(&self) -> [Complex<T>; SIZE] {
        // SAFETY:
        // The function will only return None if the iterator is exhausted before the array
        // is filled. The compiler ensures that the size of this array is always equal to the size
        // of the array passed in, and therefore the size of the iterator, by the SIZE const
        // generic argument. Therefore this function will never return a None value.
        //
        // TODO: Remove this and call `map_array_init` once this commit gets released:
        // https://github.com/Manishearth/array-init/commit/9496096148b4933416a2435de65b9fa844872127
        let mut buffer: [Complex<T>; SIZE] = unsafe {
            array_init::from_iter(
                self.iter()
                    .map(|sample| Complex::new(*sample, T::default())),
            )
            .unwrap_unchecked()
        };

        get_fft_algorithm::<T>(SIZE).process(&mut buffer);
        buffer
    }
}

impl<T: FftNum, const SIZE: usize> Fft<T, SIZE> for [Complex<T>; SIZE] {
    fn fft(&self) -> [Complex<T>; SIZE] {
        // Copy into a new buffer
        let mut buffer: [Complex<T>; SIZE] = *self;
        get_fft_algorithm::<T>(SIZE).process(&mut buffer);
        buffer
    }
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

impl<T: FftNum + Default, const SIZE: usize> Ifft<T, SIZE> for [Complex<T>; SIZE] {
    fn ifft(&self) -> [Complex<T>; SIZE] {
        let mut buffer = *self;
        get_inverse_fft_algorithm::<T>(SIZE).process(&mut buffer);
        buffer
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    const ARBITRARY_EVEN_TEST_ARRAY: [f64; 6] = [1.5, 3.0, 2.1, 3.2, 2.2, 3.1];
    const ARBITRARY_ODD_TEST_ARRAY: [f64; 7] = [1.5, 3.0, 2.1, 3.2, 2.2, 3.1, 1.2];

    //TODO: Figure out why this error creeps in and if there is an appropriate constant already
    // defined.
    const ACCEPTABLE_ERROR: f64 = 0.00000000000001;

    fn fft_and_ifft_are_inverse_operations<const SIZE: usize>(array: [f64; SIZE]) {
        let converted: Vec<_> = array
            .fft()
            .ifft()
            .iter_mut()
            .map(|sample| *sample / array.len() as f64)
            .collect();
        assert_eq!(array.len(), converted.len());
        for (converted, original) in converted.iter().zip(array.iter()) {
            approx::assert_ulps_eq!(converted.re, original, epsilon = ACCEPTABLE_ERROR);
            approx::assert_ulps_eq!(converted.im, 0.0);
        }
    }

    #[test]
    fn fft_and_ifft_are_inverse_operations_even() {
        fft_and_ifft_are_inverse_operations(ARBITRARY_EVEN_TEST_ARRAY);
    }

    #[test]
    fn fft_and_ifft_are_inverse_operations_odd() {
        fft_and_ifft_are_inverse_operations(ARBITRARY_ODD_TEST_ARRAY);
    }
}
