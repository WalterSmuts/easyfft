//! Provides an easy FFT api for arrays.
//!
//! ### Example:
//! ```rust
//! use approx::assert_ulps_eq;
//! use easyfft::num_complex::Complex;
//! use easyfft::const_size::Fft;
//! use easyfft::const_size::Ifft;
//!
//! // Define a complex-valued signal
//! let complex_signal = [Complex::new(1.0_f64, 0.0); 10];
//!
//! // Call `.fft()` on the signal to obtain it's discrete fourier transform
//! let complex_signal_dft: [Complex<f64>; 10] = complex_signal.fft();
//!
//! // Call `.ifft()` on the frequency domain signal to obtain it's inverse
//! let complex_signal_dft_idft: [Complex<f64>; 10] = complex_signal_dft.ifft();
//!
//! // Verify the resulting signal is a scaled value of the original signal
//! for (original, manipulated) in complex_signal.iter().zip(complex_signal_dft_idft) {
//!     assert_ulps_eq!(manipulated.re, original.re * complex_signal.len() as f64);
//!     assert_ulps_eq!(manipulated.im, original.im * complex_signal.len() as f64);
//! }
//! ```
use array_init::map_array_init;
use std::cell::UnsafeCell;

#[rustfmt::skip]
use ::realfft::num_complex::Complex;
#[rustfmt::skip]
use ::realfft::FftNum;

use crate::get_fft_algorithm;
use crate::get_inverse_fft_algorithm;

#[cfg(feature = "const-realfft")]
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

/// A trait for performing fast in-place DFT's on structs representing complex signals with a size
/// known at compile time.
pub trait FftMut<T, const SIZE: usize> {
    /// Perform a complex-valued in-place FFT on a signal.
    fn fft_mut(&mut self);
}

/// A trait for performing fast in-place IDFT's on structs representing complex signals with a size
/// known at compile time.
pub trait IfftMut<T, const SIZE: usize> {
    /// Perform a complex-valued in-place IFFT on a signal.
    fn ifft_mut(&mut self);
}

trait StaticScratchFft<T: FftNum, const SIZE: usize>: rustfft::Fft<T> {
    fn process_with_static_scratch(&self, buffer: &mut [Complex<T>; SIZE]);
}

impl<T: FftNum + Default, U: ?Sized + rustfft::Fft<T>, const SIZE: usize> StaticScratchFft<T, SIZE>
    for U
{
    fn process_with_static_scratch(&self, buffer: &mut [Complex<T>; SIZE]) {
        let scratch_buffer_pointer =
            generic_singleton::get_or_init(|| UnsafeCell::new([Complex::default(); SIZE])).get();
        // SAFETY:
        // Issue:
        // * The pointer must be properly aligned.
        // * It must be "dereferenceable" in the sense defined in [the module documentation].
        // * The pointer must point to an initialized instance of `T`.
        // Proof:
        // These invariants are all guaranteed by the generic_singleton crate.
        //
        // Issue:
        // * You must enforce Rust's aliasing rules, since the returned lifetime `'a` is
        //   arbitrarily chosen and does not necessarily reflect the actual lifetime of the data.
        //   In particular, while this reference exists, the memory the pointer points to must
        //   not get accessed (read or written) through any other pointer.
        //
        // Proof:
        // The pointer points towards thread-local storage and is only converted to a reference in
        // this function, without leaking references. Therefore the exclusive reference invariant
        // should hold.
        let scratch_buffer_ref = unsafe { scratch_buffer_pointer.as_mut().unwrap_unchecked() };
        self.process_with_scratch(buffer, scratch_buffer_ref);
    }
}

impl<T: FftNum + Default, const SIZE: usize> Fft<T, SIZE> for [T; SIZE] {
    fn fft(&self) -> [Complex<T>; SIZE] {
        let mut buffer = map_array_init(self, |sample| Complex::new(*sample, T::default()));

        get_fft_algorithm::<T>(SIZE).process_with_static_scratch(&mut buffer);
        buffer
    }
}

impl<T: FftNum + std::default::Default, const SIZE: usize> Fft<T, SIZE> for [Complex<T>; SIZE] {
    fn fft(&self) -> [Complex<T>; SIZE] {
        // Copy into a new buffer
        let mut buffer: [Complex<T>; SIZE] = *self;
        get_fft_algorithm::<T>(SIZE).process_with_static_scratch(&mut buffer);
        buffer
    }
}

impl<T: FftNum + Default, const SIZE: usize> Ifft<T, SIZE> for [Complex<T>; SIZE] {
    fn ifft(&self) -> [Complex<T>; SIZE] {
        let mut buffer = *self;
        get_inverse_fft_algorithm::<T>(SIZE).process_with_static_scratch(&mut buffer);
        buffer
    }
}

impl<T: FftNum + std::default::Default, const SIZE: usize> FftMut<T, SIZE> for [Complex<T>; SIZE] {
    fn fft_mut(&mut self) {
        get_fft_algorithm::<T>(SIZE).process_with_static_scratch(self);
    }
}

impl<T: FftNum + Default, const SIZE: usize> IfftMut<T, SIZE> for [Complex<T>; SIZE] {
    fn ifft_mut(&mut self) {
        get_inverse_fft_algorithm::<T>(SIZE).process_with_static_scratch(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARBITRARY_EVEN_TEST_ARRAY: [f64; 6] = [1.5, 3.0, 2.1, 3.2, 2.2, 3.1];
    const ARBITRARY_ODD_TEST_ARRAY: [f64; 7] = [1.5, 3.0, 2.1, 3.2, 2.2, 3.1, 1.2];

    //TODO: Figure out why this error creeps in and if there is an appropriate constant already
    // defined.
    const ACCEPTABLE_ERROR: f64 = 0.000_000_000_000_01;

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
