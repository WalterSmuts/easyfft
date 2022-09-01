#![deny(missing_docs)]
#![doc = include_str!("../README.md")]

use rustfft::FftPlanner;
use std::sync::Arc;

pub use rustfft::num_complex::Complex;
pub use rustfft::FftNum;

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
        // TODO: Remove unnesasary initialization
        let mut buffer: [Complex<T>; SIZE] = [Complex::default(); SIZE];

        let vec: Vec<_> = self
            .iter()
            .map(|sample| Complex::new(*sample, T::default()))
            .collect();

        buffer.copy_from_slice(vec.as_slice());
        get_fft_algorithm::<T, SIZE>().process(&mut buffer);
        buffer
    }
}

// TODO: Add a cache
fn get_fft_algorithm<T: FftNum, const SIZE: usize>() -> Arc<dyn rustfft::Fft<T>> {
    FftPlanner::<T>::new().plan_fft_forward(SIZE)
}

// TODO: Add a cache
fn get_inverse_fft_algorithm<T: FftNum, const SIZE: usize>() -> Arc<dyn rustfft::Fft<T>> {
    FftPlanner::<T>::new().plan_fft_inverse(SIZE)
}

impl<T: FftNum + Default, const SIZE: usize> Ifft<T, SIZE> for [Complex<T>; SIZE] {
    fn ifft(&self) -> [Complex<T>; SIZE] {
        let mut buffer = *self;
        get_inverse_fft_algorithm::<T, SIZE>().process(&mut buffer);
        buffer
    }
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
