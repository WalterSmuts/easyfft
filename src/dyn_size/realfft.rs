//! Provides an easy real-valued FFT api for slices.
//!
//! ### Example:
//! ```rust
//! use approx::assert_ulps_eq;
//! use easyfft::dyn_size::realfft::DynRealFft;
//! use easyfft::dyn_size::realfft::DynRealIfft;
//!
//! // Define a real-valued signal slice
//! let real_signal: &[f64] = &[1.0_f64; 10];
//!
//! // Call `.real_fft()` on the signal to obtain it's discrete fourier transform
//! let real_signal_dft = real_signal.real_fft();
//! // Call `.real_ifft` on the DynRealDft signal to obtain it's real inverse
//! let real_signal_dft_idft: &[f64] = &real_signal_dft.real_ifft();
//!
//! // Verify the signals are original and manipulated signals are the same length
//! assert_eq!(real_signal.len(), real_signal_dft_idft.len());
//!
//! // Verify the resulting ifft is a scaled version of the original signal
//! for (original, manipulated) in real_signal.iter().zip(real_signal_dft_idft) {
//!     assert_ulps_eq!(*manipulated, original * 10.0);
//! }
//! ```
use rustfft::num_complex::Complex;
use rustfft::FftNum;
use std::ops::Deref;

use crate::get_inverse_real_fft_algorithm;
use crate::get_real_fft_algorithm;

/// A trait for performing fast DFT's on structs representing real signals with a size not known at
/// compile time.
pub trait DynRealFft<T> {
    /// Perform a real-valued FFT on a signal with input size `SIZE` and output size `SIZE / 2 + 1`.
    fn real_fft(&self) -> DynRealDft<T>;
}

/// A trait for performing fast IDFT's on structs representing real signals with a size not known
/// at compile time.
pub trait DynRealIfft<T> {
    /// Perform a real-valued IFFT on a signal which originally had input size `SIZE`.
    fn real_ifft(&self) -> Box<[T]>;
}

// TODO: Define constructor for creating this type manually
/// The result of calling [`DynRealFft::real_fft`].
///
/// As [explained] by the author of the [realfft crate], a real valued signal can have some
/// optimizations applied when calculating it's [discrete fourier transform]. This involves only
/// returning half the complex frequency domain signal since the other half can be inferred because
/// the DFT of a real signal is [known to be symmetric]. This poses a problem when attempting to do
/// the inverse discrete fourier transform since a signal of size `SIZE` would return a
/// complex signal of size `SIZE / 2 - 1`. Note that `SIZE` gets mapped to
/// `SIZE / 2 + 1` and the index of an array is a natural number, so we're working with lossy
/// integer division here. Specifically, observe that __BOTH__ a signal of length `5` and
/// `4` would be mapped to a DFT of length `3`. The length of the resulting signal is not enough to
/// determine the appropriate inverse.
///
/// The solution is to wrap the array in a different type which contains extra type information.
/// This newly created type is the [`DynRealDft`] and has an extra field indicating which sized
/// signal was used to create it.
///
/// [explained]: https://docs.rs/realfft/latest/realfft/index.html#real-to-complex
/// [discrete fourier transform]: https://en.wikipedia.org/wiki/Discrete_Fourier_transform
/// [realfft crate]: https://docs.rs/realfft/latest/realfft/index.html
/// [known to be symmetric]: https://en.wikipedia.org/wiki/Discrete_Fourier_transform#DFT_of_real_and_purely_imaginary_signals
/// [phantom type]: https://doc.rust-lang.org/rust-by-example/generics/phantom.html
#[derive(Debug, Clone)]
pub struct DynRealDft<T> {
    original_length: usize,
    inner: Box<[Complex<T>]>,
}

impl<T> Deref for DynRealDft<T> {
    type Target = [Complex<T>];

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> DynRealDft<T> {
    /// Get a slice of all the frequency bins excluding the first (and, if the original signal has
    /// an even number of samples, last) bin(s).
    #[must_use]
    pub fn get_frequency_bins(&self) -> &[Complex<T>] {
        // TODO: Consider an unchecked unwrap
        let wanted_len = self.original_length - 1;
        &self.inner[1..wanted_len / 2]
    }

    /// Get a mutable slice of all the frequency bins excluding the first (and, if the original
    /// signal has an even number of samples, last) bin(s).
    pub fn get_frequency_bins_mut(&mut self) -> &mut [Complex<T>] {
        // TODO: Consider an unchecked unwrap
        let wanted_len = self.original_length - 1;
        &mut self.inner[1..wanted_len / 2]
    }

    /// Get an immutable reference to the constant offset of the signal, i.e. the zeroth frequency
    /// bin.
    #[must_use]
    pub fn get_offset(&self) -> &T {
        &self.inner[0].re
    }

    /// Get an mutable reference to the constant offset of the signal, i.e. the zeroth frequency
    /// bin.
    pub fn get_offset_mut(&mut self) -> &mut T {
        &mut self.inner[0].re
    }
}

impl<T> From<DynRealDft<T>> for Box<[Complex<T>]> {
    fn from(dyn_real_dft: DynRealDft<T>) -> Self {
        dyn_real_dft.inner
    }
}

impl<T: FftNum + Default> DynRealFft<T> for [T] {
    fn real_fft(&self) -> DynRealDft<T> {
        let r2c = get_real_fft_algorithm::<T>(self.len());

        // TODO: Remove default dependency and unnesasary initialization
        let mut output = Vec::with_capacity(self.len() / 2 + 1);
        for _ in 0..self.len() / 2 + 1 {
            output.push(Complex::default());
        }

        // TODO: Consider an unchecked unwrap
        r2c.process(&mut self.to_vec(), &mut output).unwrap();
        DynRealDft {
            inner: output.into_boxed_slice(),
            original_length: self.len(),
        }
    }
}

impl<T: FftNum + Default> DynRealIfft<T> for DynRealDft<T> {
    fn real_ifft(&self) -> Box<[T]> {
        let c2r = get_inverse_real_fft_algorithm::<T>(self.original_length);

        let mut output = Vec::with_capacity(self.original_length);
        for _ in 0..self.original_length {
            output.push(T::default());
        }
        // TODO: Consider an unchecked unwrap
        c2r.process(
            &mut Into::<Box<[Complex<T>]>>::into(self.clone()),
            &mut output,
        )
        .unwrap();
        output.into_boxed_slice()
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

    fn real_fft_and_real_ifft_are_inverse_operations(array: &[f64]) {
        let converted: Vec<_> = array
            .real_fft()
            .real_ifft()
            .iter_mut()
            .map(|sample| *sample / array.len() as f64)
            .collect();
        assert_eq!(array.len(), converted.len());
        for (converted, original) in converted.iter().zip(array.iter()) {
            approx::assert_ulps_eq!(converted, original, epsilon = ACCEPTABLE_ERROR);
        }
    }

    #[test]
    fn real_fft_and_real_ifft_are_inverse_operations_even() {
        real_fft_and_real_ifft_are_inverse_operations(&ARBITRARY_EVEN_TEST_ARRAY);
    }

    #[test]
    fn real_fft_and_real_ifft_are_inverse_operations_odd() {
        real_fft_and_real_ifft_are_inverse_operations(&ARBITRARY_ODD_TEST_ARRAY);
    }
}
