//! Provides an easy FFT api for slices.
//!
//! ### Example:
//! ```rust
//! use approx::assert_ulps_eq;
//! use easyfft::Complex;
//! use easyfft::dyn_size::DynFft;
//! use easyfft::dyn_size::DynIfft;
//!
//! // Define a complex-valued signal and convert to a slice
//! let complex_signal: &[Complex<f64>] = &[Complex::new(1.0_f64, 0.0); 10];
//!
//! // Call `.fft()` on the signal to obtain it's discrete fourier transform
//! let complex_signal_dft: &[Complex<f64>] = &complex_signal.fft();
//!
//! // Call `.ifft()` on the frequency domain signal to obtain it's inverse
//! let complex_signal_dft_idft: &[Complex<f64>] = &complex_signal_dft.ifft();
//!
//! // Verify the lengths of the original and mutated signals are equal
//! assert_eq!(complex_signal.len(), complex_signal_dft_idft.len());
//!
//! // Verify the mutated signal is a scaled version of the original signal
//! for (original, manipulated) in complex_signal.iter().zip(complex_signal_dft_idft) {
//!     assert_ulps_eq!(manipulated.re, original.re * complex_signal.len() as f64);
//!     assert_ulps_eq!(manipulated.im, original.im * complex_signal.len() as f64);
//! }
//! ```
use rustfft::num_complex::Complex;
use rustfft::FftNum;

#[cfg(feature = "realfft")]
pub mod realfft;

/// A trait for performing fast DFT's on structs representing complex signals with a size not known
/// at compile time.
pub trait DynFft<T> {
    /// Perform a complex-valued FFT on a signal with unknown sizes at compile time.
    fn fft(&self) -> Box<[Complex<T>]>;
}

/// A trait for performing fast IDFT's on structs representing complex signals with a size not
/// known at compile time.
pub trait DynIfft<T> {
    /// Perform a complex-valued IFFT on a signal with unknown sizes at compile time.
    fn ifft(&self) -> Box<[Complex<T>]>;
}

impl<T: FftNum + Default> DynFft<T> for [Complex<T>] {
    fn fft(&self) -> Box<[Complex<T>]> {
        // TODO: Remove unnesasary initialization
        let mut buffer = Vec::with_capacity(self.len());
        for _ in 0..self.len() {
            buffer.push(Complex::default())
        }

        buffer.clone_from_slice(self);

        crate::get_fft_algorithm::<T>(self.len()).process(&mut buffer);
        buffer.into_boxed_slice()
    }
}

impl<T: FftNum + Default> DynIfft<T> for [Complex<T>] {
    fn ifft(&self) -> Box<[Complex<T>]> {
        // TODO: Remove unnesasary initialization
        let mut buffer = Vec::with_capacity(self.len());
        for _ in 0..self.len() {
            buffer.push(Complex::default())
        }
        buffer.copy_from_slice(self);
        crate::get_inverse_fft_algorithm::<T>(self.len()).process(&mut buffer);
        buffer.into_boxed_slice()
    }
}
