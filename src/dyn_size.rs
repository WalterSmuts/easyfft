//! Provides an easy FFT api for slices.
//!
//! ### Example:
//! ```rust
//! use approx::assert_ulps_eq;
//! use easyfft::num_complex::Complex;
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
use std::cell::UnsafeCell;
use std::collections::HashMap;

use rustfft::num_complex::Complex;
use rustfft::Fft;
use rustfft::FftNum;

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

/// A trait for performing fast in-place DFT's on structs representing complex signals with a size
/// not known at compile time.
pub trait DynFftMut<T> {
    /// Perform a complex-valued in-place FFT on a signal.
    fn fft_mut(&mut self);
}

/// A trait for performing fast in-place IDFT's on structs representing complex signals with a size
/// not known at compile time.
pub trait DynIfftMut<T> {
    /// Perform a complex-valued in-place IFFT on a signal.
    fn ifft_mut(&mut self);
}

trait StaticScratchFft<T: FftNum>: Fft<T> {
    fn process_with_static_scratch(&self, buffer: &mut [Complex<T>]);
}

impl<T: FftNum + Default, U: ?Sized + Fft<T>> StaticScratchFft<T> for U {
    fn process_with_static_scratch(&self, buffer: &mut [Complex<T>]) {
        let map_pointer = generic_singleton::get_or_init(|| {
            UnsafeCell::new(HashMap::<usize, Box<[Complex<T>]>>::new())
        })
        .get();
        // SAFETY:
        //
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
        let map = unsafe { map_pointer.as_mut().unwrap_unchecked() };
        let len = self.get_inplace_scratch_len();
        let scratch = map
            .entry(len)
            .or_insert_with(|| vec![Complex::default(); len].into_boxed_slice());
        self.process_with_scratch(buffer, scratch);
    }
}

impl<T: FftNum + Default> DynFft<T> for [T] {
    fn fft(&self) -> Box<[Complex<T>]> {
        // TODO: Remove unnesasary initialization
        let mut buffer = Vec::with_capacity(self.len());
        for sample in self {
            buffer.push(Complex::new(*sample, T::default()));
        }

        let algorithm = crate::get_fft_algorithm::<T>(self.len());
        algorithm.process_with_static_scratch(&mut buffer);
        buffer.into_boxed_slice()
    }
}

impl<T: FftNum + Default> DynFft<T> for [Complex<T>] {
    fn fft(&self) -> Box<[Complex<T>]> {
        // TODO: Remove unnesasary initialization
        let mut buffer = vec![Complex::default(); self.len()];
        buffer.clone_from_slice(self);

        crate::get_fft_algorithm::<T>(self.len()).process_with_static_scratch(&mut buffer);
        buffer.into_boxed_slice()
    }
}

impl<T: FftNum + Default> DynIfft<T> for [Complex<T>] {
    fn ifft(&self) -> Box<[Complex<T>]> {
        // TODO: Remove unnesasary initialization
        let mut buffer = vec![Complex::default(); self.len()];
        buffer.copy_from_slice(self);
        crate::get_inverse_fft_algorithm::<T>(self.len()).process_with_static_scratch(&mut buffer);
        buffer.into_boxed_slice()
    }
}

impl<T: FftNum + std::default::Default> DynFftMut<T> for [Complex<T>] {
    fn fft_mut(&mut self) {
        crate::get_fft_algorithm::<T>(self.len()).process_with_static_scratch(self);
    }
}

impl<T: FftNum + std::default::Default> DynIfftMut<T> for [Complex<T>] {
    fn ifft_mut(&mut self) {
        crate::get_inverse_fft_algorithm::<T>(self.len()).process_with_static_scratch(self);
    }
}
