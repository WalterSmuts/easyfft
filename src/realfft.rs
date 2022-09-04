//! Traits and structs for real-valued discrete fourier transforms.
//!
//! ### Requires `nightly`!
//! Unfortunately this module depends on the unstable [`generic_const_exprs`] feature of rust. That
//! means you can only use it via the nightly compiler. Once [`generic_const_exprs`] lands in stable,
//! this requirement will be lifted. For now I've feature-gated the [crate::realfft]  module so you
//! need to explicitly enable it in your `Cargo.toml`:
//! ```text
//! easyfft = { version = "LATEST_VERSION", features = ["realfft"] }
//! ```
//!
//! ### Real Signal Example:
//! ```rust
//! #![allow(incomplete_features)]
//! #![feature(generic_const_exprs)]
//!
//! use approx::assert_ulps_eq;
//! use easyfft::realfft::RealFft;
//! use easyfft::realfft::RealIfft;
//!
//! // Define a real-valued signal
//! let real_signal = [1.0_f64; 10];
//! // Call `.real_fft()` on the signal to obtain it's discrete fourier transform
//! let real_signal_dft = real_signal.real_fft();
//! // Call `.real_ifft` on the RealDft signal to obtain it's real inverse
//! let real_signal_dft_idft: [f64; 10] = real_signal_dft.real_ifft();
//!
//! // Verify the resulting ifft is a scaled version of the original signal
//! for (original, manipulated) in real_signal.iter().zip(real_signal_dft_idft) {
//!     assert_ulps_eq!(manipulated, original * 10.0);
//! }
//!
//! ```
//! [`generic_const_exprs`]: https://github.com/rust-lang/rust/issues/76560

use crate::generic_singleton;
use realfft::ComplexToReal;
use realfft::RealFftPlanner;
use realfft::RealToComplex;
use rustfft::num_complex::Complex;
use rustfft::FftNum;
use std::cell::RefCell;
use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;

/// A trait for performing fast DFT's on structs representing real signals with a size known at
/// compile time.
pub trait RealFft<T, const SIZE: usize>
where
    [T; SIZE / 2 + 1]: Sized,
{
    /// Perform a real-valued FFT on a signal with input size `SIZE` and output size `SIZE / 2 + 1`.
    fn real_fft(&self) -> RealDft<T, SIZE>;
}

/// A trait for performing fast IDFT's on structs representing real signals with a size known at
/// compile time.
pub trait RealIfft<T, const SIZE: usize> {
    /// Perform a real-valued IFFT on a signal which originally had input size `SIZE`.
    fn real_ifft(&self) -> [T; SIZE];
}

// TODO: Define constructor for creating this type manually
/// The result of calling [RealFft::real_fft].
///
/// As [explained] by the author of the [realfft crate], a real valued signal can have some
/// optimizations applied when calculating it's [discrete fourier transform]. This involves only
/// returning half the complex frequency domain signal since the other half can be inferred because
/// the DFT of a real signal is [known to be symmetric]. This poses a problem when attempting to do
/// the inverse discrete fourier transform since a signal of type `[Xr; SIZE]` would return a
/// complex signal of type `Complex<Xre, Xim>; SIZE / 2 -1]`. Note that `[_; SIZE]` gets mapped to
/// `[_; SIZE / 2 + 1]` and the index of an array is a natural number, so we're working with lossy
/// integer division here. Specifically, observe that __BOTH__ a signal of type `[_; 5]` and
/// `[_; 4]` would be mapped to a DFT of type `[_; 3]`. This means the IDFT cannot be unambiguously
/// determined by the type of the DFT.
///
/// The solution is to wrap the array in a different type which contains extra type information.
/// This newly created type is the [RealDft] and has the same memory representation as our original
/// type, but is blessed with the knowledge of its origins.
///
///
/// ### Caution!
/// Currently you can still cause a [panic!] by modifying the [RealDft] struct via the [DerefMut]
/// trait to a value that does not have a defined IDFT. This is left open for now until I figure
/// out how to define the operations that should be allowed on [RealDft] to keep it valid. Same
/// blocker is in place for constructing a [RealDft] using a `new` constructor.
///
/// [explained]: https://docs.rs/realfft/latest/realfft/index.html#real-to-complex
/// [discrete fourier transform]: https://en.wikipedia.org/wiki/Discrete_Fourier_transform
/// [realfft crate]: https://docs.rs/realfft/latest/realfft/index.html
/// [know to be symmetric]: https://en.wikipedia.org/wiki/Discrete_Fourier_transform#DFT_of_real_and_purely_imaginary_signals
/// [phantom type]: https://doc.rust-lang.org/rust-by-example/generics/phantom.html
#[derive(Debug)]
pub struct RealDft<T, const SIZE: usize>
where
    [T; SIZE / 2 + 1]: Sized,
{
    inner: [Complex<T>; SIZE / 2 + 1],
}

impl<T, const SIZE: usize> Deref for RealDft<T, SIZE>
where
    [T; SIZE / 2 + 1]: Sized,
{
    type Target = [Complex<T>; SIZE / 2 + 1];

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

// TODO: Figure out how to restrict user to make invalid modifications
impl<T, const SIZE: usize> DerefMut for RealDft<T, SIZE>
where
    [T; SIZE / 2 + 1]: Sized,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<T, const SIZE: usize> From<RealDft<T, SIZE>> for [Complex<T>; SIZE / 2 + 1]
where
    [T; SIZE / 2 + 1]: Sized,
{
    fn from(real_dft: RealDft<T, SIZE>) -> Self {
        real_dft.inner
    }
}

impl<T: FftNum + Default, const SIZE: usize> RealFft<T, SIZE> for [T; SIZE]
where
    [T; SIZE / 2 + 1]: Sized,
{
    fn real_fft(&self) -> RealDft<T, SIZE> {
        let r2c = get_real_fft_algorithm::<T, SIZE>();

        // TODO: Remove default dependency and unnesasary initialization
        let mut output = [Complex::default(); SIZE / 2 + 1];

        // TODO: remove this clone
        r2c.process(&mut self.clone(), &mut output).unwrap();
        RealDft { inner: output }
    }
}

impl<T: FftNum + Default, const SIZE: usize> RealIfft<T, SIZE> for RealDft<T, SIZE>
where
    [T; SIZE / 2 + 1]: Sized,
{
    fn real_ifft(&self) -> [T; SIZE] {
        let c2r = get_inverse_real_fft_algorithm::<T, SIZE>();

        // TODO: Remove default dependency and unnesasary initialization
        let mut output = [T::default(); SIZE];
        c2r.process(&mut (*self).clone(), &mut output).unwrap();
        output
    }
}

// TODO: Consider using UnsafeCell to avoid runtime borrow-checking.
fn get_real_fft_algorithm<T: FftNum, const SIZE: usize>() -> Arc<dyn RealToComplex<T>> {
    generic_singleton::get_or_init(|| RefCell::new(RealFftPlanner::new()))
        .borrow_mut()
        .plan_fft_forward(SIZE)
}

// TODO: Consider using UnsafeCell to avoid runtime borrow-checking.
fn get_inverse_real_fft_algorithm<T: FftNum, const SIZE: usize>() -> Arc<dyn ComplexToReal<T>> {
    generic_singleton::get_or_init(|| RefCell::new(RealFftPlanner::new()))
        .borrow_mut()
        .plan_fft_inverse(SIZE)
}
#[cfg(test)]
mod tests {
    use super::*;

    const ARBITRARY_EVEN_TEST_ARRAY: [f64; 6] = [1.5, 3.0, 2.1, 3.2, 2.2, 3.1];
    const ARBITRARY_ODD_TEST_ARRAY: [f64; 7] = [1.5, 3.0, 2.1, 3.2, 2.2, 3.1, 1.2];

    //TODO: Figure out why this error creeps in and if there is an appropriate constant already
    // defined.
    const ACCEPTABLE_ERROR: f64 = 0.00000000000001;

    fn real_fft_and_real_ifft_are_inverse_operations<const SIZE: usize>(array: [f64; SIZE])
    where
        [f64; SIZE / 2 + 1]: Sized,
    {
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
        real_fft_and_real_ifft_are_inverse_operations(ARBITRARY_EVEN_TEST_ARRAY);
    }

    #[test]
    fn real_fft_and_real_ifft_are_inverse_operations_odd() {
        real_fft_and_real_ifft_are_inverse_operations(ARBITRARY_ODD_TEST_ARRAY);
    }
}
