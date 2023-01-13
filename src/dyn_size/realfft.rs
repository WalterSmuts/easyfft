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
use realfft::num_traits::NumAssign;
use realfft::ComplexToReal;
use realfft::RealToComplex;
use rustfft::num_complex::Complex;
use rustfft::FftNum;
use std::collections::HashMap;
#[cfg(feature = "fallible")]
use std::ops::Add;
#[cfg(feature = "fallible")]
use std::ops::AddAssign;
use std::ops::Deref;
use std::ops::Mul;
use std::ops::MulAssign;

use crate::with_inverse_real_fft_algorithm;
use crate::with_real_fft_algorithm;

/// A trait for performing fast DFT's on structs representing real signals with a size not known at
/// compile time.
pub trait DynRealFft<T> {
    /// Perform a real-valued FFT on a signal with input size `self.len()` and output size `self.len() / 2 + 1`.
    fn real_fft(&self) -> DynRealDft<T>;

    #[cfg(feature = "fallible")]
    /// Perform a real-valued FFT on a signal with input size `self.len()` and output size `self.len() / 2 + 1`
    /// using a struct that's already allocated. This is to avoid an allocation when you've already
    /// got a buffer.
    ///
    /// # Panics
    /// Panics if the output's original length is not equal to the input buffer.
    fn real_fft_using(&self, output: &mut DynRealDft<T>);
}

/// A trait for performing fast IDFT's on structs representing real signals with a size not known
/// at compile time.
pub trait DynRealIfft<T> {
    /// Perform a real-valued IFFT on a signal which originally had input size `self.len()`.
    fn real_ifft(&self) -> Box<[T]>;

    #[cfg(feature = "fallible")]
    /// Perform a real-valued IFFT on a signal which originally had input size `self.len()` using a
    /// buffer that's already allocated. This is to avoid an allocation when you've already got a
    /// buffer.
    ///
    /// # Panics
    /// Panics if the `original_length` is not equal to `output.len()`.
    fn real_ifft_using(&self, output: &mut [T]);
}

trait StaticScratchComplexToReal<T: FftNum>: ComplexToReal<T> {
    unsafe fn process_with_static_scratch(&self, input: &[Complex<T>], output: &mut [T]);
}

trait PrivateRealFftUsing<T> {
    fn real_fft_using(&self, output: &mut DynRealDft<T>);
}

// The caller needs to ensure the following holds: input_length == output_length / 2 + 1
impl<T: FftNum + Default, U: ?Sized + ComplexToReal<T>> StaticScratchComplexToReal<T> for U {
    unsafe fn process_with_static_scratch(&self, input: &[Complex<T>], output: &mut [T]) {
        debug_assert_eq!(input.len(), output.len() / 2 + 1);
        generic_singleton::get_or_init_thread_local!(
            HashMap::<usize, Box<[Complex<T>]>>::new,
            |input_clone_map| {
                generic_singleton::get_or_init_thread_local!(
                    || { HashMap::<usize, Box<[Complex<T>]>>::new() },
                    |scratch_buffer_map| {
                        let scratch_buffer_len = self.get_scratch_len();

                        let scratch =
                            scratch_buffer_map
                                .entry(scratch_buffer_len)
                                .or_insert_with(|| {
                                    vec![Complex::default(); scratch_buffer_len].into_boxed_slice()
                                });
                        let input_clone = input_clone_map.entry(input.len()).or_insert_with(|| {
                            vec![Complex::default(); input.len()].into_boxed_slice()
                        });

                        input_clone.copy_from_slice(input);
                        self.process_with_scratch(input_clone, output, scratch)
                            .unwrap_unchecked();
                    }
                );
            }
        );
    }
}

trait StaticScratchRealToComplex<T: FftNum>: RealToComplex<T> {
    unsafe fn process_with_static_scratch(&self, input: &[T], output: &mut [Complex<T>]);
}

// The caller needs to ensure the following holds: input_length / 2 + 1 == output_length
impl<T: FftNum + Default, U: ?Sized + RealToComplex<T>> StaticScratchRealToComplex<T> for U {
    unsafe fn process_with_static_scratch(&self, input: &[T], output: &mut [Complex<T>]) {
        debug_assert_eq!(input.len() / 2 + 1, output.len());

        generic_singleton::get_or_init_thread_local!(
            HashMap::<usize, Box<[T]>>::new,
            |input_clone_map| {
                generic_singleton::get_or_init_thread_local!(
                    HashMap::<usize, Box<[Complex<T>]>>::new,
                    |scratch_buffer_map| {
                        let scratch_buffer_len = self.get_scratch_len();

                        let scratch =
                            scratch_buffer_map
                                .entry(scratch_buffer_len)
                                .or_insert_with(|| {
                                    vec![Complex::default(); scratch_buffer_len].into_boxed_slice()
                                });
                        let input_clone = input_clone_map
                            .entry(input.len())
                            .or_insert_with(|| vec![T::default(); input.len()].into_boxed_slice());

                        input_clone.copy_from_slice(input);
                        self.process_with_scratch(input_clone, output, scratch)
                            .unwrap_unchecked();
                    }
                );
            }
        );
    }
}

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

#[cfg(feature = "fallible")]
impl<T: Default + Copy> DynRealDft<T> {
    /// Create a new `DynRealDft` struct.
    ///
    /// You need to provide the size of the real-valued signal you expect it to produce as a
    /// function arguement. Requires the `fallible` feature.
    ///
    /// # Panics
    /// Panics if the `original_length / 2 + 1` is not equal to `frequency_bins.len() + 1`.
    ///
    /// If you don't like the idea of your code possibly panicking, consider using the
    /// [`crate::const_size::realfft`] module instead.
    /// ```
    /// use easyfft::dyn_size::realfft::DynRealDft;
    /// use easyfft::dyn_size::realfft::DynRealIfft;
    /// use easyfft::num_complex::Complex;
    ///
    /// // Initialize a DynRealDft struct that would produce an signal of length 4 after calling real_ifft().
    /// let real_dft_4 = DynRealDft::new(10.0, &[Complex::new(1.0, 2.0), Complex::default()], 4);
    /// assert_eq!(real_dft_4.real_ifft().len(), 4);
    /// // Initialize a DynRealDft struct that would produce an signal of length 5 after calling real_ifft().
    /// let real_dft_5 = DynRealDft::new(10.0, &[Complex::new(1.0, 2.0), Complex::default()], 5);
    /// assert_eq!(real_dft_5.real_ifft().len(), 5);
    /// ```
    pub fn new(zeroth_bin: T, frequency_bins: &[Complex<T>], original_length: usize) -> Self {
        assert_eq!(original_length / 2 + 1, frequency_bins.len() + 1);
        let inner = [&[Complex::new(zeroth_bin, T::default())], frequency_bins].concat();
        Self {
            original_length,
            inner: inner.into_boxed_slice(),
        }
    }
}

impl<T> Deref for DynRealDft<T> {
    type Target = [Complex<T>];

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[cfg(feature = "fallible")]
impl<T: Default + FftNum> Add for &DynRealDft<T> {
    type Output = DynRealDft<T>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.len(), rhs.len());
        let mut inner = self.inner.clone();
        for (i, r) in inner.iter_mut().zip(rhs.iter()) {
            *i = *i + r;
        }
        DynRealDft {
            original_length: self.original_length,
            inner,
        }
    }
}

#[cfg(feature = "fallible")]
impl<T: Default + FftNum> AddAssign<&Self> for DynRealDft<T> {
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(self.len(), rhs.len());
        for (i, r) in self.inner.iter_mut().zip(rhs.iter()) {
            *i = *i + r;
        }
    }
}

#[cfg(feature = "fallible")]
impl<T: Default + FftNum> Mul for &DynRealDft<T> {
    type Output = DynRealDft<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.len(), rhs.len());
        let mut inner = Vec::with_capacity(self.len());
        for index in 0..self.len() {
            inner.push(self[index] * rhs[index]);
        }
        DynRealDft {
            inner: inner.into_boxed_slice(),
            original_length: self.original_length,
        }
    }
}

#[cfg(feature = "fallible")]
impl<T: Default + FftNum + NumAssign> MulAssign<&Self> for DynRealDft<T> {
    fn mul_assign(&mut self, rhs: &Self) {
        assert_eq!(self.len(), rhs.len());
        for (bin_self, bin_rhs) in self.inner.iter_mut().zip(rhs.iter()) {
            *bin_self *= bin_rhs;
        }
    }
}

impl<T: Default + FftNum> Mul<T> for &DynRealDft<T> {
    type Output = DynRealDft<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut inner = Vec::with_capacity(self.len());
        for index in 0..self.len() {
            inner.push(self[index] * rhs);
        }
        DynRealDft {
            inner: inner.into_boxed_slice(),
            original_length: self.original_length,
        }
    }
}

impl<T: Default + FftNum + NumAssign> MulAssign<T> for DynRealDft<T> {
    fn mul_assign(&mut self, rhs: T) {
        for bin_self in self.inner.iter_mut() {
            *bin_self *= rhs;
        }
    }
}

#[cfg(feature = "fallible")]
impl<T: Default + FftNum> Mul<&[T]> for &DynRealDft<T> {
    type Output = DynRealDft<T>;

    fn mul(self, rhs: &[T]) -> Self::Output {
        assert_eq!(self.len(), rhs.len());
        let mut inner = Vec::with_capacity(self.len());
        for index in 0..self.len() {
            inner.push(self[index] * rhs[index]);
        }
        DynRealDft {
            inner: inner.into_boxed_slice(),
            original_length: self.original_length,
        }
    }
}

#[cfg(feature = "fallible")]
impl<T: Default + FftNum + NumAssign> MulAssign<&[T]> for DynRealDft<T> {
    fn mul_assign(&mut self, rhs: &[T]) {
        assert_eq!(self.len(), rhs.len());
        for (bin_self, bin_rhs) in self.inner.iter_mut().zip(rhs.iter()) {
            *bin_self *= bin_rhs;
        }
    }
}

impl<T> DynRealDft<T> {
    /// Get a slice of all the frequency bins excluding the first (and, if the original signal has
    /// an even number of samples, last) bin(s).
    #[must_use]
    pub fn get_frequency_bins(&self) -> &[Complex<T>] {
        let wanted_len = self.original_length - 1;
        &self.inner[1..wanted_len / 2]
    }

    /// Get a mutable slice of all the frequency bins excluding the first (and, if the original
    /// signal has an even number of samples, last) bin(s).
    pub fn get_frequency_bins_mut(&mut self) -> &mut [Complex<T>] {
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

impl<T: FftNum + Default> DynRealDft<T> {
    #[cfg(feature = "fallible")]
    /// Allocate a default DFT with provided `original_length`.
    #[must_use]
    pub fn default(original_length: usize) -> Self {
        let inner = vec![Complex::default(); original_length / 2 + 1].into_boxed_slice();
        Self {
            original_length,
            inner,
        }
    }
}
impl<T: FftNum> DynRealDft<T> {
    #[cfg(feature = "fallible")]
    /// Copy from a slice of Complex values.
    ///
    /// # Panics
    /// * Panics if the `original_length / 2 + 1` is not equal to `slice.len()`.
    /// * If the first element of the slice has a non-zero imaginary component or the last element if
    /// `original_length` is odd.
    pub fn copy_from_slice(&mut self, slice: &[Complex<T>]) {
        assert_eq!(self.original_length / 2 + 1, slice.len());
        assert_eq!(slice[0].im, T::from_f32(0.0).unwrap());
        if self.original_length % 2 == 0 {
            assert_eq!(slice[slice.len() - 1].im, T::from_f32(0.0).unwrap());
        }
        self.inner.copy_from_slice(slice);
    }
}

impl<T> From<DynRealDft<T>> for Box<[Complex<T>]> {
    fn from(dyn_real_dft: DynRealDft<T>) -> Self {
        dyn_real_dft.inner
    }
}

impl<T: FftNum + Default> DynRealFft<T> for [T] {
    fn real_fft(&self) -> DynRealDft<T> {
        // TODO: Remove default dependency and unnesasary initialization
        // Pending issue: https://github.com/ejmahler/RustFFT/issues/105
        let output = vec![Complex::default(); self.len() / 2 + 1];
        let mut output = DynRealDft {
            inner: output.into_boxed_slice(),
            original_length: self.len(),
        };
        self.real_fft_using(&mut output);
        output
    }

    #[cfg(feature = "fallible")]
    fn real_fft_using(&self, output: &mut DynRealDft<T>) {
        assert_eq!(self.len(), output.original_length);
        with_real_fft_algorithm::<T>(self.len(), |r2c| {
            // SAFETY:
            // The error case only happens when the size of the input and output and fft algorithm are
            // not consistent. Since all these are calculated inside this function and have been double
            // checked and tested, we can be sure they won't be inconsistent.
            unsafe {
                r2c.process_with_static_scratch(self, &mut output.inner);
            }
        });
    }
}

impl<T: FftNum + Default> DynRealIfft<T> for DynRealDft<T> {
    fn real_ifft(&self) -> Box<[T]> {
        let mut output = vec![T::default(); self.original_length];
        self.real_ifft_using(&mut output);
        output.into_boxed_slice()
    }

    #[cfg(feature = "fallible")]
    fn real_ifft_using(&self, output: &mut [T]) {
        assert_eq!(self.original_length, output.len());
        with_inverse_real_fft_algorithm::<T>(self.original_length, |c2r| {
            // SAFETY:
            // The error case only happens when the size of the input and output and fft algorithm are
            // not consistent. Since all these are calculated inside this function and have been double
            // checked and tested, we can be sure they won't be inconsistent.
            unsafe {
                c2r.process_with_static_scratch(self, output);
            }
        });
    }
}

// TODO: Investigate ways to de-duplicate this logic
#[cfg(not(feature = "fallible"))]
impl<T: FftNum + Default> PrivateRealFftUsing<T> for [T] {
    fn real_fft_using(&self, output: &mut DynRealDft<T>) {
        debug_assert_eq!(self.len(), output.original_length);
        let r2c = with_real_fft_algorithm::<T>(self.len());

        // SAFETY:
        // The error case only happens when the size of the input and output and fft algorithm are
        // not consistent. Since all these are calculated inside this function and have been double
        // checked and tested, we can be sure they won't be inconsistent.
        unsafe {
            r2c.process_with_static_scratch(self, &mut output.inner);
        }
    }
}

// TODO: Investigate ways to de-duplicate this logic
#[cfg(not(feature = "fallible"))]
impl<T: FftNum + Default> DynRealDft<T> {
    fn real_ifft_using(&self, output: &mut [T]) {
        debug_assert_eq!(self.original_length, output.len());
        let c2r = with_inverse_real_fft_algorithm::<T>(self.original_length);

        // SAFETY:
        // The error case only happens when the size of the input and output and fft algorithm are
        // not consistent. Since all these are calculated inside this function and have been double
        // checked and tested, we can be sure they won't be inconsistent.
        unsafe {
            c2r.process_with_static_scratch(self, output);
        }
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
