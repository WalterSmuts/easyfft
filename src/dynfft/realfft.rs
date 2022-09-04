//! WIP - No docs for now
use crate::generic_singleton;
use realfft::ComplexToReal;
use realfft::RealFftPlanner;
use realfft::RealToComplex;
use rustfft::num_complex::Complex;
use rustfft::FftNum;
use std::cell::RefCell;
use std::ops::Deref;
use std::sync::Arc;

/// A trait for performing fast DFT's on structs representing real signals with a size unknown at
/// compile time.
pub trait DynRealFft<T> {
    /// Perform a real-valued FFT on a signal with input size `SIZE` and output size `SIZE / 2 + 1`.
    fn real_fft(&self) -> DynRealDft<T>;
}

/// A trait for performing fast IDFT's on structs representing real signals with a size unknown at
/// compile time.
pub trait DynRealIfft<T> {
    /// Perform a real-valued IFFT on a signal which originally had input size `SIZE`.
    fn real_ifft(&self) -> Box<[T]>;
}

// TODO: Define constructor for creating this type manually
/// The result of calling [RealFft::real_fft].
///
/// As [explained] by the author of the [realfft crate], a real valued signal can have some
/// optimizations applied when calculating it's [discrete fourier transform]. This involves only
/// returning half the complex frequency domain signal since the other half can be inferred because
/// the DFT of a real signal is [unknown to be symmetric]. This poses a problem when attempting to do
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
/// [explained]: https://docs.rs/realfft/latest/realfft/index.html#real-to-complex
/// [discrete fourier transform]: https://en.wikipedia.org/wiki/Discrete_Fourier_transform
/// [realfft crate]: https://docs.rs/realfft/latest/realfft/index.html
/// [know to be symmetric]: https://en.wikipedia.org/wiki/Discrete_Fourier_transform#DFT_of_real_and_purely_imaginary_signals
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

// TODO: Consider using UnsafeCell to avoid runtime borrow-checking.
fn get_real_fft_algorithm<T: FftNum>(size: usize) -> Arc<dyn RealToComplex<T>> {
    generic_singleton::get_or_init(|| RefCell::new(RealFftPlanner::new()))
        .borrow_mut()
        .plan_fft_forward(size)
}

// TODO: Consider using UnsafeCell to avoid runtime borrow-checking.
fn get_inverse_real_fft_algorithm<T: FftNum>(size: usize) -> Arc<dyn ComplexToReal<T>> {
    generic_singleton::get_or_init(|| RefCell::new(RealFftPlanner::new()))
        .borrow_mut()
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
