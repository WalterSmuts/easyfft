//! WIP - No docs for now
use rustfft::num_complex::Complex;
use rustfft::FftNum;

#[cfg(feature = "realfft")]
pub mod realfft;

/// A trait for performing fast DFT's on structs representing complex signals with a size unknown
/// at compile time.
pub trait DynFft<T> {
    /// Perform a complex-valued FFT on a signal with input and outpiut size `SIZE`.
    fn fft(&self) -> Box<[Complex<T>]>;
}

/// A trait for performing fast IDFT's on structs representing complex signals with a size unknown
/// at compile time.
pub trait DynIfft<T> {
    /// Perform a complex-valued IFFT on a signal with unknown sizes.
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
