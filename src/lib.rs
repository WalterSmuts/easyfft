// TODO: Remove once feature hits stable:
// https://github.com/rust-lang/rust/issues/76560
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use realfft::{ComplexToReal, RealToComplex};
use realfft::{FftNum, RealFftPlanner};
use rustfft::{num_complex::Complex, FftPlanner};
use std::marker::PhantomData;
use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;

// TODO: Move to using const_guards once issue is resolved:
// https://github.com/Mari-W/const_guards/issues/4
mod type_restriction;
use type_restriction::ConstCheck;
use type_restriction::True;

pub trait Parity {}
#[derive(Debug)]
struct Even;
#[derive(Debug)]
struct Odd;

// TODO: Define constructor for creating this type manually
#[derive(Debug)]
pub struct RealDft<T, const SIZE: usize, U: Parity> {
    inner: [Complex<T>; SIZE],
    phantom: PhantomData<U>,
}

impl Parity for Even {}
impl Parity for Odd {}

impl<T, const SIZE: usize, U: Parity> Deref for RealDft<T, SIZE, U> {
    type Target = [Complex<T>; SIZE];

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

// TODO: Figure out how to restrict user to make invalid modifications
impl<T, const SIZE: usize, U: Parity> DerefMut for RealDft<T, SIZE, U> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

pub trait RealFft<T, const SIZE: usize, U: Parity> {
    fn real_fft(&self) -> RealDft<T, { SIZE / 2 + 1 }, U>;
}

pub trait Fft<T, const SIZE: usize> {
    fn fft(&self) -> [Complex<T>; SIZE];
}

pub trait RealIfft<T, const SIZE: usize, const OUTPUT_SIZE: usize> {
    type Output;

    fn real_ifft(&self) -> [Self::Output; OUTPUT_SIZE];
}

pub trait Ifft<T, const SIZE: usize> {
    fn ifft(&self) -> [Complex<T>; SIZE];
}

impl<T: FftNum + Default, const SIZE: usize> RealFft<T, SIZE, Even> for [T; SIZE]
where
    ConstCheck<{ SIZE % 2 == 0 }>: True,
{
    fn real_fft(&self) -> RealDft<T, { SIZE / 2 + 1 }, Even> {
        let r2c = get_real_fft_algorithm::<T, SIZE>();

        // TODO: Remove default dependency and unnesasary initialization
        let mut output = [Complex::default(); SIZE / 2 + 1];

        // TODO: remove this clone
        r2c.process(&mut self.clone(), &mut output).unwrap();
        RealDft {
            inner: output,
            phantom: PhantomData,
        }
    }
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

impl<T: FftNum + Default, const SIZE: usize> RealFft<T, SIZE, Odd> for [T; SIZE]
where
    ConstCheck<{ SIZE % 2 == 1 }>: True,
{
    fn real_fft(&self) -> RealDft<T, { SIZE / 2 + 1 }, Odd> {
        let r2c = get_real_fft_algorithm::<T, SIZE>();

        // TODO: Remove unnesasary initialization
        let mut output = [Complex::default(); SIZE / 2 + 1];

        // TODO: remove this clone
        r2c.process(&mut self.clone(), &mut output).unwrap();
        RealDft {
            inner: output,
            phantom: PhantomData,
        }
    }
}

// TODO: Add a cache
fn get_real_fft_algorithm<T: FftNum, const SIZE: usize>() -> Arc<dyn RealToComplex<T>> {
    let mut real_planner = RealFftPlanner::<T>::new();
    real_planner.plan_fft_forward(SIZE)
}

// TODO: Add a cache
fn get_inverse_real_fft_algorithm<T: FftNum, const SIZE: usize>() -> Arc<dyn ComplexToReal<T>> {
    let mut real_planner = RealFftPlanner::<T>::new();
    real_planner.plan_fft_inverse(SIZE)
}

// TODO: Add a cache
fn get_fft_algorithm<T: FftNum, const SIZE: usize>() -> Arc<dyn rustfft::Fft<T>> {
    FftPlanner::<T>::new().plan_fft_forward(SIZE)
}

// TODO: Add a cache
fn get_inverse_fft_algorithm<T: FftNum, const SIZE: usize>() -> Arc<dyn rustfft::Fft<T>> {
    FftPlanner::<T>::new().plan_fft_inverse(SIZE)
}

impl<T: FftNum + Default, const SIZE: usize> RealIfft<T, SIZE, { (SIZE - 1) * 2 }>
    for RealDft<T, SIZE, Even>
{
    type Output = T;

    fn real_ifft(&self) -> [Self::Output; (SIZE - 1) * 2] {
        let c2r = get_inverse_real_fft_algorithm::<T, { (SIZE - 1) * 2 }>();

        // TODO: Remove default dependency and unnesasary initialization
        let mut output = [T::default(); { (SIZE - 1) * 2 }];
        c2r.process(&mut (*self).clone(), &mut output).unwrap();
        output
    }
}

impl<T: FftNum + Default, const SIZE: usize> RealIfft<T, SIZE, { (SIZE - 1) * 2 + 1 }>
    for RealDft<T, SIZE, Odd>
{
    type Output = T;

    fn real_ifft(&self) -> [Self::Output; (SIZE - 1) * 2 + 1] {
        let c2r = get_inverse_real_fft_algorithm::<T, { (SIZE - 1) * 2 + 1 }>();

        // TODO: Remove default dependency and unnesasary initialization
        let mut output = [T::default(); { (SIZE - 1) * 2 + 1 }];
        c2r.process(&mut (*self).clone(), &mut output).unwrap();
        output
    }
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

    #[test]
    fn fft_and_ifft_are_inverse_operations_even() {
        let converted: Vec<_> = ARBITRARY_EVEN_TEST_ARRAY
            .fft()
            .ifft()
            .iter_mut()
            .map(|sample| *sample / ARBITRARY_EVEN_TEST_ARRAY.len() as f64)
            .collect();
        for (converted, original) in converted.iter().zip(ARBITRARY_EVEN_TEST_ARRAY.iter()) {
            approx::assert_ulps_eq!(converted.re, original, epsilon = ACCEPTABLE_ERROR);
            approx::assert_ulps_eq!(converted.im, 0.0);
        }
    }

    #[test]
    fn fft_and_ifft_are_inverse_operations_odd() {
        let converted: Vec<_> = ARBITRARY_ODD_TEST_ARRAY
            .fft()
            .ifft()
            .iter_mut()
            .map(|sample| *sample / ARBITRARY_ODD_TEST_ARRAY.len() as f64)
            .collect();
        for (converted, original) in converted.iter().zip(ARBITRARY_ODD_TEST_ARRAY.iter()) {
            approx::assert_ulps_eq!(converted.re, original, epsilon = ACCEPTABLE_ERROR);
            approx::assert_ulps_eq!(converted.im, 0.0);
        }
    }

    #[test]
    fn real_fft_and_real_ifft_are_inverse_operations_even() {
        let converted: Vec<_> = ARBITRARY_EVEN_TEST_ARRAY
            .real_fft()
            .real_ifft()
            .iter_mut()
            .map(|sample| *sample / ARBITRARY_EVEN_TEST_ARRAY.len() as f64)
            .collect();
        for (converted, original) in converted.iter().zip(ARBITRARY_EVEN_TEST_ARRAY.iter()) {
            approx::assert_ulps_eq!(converted, original);
        }
    }

    #[test]
    fn real_fft_and_real_ifft_are_inverse_operations_odd() {
        let converted: Vec<_> = ARBITRARY_ODD_TEST_ARRAY
            .real_fft()
            .real_ifft()
            .iter_mut()
            .map(|sample| *sample / ARBITRARY_ODD_TEST_ARRAY.len() as f64)
            .collect();
        for (converted, original) in converted.iter().zip(ARBITRARY_ODD_TEST_ARRAY.iter()) {
            approx::assert_ulps_eq!(converted, original, epsilon = ACCEPTABLE_ERROR);
        }
    }
}
