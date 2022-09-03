// TODO: Remove once feature hits stable:
// https://github.com/rust-lang/rust/issues/76560
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![deny(missing_docs)]
#![deny(clippy::undocumented_unsafe_blocks)]
#![doc = include_str!("../README.md")]

use realfft::ComplexToReal;
use realfft::RealFftPlanner;
use realfft::RealToComplex;
use rustfft::FftPlanner;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;

pub use rustfft::num_complex::Complex;
pub use rustfft::FftNum;

// TODO: Move to using const_guards once issue is resolved:
// https://github.com/Mari-W/const_guards/issues/4
mod type_restriction;
use type_restriction::ConstCheck;
use type_restriction::True;

/// Parity is a property possessed by a number and can either be `Even` or `Odd`.
///
/// `Even` parity is defined by `N % 2 == 0` and `Odd` parity is defined by `N % 2 == 0`.
/// Internally `constfft` implements [Parity] for an `Even` and `Odd` marker `struct`.
pub trait Parity {}
#[derive(Debug)]
struct Even;
#[derive(Debug)]
struct Odd;

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
/// type, but is blessed with the knowledge of its origins. Specifically it contains a [phantom
/// type] parameterized by a [Parity]. This lets the compiler figure out which variant of
/// [RealIfft::real_ifft] to use to calculate the IDFT.
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

/// A trait for performing fast DFT's on structs representing real signals with a size known at
/// compile time.
pub trait RealFft<T, const SIZE: usize, U: Parity> {
    /// Perform a real-valued FFT on a signal with input size `SIZE` and output size `SIZE / 2 + 1`.
    fn real_fft(&self) -> RealDft<T, { SIZE / 2 + 1 }, U>;
}

/// A trait for performing fast DFT's on structs representing complex signals with a size known at
/// compile time.
pub trait Fft<T, const SIZE: usize> {
    /// Perform a complex-valued FFT on a signal with input and outpiut size `SIZE`.
    fn fft(&self) -> [Complex<T>; SIZE];
}

/// A trait for performing fast IDFT's on structs representing real signals with a size known at
/// compile time.
pub trait RealIfft<T, const SIZE: usize, const OUTPUT_SIZE: usize> {
    /// The underlying type of the elements of the signal.
    type Output;

    /// Perform a real-valued FFT on a signal with input size `SIZE` and output size depending on
    /// the parity of the input.
    ///
    /// If the input is tagged with 'Even' parity then the output size will be `(SIZE - 1) * 2`.
    /// If the input is tagged with 'Odd' parity then the output size will be `(SIZE - 1) * 2 + 1`.
    fn real_ifft(&self) -> [Self::Output; OUTPUT_SIZE];
}

/// A trait for performing fast IDFT's on structs representing complex signals with a size known at
/// compile time.
pub trait Ifft<T, const SIZE: usize> {
    /// Perform a complex-valued IFFT on a signal with input and outpiut size `SIZE`.
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
        // SAFETY:
        // The function will only return None if the iterator is exhausted before the array
        // is filled. The compiler ensures that the size of this array is always equal to the size
        // of the array passed in, and therefore the size of the iterator, by the SIZE const
        // generic argument. Therefore this function will never return a None value.
        //
        // TODO: Remove this and call `map_array_init` once this commit gets released:
        // https://github.com/Manishearth/array-init/commit/9496096148b4933416a2435de65b9fa844872127
        let mut buffer: [Complex<T>; SIZE] = unsafe {
            array_init::from_iter(
                self.iter()
                    .map(|sample| Complex::new(*sample, T::default())),
            )
            .unwrap_unchecked()
        };

        get_fft_algorithm::<T, SIZE>().process(&mut buffer);
        buffer
    }
}

impl<T: FftNum, const SIZE: usize> Fft<T, SIZE> for [Complex<T>; SIZE] {
    fn fft(&self) -> [Complex<T>; SIZE] {
        // Copy into a new buffer
        let mut buffer: [Complex<T>; SIZE] = *self;
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

mod generic_singleton {
    use anymap::AnyMap;
    use std::cell::RefCell;

    /// Get a static reference to a generic singleton or initialize it if it doesn't exist.
    ///
    /// # Panics
    /// Initialization will panic if the init function calls get_or_init during initialization.
    pub fn get_or_init<T: 'static>(init: fn() -> T) -> &'static T {
        // TODO: Consider using UnsafeCell to avoid runtime borrow-checking.
        thread_local! {
            static REF_CELL_MAP: RefCell<AnyMap> = RefCell::new(AnyMap::new());
        };
        REF_CELL_MAP.with(|map_cell| {
            let mut map = map_cell.borrow_mut();
            if !map.contains::<T>() {
                map.insert(init());
            }
            // SAFETY:
            // The function will only return None if the item is not present. Since we always add the
            // item if it's not present two lines above and never remove items, we can be sure that
            // this function will always return `Some`.
            let t_ref = unsafe { map.get::<T>().unwrap_unchecked() };
            let ptr = t_ref as *const T;
            // SAFETY:
            // Check: The pointer must be properly aligned.
            // Proof: The pointer is obtained from a valid reference so it must be aligned.
            //
            // Check: It must be “dereferenceable” in the sense defined in the module documentation.
            // Proof: The pointer is obtained from a valid reference it musts be dereferenceable.
            //
            // Check: The pointer must point to an initialized instance of T.
            // Proof: The AnyMap crate provides this guarantee.
            //
            // Check: You must enforce Rust’s aliasing rules, since the returned lifetime 'a is
            //        arbitrarily chosen and does not necessarily reflect the actual lifetime of the data.
            //        In particular, while this reference exists, the memory the pointer points to
            //        must not get mutated (except inside UnsafeCell).
            // Proof: We return a shared reference and therefore cannot be mutated unless T is
            //        guarded with the normal rust memory protection constructs using UnsafeCell.
            //        The data could be dropped if we ever removed it from this map however. Care
            //        must be taken to never introduce any logic that would remove T from the map.
            let optional_ref = unsafe { ptr.as_ref() };
            // SAFETY:
            // This requires the pointer not to be null. We obtained the pointer one line above
            // from a valid reference, therefore this is considered safe to do.
            unsafe { optional_ref.unwrap_unchecked() }
        })
    }
}

// TODO: Consider using UnsafeCell to avoid runtime borrow-checking.
fn get_fft_algorithm<T: FftNum, const SIZE: usize>() -> Arc<dyn rustfft::Fft<T>> {
    generic_singleton::get_or_init(|| RefCell::new(FftPlanner::new()))
        .borrow_mut()
        .plan_fft_forward(SIZE)
}

// TODO: Consider using UnsafeCell to avoid runtime borrow-checking.
fn get_inverse_fft_algorithm<T: FftNum, const SIZE: usize>() -> Arc<dyn rustfft::Fft<T>> {
    generic_singleton::get_or_init(|| RefCell::new(FftPlanner::new()))
        .borrow_mut()
        .plan_fft_inverse(SIZE)
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
