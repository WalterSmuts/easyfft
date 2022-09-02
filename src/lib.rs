#![deny(missing_docs)]
#![deny(clippy::undocumented_unsafe_blocks)]
#![doc = include_str!("../README.md")]

use rustfft::FftPlanner;
use std::cell::RefCell;
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
