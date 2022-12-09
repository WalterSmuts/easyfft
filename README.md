# easyfft
A Rust library crate providing an [FFT] API for arrays and slices. This crate wraps the
[rustfft] and [realfft] crates that does the heavy lifting behind the scenes.

### Example
The `nightly` dependent features are commented out.
```rust
// NOTE: Only required for real arrays
// #![allow(incomplete_features)]
// #![feature(generic_const_exprs)]

use easyfft::prelude::*;
use easyfft::Complex;

// Complex arrays
let complex_array = [Complex::new(1.0, 0.0); 100];
let complex_array_dft = complex_array.fft();
let _complex_array_dft_idft = complex_array_dft.ifft();

// Real to complex arrays
let real_array = [1.0; 100];
let _real_array_dft = real_array.fft();

// // Real arrays
// let real_array = [1.0; 100];
// let real_array_dft = real_array.real_fft();
// let _real_array_dft_idft = real_array_dft.real_ifft();

// Complex slices
let complex_slice: &[_] = &[Complex::new(1.0, 0.0); 100];
let complex_slice_dft = complex_slice.fft();
let _complex_slice_dft_idft = complex_slice_dft.ifft();

// Real to complex slices
let real_slice: &[_] = &[1.0; 100];
let _real_slice_dft = real_slice.fft();

// Real slices
let real_slice: &[_] = &[1.0; 100];
let real_slice_dft = real_slice.real_fft();
let _real_slice_dft_idft = real_slice_dft.real_ifft();

// In-place mutation on complex -> complex transforms
let mut complex_slice = [Complex::new(1.0, 0.0); 100];
complex_slice.fft_mut();
complex_slice.ifft_mut();
let mut complex_array = [Complex::new(1.0, 0.0); 100];
complex_array.fft_mut();
complex_array.ifft_mut();
```

### Current limitations
* The `const-realfft` feature requires the `nightly` compiler because it depends on
  the [generic_const_exprs] feature
* There are no methods for in-place mutation for complex -> real or real ->
  complex transforms.

### The `fallible` feature
The `DynRealDft` struct has some associated operations which can panic. This is
because the rust language does not have the ability to encode properties of the
length of slices in the type system. This might become possible in the future
if the rust team manages to extend const generics to fully fledged
[dependent types]. For now, we're limited to using arrays where we can ensure
these properties. If safety is your primary concern I recommend you take a step
back and consider if you REALLY need to work with slices instead of arrays.
Many applications can get away with knowing the size of their signal at compile
time. You can opt out of these panic-able operations by removing the `fallible`
feature flag, which is enabled by default.

#### Footnotes
[^panic]: While this could be true in theory, in practice it most probably is not.
There could be bugs in this crate or it's dependencies that may cause a panic,
but in theory all the runtime panics have been moved to compile time errors.


[FFT]: https://en.wikipedia.org/wiki/Fast_Fourier_transform
[rustfft]: https://docs.rs/rustfft/latest/rustfft/
[realfft]: https://docs.rs/realfft/latest/realfft/
[arrays]: https://doc.rust-lang.org/std/primitive.array.html
[generic_const_exprs]: https://github.com/rust-lang/rust/issues/76560
[Result]: https://doc.rust-lang.org/std/result/enum.Result.html
[Error]: https://doc.rust-lang.org/std/result/enum.Result.html#variant.Err
[realfft module]: https://docs.rs/easyfft/latest/easyfft/realfft/index.html
[dependent types]: https://en.wikipedia.org/wiki/Dependent_type
