# constfft
A Rust library crate providing an [FFT] API for arrays. This crate wraps the
[rustfft] crate that does the heavy lifting behind the scenes. Use `constfft`
if you're working with [arrays], i.e. you know the size of the signal at
compile time.

### Advantages
* If it compiles, your code **won't panic™** in this library[^panic]
* Ergonomic API

### Current limitations
* No caching of the intermediate context/structs required to do the fft calculation
* No implementation for slices
* No implementation for `real_[i]fft`: [issue][real_fft_issue]

### Possible future plans
Currently I don't see any reason why the same API (minus the compile time size
checks) cannot be implementable on slices. This seems like a much nicer API to
work with and AFAICT does not depend on the types to be arrays. If this is the
case I'd implement the same API on slices and rename this crate to `easyfft`.

#### Footnotes
[^panic]: While this could be true in theory, in practice it probably is not.
There could be bugs in this crate or it's dependencies that may cause a panic,
but in theory all the runtime panics have been moved to compile time errors.

[FFT]: https://en.wikipedia.org/wiki/Fast_Fourier_transform
[rustfft]: https://docs.rs/rustfft/latest/rustfft/
[arrays]: https://doc.rust-lang.org/std/primitive.array.html
[generic_const_exprs]: https://github.com/rust-lang/rust/issues/76560
[real_fft_issue]: https://github.com/WalterSmuts/constfft/issues/1
