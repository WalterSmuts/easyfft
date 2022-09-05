# easyfft
A Rust library crate providing an [FFT] API for arrays and slices. This crate wraps the
[rustfft] and [realfft] crates that does the heavy lifting behind the scenes.

### Advantages
* If it compiles, your code **won't panic™** in this library[^panic]
* No [Result] return type for you to worry about
* Ergonomic API

### Current limitations
* The `realfft` feature requires the `nightly` compiler because it depends on
  the [generic_const_exprs] feature

#### Footnotes
[^panic]: While this could be true in theory, in practice it most probably is not.
There could be other bugs in this crate or it's dependencies that may cause a
panic, but in theory all the runtime panics have been moved to compile time
errors.

[FFT]: https://en.wikipedia.org/wiki/Fast_Fourier_transform
[rustfft]: https://docs.rs/rustfft/latest/rustfft/
[realfft]: https://docs.rs/realfft/latest/realfft/
[arrays]: https://doc.rust-lang.org/std/primitive.array.html
[generic_const_exprs]: https://github.com/rust-lang/rust/issues/76560
[Result]: https://doc.rust-lang.org/std/result/enum.Result.html
[realfft module]: https://docs.rs/easyfft/latest/easyfft/realfft/index.html
