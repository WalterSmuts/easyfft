# constfft
A Rust library crate providing an [FFT] API for arrays. This crate wraps the
[rustfft] and [realfft] crates that does the heavy lifting behind the scenes.
Use `constfft` if you're working with [arrays], i.e. you know the size of the
signal at compile time.

### Advantages
* Compile time size checking
* If it compiles, your code should not panic

### Current limitations
* No caching of the intermediate context/structs required to do the fft calculation.
* Even/Odd bounds don't seem to propogate back up the call chain (TODO: Add
  example and link to rust issue)
* No implementation for slices

### Possible future plans
Currently I don't see any reason why the same API (minus the compile time size
checks) cannot be implementable on slices. This seems like a much nicer API to
work with and AFAICT does not depend on the types to be arrays. If this is the
case I'd implement the same API on slices and rename this crate to `easyfft`.

[FFT]: https://en.wikipedia.org/wiki/Fast_Fourier_transform
[rustfft]: https://docs.rs/rustfft/latest/rustfft/
[realfft]: https://docs.rs/realfft/latest/realfft/
[arrays]: https://doc.rust-lang.org/std/primitive.array.html://docs.rs/realfft/0.3.0/realfft/
