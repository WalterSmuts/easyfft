### Use [easyfft] instead
I realised that `constfft`'s primary feature, the lack of errors/panics and
implementation directly on arrays, does not have to be limited to arrays. So I
implemented it on slice too. This means the name `constfft` is a bit of a
misnomer, so [easyfft] was born. [easyfft] is strictly more powerful than
`constfft`.

[easyfft]: https://crates.io/crates/easyfft
