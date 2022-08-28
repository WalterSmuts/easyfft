fn main() {
    same_as_example()
}

fn same_as_example() {
    use constfft::RealFft;
    use constfft::RealIfft;
    [1.0_f64; 3].real_fft().real_ifft();
}
