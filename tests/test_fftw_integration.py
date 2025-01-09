import numpy as np
import pyfftw.interfaces.numpy_fft as fftw
import time
import unittest

class TestFFTWIntegration(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 44100
        self.duration = 1.0
        self.t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        self.sound = np.sin(2 * np.pi * 440 * self.t)

    def test_fftw_correctness(self):
        # Compute FFT using numpy
        np_fft_result = np.fft.fft(self.sound)
        np_freqs = np.fft.fftfreq(len(self.sound), 1/self.sample_rate)

        # Compute FFT using FFTW
        fftw_fft_result = fftw.fft(self.sound)
        fftw_freqs = fftw.fftfreq(len(self.sound), 1/self.sample_rate)

        # Compare the results
        np.testing.assert_array_almost_equal(np_fft_result, fftw_fft_result, decimal=5)
        np.testing.assert_array_almost_equal(np_freqs, fftw_freqs, decimal=5)

    def test_fftw_performance(self):
        input_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
        np_times = []
        fftw_times = []

        for size in input_sizes:
            sound = np.sin(2 * np.pi * 440 * np.linspace(0, self.duration, size, endpoint=False))

            # Measure numpy FFT time
            start_time = time.time()
            np.fft.fft(sound)
            np_times.append(time.time() - start_time)

            # Measure FFTW time
            start_time = time.time()
            fftw.fft(sound)
            fftw_times.append(time.time() - start_time)

        print("Input Sizes:", input_sizes)
        print("NumPy FFT Times:", np_times)
        print("FFTW FFT Times:", fftw_times)

if __name__ == '__main__':
    unittest.main()
