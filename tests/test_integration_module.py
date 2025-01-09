import unittest
import numpy as np
from src.integration_module import (
    integrate_theoretical_acoustics_with_practical_synthesis,
    integrate_new_tools_with_existing_tools
)
from src.harmonic_sounds_module import generate_harmonic_sound, generate_noise
from src.acoustic_analysis_module import generate_synthetic_speech

class TestIntegrationModule(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 44100
        self.duration = 1.0
        self.t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        self.sound = np.sin(2 * np.pi * 440 * self.t)
        self.sine_waves = [np.sin(2 * np.pi * 440 * self.t), np.sin(2 * np.pi * 660 * self.t)]
        self.noise_components = [np.random.normal(0, 1, len(self.t)), np.random.normal(0, 1, len(self.t))]
        self.spectral_envelopes = [np.linspace(1, 0, len(self.t)), np.linspace(0, 1, len(self.t))]
        self.pitch = 100
        self.formant_freqs = [500, 1500, 2500]
        self.formant_bandwidths = [50, 75, 100]

    def test_integrate_theoretical_acoustics_with_practical_synthesis(self):
        integrated_sound = integrate_theoretical_acoustics_with_practical_synthesis(self.sound, self.sample_rate)
        self.assertEqual(len(integrated_sound), len(self.sound))
        self.assertTrue(np.any(integrated_sound))

    def test_integrate_new_tools_with_existing_tools(self):
        integrated_sound = integrate_new_tools_with_existing_tools(
            self.sine_waves,
            self.noise_components,
            self.spectral_envelopes,
            self.pitch,
            self.formant_freqs,
            self.formant_bandwidths,
            self.duration,
            self.sample_rate
        )
        self.assertEqual(len(integrated_sound), len(self.t))
        self.assertTrue(np.any(integrated_sound))

if __name__ == '__main__':
    unittest.main()
