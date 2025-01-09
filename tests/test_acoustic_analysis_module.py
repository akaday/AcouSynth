import unittest
import numpy as np
from src.acoustic_analysis_module import (
    filter_noise_for_formants,
    manipulate_spectral_envelope,
    apply_subharmonics,
    apply_jitter_effects,
    apply_pitch_modulation
)

class TestAcousticAnalysisModule(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 44100
        self.duration = 1.0
        self.t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        self.sound = np.sin(2 * np.pi * 440 * self.t)
        self.noise = np.random.normal(0, 1, int(self.sample_rate * self.duration))

    def test_filter_noise_for_formants(self):
        formant_freqs = [500, 1500, 2500]
        bandwidths = [50, 75, 100]
        filtered_noise = filter_noise_for_formants(self.noise, formant_freqs, bandwidths, self.sample_rate)
        self.assertEqual(len(filtered_noise), len(self.noise))
        self.assertTrue(np.any(filtered_noise))

    def test_manipulate_spectral_envelope(self):
        envelope = np.linspace(1, 0, len(self.sound))
        manipulated_sound = manipulate_spectral_envelope(self.sound, envelope, self.sample_rate)
        self.assertEqual(len(manipulated_sound), len(self.sound))
        self.assertTrue(np.any(manipulated_sound))

    def test_apply_subharmonics(self):
        subharmonic_freqs = [220, 110]
        amplitudes = [0.5, 0.25]
        subharmonic_sound = apply_subharmonics(self.sound, subharmonic_freqs, amplitudes, self.sample_rate)
        self.assertEqual(len(subharmonic_sound), len(self.sound))
        self.assertTrue(np.any(subharmonic_sound))

    def test_apply_jitter_effects(self):
        jitter_amount = 0.05
        jittered_sound = apply_jitter_effects(self.sound, jitter_amount, self.sample_rate)
        self.assertEqual(len(jittered_sound), len(self.sound))
        self.assertTrue(np.any(jittered_sound))

    def test_apply_pitch_modulation(self):
        modulation_freq = 5
        modulation_depth = 0.1
        modulated_sound = apply_pitch_modulation(self.sound, modulation_freq, modulation_depth, self.sample_rate)
        self.assertEqual(len(modulated_sound), len(self.sound))
        self.assertTrue(np.any(modulated_sound))

if __name__ == '__main__':
    unittest.main()
