import unittest
import numpy as np
from src.harmonic_sounds_module import (
    generate_harmonic_sound,
    generate_formant_sound,
    generate_noise,
    combine_sine_and_noise,
    generate_syllabic_sound,
    real_time_spectral_analysis,
    control_parameters
)

class TestHarmonicSoundsModule(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 44100
        self.duration = 1.0
        self.t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        self.fundamental_freq = 440
        self.harmonics = [(1, 1.0), (2, 0.5), (3, 0.25)]
        self.formants = [(500, 50), (1500, 75), (2500, 100)]
        self.vowels = [np.sin(2 * np.pi * 440 * self.t), np.sin(2 * np.pi * 660 * self.t)]
        self.consonants = [np.sin(2 * np.pi * 880 * self.t), np.sin(2 * np.pi * 1100 * self.t)]
        self.structure = [('vowel', 0), ('consonant', 1), ('vowel', 1)]
        self.amplitude_envelope = np.linspace(1, 0, len(self.t))
        self.harmonic_content = [(1, 1.0), (2, 0.5), (3, 0.25)]
        self.noise_component = np.random.normal(0, 1, len(self.t))
        self.formant_frequencies = [500, 1500, 2500]
        self.temporal_evolution = {'attack': 0.1, 'sustain': 0.7, 'decay': 0.1, 'release': 0.1}

    def test_generate_harmonic_sound(self):
        sound = generate_harmonic_sound(self.fundamental_freq, self.harmonics, self.duration, self.sample_rate)
        self.assertEqual(len(sound), len(self.t))
        self.assertTrue(np.any(sound))

    def test_generate_formant_sound(self):
        sound = generate_formant_sound(self.fundamental_freq, self.formants, self.duration, self.sample_rate)
        self.assertEqual(len(sound), len(self.t))
        self.assertTrue(np.any(sound))

    def test_generate_noise(self):
        noise = generate_noise(self.duration, self.sample_rate)
        self.assertEqual(len(noise), len(self.t))
        self.assertTrue(np.any(noise))

    def test_combine_sine_and_noise(self):
        sine_wave = np.sin(2 * np.pi * self.fundamental_freq * self.t)
        noise_component = np.random.normal(0, 1, len(self.t))
        combined_sound = combine_sine_and_noise(sine_wave, noise_component, noise_level=0.5)
        self.assertEqual(len(combined_sound), len(self.t))
        self.assertTrue(np.any(combined_sound))

    def test_generate_syllabic_sound(self):
        syllabic_sound = generate_syllabic_sound(self.vowels, self.consonants, self.structure, self.duration, self.sample_rate)
        self.assertEqual(len(syllabic_sound), len(self.t) * len(self.structure))
        self.assertTrue(np.any(syllabic_sound))

    def test_real_time_spectral_analysis(self):
        sound = np.sin(2 * np.pi * self.fundamental_freq * self.t)
        frequencies, times, spectrogram_data = real_time_spectral_analysis(sound, self.sample_rate)
        self.assertTrue(np.any(frequencies))
        self.assertTrue(np.any(times))
        self.assertTrue(np.any(spectrogram_data))

    def test_control_parameters(self):
        sound = np.sin(2 * np.pi * self.fundamental_freq * self.t)
        manipulated_sound = control_parameters(
            sound,
            self.amplitude_envelope,
            self.harmonic_content,
            self.noise_component,
            self.formant_frequencies,
            self.temporal_evolution,
            self.sample_rate
        )
        self.assertEqual(len(manipulated_sound), len(self.t))
        self.assertTrue(np.any(manipulated_sound))

if __name__ == '__main__':
    unittest.main()
