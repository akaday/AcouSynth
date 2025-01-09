import numpy as np

def generate_harmonic_sound(fundamental_freq, harmonics, duration, sample_rate=44100):
    """
    Generate a harmonic sound with given fundamental frequency and harmonics.

    Parameters:
    - fundamental_freq: The fundamental frequency of the sound (in Hz).
    - harmonics: A list of tuples, where each tuple contains the harmonic number and its amplitude.
    - duration: The duration of the sound (in seconds).
    - sample_rate: The sample rate of the sound (in samples per second).

    Returns:
    - A numpy array containing the generated harmonic sound.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sound = np.zeros_like(t)
    for harmonic, amplitude in harmonics:
        sound += amplitude * np.sin(2 * np.pi * harmonic * fundamental_freq * t)
    return sound

def generate_formant_sound(fundamental_freq, formants, duration, sample_rate=44100):
    """
    Generate a sound with given fundamental frequency and formants.

    Parameters:
    - fundamental_freq: The fundamental frequency of the sound (in Hz).
    - formants: A list of tuples, where each tuple contains the formant frequency and its bandwidth.
    - duration: The duration of the sound (in seconds).
    - sample_rate: The sample rate of the sound (in samples per second).

    Returns:
    - A numpy array containing the generated formant sound.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sound = np.sin(2 * np.pi * fundamental_freq * t)
    for formant_freq, bandwidth in formants:
        sound *= np.exp(-bandwidth * t) * np.sin(2 * np.pi * formant_freq * t)
    return sound
