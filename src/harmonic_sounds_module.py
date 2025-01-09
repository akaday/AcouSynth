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

def generate_noise(duration, sample_rate=44100):
    """
    Generate a noise component.

    Parameters:
    - duration: The duration of the noise (in seconds).
    - sample_rate: The sample rate of the noise (in samples per second).

    Returns:
    - A numpy array containing the generated noise.
    """
    return np.random.normal(0, 1, int(sample_rate * duration))

def combine_sine_and_noise(sine_wave, noise_component, noise_level=0.5):
    """
    Combine a sine wave and a noise component.

    Parameters:
    - sine_wave: A numpy array containing the sine wave data.
    - noise_component: A numpy array containing the noise component data.
    - noise_level: The level of the noise component to be combined with the sine wave (default is 0.5).

    Returns:
    - A numpy array containing the combined sound.
    """
    return sine_wave + noise_level * noise_component
