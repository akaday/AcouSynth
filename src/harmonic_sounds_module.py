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

def synthesize_vocal_sound(fundamental_freq, formants, harmonics, duration, sample_rate=44100):
    """
    Synthesize a vocal-like sound.

    Parameters:
    - fundamental_freq: The fundamental frequency of the sound (in Hz).
    - formants: A list of tuples, where each tuple contains the formant frequency and its bandwidth.
    - harmonics: A list of tuples, where each tuple contains the harmonic number and its amplitude.
    - duration: The duration of the sound (in seconds).
    - sample_rate: The sample rate of the sound (in samples per second).

    Returns:
    - A numpy array containing the synthesized vocal sound.
    """
    harmonic_sound = generate_harmonic_sound(fundamental_freq, harmonics, duration, sample_rate)
    formant_sound = generate_formant_sound(fundamental_freq, formants, duration, sample_rate)
    noise = generate_noise(duration, sample_rate)
    vocal_sound = combine_sine_and_noise(harmonic_sound + formant_sound, noise)
    return vocal_sound

def synthesize_instrument_sound(fundamental_freq, harmonics, duration, sample_rate=44100):
    """
    Synthesize an instrument sound.

    Parameters:
    - fundamental_freq: The fundamental frequency of the sound (in Hz).
    - harmonics: A list of tuples, where each tuple contains the harmonic number and its amplitude.
    - duration: The duration of the sound (in seconds).
    - sample_rate: The sample rate of the sound (in samples per second).

    Returns:
    - A numpy array containing the synthesized instrument sound.
    """
    harmonic_sound = generate_harmonic_sound(fundamental_freq, harmonics, duration, sample_rate)
    noise = generate_noise(duration, sample_rate)
    instrument_sound = combine_sine_and_noise(harmonic_sound, noise)
    return instrument_sound

def synthesize_non_verbal_communication(fundamental_freq, formants, duration, sample_rate=44100):
    """
    Synthesize non-verbal communication sounds.

    Parameters:
    - fundamental_freq: The fundamental frequency of the sound (in Hz).
    - formants: A list of tuples, where each tuple contains the formant frequency and its bandwidth.
    - duration: The duration of the sound (in seconds).
    - sample_rate: The sample rate of the sound (in samples per second).

    Returns:
    - A numpy array containing the synthesized non-verbal communication sound.
    """
    formant_sound = generate_formant_sound(fundamental_freq, formants, duration, sample_rate)
    noise = generate_noise(duration, sample_rate)
    non_verbal_sound = combine_sine_and_noise(formant_sound, noise)
    return non_verbal_sound

def synthesize_sound_effect(fundamental_freq, harmonics, duration, sample_rate=44100):
    """
    Synthesize sound effects.

    Parameters:
    - fundamental_freq: The fundamental frequency of the sound (in Hz).
    - harmonics: A list of tuples, where each tuple contains the harmonic number and its amplitude.
    - duration: The duration of the sound (in seconds).
    - sample_rate: The sample rate of the sound (in samples per second).

    Returns:
    - A numpy array containing the synthesized sound effect.
    """
    harmonic_sound = generate_harmonic_sound(fundamental_freq, harmonics, duration, sample_rate)
    noise = generate_noise(duration, sample_rate)
    sound_effect = combine_sine_and_noise(harmonic_sound, noise)
    return sound_effect
