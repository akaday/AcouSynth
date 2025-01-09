import numpy as np
from scipy.signal import find_peaks
from src.harmonic_sounds_module import combine_sine_and_noise

def integrate_theoretical_acoustics_with_practical_synthesis(sound, sample_rate=44100):
    """
    Integrate theoretical acoustics with practical synthesis.

    Parameters:
    - sound: A numpy array containing the sound data.
    - sample_rate: The sample rate of the sound (in samples per second).

    Returns:
    - A numpy array containing the integrated sound.
    """
    freqs, spectrum = analyze_frequency_spectrum(sound, sample_rate)
    formants = detect_formants(sound, sample_rate)
    harmonic_ratios = calculate_harmonic_ratios(sound, sample_rate)
    
    integrated_sound = generate_harmonic_sound(freqs[0], harmonic_ratios, len(sound) / sample_rate, sample_rate)
    for formant_freq, bandwidth in formants:
        integrated_sound *= np.exp(-bandwidth * np.linspace(0, len(sound) / sample_rate, len(sound))) * np.sin(2 * np.pi * formant_freq * np.linspace(0, len(sound) / sample_rate, len(sound)))
    
    noise_component = generate_noise(len(sound) / sample_rate, sample_rate)
    integrated_sound = combine_sine_and_noise(integrated_sound, noise_component)
    
    return integrated_sound

def analyze_frequency_spectrum(sound, sample_rate=44100):
    """
    Analyze the frequency spectrum of a given sound.

    Parameters:
    - sound: A numpy array containing the sound data.
    - sample_rate: The sample rate of the sound (in samples per second).

    Returns:
    - A tuple containing the frequencies and their corresponding amplitudes.
    """
    n = len(sound)
    freqs = np.fft.fftfreq(n, 1/sample_rate)
    spectrum = np.abs(np.fft.fft(sound))
    return freqs[:n//2], spectrum[:n//2]

def detect_formants(sound, sample_rate=44100, num_formants=5):
    """
    Detect formants in a given sound.

    Parameters:
    - sound: A numpy array containing the sound data.
    - sample_rate: The sample rate of the sound (in samples per second).
    - num_formants: The number of formants to detect.

    Returns:
    - A list of tuples, where each tuple contains the formant frequency and its bandwidth.
    """
    freqs, spectrum = analyze_frequency_spectrum(sound, sample_rate)
    peaks, _ = find_peaks(spectrum, height=np.max(spectrum)/num_formants)
    formants = [(freqs[peak], spectrum[peak]) for peak in peaks[:num_formants]]
    return formants

def calculate_harmonic_ratios(sound, sample_rate=44100):
    """
    Calculate the harmonic ratios of a given sound.

    Parameters:
    - sound: A numpy array containing the sound data.
    - sample_rate: The sample rate of the sound (in samples per second).

    Returns:
    - A list of harmonic ratios.
    """
    freqs, spectrum = analyze_frequency_spectrum(sound, sample_rate)
    fundamental_freq = freqs[np.argmax(spectrum)]
    harmonic_ratios = [freq / fundamental_freq for freq in freqs if freq % fundamental_freq == 0]
    return harmonic_ratios

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
