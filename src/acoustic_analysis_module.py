import numpy as np
from scipy.signal import find_peaks

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
