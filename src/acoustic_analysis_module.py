import pyfftw.interfaces.numpy_fft as fftw
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
    freqs = fftw.fftfreq(n, 1/sample_rate)
    spectrum = np.abs(fftw.fft(sound))
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

def filter_noise_for_formants(noise, formant_freqs, bandwidths, sample_rate=44100):
    """
    Filter noise components to create formants.

    Parameters:
    - noise: A numpy array containing the noise data.
    - formant_freqs: A list of formant frequencies to be emphasized.
    - bandwidths: A list of bandwidths for each formant frequency.
    - sample_rate: The sample rate of the noise (in samples per second).

    Returns:
    - A numpy array containing the filtered noise.
    """
    t = np.linspace(0, len(noise) / sample_rate, len(noise), endpoint=False)
    filtered_noise = noise.copy()
    for formant_freq, bandwidth in zip(formant_freqs, bandwidths):
        filtered_noise *= np.exp(-bandwidth * t) * np.sin(2 * np.pi * formant_freq * t)
    return filtered_noise

def manipulate_spectral_envelope(sound, envelope, sample_rate=44100):
    """
    Manipulate the spectral envelope to shape the timbre of the sound over time.

    Parameters:
    - sound: A numpy array containing the sound data.
    - envelope: A numpy array containing the spectral envelope data.
    - sample_rate: The sample rate of the sound (in samples per second).

    Returns:
    - A numpy array containing the sound with manipulated spectral envelope.
    """
    t = np.linspace(0, len(sound) / sample_rate, len(sound), endpoint=False)
    manipulated_sound = sound * envelope
    return manipulated_sound

def apply_subharmonics(sound, subharmonic_freqs, amplitudes, sample_rate=44100):
    """
    Generate lower harmonics for deeper tones.

    Parameters:
    - sound: A numpy array containing the sound data.
    - subharmonic_freqs: A list of subharmonic frequencies to be added.
    - amplitudes: A list of amplitudes for each subharmonic frequency.
    - sample_rate: The sample rate of the sound (in samples per second).

    Returns:
    - A numpy array containing the sound with added subharmonics.
    """
    t = np.linspace(0, len(sound) / sample_rate, len(sound), endpoint=False)
    subharmonic_sound = sound.copy()
    for subharmonic_freq, amplitude in zip(subharmonic_freqs, amplitudes):
        subharmonic_sound += amplitude * np.sin(2 * np.pi * subharmonic_freq * t)
    return subharmonic_sound

def apply_jitter_effects(sound, jitter_amount, sample_rate=44100):
    """
    Introduce random variations in pitch, amplitude, or timing for more organic or "shaky" sound characteristics.

    Parameters:
    - sound: A numpy array containing the sound data.
    - jitter_amount: The amount of jitter to be applied.
    - sample_rate: The sample rate of the sound (in samples per second).

    Returns:
    - A numpy array containing the sound with applied jitter effects.
    """
    t = np.linspace(0, len(sound) / sample_rate, len(sound), endpoint=False)
    jittered_sound = sound * (1 + jitter_amount * np.random.randn(len(sound)))
    return jittered_sound

def apply_pitch_modulation(sound, modulation_freq, modulation_depth, sample_rate=44100):
    """
    Control pitch bending and vibrato effects.

    Parameters:
    - sound: A numpy array containing the sound data.
    - modulation_freq: The frequency of the pitch modulation.
    - modulation_depth: The depth of the pitch modulation.
    - sample_rate: The sample rate of the sound (in samples per second).

    Returns:
    - A numpy array containing the sound with applied pitch modulation.
    """
    t = np.linspace(0, len(sound) / sample_rate, len(sound), endpoint=False)
    modulated_sound = sound * np.sin(2 * np.pi * modulation_freq * t) * modulation_depth
    return modulated_sound

def generate_synthetic_speech(pitch, formant_freqs, formant_bandwidths, duration, sample_rate=44100):
    """
    Generate synthetic speech using advanced synthesis techniques.

    Parameters:
    - pitch: The pitch of the synthetic speech (in Hz).
    - formant_freqs: A list of formant frequencies to be emphasized.
    - formant_bandwidths: A list of bandwidths for each formant frequency.
    - duration: The duration of the synthetic speech (in seconds).
    - sample_rate: The sample rate of the synthetic speech (in samples per second).

    Returns:
    - A numpy array containing the generated synthetic speech.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    speech = np.sin(2 * np.pi * pitch * t)
    for formant_freq, bandwidth in zip(formant_freqs, formant_bandwidths):
        speech *= np.exp(-bandwidth * t) * np.sin(2 * np.pi * formant_freq * t)
    return speech
