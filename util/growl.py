import numpy as np
from scipy.signal import butter, sosfilt, square

_MAX_VIBRATO_CENTS = 100.0
_HP_CUTOFF_HZ      = 20.0
_MIN_NYQ_FRAC      = 0.01


def _highpass(audio: np.ndarray, sr: int, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    nyq   = sr / 2
    norm  = np.clip(cutoff / nyq, _MIN_NYQ_FRAC, 0.99)
    sos   = butter(4, norm, "high", output="sos")
    high  = sosfilt(sos, audio)
    high  = np.asarray(high)
    return high, audio - high


def _square_lfo(num_samples: int, sr: int, freq: float) -> np.ndarray:
    t = np.arange(num_samples) / sr
    return square(2 * np.pi * freq * t)


def _apply_pitch_modulation(band: np.ndarray,
                            sr: int,
                            lfo: np.ndarray,
                            strength: float) -> np.ndarray:
    cents   = lfo * strength * _MAX_VIBRATO_CENTS
    ratio   = 2 ** (cents / 1200.0)
    cumsum  = np.cumsum(ratio) - ratio[0]
    ideal   = np.arange(len(band)) * np.mean(ratio)
    drift   = cumsum - ideal

    if len(band) > 100:
        hp_norm = _HP_CUTOFF_HZ / (sr / 2)
        drift   = sosfilt(butter(2, hp_norm, "high", output="sos"), drift)

    idx       = np.clip(np.arange(len(band)) + drift, 0, len(band) - 1)
    modulated = np.interp(idx, np.arange(len(band)), band)

    rms_orig  = np.sqrt(np.mean(band ** 2))
    rms_new   = np.sqrt(np.mean(modulated ** 2))
    if rms_new > 1e-10:
        modulated *= rms_orig / rms_new
    return modulated


def growl(audio: np.ndarray,
          sample_rate: int,
          *,
          frequency: float = 80.0,
          strength: float = 0.5,
          freq_low: float = 400.0) -> np.ndarray:
    if strength == 0 or frequency <= 0:
        return audio.copy()

    band, complement = _highpass(audio, sample_rate, freq_low)
    lfo              = _square_lfo(len(audio), sample_rate, frequency)
    band             = _apply_pitch_modulation(band, sample_rate, lfo, strength)

    return complement + band