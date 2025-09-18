#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
whisperize_hifisampler.py

Optimized, production-ready single-file module + CLI to convert an input WAV
(or numpy waveform) into a "constant whisper" (tn_fnds style) by preserving
spectral envelope (mel->linear->stft envelope) and re-synthesizing using a
noise source per-frame. Designed to be embedded into hifisampler's feature
pipeline or run standalone for batch preprocessing of voicebank samples.

Features / design points:
 - Uses librosa (if available) for high-quality mel<->stft inverse path.
 - Robust STFT fallback if librosa is unavailable.
 - Fast frame-wise FFT shaping + overlap-add with Hann window.
 - Optional hybrid consonant-preserve mode that keeps short high-energy
   transient segments from the original waveform to maintain crisp consonants.
 - Small "preserve_harmonics" mixing param for realistic, less-thin results.
 - CLI friendly and exposes `whisperize_wave()` for direct import into
   Python-based pipelines (e.g. hifisampler backend/feature generator).

Dependencies (recommended):
  pip install numpy scipy soundfile librosa tqdm

License: MIT (you can re-license when bundling with hifisampler per their
Apache-2.0 license rules — keep a notice).
"""

from __future__ import annotations
import argparse
import math
import os
import sys
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import get_window
from scipy.fft import rfft, irfft
from scipy.signal import lfilter

try:
    import librosa
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

# -----------------------------
# Utilities
# -----------------------------

def _pink_noise_voss(n: int) -> np.ndarray:
    """Generate pink noise using Voss-McCartney-ish approach (fast approximation).
    Returns float32 array length n.
    """
    # number of random sources
    n_rows = 16
    n_cols = int(np.ceil(n / n_rows))
    # random matrix and cumulative sum across rows
    mat = np.random.randn(n_rows, n_cols).astype(np.float32)
    cum = np.cumsum(mat, axis=0)
    flat = cum.flatten()[:n]
    # normalize
    flat = flat - flat.mean()
    flat = flat / (np.std(flat) + 1e-9)
    return flat.astype(np.float32)


def _white_noise(n: int) -> np.ndarray:
    return np.random.normal(0.0, 1.0, n).astype(np.float32)


# -----------------------------
# Envelope extraction helpers
# -----------------------------

def _stft_frames(y: np.ndarray, n_fft: int, hop_length: int, window: str = "hann") -> Tuple[np.ndarray, np.ndarray]:
    """Return framed windowed time-domain matrix and hann window (both float32).
    Frames shape: (n_frames, n_fft)
    """
    win = get_window(window, n_fft, fftbins=True).astype(np.float32)
    pad = n_fft // 2
    y_pad = np.pad(y, pad, mode="reflect")
    hop = hop_length
    n_frames = 1 + (len(y_pad) - n_fft) // hop
    # stride trick for efficiency
    frames = np.lib.stride_tricks.as_strided(
        y_pad,
        shape=(n_frames, n_fft),
        strides=(y_pad.strides[0] * hop, y_pad.strides[0]),
        writeable=False,
    ).copy()
    frames *= win[None, :]
    return frames, win


def compute_linear_envelope(y: np.ndarray, sr: int, n_fft: int, hop_length: int, use_librosa: bool = True, n_mels: int = 80) -> Tuple[np.ndarray, int]:
    """Compute per-frame linear magnitude envelopes (n_frames x (n_fft//2+1)).

    When librosa is present and use_librosa True, we compute log-mel
    spectrogram and invert mel->stft magnitude using librosa's helpers for
    smoother envelope extraction. Otherwise we fall back to plain STFT magnitude
    with light smoothing.
    Returns (S_env, n_frames)
    """
    if y.ndim > 1:
        y = y.mean(axis=1)
    if HAS_LIBROSA and use_librosa:
        # librosa path: compute mel-power spectrogram then invert to stft magnitude
        # 1) stft -> mel -> log-mel (like many vocoder frontends)
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann', center=True)
        mag = np.abs(S)
        # convert to mel-power
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        mel_spec = mel_basis.dot(mag)
        # small floor and smoothing in mel domain
        mel_spec = np.maximum(mel_spec, 1e-8)
        # invert mel -> approximate stft magnitude using librosa's pseudo-inverse
        # librosa has feature.inverse.mel_to_stft
        try:
            mag_approx = librosa.feature.inverse.mel_to_stft(mel_spec, sr=sr, n_fft=n_fft, power=1.0)
            mag_approx = np.maximum(mag_approx, 1e-8)
            S_env = mag_approx.T.astype(np.float32)  # shape (n_frames, n_fft//2+1)
            return S_env, S_env.shape[0]
        except Exception:
            # fall back to STFT path below
            pass
    # fallback: direct STFT magnitude + smoothing
    frames, _ = _stft_frames(y, n_fft, hop_length)
    # compute rfft magnitudes per frame
    S = np.abs(np.fft.rfft(frames, axis=1))
    # frequency smoothing (moving average across freq bins)
    def _smooth_freq(spec: np.ndarray, k: int = 5) -> np.ndarray:
        kernel = np.ones(k, dtype=np.float32) / float(k)
        return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), 1, spec)
    S_smooth = _smooth_freq(S, k=5)
    S_smooth = np.maximum(S_smooth, 1e-8)
    return S_smooth.astype(np.float32), S_smooth.shape[0]


# -----------------------------
# Core: whisperize_wave
# -----------------------------

def whisperize_wave(y: np.ndarray,
                     sr: int,
                     n_fft: int = 2048,
                     hop_length: int = 256,
                     noise: str = 'white',
                     hybrid: bool = True,
                     consonant_db: float = -40.0,
                     whisper_strength: float = 1.0,
                     preserve_harmonics: float = 0.0,
                     use_librosa: Optional[bool] = None,
                     progress: bool = False) -> np.ndarray:
    """Convert waveform y -> whispered waveform.

    Args:
      y: 1D float32 waveform (mono). If multi-channel, averaged to mono.
      sr: sample rate
      n_fft, hop_length: STFT params
      noise: 'white' or 'pink'
      hybrid: keep high-energy short transients from original
      consonant_db: dB threshold to detect frames for hybrid copy
      whisper_strength: 0..1 (1 full whisper), interpolation with original
      preserve_harmonics: small 0..0.05 to mix a fraction of original back
      use_librosa: override automatic librosa usage
    Returns:
      float32 waveform same length as input (clipped/normalized)
    """
    if use_librosa is None:
        use_librosa = HAS_LIBROSA
    # mono
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    S_env, n_frames = compute_linear_envelope(y, sr, n_fft, hop_length, use_librosa=use_librosa)
    frame_len = n_fft
    win = get_window('hann', frame_len, fftbins=True).astype(np.float32)

    out_frames = np.zeros((n_frames, frame_len), dtype=np.float32)
    iterator = range(n_frames)
    if progress and TQDM:
        iterator = tqdm(iterator, desc='whisper frames')

    for i in iterator:
        # noise source
        if noise == 'white':
            src = _white_noise(frame_len)
        else:
            src = _pink_noise_voss(frame_len)
        # FFT, shape by envelope
        src_spec = rfft(src)
        env = S_env[i]
        # if envelopes length != src_spec length, pad/trim
        L = src_spec.shape[0]
        if env.shape[0] != L:
            # resample envelope to match L
            env = np.interp(np.linspace(0, env.shape[0]-1, num=L), np.arange(env.shape[0]), env)
        shaped = src_spec * (env.astype(np.complex64))
        frame_t = irfft(shaped, n=frame_len).astype(np.float32)
        out_frames[i] = frame_t * win

    # overlap-add
    out = np.zeros(( (n_frames - 1) * hop_length + frame_len, ), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        out[start:start+frame_len] += out_frames[i]
    # window sum compensation
    win_sum = np.zeros_like(out)
    for i in range(n_frames):
        start = i * hop_length
        win_sum[start:start+frame_len] += win
    mask = win_sum > 1e-8
    out[mask] /= win_sum[mask]
    out = out[:len(y)]

    # hybrid consonant preserve
    if hybrid:
        # compute frame energies on original
        frames_orig, _ = _stft_frames(y, n_fft, hop_length)
        e = np.sum((frames_orig * win[None, :])**2, axis=1)
        e_db = 10.0 * np.log10(np.maximum(e, 1e-12))
        consonant_flags = e_db > consonant_db
        for i in range(n_frames):
            if consonant_flags[i]:
                s = i * hop_length
                e = s + frame_len
                if e > len(y):
                    continue
                # crossfade original to preserve transient
                alpha = 0.92
                out[s:e] = alpha * (y[s:e] * win) + (1.0 - alpha) * out[s:e]

    # tiny harmonic mixing
    if preserve_harmonics > 0.0:
        mix = preserve_harmonics * y[:len(out)]
        out = (1.0 - preserve_harmonics) * out + mix

    # mix with original according to whisper_strength
    if whisper_strength < 1.0:
        out = whisper_strength * out + (1.0 - whisper_strength) * y[:len(out)]

    # normalize
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 0:
        out = out / peak * 0.98
    return out


# -----------------------------
# CLI wrapper
# -----------------------------

def _parse_args():
    p = argparse.ArgumentParser(prog='whisperize_hifisampler', description='Create constant whisper WAVs')
    p.add_argument('input', help='input wav file path')
    p.add_argument('output', help='output wav file path')
    p.add_argument('--sr', type=int, default=44100)
    p.add_argument('--n_fft', type=int, default=2048)
    p.add_argument('--hop', type=int, default=256)
    p.add_argument('--noise', choices=['white', 'pink'], default='white')
    p.add_argument('--no-hybrid', action='store_true', help='disable hybrid consonant preservation')
    p.add_argument('--consonant-db', type=float, default=-40.0)
    p.add_argument('--whisper-strength', type=float, default=1.0)
    p.add_argument('--preserve-harmonics', type=float, default=0.0)
    p.add_argument('--no-resample', action='store_true')
    p.add_argument('--progress', action='store_true')
    return p.parse_args()


def cli_main():
    args = _parse_args()
    x, sr0 = sf.read(args.input)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if (sr0 != args.sr) and (not args.no_resample):
        if HAS_LIBROSA:
            x = librosa.resample(x.astype(np.float32), sr0, args.sr)
        else:
            from scipy.signal import resample
            n_new = int(len(x) * args.sr / sr0)
            x = resample(x, n_new)
        sr0 = args.sr
    out = whisperize_wave(x.astype(np.float32), sr0,
                          n_fft=args.n_fft, hop_length=args.hop,
                          noise=args.noise,
                          hybrid=not args.no_hybrid,
                          consonant_db=args.consonant_db,
                          whisper_strength=args.whisper_strength,
                          preserve_harmonics=args.preserve_harmonics,
                          progress=args.progress)
    sf.write(args.output, out.astype(np.float32), sr0)
    print('Wrote', args.output)


if __name__ == '__main__':
    cli_main()
