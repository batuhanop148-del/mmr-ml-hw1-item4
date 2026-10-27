"""
mmr_data.py - load, clean and tabulate real MMR (magneto-mechanical resonator) recordings.

Raw data: Reiss et al., "Parameter Estimation for Model-Based Sensing of Magneto-Mechanical
Resonators", Communications Physics (2026). Dataset DOI 10.15480/882.16742 (CC0 1.0).
The HDF5 layout below was taken from the authors' loader (readDataFromHDF5 in
github.com/IBIResearch/MMR-Parameter-Estimation, utils/utils.jl):
    /data                    -> Julia array data[sample, channel, frame]
    /sequence/baseFrequency  -> scalar
    /sequence/decimation     -> scalar      fs = baseFrequency / decimation
Julia stores arrays column-major, so h5py usually returns the REVERSED shape
(frame, channel, sample). load_raw() does not assume this: it identifies the sample axis
as the longest one and reorders to (frame, channel, sample).
"""
from __future__ import annotations
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.signal import butter, sosfiltfilt


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def sha256(path, chunk=2**20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def describe_h5(path) -> list[str]:
    """Return one line per group/dataset/attribute, so the file can be inspected before use."""
    lines = []
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                lines.append(f"DATASET {name}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                lines.append(f"GROUP   {name}")
            for k, v in obj.attrs.items():
                lines.append(f"   attr {k} = {v!r}")
        f.visititems(visit)
    return lines


def load_raw(path):
    """Return data (frame, channel, sample) as float64, fs [Hz], and a dict with load facts."""
    with h5py.File(path, "r") as f:
        raw = f["data"][()]
        base = float(np.squeeze(f["sequence/baseFrequency"][()]))
        dec = float(np.squeeze(f["sequence/decimation"][()]))
    info = {"stored_shape": tuple(raw.shape), "stored_dtype": str(raw.dtype)}
    if raw.ndim != 3:
        raise ValueError(f"expected a 3-D /data array, got shape {raw.shape} - inspect with describe_h5()")
    s_ax = int(np.argmax(raw.shape))                       # longest axis = time samples
    others = [a for a in range(3) if a != s_ax]
    # Julia order is (sample, channel, frame) -> reversed (frame, channel, sample).
    # Of the two remaining axes, the one that was the Julia "frame" axis is the one
    # farthest from the sample axis.
    frame_ax = max(others, key=lambda a: abs(a - s_ax))
    chan_ax = [a for a in others if a != frame_ax][0]
    data = np.transpose(raw, (frame_ax, chan_ax, s_ax)).astype(np.float64)
    info.update(axis_order_used="(frame, channel, sample)", fs=base / dec,
                baseFrequency=base, decimation=dec)
    return data, base / dec, info


# --------------------------------------------------------------------------------------
# Cleaning helpers
# --------------------------------------------------------------------------------------
def bandpass(x, fs, band, order=4):
    """Zero-phase Butterworth band-pass along the last axis (same idea as the authors' filtfilt)."""
    sos = butter(order, band, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def clipped_fraction(x):
    """Share of samples sitting exactly at the per-signal min or max - a saturation (clipping) symptom."""
    mx = np.max(np.abs(x), axis=-1, keepdims=True)
    return np.mean(np.isclose(np.abs(x), mx), axis=-1)


def envelope_rms(x, fs, win_s=0.01):
    """Moving RMS envelope along the last axis."""
    n = max(1, int(round(win_s * fs)))
    k = np.ones(n) / n
    return np.sqrt(np.apply_along_axis(lambda v: np.convolve(v ** 2, k, mode="same"), -1, x))


# --------------------------------------------------------------------------------------
# Reference sinusoid fit (non-linear least squares in f, linear least squares in A, phase)
# --------------------------------------------------------------------------------------
def _lin_fit(x, f, fs, tau=None):
    """Fit a*cos + b*sin. If tau is given, the basis decays as exp(-(t - t_mid)/tau), so the
    returned amplitude refers to the MIDDLE of x and the decay no longer shows up as residual."""
    t = np.arange(x.size) / fs
    env = np.exp(-(t - t.mean()) / tau) if tau else 1.0
    M = np.stack([env * np.cos(2 * np.pi * f * t), env * np.sin(2 * np.pi * f * t)], axis=1)
    (a, b), *_ = np.linalg.lstsq(M, x, rcond=None)
    r = x - M @ np.array([a, b])
    return np.hypot(a, b), np.arctan2(-b, a), float(np.sqrt(np.mean(r ** 2)))


def fit_sinusoid(x, fs, f_band, zp=8, tau=None):
    """Single-tone fit (optionally exponentially decaying with time constant tau).
    Phase refers to the first sample of x; amplitude to its middle. Returns f, A, phi, rms residual."""
    n = x.size
    nfft = zp * n
    X = np.abs(np.fft.rfft(x * np.hanning(n), nfft))
    fr = np.fft.rfftfreq(nfft, 1 / fs)
    band = np.where((fr >= f_band[0]) & (fr <= f_band[1]))[0]
    k = band[np.argmax(X[band])]
    step = fs / n
    lo, hi = max(f_band[0], fr[k] - step), min(f_band[1], fr[k] + step)
    f = minimize_scalar(lambda ff: _lin_fit(x, ff, fs, tau)[2], bounds=(lo, hi),
                        method="bounded", options={"xatol": 1e-5}).x
    A, phi, res = _lin_fit(x, f, fs, tau)
    return f, A, phi, res


# --------------------------------------------------------------------------------------
# Table builder: one row = one short window from one frame of one channel
# --------------------------------------------------------------------------------------
def build_table(sig, fs, frames, channel, f_band, win_s=0.05, ref_s=0.2,
                n_win=10, min_rel_amp=0.3):
    """
    sig     : cleaned signals (frame, channel, sample)
    Feature : the win_s-long window (raw samples, stored separately in X)
    Target  : f, A, phi from a reference fit on a longer ref_s segment centred on the window.
              The longer segment has lower variance, so it serves as a pseudo-label
              (no ground truth exists in real data).
    Returns (table DataFrame, X array of shape (rows, n_samples)).
    """
    nw, nr = int(round(win_s * fs)), int(round(ref_s * fs))
    rows, X = [], []
    for fr in frames:
        x = sig[fr, channel]
        env = envelope_rms(x, fs)
        peak = int(np.argmax(env))
        # start a little after the envelope peak, where the excitation is assumed to be over
        # and the free ring-down has begun (ASSUMPTION - check against the README.pdf timing)
        start0 = peak + int(round(0.01 * fs))
        # decay time constant of this frame: straight-line fit to log(envelope) over the usable part
        usable = np.where(env[start0:] >= min_rel_amp * env[peak])[0] + start0
        slope = np.polyfit(usable / fs, np.log(env[usable]), 1)[0]
        tau = -1.0 / slope if slope < 0 else None
        for i in range(n_win):
            s = start0 + i * nw
            c0 = s + nw // 2 - nr // 2                       # reference segment, centred on window
            if c0 < 0:
                # cannot centre the reference -> skip; a shifted reference would give neighbouring
                # windows the SAME label, and with frequency drift a biased one
                continue
            if c0 + nr > x.size:
                break
            w = x[s:s + nw]
            amp_rel = env[s + nw // 2] / env[peak]
            if amp_rel < min_rel_amp:                       # signal decayed too far - stop
                break
            f, A, phi_c, res = fit_sinusoid(x[c0:c0 + nr], fs, f_band, tau=tau)
            # move the reference phase from the segment start (c0) to the window start (s)
            phi = (phi_c + 2 * np.pi * f * (s - c0) / fs + np.pi) % (2 * np.pi) - np.pi
            rows.append(dict(frame=fr, channel=channel, t_start_s=s / fs, n_samples=nw,
                             fs_hz=fs, rel_amplitude=amp_rel,
                             snr_db_est=-20 * np.log10(res / (A / np.sqrt(2))),   # signal RMS / fit-residual RMS
                             tau_frame_s=tau, f_ref_hz=f, A_ref=A, phi_ref_rad=phi,
                             ref_residual_rel=res / (A / np.sqrt(2))))
            X.append(w.astype(np.float32))
    return pd.DataFrame(rows), np.array(X)
