"""
Analog modulation and demodulation simulation for DSB-SC, AM, SSB, and FM.

The script generates or loads an audio baseband signal m(t), modulates it with
four analog modulation methods, adds AWGN at high/low SNR, demodulates the noisy
signals, calculates input/output SNR, and saves plots plus WAV files.

Default usage:
    python am_dsb_ssb_fm_simulation.py

Optional WAV input:
    python am_dsb_ssb_fm_simulation.py --audio path/to/input.wav

If --audio is omitted, the script first looks for a WAV file in ./myvioce.
"""

from __future__ import annotations

import argparse
import csv
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_AUDIO_DIR = Path("myvioce")


@dataclass(frozen=True)
class SimConfig:
    fs: int = 48_000
    duration: float = 2.0
    fc: float = 8_000.0
    message_bandwidth: float = 3_000.0
    am_modulation_index: float = 0.65
    fm_frequency_deviation: float = 1_800.0
    high_snr_db: float = 30.0
    low_snr_db: float = 8.0
    random_seed: int = 20260427


def normalize(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = float(np.max(np.abs(x)))
    if max_abs < 1e-12:
        return x.copy()
    return peak * x / max_abs


def rms_power(x: np.ndarray) -> float:
    return float(np.mean(np.square(x)))


def snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    noise = test - reference
    p_signal = rms_power(reference)
    p_noise = rms_power(noise)
    if p_noise < 1e-18:
        return float("inf")
    return 10.0 * math.log10(p_signal / p_noise)


def add_awgn(x: np.ndarray, target_snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p_signal = rms_power(x)
    p_noise = p_signal / (10.0 ** (target_snr_db / 10.0))
    noise = rng.normal(0.0, math.sqrt(p_noise), size=x.shape)
    return x + noise


def fft_filter(x: np.ndarray, fs: int, pass_mask: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft(x)
    filtered = np.fft.ifft(spectrum * pass_mask)
    return np.real(filtered)


def lowpass(x: np.ndarray, fs: int, cutoff_hz: float) -> np.ndarray:
    freqs = np.fft.fftfreq(len(x), d=1.0 / fs)
    mask = np.abs(freqs) <= cutoff_hz
    return fft_filter(x, fs, mask)


def bandpass(x: np.ndarray, fs: int, low_hz: float, high_hz: float) -> np.ndarray:
    freqs = np.fft.fftfreq(len(x), d=1.0 / fs)
    mask = (np.abs(freqs) >= low_hz) & (np.abs(freqs) <= high_hz)
    return fft_filter(x, fs, mask)


def analytic_signal(x: np.ndarray) -> np.ndarray:
    """Return analytic signal using an FFT Hilbert transform."""
    n = len(x)
    spectrum = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * h)


def find_default_audio(audio_dir: Path = DEFAULT_AUDIO_DIR) -> Path | None:
    wav_files = sorted(audio_dir.glob("*.wav"))
    return wav_files[0] if wav_files else None


def choose_active_segment(data: np.ndarray, target_len: int, fs: int) -> np.ndarray:
    if len(data) <= target_len:
        return np.pad(data, (0, target_len - len(data)))

    frame_len = max(1, int(0.05 * fs))
    hop = max(1, frame_len // 2)
    best_start = 0
    best_power = -1.0
    max_start = len(data) - target_len

    for start in range(0, max_start + 1, hop):
        segment = data[start : start + target_len]
        power = rms_power(segment)
        if power > best_power:
            best_power = power
            best_start = start

    return data[best_start : best_start + target_len]


def load_wav_mono(path: Path, fs: int, duration: float, bandwidth_hz: float) -> np.ndarray:
    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        sampwidth = reader.getsampwidth()
        source_fs = reader.getframerate()
        raw = reader.readframes(reader.getnframes())

    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    if source_fs != fs:
        old_t = np.arange(len(data)) / source_fs
        new_len = int(len(data) * fs / source_fs)
        new_t = np.arange(new_len) / fs
        data = np.interp(new_t, old_t, data)

    target_len = int(fs * duration)
    data = choose_active_segment(data, target_len, fs)
    data = lowpass(data, fs, min(0.45 * fs, bandwidth_hz))
    return normalize(data)


def synthesize_message(cfg: SimConfig) -> np.ndarray:
    """Create a speech-like personalized audio waveform for repeatable tests."""
    t = np.arange(int(cfg.fs * cfg.duration)) / cfg.fs
    vowel_a = 0.45 * np.sin(2 * np.pi * 180 * t)
    vowel_b = 0.28 * np.sin(2 * np.pi * 360 * t + 0.2)
    vowel_c = 0.18 * np.sin(2 * np.pi * 720 * t + 0.5)
    tone = 0.20 * np.sin(2 * np.pi * 1_120 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2.0 * t))
    envelope = 0.55 + 0.35 * np.sin(2 * np.pi * 3.0 * t) ** 2
    message = envelope * (vowel_a + vowel_b + vowel_c + tone)
    message += 0.08 * np.sin(2 * np.pi * 2_200 * t + 0.4)
    return normalize(lowpass(message, cfg.fs, cfg.message_bandwidth))


def dsb_sc_modulate(m: np.ndarray, t: np.ndarray, cfg: SimConfig) -> np.ndarray:
    return m * np.cos(2 * np.pi * cfg.fc * t)


def dsb_sc_demodulate(r: np.ndarray, t: np.ndarray, cfg: SimConfig) -> np.ndarray:
    mixed = 2.0 * r * np.cos(2 * np.pi * cfg.fc * t)
    return normalize(lowpass(mixed, cfg.fs, cfg.message_bandwidth))


def am_modulate(m: np.ndarray, t: np.ndarray, cfg: SimConfig) -> np.ndarray:
    carrier = np.cos(2 * np.pi * cfg.fc * t)
    return (1.0 + cfg.am_modulation_index * m) * carrier


def am_envelope_demodulate(r: np.ndarray, cfg: SimConfig) -> np.ndarray:
    envelope = np.abs(analytic_signal(r))
    baseband = (envelope - np.mean(envelope)) / cfg.am_modulation_index
    return normalize(lowpass(baseband, cfg.fs, cfg.message_bandwidth))


def ssb_modulate_usb(m: np.ndarray, t: np.ndarray, cfg: SimConfig) -> np.ndarray:
    mh = np.imag(analytic_signal(m))
    return m * np.cos(2 * np.pi * cfg.fc * t) - mh * np.sin(2 * np.pi * cfg.fc * t)


def ssb_demodulate(r: np.ndarray, t: np.ndarray, cfg: SimConfig) -> np.ndarray:
    mixed = 2.0 * r * np.cos(2 * np.pi * cfg.fc * t)
    return normalize(lowpass(mixed, cfg.fs, cfg.message_bandwidth))


def fm_modulate(m: np.ndarray, t: np.ndarray, cfg: SimConfig) -> np.ndarray:
    integral_m = np.cumsum(m) / cfg.fs
    phase = 2 * np.pi * cfg.fc * t + 2 * np.pi * cfg.fm_frequency_deviation * integral_m
    return np.cos(phase)


def fm_demodulate(r: np.ndarray, t: np.ndarray, cfg: SimConfig) -> np.ndarray:
    del t
    analytic = analytic_signal(r)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase, prepend=phase[0]) * cfg.fs / (2 * np.pi)
    baseband = (inst_freq - cfg.fc) / cfg.fm_frequency_deviation
    return normalize(lowpass(baseband, cfg.fs, cfg.message_bandwidth))


def spectrum_db(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    window = np.hanning(len(x))
    spec = np.fft.fftshift(np.fft.fft(x * window))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(x), d=1.0 / fs))
    mag = 20 * np.log10(np.abs(spec) / len(x) + 1e-12)
    return freqs, mag


def save_wav(path: Path, x: np.ndarray, fs: int) -> None:
    y = normalize(x, 0.98)
    pcm = np.clip(y * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(fs)
        writer.writeframes(pcm.tobytes())


def plot_modulation(
    out_path: Path,
    name: str,
    m: np.ndarray,
    s: np.ndarray,
    demod_high: np.ndarray,
    demod_low: np.ndarray,
    t: np.ndarray,
    cfg: SimConfig,
) -> None:
    n_time = min(len(t), int(0.015 * cfg.fs))
    f_m, sp_m = spectrum_db(m, cfg.fs)
    f_s, sp_s = spectrum_db(s, cfg.fs)

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(f"{name} modulation and demodulation", fontsize=15)

    axes[0, 0].plot(t[:n_time] * 1_000, m[:n_time], lw=1.1)
    axes[0, 0].set_title("Baseband m(t)")
    axes[0, 0].set_xlabel("Time / ms")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t[:n_time] * 1_000, s[:n_time], lw=1.0)
    axes[0, 1].set_title("Modulated signal s(t)")
    axes[0, 1].set_xlabel("Time / ms")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(f_m / 1_000, sp_m, lw=1.0)
    axes[1, 0].set_xlim(-5, 5)
    axes[1, 0].set_ylim(np.max(sp_m) - 90, np.max(sp_m) + 5)
    axes[1, 0].set_title("Spectrum of m(t)")
    axes[1, 0].set_xlabel("Frequency / kHz")
    axes[1, 0].set_ylabel("Magnitude / dB")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(f_s / 1_000, sp_s, lw=1.0)
    axes[1, 1].set_xlim(-15, 15)
    axes[1, 1].set_ylim(np.max(sp_s) - 90, np.max(sp_s) + 5)
    axes[1, 1].set_title("Spectrum of s(t)")
    axes[1, 1].set_xlabel("Frequency / kHz")
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].plot(t[:n_time] * 1_000, m[:n_time], label="original", lw=1.2)
    axes[2, 0].plot(t[:n_time] * 1_000, demod_high[:n_time], label="demod high SNR", lw=1.0)
    axes[2, 0].set_title("Demodulated output, high SNR")
    axes[2, 0].set_xlabel("Time / ms")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()

    axes[2, 1].plot(t[:n_time] * 1_000, m[:n_time], label="original", lw=1.2)
    axes[2, 1].plot(t[:n_time] * 1_000, demod_low[:n_time], label="demod low SNR", lw=1.0)
    axes[2, 1].set_title("Demodulated output, low SNR")
    axes[2, 1].set_xlabel("Time / ms")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_case(
    name: str,
    modulate,
    demodulate,
    m: np.ndarray,
    t: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
    out_dir: Path,
) -> list[dict[str, str]]:
    s = modulate(m, t, cfg)
    rows: list[dict[str, str]] = []
    demod_outputs: dict[str, np.ndarray] = {}

    for label, target_snr in [("high", cfg.high_snr_db), ("low", cfg.low_snr_db)]:
        r = add_awgn(s, target_snr, rng)
        y = demodulate(r, t, cfg) if name != "AM" else demodulate(r, cfg)
        input_snr = snr_db(s, r)
        output_snr = snr_db(m, y)
        demod_outputs[label] = y
        save_wav(out_dir / f"{name.lower()}_{label}_snr_demod.wav", y, cfg.fs)
        rows.append(
            {
                "scheme": name,
                "snr_case": label,
                "target_input_snr_db": f"{target_snr:.2f}",
                "measured_input_snr_db": f"{input_snr:.2f}",
                "measured_output_snr_db": f"{output_snr:.2f}",
            }
        )

    plot_modulation(
        out_dir / f"{name.lower()}_summary.png",
        name,
        m,
        s,
        demod_outputs["high"],
        demod_outputs["low"],
        t,
        cfg,
    )
    save_wav(out_dir / f"{name.lower()}_modulated_preview.wav", s, cfg.fs)
    return rows


def write_report(out_dir: Path, cfg: SimConfig, rows: list[dict[str, str]], message_source: str) -> None:
    csv_path = out_dir / "snr_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fm_bandwidth = 2 * (cfg.fm_frequency_deviation + cfg.message_bandwidth)
    report = [
        "# Analog Modulation Simulation Results",
        "",
        "## Parameters",
        f"- Message source = {message_source}",
        f"- Sampling frequency fs = {cfg.fs} Hz",
        f"- Signal duration = {cfg.duration:.2f} s",
        f"- Carrier frequency fc = {cfg.fc:.1f} Hz",
        f"- Message bandwidth W = {cfg.message_bandwidth:.1f} Hz",
        f"- AM modulation index ma = {cfg.am_modulation_index:.2f}",
        f"- FM frequency deviation delta_f = {cfg.fm_frequency_deviation:.1f} Hz",
        f"- FM Carson bandwidth B ~= 2(delta_f + W) = {fm_bandwidth:.1f} Hz",
        f"- High/low input SNR targets = {cfg.high_snr_db:.1f} dB / {cfg.low_snr_db:.1f} dB",
        "",
        "## SNR Results",
        "",
        "| Scheme | Case | Target input SNR / dB | Measured input SNR / dB | Output SNR / dB |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['scheme']} | {row['snr_case']} | {row['target_input_snr_db']} | "
            f"{row['measured_input_snr_db']} | {row['measured_output_snr_db']} |"
        )
    report.extend(
        [
            "",
            "## Notes",
            "- DSB-SC and SSB use coherent detection.",
            "- AM uses envelope detection, so low SNR and over-modulation are more damaging.",
            "- SSB is generated as an upper-sideband signal with an FFT Hilbert transform.",
            "- FM is demodulated by differentiating the unwrapped analytic-signal phase.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DSB-SC, AM, SSB, and FM simulation")
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Optional mono/stereo WAV file as m(t). If omitted, the first .wav in ./myvioce is used.",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--duration", type=float, default=2.0, help="Simulation duration in seconds")
    parser.add_argument("--fs", type=int, default=48_000, help="Sampling frequency in Hz")
    parser.add_argument("--fc", type=float, default=8_000.0, help="Carrier frequency in Hz")
    parser.add_argument("--high-snr", type=float, default=30.0, help="High input SNR in dB")
    parser.add_argument("--low-snr", type=float, default=8.0, help="Low input SNR in dB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimConfig(
        fs=args.fs,
        duration=args.duration,
        fc=args.fc,
        high_snr_db=args.high_snr,
        low_snr_db=args.low_snr,
    )
    if cfg.fc + cfg.message_bandwidth >= cfg.fs / 2:
        raise ValueError("fc + message bandwidth must be lower than Nyquist frequency.")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.random_seed)
    t = np.arange(int(cfg.fs * cfg.duration)) / cfg.fs

    audio_path = args.audio or find_default_audio()
    if audio_path:
        m = load_wav_mono(audio_path, cfg.fs, cfg.duration, cfg.message_bandwidth)
        message_source = str(audio_path)
    else:
        m = synthesize_message(cfg)
        message_source = "synthetic speech-like signal"
    save_wav(out_dir / "baseband_message.wav", m, cfg.fs)

    all_rows: list[dict[str, str]] = []
    all_rows += run_case("DSB-SC", dsb_sc_modulate, dsb_sc_demodulate, m, t, cfg, rng, out_dir)
    all_rows += run_case("AM", am_modulate, am_envelope_demodulate, m, t, cfg, rng, out_dir)
    all_rows += run_case("SSB", ssb_modulate_usb, ssb_demodulate, m, t, cfg, rng, out_dir)
    all_rows += run_case("FM", fm_modulate, fm_demodulate, m, t, cfg, rng, out_dir)
    write_report(out_dir, cfg, all_rows, message_source)

    print(f"Message source: {message_source}")
    print(f"Done. Results saved to: {out_dir.resolve()}")
    print("Key files: summary.md, snr_results.csv, *_summary.png, *_demod.wav")


if __name__ == "__main__":
    main()
