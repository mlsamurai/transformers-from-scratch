"""Static figures and cover of the second course's lesson 14, Whisper: speech recognition as an encoder–decoder."""

import json
import math

import numpy as np
from huggingface_hub import hf_hub_download

from tools.figures import arch
from tools.figures import cover as cv
from tools.figures.lib import Figure

TINY = "openai/whisper-tiny"
EXAMPLE = [" He", " hoped", " there", " would", " be", " stew"]  # LibriSpeech test-clean 1089-134686-0000


def config(repo, name="config.json"):
    with open(hf_hub_download(repo, name)) as f:
        return json.load(f)


def mel_filters(n_mels=80, n_fft=400, sr=16000):
    """Slaney-style triangular mel filters, the bank Whisper's front end uses (same code as in the lesson)."""
    def hz_to_mel(f):
        f = np.asarray(f, float)
        return np.where(f >= 1000, 15 + np.log(np.maximum(f, 1e-9) / 1000) / (np.log(6.4) / 27), 3 * f / 200)

    def mel_to_hz(m):
        m = np.asarray(m, float)
        return np.where(m >= 15, 1000 * np.exp((np.log(6.4) / 27) * (m - 15)), 200 * m / 3)

    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2))
    ramps = pts[:, None] - freqs[None, :]
    lower, upper = -ramps[:-2] / np.diff(pts)[:-1, None], ramps[2:] / np.diff(pts)[1:, None]
    return freqs, np.maximum(0, np.minimum(lower, upper))


def wave(fig, x, y, w, h, n=120, seed=1):
    """A short stretch of speech-like waveform: bursts of a few tones under a slow envelope."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    env = 0.25 + 0.75 * np.abs(np.sin(2.7 * np.pi * t)) * (rng.random(n) * 0.3 + 0.7)
    s = env * np.sin(2 * np.pi * 9 * t) * np.cos(2 * np.pi * 2.3 * t)
    pts = [(x + w * t[i], y + h / 2 - s[i] * h / 2) for i in range(n)]
    fig.line(*pts, role="data", width=1.2)


def spectro(fig, x, y, cols=12, rows=7, cell=8, gap=1, seed=2):
    """A tiny spectrogram: tinted cells, lower rows more often strong, the way speech energy sits."""
    rng = np.random.default_rng(seed)
    for r in range(rows):
        for c in range(cols):
            strong = rng.random() < 0.55 - 0.06 * r + 0.25 * math.sin(c / 1.7) ** 2 * (r < 4)
            fig.rect(x + c * (cell + gap), y + (rows - 1 - r) * (cell + gap), cell, cell, role="enc", tint=not strong)


# ---------------------------------------------------------------- hero

HERO_ALT = ("Whisper in one picture. Thirty seconds of sound become a log-mel spectrogram of 80 rows by 3,000 frames. Two "
            "convolutions, the second with stride 2, turn it into 1,500 vectors, which a Transformer encoder reads. A "
            "decoder writes the transcript token by token, starting from special tokens that name the task and the "
            "language, and attends to the encoder's vectors through cross-attention")


def hero() -> Figure:
    c = config(TINY)
    p = config(TINY, "preprocessor_config.json")
    frames = p["nb_max_frames"]
    fig = Figure(760, 340, alt=HERO_ALT)
    fig.text(14, 28, "Whisper turns 30 seconds of sound into text with an encoder and a decoder", kind="title")
    fig.text(14, 50, f"whisper-tiny: {c['num_mel_bins']} mel bins × {frames:,} frames in, {c['max_source_positions']:,} "
                     f"vectors of {c['d_model']} out, a decoder that writes tokens", kind="note")
    wave(fig, 18, 86, 96, 44)
    fig.text(66, 148, "16 kHz sound", kind="small", anchor="middle")
    fig.arrow((120, 108), (140, 108))
    spectro(fig, 144, 76)
    fig.text(197, 148, f"log-mel, {c['num_mel_bins']}×{frames:,}", kind="small", anchor="middle")
    fig.arrow((256, 108), (276, 108))
    fig.box(280, 88, 116, 40, role="enc", label="2 convolutions")
    fig.text(338, 148, "the second has stride 2", kind="small", anchor="middle")
    fig.arrow((397, 108), (417, 108))
    fig.box(421, 88, 120, 40, role="enc", label=f"encoder ×{c['encoder_layers']}")
    fig.arrow((542, 108), (562, 108))
    for k in range(7):
        fig.rect(566 + k * 12, 98, 8, 20, role="enc", tint=k % 2 == 1)
    fig.text(606, 148, f"{c['max_source_positions']:,} vectors", kind="small", anchor="middle")
    fig.arrow((606, 166), (606, 200))
    fig.text(614, 188, "cross-attention", kind="small")
    fig.box(250, 204, 400, 40, role="dec", label=f"decoder ×{c['decoder_layers']}, one token at a time")
    fig.arrow((450, 245), (450, 266))
    tokens = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"] + EXAMPLE[:3] + ["…"]
    fig.row(tokens, 20, 272, role="dec", strong=lambda i, t: i < 4, size=10.5, h=26)
    fig.text(14, 326, "the first four tokens choose the task; the rest is the transcript", kind="note")
    return fig


# ---------------------------------------------------------------- front end

FRONT_ALT = ("How Whisper hears. The 16 kHz waveform is cut into overlapping windows of 400 samples (25 ms) every 160 "
             "samples (10 ms); each window becomes 201 frequency bins by a Fourier transform; 80 triangular mel filters, "
             "narrow at low frequencies and wide at high ones, sum the energy into 80 bands; the result is log-compressed, "
             "clamped to 8 orders of magnitude below its maximum and scaled to about -1 to 1")


def frontend() -> Figure:
    p = config(TINY, "preprocessor_config.json")
    n_fft, hop, sr = p["n_fft"], p["hop_length"], p["sampling_rate"]
    freqs, bank = mel_filters(p["feature_size"], n_fft, sr)
    fig = Figure(760, 330, alt=FRONT_ALT)
    fig.text(14, 28, "From a waveform to a log-mel spectrogram", kind="title")
    fig.text(14, 50, f"windows of {n_fft} samples ({1000 * n_fft / sr:.0f} ms) every {hop} ({1000 * hop / sr:.0f} ms); "
                     f"{n_fft // 2 + 1} frequencies; {p['feature_size']} mel bands; a log", kind="note")
    # 1. windows over the waveform
    wave(fig, 20, 96, 200, 50, n=200, seed=3)
    for k in range(4):
        x = 30 + k * 34
        fig.line((x, 156 + 8 * k), (x + 80, 156 + 8 * k), role="enc", width=2)
    fig.text(20, 214, f"{n_fft}-sample windows, hop {hop}", kind="small")
    fig.text(20, 230, "a Hann taper on each", kind="small")
    fig.arrow((228, 121), (250, 121))
    # 2. one window's spectrum
    fig.box(254, 84, 90, 76, role="plain")
    for k in range(18):
        h = 8 + 50 * math.exp(-k / 5) * (0.6 + 0.4 * math.sin(k * 1.9) ** 2)
        fig.rect(260 + k * 4.5, 154 - h, 3, h, role="data", tint=k % 3 == 1)
    fig.text(299, 178, f"|FFT|², {n_fft // 2 + 1} bins", kind="small", anchor="middle")
    fig.arrow((350, 121), (372, 121))
    # 3. the real mel filterbank
    x0, y0, w, h = 380, 82, 220, 80
    fig.line((x0, y0 + h), (x0 + w, y0 + h), role="muted", width=1)
    for m in range(0, bank.shape[0], 2):
        nz = np.nonzero(bank[m])[0]
        lo, hi = max(nz[0] - 1, 0), min(nz[-1] + 1, len(freqs) - 1)
        pts = [(x0 + w * freqs[i] / freqs[-1], y0 + h - h * bank[m, i] / bank[m].max()) for i in range(lo, hi + 1)]
        fig.line(*pts, role="enc", width=0.9)
    fig.text(x0, y0 + h + 18, "0 Hz", kind="small")
    fig.text(x0 + w, y0 + h + 18, f"{sr // 2:,} Hz", kind="small", anchor="end")
    fig.text(x0 + w / 2, y0 + h + 36, f"{bank.shape[0]} triangular filters (every second drawn)", kind="small",
             anchor="middle")
    fig.arrow((606, 121), (628, 121))
    # 4. log and scale
    fig.box(632, 96, 116, 50, role="att", label="log10")
    fig.text(690, 168, "clamp at max − 8,", kind="small", anchor="middle")
    fig.text(690, 184, "then (x + 4) / 4", kind="small", anchor="middle")
    fig.text(14, 286, f"30 s × {sr // hop} frames per second = {30 * sr // hop:,} frames; shorter clips are padded "
                      "with silence, longer ones cut", kind="note")
    fig.text(14, 308, "like the ear, the bank resolves low frequencies finely and high ones coarsely", kind="note")
    return fig


# ---------------------------------------------------------------- tokens: the task format

TOKENS_ALT = ("Whisper's task format. Every output starts with the start-of-transcript token, then a language token, then "
              "transcribe or translate, then either no-timestamps or timestamp tokens around each segment. The same "
              "decoder transcribes English, writes timestamps, or translates Russian speech into English text, depending "
              "only on these first tokens. Asked for the first token after start-of-transcript, it names the language")


def tokens() -> Figure:
    fig = Figure(760, 300, alt=TOKENS_ALT)
    fig.text(14, 28, "One decoder for several tasks, chosen by the first tokens", kind="title")
    fig.text(14, 50, "special tokens are ordinary vocabulary entries; the model learned what follows each of them",
             kind="note")
    rows = [
        ("transcribe", ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"] + EXAMPLE[:4]),
        ("with times", ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|0.00|>"] + EXAMPLE[:3] + ["<|1.84|>"]),
        ("translate", ["<|startoftranscript|>", "<|ru|>", "<|translate|>", "<|notimestamps|>", " Hello", ","]),
        ("find language", ["<|startoftranscript|>", "→ <|ru|> 0.97"]),
    ]
    y = 78
    for label, toks in rows:
        fig.text(14, y + 18, label, kind="small")
        fig.row(toks, 110, y, role="dec", strong=lambda i, t: t.startswith("<|") or t.startswith("→"), size=10.5, h=26)
        y += 50
    fig.text(14, 286, "timestamps come in steps of 0.02 s; the transcript between them is one segment", kind="note")
    return fig


# ---------------------------------------------------------------- long recordings

LONG_ALT = ("How Whisper transcribes a recording longer than 30 seconds. It decodes one 30-second window, moves the window "
            "to the last complete segment and decodes again, passing the previous text as a prompt. Each window is "
            "first decoded greedily at temperature 0; if the output repeats itself (its compression ratio is above "
            "2.4) or is unlikely (mean log-probability below -1), it is decoded again at a higher temperature, up to 1. "
            "A window whose no-speech probability is above 0.6 and whose text is unlikely is treated as silence")


def longform() -> Figure:
    fig = Figure(760, 330, alt=LONG_ALT)
    fig.text(14, 28, "Long recordings are read 30 seconds at a time, with a fallback for bad windows", kind="title")
    fig.text(14, 50, "the defaults of openai/whisper's transcribe(); the model itself only ever sees 30 seconds",
             kind="note")
    x0, y0 = 20, 84
    fig.line((x0, y0 + 16), (x0 + 720, y0 + 16), role="muted", width=1)
    for k, (a, b) in enumerate([(0, 300), (250, 550), (510, 720)]):
        fig.box(x0 + a, y0 + 4 + (k % 2) * 30, b - a, 24, role="data" if k != 1 else "plain",
                label=f"window {k + 1}: 30 s")
    fig.text(x0, y0 + 76, "each window starts at the end of the last complete segment of the one before",
             kind="small")
    y = 190
    fig.box(20, y, 150, 40, role="dec", label="decode at T = 0")
    fig.arrow((171, y + 20), (199, y + 20))
    fig.box(203, y, 230, 40, role="plain", label="repeats or unlikely?")
    fig.text(318, y + 56, "compression ratio > 2.4, or", kind="small", anchor="middle")
    fig.text(318, y + 72, "mean log-probability < −1", kind="small", anchor="middle")
    fig.arrow((434, y + 20), (462, y + 20))
    fig.box(466, y, 274, 40, role="att", label="retry at T = 0.2, 0.4 … 1.0")
    fig.text(14, 312, "a window with no-speech probability > 0.6 and an unlikely transcript is skipped as silence",
             kind="note")
    return fig


# ---------------------------------------------------------------- word timing through cross-attention

ALIGN_ALT = ("How Whisper finds when each word was said. A few cross-attention heads, the alignment heads, look at the "
             "part of the sound each text token comes from. Their attention, tokens by audio frames, forms a bright "
             "diagonal band; dynamic time warping finds the monotonic path through it, and the path gives each word a "
             "start and an end time")


def alignment() -> Figure:
    fig = Figure(760, 320, alt=ALIGN_ALT)
    fig.text(14, 28, "Word times from where the decoder looks in the sound", kind="title")
    fig.text(14, 50, "a few cross-attention heads track the audio; dynamic time warping turns their attention into "
                     "a path", kind="note")
    x0, y0, cw, ch = 150, 76, 14, 26
    words = EXAMPLE
    ncols = 30
    rng = np.random.default_rng(4)
    path_x = np.linspace(2, ncols - 3, len(words)).round().astype(int)
    for r, word in enumerate(words):
        fig.text(x0 - 10, y0 + r * ch + 18, word.strip(), kind="mono", anchor="end")
        for c in range(ncols):
            d = abs(c - path_x[r])
            strong = d <= 1 or (d == 2 and rng.random() < 0.3)
            fig.rect(x0 + c * cw, y0 + r * ch, cw - 2, ch - 2, role="att", tint=not strong)
    pts = [(x0 + path_x[r] * cw + cw / 2, y0 + r * ch + ch / 2) for r in range(len(words))]
    fig.line(*pts, role="dec", width=2.2)
    fig.text(x0, y0 + len(words) * ch + 22, "audio frames, 20 ms each →", kind="small")
    fig.text(x0 + ncols * cw + 16, y0 + 30, "the path's column", kind="small")
    fig.text(x0 + ncols * cw + 16, y0 + 46, "at each word gives", kind="small")
    fig.text(x0 + ncols * cw + 16, y0 + 62, "its start and end", kind="small")
    fig.text(14, 304, "openai/whisper stores the alignment heads per model; the lesson reads them from the "
                      "generation config", kind="note")
    return fig


# ---------------------------------------------------------------- passport

def passport() -> Figure:
    c = config(TINY)
    rows = [
        ("published", "September 2022, OpenAI; MIT licence (code and weights)"),
        ("type", f"encoder–decoder Transformer, {c['encoder_layers']} + {c['decoder_layers']} blocks of {c['d_model']}"),
        ("audio in", f"{c['num_mel_bins']}-band log-mel, 30 s, {c['max_source_positions']:,} encoder positions"),
        ("text out", f"byte-level BPE, {c['vocab_size']:,} tokens, up to {c['max_target_positions']} positions"),
        ("tasks", "transcription, translation into English, language ID, timestamps"),
        ("training", "680,000 hours of weakly supervised audio with transcripts from the web"),
        ("good at", "clean speech in many languages, robust to accents and noise"),
        ("not for", "streaming, silence without a voice detector, exact verbatim records"),
    ]
    alt = ("whisper-tiny's passport. Published September 2022 by OpenAI under the MIT licence. An encoder–decoder "
           f"Transformer with {c['encoder_layers']} encoder and {c['decoder_layers']} decoder blocks of width "
           f"{c['d_model']}. Audio in: a {c['num_mel_bins']}-band log-mel spectrogram of 30 seconds, "
           f"{c['max_source_positions']:,} encoder positions. Text out: byte-level BPE with {c['vocab_size']:,} tokens. "
           "Tasks: transcription, translation into English, language identification, timestamps. Trained on 680,000 "
           "hours of weakly supervised web audio. Good at clean speech in many languages; not for streaming, silence "
           "without a voice detector, or exact verbatim records")
    return arch.passport("whisper-tiny", rows, alt=alt)


# ---------------------------------------------------------------- cover

def _cover():
    c = config(TINY)
    out = [cv.label(cv.MOTIF_X0, 150, "THIRTY SECONDS OF SOUND AS A SPECTROGRAM")]
    rng = np.random.default_rng(7)
    cell, gap, cols, rows = 18, 5, 16, 6
    x0, y0 = cv.MOTIF_X0, 172
    for r in range(rows):
        for q in range(cols):
            hot = rng.random() < 0.18 + 0.5 * (math.sin(q / 2.3) ** 2) * (r >= 3)
            fill = cv.RED if hot and r >= 3 and q % 5 in (1, 2) else "none"
            out.append(f'<rect x="{x0 + q * (cell + gap)}" y="{y0 + r * (cell + gap)}" width="{cell}" height="{cell}" '
                       f'rx="3" fill="{fill}" stroke="{cv.LINE}" stroke-width="1.2"/>')
    ry = y0 + rows * (cell + gap) + 56
    out.append(cv.label(cv.MOTIF_X0, ry - 16, f"…WRITTEN DOWN BY A DECODER, {c['decoder_layers']} BLOCKS DEEP"))
    svg, spans, right = cv.chip_row(["<|en|>", "He", "hoped"], cv.MOTIF_X0, ry, size=18,
                                    kinds={0: "hot"})
    out.append(svg)
    cv.fits(right)
    return cv.frame("a14_whisper.ipynb", ["Speech", "recognition", "with Whisper"],
                    ["How an encoder–decoder", "turns speech into text"], "".join(out), footer=arch.COURSE)


FIGURES = {
    "hero": hero,
    "frontend": frontend,
    "tokens": tokens,
    "longform": longform,
    "alignment": alignment,
    "passport": passport,
    "family": lambda: arch.family(14),
}
COVERS = {"en": _cover}
