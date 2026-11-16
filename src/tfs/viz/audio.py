"""A spectrogram you can listen to: the picture, the sound, and a playhead that moves over the picture.

`audio_spectrogram(audio, sr, values)` draws a time-frequency image with matplotlib and returns an HTML block that
holds the PNG, the sound as WAV and a small script. In Jupyter and Colab the script moves a line over the
spectrogram while the sound plays and seeks when the picture is clicked; where scripts are blocked the picture and
the player still work. The blog exporter recognises the block (class tfs-audio-spec), stores the image and the
sound next to the post and keeps the same behaviour through the site's audio-spec.js.
"""

from __future__ import annotations

import base64
import html
import io
import wave

import matplotlib.pyplot as plt
import numpy as np

# the axes' horizontal extent, in figure fractions; the image is saved without cropping, so the playhead can map
# time to pixels exactly. Vertical margins are in inches, so a tall picture doesn't get wide empty bands.
LEFT, RIGHT = 0.07, 0.985
BOTTOM_IN, TOP_IN = 0.55, 0.4

SCRIPT = """
(function () {
  var box = document.currentScript.parentElement;
  var img = box.querySelector("img"), audio = box.querySelector("audio"), line = box.querySelector(".tfs-playhead");
  var x0 = parseFloat(box.dataset.x0), x1 = parseFloat(box.dataset.x1), dur = parseFloat(box.dataset.duration);
  function draw() {
    var f = Math.min(audio.currentTime / dur, 1);
    line.style.left = (100 * (x0 + (x1 - x0) * f)) + "%";
    line.style.opacity = audio.currentTime > 0 ? 1 : 0;
    if (!audio.paused) { requestAnimationFrame(draw); }
  }
  audio.addEventListener("play", draw);
  audio.addEventListener("seeked", draw);
  img.addEventListener("click", function (e) {
    var r = img.getBoundingClientRect(), f = ((e.clientX - r.left) / r.width - x0) / (x1 - x0);
    if (f >= 0 && f <= 1) { audio.currentTime = f * dur; audio.play(); }
  });
})();
"""
STYLE = ("position:relative;display:inline-block;max-width:100%;",
         "position:absolute;top:%.1f%%;height:%.1f%%;width:2px;margin-left:-1px;background:#e8433a;opacity:0;"
         "pointer-events:none;")


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """16-bit mono PCM WAV, peak-normalised only if the signal would clip."""
    x = np.asarray(audio, dtype=np.float64)
    peak = np.abs(x).max() if len(x) else 0.0
    if peak > 1:
        x = x / peak
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buffer.getvalue()


def audio_spectrogram(audio, sr: int, values, *, title: str | None = None, alt: str = "", ylabel: str = "mel bin",
                      labels=None, left: float = LEFT, cmap: str = "magma", vmin=None, vmax=None, figsize=(9, 2.6)):
    """An HTML block: the image of `values` (a [rows, frames] array over the clip's duration), the sound, a playhead.

    values  what to draw over time: a log-mel spectrogram, an attention map, anything whose columns span the clip
    labels  optional name for each row, e.g. the tokens of an attention map; then row 0 is drawn at the top
    left    where the axes start, in figure fractions; widen it for long row labels
    """
    from IPython.display import HTML

    duration = len(audio) / sr
    rows = values.shape[0]
    fig = plt.figure(figsize=figsize)
    bottom, top = BOTTOM_IN / figsize[1], 1 - TOP_IN / figsize[1]
    ax = fig.add_axes([left, bottom, RIGHT - left, top - bottom])
    ax.imshow(values, origin="upper" if labels is not None else "lower", aspect="auto", cmap=cmap,
              extent=(0, duration, 0, rows), vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlabel("time, s")
    ax.set_ylabel(ylabel)
    ax.grid(False)  # grid lines would cross the picture
    if labels is not None:
        ax.set_yticks([rows - i - 0.5 for i in range(rows)], [str(t) for t in labels], fontsize=7)
    if title:
        ax.set_title(title, loc="left")
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=150, bbox_inches=None)  # no cropping: the axes stay where LEFT/RIGHT say
    plt.close(fig)
    top, height = 100 * (1 - top), 100 * (top - bottom)
    block = (
        f'<div class="tfs-audio-spec" data-x0="{left}" data-x1="{RIGHT}" data-duration="{duration:.3f}">'
        f'<div class="tfs-spec" style="{STYLE[0]}">'  # the playhead is positioned over the image only
        f'<img src="data:image/png;base64,{base64.b64encode(png.getvalue()).decode()}" alt="{html.escape(alt)}" '
        f'style="max-width:100%;cursor:pointer;display:block">'
        f'<div class="tfs-playhead" style="{STYLE[1] % (top, height)}"></div></div>'
        f'<audio controls preload="metadata" style="width:100%" '
        f'src="data:audio/wav;base64,{base64.b64encode(wav_bytes(audio, sr)).decode()}"></audio>'
        f"<script>{SCRIPT}</script></div>"
    )
    return HTML(block)
