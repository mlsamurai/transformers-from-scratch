import io
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import export_post  # noqa: E402

from tfs.viz import audio_spectrogram, wav_bytes  # noqa: E402

pytest.importorskip("IPython")
SR = 16000


@pytest.fixture
def clip():
    t = np.arange(2 * SR) / SR
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_wav_bytes_is_16_bit_mono_and_only_rescales_when_it_would_clip(clip):
    with wave.open(io.BytesIO(wav_bytes(clip, SR))) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, SR, len(clip))
        pcm = np.frombuffer(w.readframes(len(clip)), dtype="<i2")
    assert abs(pcm.max() / 32767 - 0.3) < 1e-3
    with wave.open(io.BytesIO(wav_bytes(5 * clip, SR))) as w:
        assert np.abs(np.frombuffer(w.readframes(len(clip)), dtype="<i2")).max() == 32767


def test_audio_spectrogram_carries_picture_sound_and_the_time_axis(clip):
    block = audio_spectrogram(clip, SR, np.random.rand(10, 200), alt="a tone").data
    assert 'class="tfs-audio-spec"' in block and 'data-duration="2.000"' in block
    assert 'src="data:image/png;base64,' in block and 'src="data:audio/wav;base64,' in block
    assert 'alt="a tone"' in block and "<script>" in block
    labelled = audio_spectrogram(clip, SR, np.random.rand(3, 200), labels=["a", "b", "c"], left=0.15).data
    assert 'data-x0="0.15"' in labelled


def test_audio_block_moves_picture_and_sound_next_to_the_post(clip, tmp_path, monkeypatch):
    monkeypatch.setattr(export_post.shutil, "which", lambda name: None)  # no ffmpeg: the WAV stays
    block = audio_spectrogram(clip, SR, np.random.rand(10, 200), alt="a tone").data
    out = export_post.audio_block(block, tmp_path, "/assets/posts/x", [0])
    assert out.startswith('<figure class="tfs-audio-spec" markdown="0"') and out.endswith("</figure>")
    assert "<script>" not in out and "base64" not in out
    assert 'src="/assets/posts/x/output-01.png"' in out and 'src="/assets/posts/x/output-01.wav"' in out
    assert 'preload="none"' in out
    assert (tmp_path / "output-01.png").stat().st_size > 0 and (tmp_path / "output-01.wav").stat().st_size > 0


def test_audio_block_turns_a_plain_player_into_a_site_player(clip, tmp_path, monkeypatch):
    from IPython.display import Audio

    monkeypatch.setattr(export_post.shutil, "which", lambda name: None)
    out = export_post.audio_block(Audio(clip, rate=SR)._repr_html_(), tmp_path, "/a", [4])
    assert out == '<audio class="tfs-audio" controls preload="none" src="/a/output-05.wav"></audio>'
