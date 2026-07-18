import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from types import MethodType
from types import SimpleNamespace

import pytest

from broadcastify_cli.asr import AsrResult, AsrSegment
from broadcastify_cli.portable_diarization import PortableSpeakerTurn
from broadcastify_cli.qwen_asr import SherpaQwen3Asr
from broadcastify_cli.transcription import (
    LocalTranscriber,
    SpeakerTurn,
    TranscriptionQualityError,
    TranscriptWord,
    format_timestamp,
    group_words,
    speaker_for_interval,
    transcript_quality_report,
)


def _fake_diarization_modules(monkeypatch, from_pretrained) -> None:
    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    torch_module.device = lambda value: value  # type: ignore[attr-defined]
    audio_module = ModuleType("pyannote.audio")
    audio_module.Pipeline = SimpleNamespace(from_pretrained=from_pretrained)  # type: ignore[attr-defined]
    pyannote_module = ModuleType("pyannote")
    pyannote_module.audio = audio_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio_module)


def test_whisper_cpp_auto_device_prefers_metal_on_macos(monkeypatch) -> None:
    monkeypatch.setattr("broadcastify_cli.transcription.sys.platform", "darwin")
    monkeypatch.setattr(
        "broadcastify_cli.transcription.find_whisper_cpp", lambda: "/opt/whisper-cli"
    )
    monkeypatch.setattr(
        "broadcastify_cli.transcription.whisper_cpp_backends",
        lambda _path: ["cpu", "metal"],
    )

    transcriber = LocalTranscriber(
        asr_engine="whisper.cpp", device="auto", load_asr=False
    )

    assert transcriber.device == "metal"


def test_diarization_reuses_cached_model_without_huggingface_token(monkeypatch) -> None:
    loaded: dict[str, object] = {}

    class FakePipeline:
        def to(self, target: object) -> None:
            loaded["target"] = target

    pipeline = FakePipeline()

    def from_pretrained(model: str, *, token: str | None) -> FakePipeline:
        loaded.update(model=model, token=token)
        return pipeline

    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    _fake_diarization_modules(monkeypatch, from_pretrained)

    transcriber = LocalTranscriber(
        asr_engine="whisper.cpp",
        device="cpu",
        diarization_device="cpu",
        diarize=True,
        load_asr=False,
    )

    assert transcriber._diarization_pipeline is pipeline
    assert loaded == {
        "model": LocalTranscriber.DIARIZATION_MODEL,
        "token": None,
        "target": "cpu",
    }


def test_diarization_without_token_explains_cache_miss(monkeypatch) -> None:
    def from_pretrained(_model: str, *, token: str | None) -> object:
        assert token is None
        raise OSError("cache miss")

    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    _fake_diarization_modules(monkeypatch, from_pretrained)

    with pytest.raises(RuntimeError, match="no usable cached speaker-label model"):
        LocalTranscriber(
            asr_engine="whisper.cpp",
            device="cpu",
            diarization_device="cpu",
            diarize=True,
            load_asr=False,
        )


def test_speaker_uses_largest_overlap_not_first_overlap() -> None:
    turns = [
        SpeakerTurn(0.0, 1.1, "SPEAKER_00"),
        SpeakerTurn(1.1, 5.0, "SPEAKER_01"),
    ]
    assert speaker_for_interval(0.9, 2.5, turns) == "SPEAKER_01"


def test_zero_duration_word_uses_containing_or_nearest_turn() -> None:
    turns = [SpeakerTurn(1.0, 2.0, "SPEAKER_00")]
    assert speaker_for_interval(1.5, 1.5, turns) == "SPEAKER_00"
    assert speaker_for_interval(2.25, 2.25, turns) == "SPEAKER_00"
    assert speaker_for_interval(3.0, 3.0, turns) is None


def test_words_group_only_while_speaker_is_unchanged() -> None:
    words = [
        TranscriptWord(0.0, 0.5, " Dispatch", "SPEAKER_00"),
        TranscriptWord(0.5, 1.0, " calling", "SPEAKER_00"),
        TranscriptWord(1.0, 1.5, " unit", "SPEAKER_01"),
    ]
    grouped = group_words(words)
    assert [segment.speaker for segment in grouped] == ["SPEAKER_00", "SPEAKER_01"]
    assert grouped[0].text == "Dispatch calling"
    assert grouped[1].text == "unit"


def test_timestamp_includes_milliseconds() -> None:
    assert format_timestamp(3661.234) == "01:01:01.234"


def test_compatible_transcript_is_a_cache_hit(tmp_path: Path) -> None:
    audio = tmp_path / "combined.mp3"
    audio.write_bytes(b"audio")
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir()
    json_path = transcript_dir / "combined.json"
    txt_path = transcript_dir / "combined.txt"
    json_path.write_text(
        json.dumps({"model": "turbo", "segments": [], "diarization_requested": False}),
        encoding="utf-8",
    )
    txt_path.write_text("", encoding="utf-8")

    transcriber = object.__new__(LocalTranscriber)
    transcriber.model_name = "turbo"
    transcriber.diarize = False

    assert transcriber._existing_transcript_is_current(audio, json_path, txt_path)


def test_existing_transcript_respects_diarization_quality_direction(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "combined.mp3"
    audio.write_bytes(b"audio")
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir()
    json_path = transcript_dir / "combined.json"
    txt_path = transcript_dir / "combined.txt"
    txt_path.write_text("", encoding="utf-8")
    base_payload = {
        "model": "turbo",
        "asr_engine": "faster-whisper",
        "segments": [],
        "diarization_requested": True,
        "diarization_completed": True,
        "diarization_model": (
            "pyannote-segmentation-3.0-int8+nemo-titanet-small"
        ),
        "diarization_engine": "sherpa-onnx",
    }
    json_path.write_text(json.dumps(base_payload), encoding="utf-8")
    transcriber = object.__new__(LocalTranscriber)
    transcriber.model_name = "turbo"
    transcriber.asr_engine = "faster-whisper"
    transcriber.diarize = True
    transcriber.diarization_engine = "community-1"

    assert not transcriber._existing_transcript_is_current(
        audio, json_path, txt_path
    )

    base_payload.update(
        diarization_engine="community-1",
        diarization_model="pyannote/speaker-diarization-community-1",
    )
    json_path.write_text(json.dumps(base_payload), encoding="utf-8")
    transcriber.diarization_engine = "sherpa-onnx"

    assert transcriber._existing_transcript_is_current(
        audio, json_path, txt_path
    )


def test_external_asr_records_actual_fallback_backend(tmp_path: Path) -> None:
    audio = tmp_path / "radio.wav"
    audio.write_bytes(b"audio")

    class FakeExternalAsr:
        @staticmethod
        def transcribe(_path: Path, progress=None) -> AsrResult:
            return AsrResult(
                text="unit responding",
                duration=1.0,
                segments=[AsrSegment(0.0, 1.0, "unit responding")],
                engine="openvino",
                backend="OpenVINO CPU (fallback from GPU)",
                metadata={
                    "model": "tiny",
                    "fallback_reason": "GPU execution failed",
                    "fallback_stage": "generation",
                },
            )

    transcriber = object.__new__(LocalTranscriber)
    transcriber._asr = None
    transcriber._external_asr = FakeExternalAsr()
    transcriber.model_name = "tiny.en"
    transcriber.asr_engine = "openvino"
    transcriber.backend_description = "OpenVINO GPU"
    transcriber.device = "openvino-gpu"
    transcriber.compute_type = "int8"
    transcriber.diarize = False
    transcriber.diarization_device = "cpu"

    transcript_path = transcriber.transcribe_file(audio)
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))

    assert payload["asr_backend"] == "OpenVINO CPU (fallback from GPU)"
    assert payload["model"] == "tiny"
    assert payload["requested_model"] == "tiny.en"
    assert payload["asr_metadata"]["fallback_stage"] == "generation"
    assert transcriber._existing_transcript_is_current(
        audio,
        transcript_path,
        transcript_path.with_suffix(".txt"),
    )


def test_transcript_quality_rejects_repetition_collapse() -> None:
    report = transcript_quality_report(
        ["15. I'll show you enough for that."] * 1_881
        + [f"variation {index}" for index in range(28)]
    )

    assert report["status"] == "rejected"
    assert report["segment_count"] == 1_909
    assert report["dominant_segment_ratio"] > 0.98


def test_repetition_collapse_is_not_saved_or_marked_current(tmp_path: Path) -> None:
    audio = tmp_path / "radio.wav"
    audio.write_bytes(b"audio")

    class RepeatingExternalAsr:
        @staticmethod
        def transcribe(_path: Path, progress=None) -> AsrResult:
            return AsrResult(
                text=" ".join(["same hallucinated sentence"] * 100),
                duration=3_600.0,
                segments=[
                    AsrSegment(
                        float(index * 30),
                        float(index * 30 + 5),
                        "same hallucinated sentence",
                    )
                    for index in range(100)
                ],
                engine="whisper.cpp",
                backend="whisper.cpp Vulkan",
                metadata={"model": "base"},
            )

    transcriber = object.__new__(LocalTranscriber)
    transcriber._asr = None
    transcriber._external_asr = RepeatingExternalAsr()
    transcriber.model_name = "base.en"
    transcriber.asr_engine = "whisper.cpp"
    transcriber.backend_description = "whisper.cpp Vulkan"
    transcriber.device = "vulkan"
    transcriber.compute_type = "ggml quantized"
    transcriber.diarize = False
    transcriber.diarization_device = "none"

    with pytest.raises(TranscriptionQualityError, match="quality check rejected"):
        transcriber.transcribe_file(audio)

    transcript_dir = tmp_path / "transcripts"
    assert not (transcript_dir / "radio.json").exists()
    assert not (transcript_dir / "radio.txt").exists()


def test_rendered_transcript_hash_detects_interrupted_file_pair(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "radio.wav"
    audio.write_bytes(b"audio")

    class FakeExternalAsr:
        @staticmethod
        def transcribe(_path: Path, progress=None) -> AsrResult:
            return AsrResult(
                text="unit responding",
                duration=1.0,
                segments=[AsrSegment(0.0, 1.0, "unit responding")],
                engine="openvino",
                backend="OpenVINO CPU",
                metadata={"model": "tiny"},
            )

    transcriber = object.__new__(LocalTranscriber)
    transcriber._asr = None
    transcriber._external_asr = FakeExternalAsr()
    transcriber.model_name = "tiny"
    transcriber.asr_engine = "openvino"
    transcriber.backend_description = "OpenVINO CPU"
    transcriber.device = "openvino-cpu"
    transcriber.compute_type = "int8"
    transcriber.diarize = False
    transcriber.diarization_device = "none"

    transcript_path = transcriber.transcribe_file(audio)
    text_path = transcript_path.with_suffix(".txt")
    text_path.write_text("partial replacement", encoding="utf-8")

    assert not transcriber._existing_transcript_is_current(
        audio,
        transcript_path,
        text_path,
    )


def test_portable_speakers_do_not_replace_external_asr_timestamp_identity(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "radio.wav"
    audio.write_bytes(b"audio")

    class FakeExternalAsr:
        @staticmethod
        def transcribe(_path: Path, progress=None) -> AsrResult:
            return AsrResult(
                text="unit responding",
                duration=1.0,
                segments=[AsrSegment(0.0, 1.0, "unit responding")],
                engine="whisper.cpp",
                backend="whisper.cpp Vulkan",
                metadata={
                    "model": "turbo",
                    "timestamp_source": "whisper.cpp-token-timestamps",
                },
            )

    transcriber = object.__new__(LocalTranscriber)
    transcriber._asr = None
    transcriber._external_asr = FakeExternalAsr()
    transcriber._diarize = lambda _path, progress=None: [
        SpeakerTurn(0.0, 1.0, "SPEAKER_00")
    ]
    transcriber.model_name = "turbo"
    transcriber.asr_engine = "whisper.cpp"
    transcriber.backend_description = "whisper.cpp Vulkan"
    transcriber.device = "vulkan"
    transcriber.compute_type = "q5_1"
    transcriber.diarize = True
    transcriber.diarization_engine = "sherpa-onnx"
    transcriber.diarization_model = (
        "pyannote-segmentation-3.0-int8+nemo-titanet-small"
    )
    transcriber.diarization_quality = "preview"
    transcriber.diarization_device = "cpu"

    transcript_path = transcriber.transcribe_file(audio)
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))

    assert (
        payload["asr_metadata"]["timestamp_source"]
        == "whisper.cpp-token-timestamps"
    )
    assert "speaker_label_quality" not in payload["asr_metadata"]


def test_qwen_reuses_diarization_turns_as_timestamped_asr_regions(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "radio.wav"
    audio.write_bytes(b"audio")
    captured: dict[str, object] = {}
    turns = [SpeakerTurn(10.0, 12.0, "SPEAKER_01")]
    qwen = object.__new__(SherpaQwen3Asr)
    qwen.backend = "sherpa-onnx CPU / Qwen3-ASR test"

    def fake_transcribe(
        _self: SherpaQwen3Asr,
        _path: Path,
        progress=None,
        *,
        segment_hints=None,
    ) -> AsrResult:
        captured["hints"] = segment_hints
        return AsrResult(
            text="stolen squad car",
            duration=20.0,
            segments=[AsrSegment(10.0, 12.0, "stolen squad car")],
            engine="qwen3-asr",
            backend=qwen.backend,
            metadata={
                "model": "qwen3-asr-0.6b-int8",
                "timestamp_source": "pyannote-exclusive-speaker-turns",
            },
        )

    qwen.transcribe = MethodType(fake_transcribe, qwen)
    transcriber = object.__new__(LocalTranscriber)
    transcriber._asr = None
    transcriber._external_asr = qwen
    transcriber._diarize = lambda _path, progress=None: turns
    transcriber.model_name = "qwen3-asr-0.6b-int8"
    transcriber.asr_engine = "qwen3-asr"
    transcriber.backend_description = qwen.backend
    transcriber.device = "cpu"
    transcriber.compute_type = "int8"
    transcriber.diarize = True
    transcriber.diarization_device = "cpu"

    transcript_path = transcriber.transcribe_file(audio)
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))

    assert captured["hints"] == [(10.0, 12.0)]
    assert payload["segments"] == [
        {
            "start": 10.0,
            "end": 12.0,
            "text": "stolen squad car",
            "speaker": "SPEAKER_01",
        }
    ]
    assert payload["words"] == []
    assert payload["speaker_turns"][0]["speaker"] == "SPEAKER_01"
    assert payload["asr_metadata"]["timestamp_source"].startswith("pyannote")


def test_diarization_turn_cache_is_parameter_and_audio_specific(tmp_path: Path) -> None:
    audio = tmp_path / "combined.mp3"
    audio.write_bytes(b"audio")
    transcriber = object.__new__(LocalTranscriber)
    transcriber.min_speakers = 2
    transcriber.max_speakers = 8
    transcriber.diarization_device = "cpu"
    turns = [SpeakerTurn(1.0, 2.5, "SPEAKER_00")]

    transcriber._save_diarization_cache(audio, turns)
    cache = json.loads(
        transcriber._diarization_cache_path(audio).read_text(encoding="utf-8")
    )
    assert transcriber._load_diarization_cache(audio) == turns
    assert cache["device"] == "cpu"

    transcriber.max_speakers = 9
    assert transcriber._load_diarization_cache(audio) is None


def test_portable_and_community_diarization_caches_are_isolated(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "combined.mp3"
    audio.write_bytes(b"audio")
    community = object.__new__(LocalTranscriber)
    community.diarization_engine = "community-1"
    community.diarization_model = LocalTranscriber.DIARIZATION_MODEL
    community.diarization_quality = "accuracy-default"
    community.diarization_device = "cpu"
    community.min_speakers = None
    community.max_speakers = None
    community._diarization_details = {}

    portable = object.__new__(LocalTranscriber)
    portable.diarization_engine = "sherpa-onnx"
    portable.diarization_model = (
        "pyannote-segmentation-3.0-int8+nemo-titanet-small"
    )
    portable.diarization_quality = "preview"
    portable.diarization_device = "cpu"
    portable.min_speakers = None
    portable.max_speakers = None
    portable._diarization_details = {"cluster_threshold": 0.95}

    community._save_diarization_cache(
        audio, [SpeakerTurn(1.0, 2.0, "SPEAKER_00")]
    )
    portable._save_diarization_cache(
        audio, [SpeakerTurn(1.0, 2.0, "SPEAKER_01")]
    )

    assert community._diarization_cache_path(audio).name == "combined.diarization.json"
    assert (
        portable._diarization_cache_path(audio).name
        == "combined.diarization.sherpa-onnx.json"
    )
    assert community._load_diarization_cache(audio) == [
        SpeakerTurn(1.0, 2.0, "SPEAKER_00")
    ]
    assert portable._load_diarization_cache(audio) == [
        SpeakerTurn(1.0, 2.0, "SPEAKER_01")
    ]


def test_portable_diarization_uses_archive_specific_chunk_checkpoint(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "combined.mp3"
    audio.write_bytes(b"audio")
    captured: dict[str, object] = {}

    class FakePortable:
        metadata = {"engine": "sherpa-onnx"}

        def process(
            self,
            path: Path,
            progress=None,
            checkpoint_path: Path | None = None,
            cleanup_checkpoint_on_success: bool = True,
        ) -> list[PortableSpeakerTurn]:
            captured["path"] = path
            captured["checkpoint_path"] = checkpoint_path
            captured["cleanup_checkpoint_on_success"] = (
                cleanup_checkpoint_on_success
            )
            assert checkpoint_path is not None
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text("completed chunks", encoding="utf-8")
            return [PortableSpeakerTurn(1.0, 2.0, "SPEAKER_00")]

    transcriber = object.__new__(LocalTranscriber)
    transcriber._portable_diarizer = FakePortable()
    transcriber._diarization_pipeline = None
    transcriber.diarization_engine = "sherpa-onnx"
    transcriber.diarization_model = (
        "pyannote-segmentation-3.0-int8+nemo-titanet-small"
    )
    transcriber.diarization_quality = "preview"
    transcriber.diarization_device = "cpu"
    transcriber.min_speakers = None
    transcriber.max_speakers = None

    turns = transcriber._diarize(audio)

    expected = (
        tmp_path
        / "transcripts"
        / "combined.diarization.sherpa-onnx.chunks.json"
    )
    assert turns == [SpeakerTurn(1.0, 2.0, "SPEAKER_00")]
    assert captured == {
        "path": audio,
        "checkpoint_path": expected,
        "cleanup_checkpoint_on_success": False,
    }
    assert transcriber._diarization_cache_path(audio).is_file()
    assert not expected.exists()


def test_portable_checkpoint_survives_failed_final_cache_write(
    monkeypatch, tmp_path: Path
) -> None:
    audio = tmp_path / "combined.mp3"
    audio.write_bytes(b"audio")
    checkpoint = (
        tmp_path
        / "transcripts"
        / "combined.diarization.sherpa-onnx.chunks.json"
    )

    class FakePortable:
        metadata = {"engine": "sherpa-onnx"}

        def process(
            self,
            path: Path,
            progress=None,
            checkpoint_path: Path | None = None,
            cleanup_checkpoint_on_success: bool = True,
        ) -> list[PortableSpeakerTurn]:
            assert path == audio
            assert checkpoint_path == checkpoint
            assert cleanup_checkpoint_on_success is False
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text("all chunks complete", encoding="utf-8")
            return [PortableSpeakerTurn(1.0, 2.0, "SPEAKER_00")]

    transcriber = object.__new__(LocalTranscriber)
    transcriber._portable_diarizer = FakePortable()
    transcriber._diarization_pipeline = None
    transcriber.diarization_engine = "sherpa-onnx"
    transcriber.diarization_model = (
        "pyannote-segmentation-3.0-int8+nemo-titanet-small"
    )
    transcriber.diarization_quality = "preview"
    transcriber.diarization_device = "cpu"
    transcriber.min_speakers = None
    transcriber.max_speakers = None
    monkeypatch.setattr(
        transcriber,
        "_save_diarization_cache",
        lambda *_args: (_ for _ in ()).throw(OSError("disk unavailable")),
    )

    with pytest.raises(OSError, match="disk unavailable"):
        transcriber._diarize(audio)

    assert checkpoint.read_text(encoding="utf-8") == "all chunks complete"


@pytest.mark.parametrize(
    ("pipeline_batch_size", "expected_batch_size"),
    [(1, 8), (32, 32)],
)
def test_diarization_reports_inner_pipeline_progress_without_lowering_model_batch(
    monkeypatch,
    tmp_path: Path,
    pipeline_batch_size: int,
    expected_batch_size: int,
) -> None:
    audio = tmp_path / "combined.mp3"
    audio.write_bytes(b"audio")
    prepared = tmp_path / "combined.pyannote.flac"
    prepared.write_bytes(b"prepared audio")
    messages: list[str] = []

    class FakePipeline:
        embedding_batch_size = pipeline_batch_size

        def __call__(self, audio: dict[str, object], *, hook=None, **_kwargs: object):
            assert audio["sample_rate"] == 16_000
            assert "waveform" in audio
            assert hook is not None
            assert self.embedding_batch_size == expected_batch_size
            hook("segmentation", None, file={"uri": "test"}, total=4, completed=1)
            hook("segmentation", None, file={"uri": "test"}, total=4, completed=4)
            hook("embeddings", None)
            return [(SimpleNamespace(start=1.0, end=2.0), "SPEAKER_00")]

    transcriber = object.__new__(LocalTranscriber)
    transcriber._diarization_pipeline = FakePipeline()
    transcriber.batch_size = 8
    transcriber.min_speakers = None
    transcriber.max_speakers = None
    transcriber.diarization_device = "cpu"
    monkeypatch.setattr(
        transcriber, "_prepare_diarization_input", lambda _path: (prepared, True)
    )
    monkeypatch.setattr(
        "broadcastify_cli.transcription.decoded_diarization_audio",
        lambda _path: nullcontext({"waveform": object(), "sample_rate": 16_000}),
    )

    turns = transcriber._diarize(audio, progress=messages.append)

    assert turns == [SpeakerTurn(1.0, 2.0, "SPEAKER_00")]
    assert "Diarization segmentation: 25% (1/4)" in messages
    assert "Diarization segmentation: 100% (4/4)" in messages
    assert "Diarization embeddings" in messages
    assert not prepared.exists()


def test_diarization_keeps_prepared_input_after_pipeline_failure(
    monkeypatch, tmp_path: Path
) -> None:
    audio = tmp_path / "combined.mp3"
    audio.write_bytes(b"audio")
    prepared = tmp_path / "combined.pyannote.flac"
    prepared.write_bytes(b"prepared audio")

    class FailingPipeline:
        embedding_batch_size = 32

        def __call__(self, _audio: dict[str, object], **_kwargs: object):
            raise RuntimeError("diarization failed")

    transcriber = object.__new__(LocalTranscriber)
    transcriber._diarization_pipeline = FailingPipeline()
    transcriber.batch_size = 8
    transcriber.min_speakers = None
    transcriber.max_speakers = None
    monkeypatch.setattr(
        transcriber, "_prepare_diarization_input", lambda _path: (prepared, True)
    )
    monkeypatch.setattr(
        "broadcastify_cli.transcription.decoded_diarization_audio",
        lambda _path: nullcontext({"waveform": object(), "sample_rate": 16_000}),
    )

    with pytest.raises(RuntimeError, match="diarization failed"):
        transcriber._diarize(audio)

    assert prepared.exists()


def test_diarization_reuses_completed_lossless_retry_input(
    monkeypatch, tmp_path: Path
) -> None:
    audio = tmp_path / "combined.mp3"
    audio.write_bytes(b"audio")
    prepared = (
        tmp_path
        / "transcripts"
        / ".cache"
        / "combined.pyannote.flac"
    )
    prepared.parent.mkdir(parents=True)
    prepared.write_bytes(b"lossless audio")
    monkeypatch.setattr(
        "broadcastify_cli.transcription.find_ffmpeg",
        lambda: (_ for _ in ()).throw(AssertionError("FFmpeg should not run")),
    )
    transcriber = object.__new__(LocalTranscriber)

    result, temporary = transcriber._prepare_diarization_input(audio)

    assert result == prepared
    assert temporary is True
