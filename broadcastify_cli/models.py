from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator

from .portable_diarization import (
    COMMUNITY_DIARIZATION_ENGINE,
    normalize_diarization_engine,
)


@dataclass(frozen=True)
class FeedSearchResult:
    feed_id: str
    name: str
    location: str = ""
    description: str = ""
    genre: str = ""
    listeners: int = 0
    status: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JobRequest:
    feed_id: str
    start_date: date
    end_date: date
    feed_name: str = ""
    output_dir: Path = Path("archives")
    combine: bool = False
    keep_originals: bool = True
    transcribe: bool = False
    diarize: bool = False
    model: str = "turbo"
    asr_engine: str = "auto"
    device: str = "auto"
    device_index: int = 0
    compute_type: str = "auto"
    asr_model_path: str | None = None
    diarization_engine: str = COMMUNITY_DIARIZATION_ENGINE
    diarization_device: str = "auto"
    download_jobs: int = 1
    batch_size: int = 8
    min_speakers: int | None = None
    max_speakers: int | None = None
    huggingface_token: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "JobRequest":
        request = cls(
            feed_id=str(value["feed_id"]).strip(),
            start_date=date.fromisoformat(str(value["start_date"])),
            end_date=date.fromisoformat(str(value["end_date"])),
            feed_name=str(value.get("feed_name") or "").strip()[:200],
            output_dir=Path(value.get("output_dir") or "archives"),
            combine=bool(value.get("combine", False)),
            keep_originals=bool(value.get("keep_originals", True)),
            transcribe=bool(value.get("transcribe", False)),
            diarize=bool(value.get("diarize", False)),
            model=str(value.get("model") or "turbo"),
            asr_engine=str(value.get("asr_engine") or "auto"),
            device=str(value.get("device") or "auto"),
            device_index=int(value.get("device_index", 0)),
            compute_type=str(value.get("compute_type") or "auto"),
            asr_model_path=(
                str(value["asr_model_path"]).strip()
                if value.get("asr_model_path")
                else None
            ),
            diarization_engine=normalize_diarization_engine(
                str(
                    value.get("diarization_engine")
                    or COMMUNITY_DIARIZATION_ENGINE
                )
            ),
            diarization_device=str(value.get("diarization_device") or "auto"),
            download_jobs=max(1, int(value.get("download_jobs", 1))),
            batch_size=max(1, int(value.get("batch_size", 8))),
            min_speakers=(
                int(value["min_speakers"])
                if value.get("min_speakers") not in {None, ""}
                else None
            ),
            max_speakers=(
                int(value["max_speakers"])
                if value.get("max_speakers") not in {None, ""}
                else None
            ),
            huggingface_token=(
                str(value["huggingface_token"]).strip()
                if value.get("huggingface_token")
                else None
            ),
        )
        request.validate()
        return request

    def validate(self) -> None:
        if not self.feed_id.isdigit():
            raise ValueError("Feed ID must contain only digits.")
        if self.start_date > self.end_date:
            raise ValueError("Start date must be on or before end date.")
        if self.end_date > date.today():
            raise ValueError("End date cannot be in the future.")
        if self.diarize and not self.transcribe:
            raise ValueError("Diarization requires transcription.")
        if self.diarize and not self.combine:
            raise ValueError(
                "Diarization requires daily combination so speaker labels and timestamps do not restart."
            )
        if self.asr_engine not in {
            "auto",
            "faster-whisper",
            "whisper.cpp",
            "openvino",
            "windows-ml",
            "qwen3-asr",
        }:
            raise ValueError(
                "ASR engine must be auto, faster-whisper, whisper.cpp, openvino, "
                "windows-ml, or qwen3-asr."
            )
        if self.device not in {
            "auto",
            "cpu",
            "cuda",
            "vulkan",
            "metal",
            "openvino-auto",
            "openvino-cpu",
            "openvino-gpu",
            "openvino-npu",
            "gpu",
            "npu",
            "windows-ml",
            "directml",
        }:
            raise ValueError("The selected transcription device is not supported.")
        if self.diarization_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Diarization device must be auto, cpu, or cuda.")
        normalize_diarization_engine(self.diarization_engine)
        if (
            self.diarization_engine == "sherpa-onnx"
            and self.diarization_device == "cuda"
        ):
            raise ValueError(
                "The sherpa-onnx speaker preview currently supports CPU or Automatic."
            )
        if self.device_index < 0:
            raise ValueError("Device index cannot be negative.")
        if self.min_speakers is not None and self.min_speakers < 1:
            raise ValueError("Minimum speakers must be at least 1.")
        if self.max_speakers is not None and self.max_speakers < 1:
            raise ValueError("Maximum speakers must be at least 1.")
        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers > self.max_speakers
        ):
            raise ValueError("Minimum speakers cannot exceed maximum speakers.")

    def dates(self) -> Iterator[date]:
        current = self.start_date
        while current <= self.end_date:
            yield current
            current += timedelta(days=1)
