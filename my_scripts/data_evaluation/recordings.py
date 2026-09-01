"""NUS recording discovery shared by annotation and evaluation tools."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .rgbd_geometry import DepthZipReader, VideoMetadata, video_metadata

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass(frozen=True)
class NUSRecording:
    data_root: Path
    subject: str
    session_id: str
    trial: str
    video_path: Path
    depth_path: Path | None
    calibration_path: Path | None
    mmpose_path: Path | None
    video: VideoMetadata
    depth: dict | None

    @property
    def subject_path(self) -> Path:
        return self.data_root / self.subject

    @property
    def identifier(self) -> str:
        return f"{self.subject}/{self.session_id}/{self.trial}"


def find_recording(
    data_root: str | Path,
    subject: str,
    session_id: str,
    trial: str,
) -> NUSRecording:
    data_root = Path(data_root).expanduser().resolve()
    subject_path = _direct_child(data_root, subject)
    if not subject_path.is_dir():
        raise FileNotFoundError(f"subject not found: {subject}")
    session_id = _component(session_id, "session")
    trial = _component(trial, "trial")
    video_dir = subject_path / "videos" / session_id
    videos = sorted(
        path for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ) if video_dir.is_dir() else []
    video_path = next((path for path in videos if path.stem == trial), None)
    if video_path is None:
        raise FileNotFoundError(
            f"recording not found: {subject}/{session_id}/{trial}"
        )
    depth_path = subject_path / "depth" / session_id / "depth.zip"
    if not depth_path.is_file():
        depth_path = None
    calibration_path = subject_path / "camera_parameters" / session_id / "calibration.json"
    if not calibration_path.is_file():
        calibration_path = None
    mmpose_path = _find_mmpose(subject_path, session_id, trial)
    depth = _depth_metadata(depth_path)
    return NUSRecording(
        data_root=data_root,
        subject=subject_path.name,
        session_id=session_id,
        trial=trial,
        video_path=video_path,
        depth_path=depth_path,
        calibration_path=calibration_path,
        mmpose_path=mmpose_path,
        video=video_metadata(str(video_path)),
        depth=depth,
    )


def discover_recordings(data_root: str | Path) -> list[NUSRecording]:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        return []
    found = []
    for subject_path in sorted(path for path in root.iterdir() if path.is_dir()):
        video_root = subject_path / "videos"
        if not video_root.is_dir():
            continue
        for session_path in sorted(path for path in video_root.iterdir() if path.is_dir()):
            for video_path in sorted(
                path for path in session_path.iterdir()
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            ):
                try:
                    found.append(
                        find_recording(root, subject_path.name, session_path.name, video_path.stem)
                    )
                except (FileNotFoundError, OSError, ValueError):
                    continue
    return found


def annotation_path(
    recording: NUSRecording,
    username: str,
) -> Path:
    username = _component(username, "username")
    return (
        recording.subject_path
        / "annotations"
        / username
        / recording.session_id
        / f"{recording.trial}.json"
    )


def task_annotation_path(
    recording: NUSRecording,
    username: str,
) -> Path:
    username = _component(username, "username")
    return (
        recording.subject_path
        / "annotations"
        / username
        / "task_reviews"
        / recording.session_id
        / f"{recording.trial}.json"
    )


def _find_mmpose(subject_path: Path, session_id: str, trial: str) -> Path | None:
    root = subject_path / "mmpose"
    if not root.is_dir():
        return None
    matches = sorted(root.rglob(f"{trial}.json"))
    session_matches = [path for path in matches if session_id in path.parts]
    return (session_matches or matches)[0] if (session_matches or matches) else None


def _depth_metadata(path: Path | None) -> dict | None:
    if path is None:
        return None
    reader = DepthZipReader(str(path))
    try:
        return {
            "width": reader.width,
            "height": reader.height,
            "frame_count": len(reader.entries),
            "fx": reader.fx,
            "fy": reader.fy,
            "cx": reader.cx,
            "cy": reader.cy,
            "orientation": reader.orientation,
        }
    finally:
        reader.close()


def _component(value: str, field: str) -> str:
    value = str(value or "").strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"invalid {field}")
    return value


def _direct_child(root: Path, name: str) -> Path:
    name = _component(name, "subject")
    path = (root / name).resolve()
    if path.parent != root:
        raise ValueError("invalid subject")
    return path
