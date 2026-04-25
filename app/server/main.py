"""
TelePT – SAM3DBody Video Processing Server
===========================================

FastAPI server that accepts a video upload, processes it to produce
per-frame 3D body meshes, and returns a ZIP of OBJ files + meta.json.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from config import HOST, MAX_UPLOAD_SIZE_MB, PORT, TEMP_DIR, USE_SAM3D
from mesh_gen import generate_stub_meshes, generate_sam3d_meshes, generate_sam3d_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sam3d_server")

# ---------------------------------------------------------------------------
# In-memory job store  {job_id: {"status": ..., "frames_done": int, "total_frames": int, ...}}
# ---------------------------------------------------------------------------
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(TEMP_DIR, exist_ok=True)
    logger.info("Server started – temp dir: %s  |  USE_SAM3D=%s", TEMP_DIR, USE_SAM3D)
    yield
    # Cleanup on shutdown
    if os.path.isdir(TEMP_DIR):
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    logger.info("Server shut down, temp dir cleaned.")


app = FastAPI(
    title="TelePT SAM3DBody Server",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow the Flutter app on any origin (local network).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "use_sam3d": USE_SAM3D}


# ---------------------------------------------------------------------------
# POST /process  –  accept video, kick off background job, return job_id
# ---------------------------------------------------------------------------
@app.post("/process")
async def process_video(video: UploadFile = File(...)):
    # --- validate upload -------------------------------------------------
    if video.content_type and not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail=f"Expected a video file, got {video.content_type}")

    # Create a unique working directory and job id
    job_id = uuid.uuid4().hex
    work_dir = tempfile.mkdtemp(dir=TEMP_DIR)
    video_path = os.path.join(work_dir, video.filename or "upload.mp4")

    # --- save uploaded video to disk -------------------------------------
    try:
        total_bytes = 0
        max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
        with open(video_path, "wb") as f:
            while chunk := await video.read(1024 * 1024):  # 1 MB chunks
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    shutil.rmtree(work_dir, ignore_errors=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (max {MAX_UPLOAD_SIZE_MB} MB)",
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc))

    logger.info("Received %s (%.1f MB) → job %s", video.filename, total_bytes / 1e6, job_id)

    # Register job
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "processing",
            "frames_done": 0,
            "total_frames": 0,
            "work_dir": work_dir,
            "zip_path": None,
            "error": None,
        }

    # --- launch background processing thread -----------------------------
    def _run():
        try:
            def _progress(frames_done: int, total_frames: int):
                with _jobs_lock:
                    _jobs[job_id]["frames_done"] = frames_done
                    _jobs[job_id]["total_frames"] = total_frames

            t0 = time.time()
            if USE_SAM3D:
                zip_path, n_frames, fps = generate_sam3d_meshes(
                    video_path, work_dir, progress_callback=_progress
                )
            else:
                zip_path, n_frames, fps = generate_stub_meshes(
                    video_path, work_dir, progress_callback=_progress
                )
            elapsed = time.time() - t0
            logger.info("Job %s: %d frames @ %.1f fps in %.2fs", job_id, n_frames, fps, elapsed)

            with _jobs_lock:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["zip_path"] = zip_path
                _jobs[job_id]["frames_done"] = n_frames
                _jobs[job_id]["total_frames"] = n_frames
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            with _jobs_lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = str(exc)
            shutil.rmtree(work_dir, ignore_errors=True)

    threading.Thread(target=_run, daemon=True).start()

    return JSONResponse({"job_id": job_id})


# ---------------------------------------------------------------------------
# POST /process_params  –  like /process but returns compact SMPL binary
# ---------------------------------------------------------------------------
@app.post("/process_params")
async def process_video_params(video: UploadFile = File(...)):
    """
    Accept a video, run SAM3DBody + MHR2SMPL conversion, return a job_id.
    Poll /progress/{job_id} then fetch /result_params/{job_id}.
    Falls back to OBJ stub when USE_SAM3D=0.
    """
    if video.content_type and not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail=f"Expected a video file, got {video.content_type}")

    job_id = uuid.uuid4().hex
    work_dir = tempfile.mkdtemp(dir=TEMP_DIR)
    video_path = os.path.join(work_dir, video.filename or "upload.mp4")

    try:
        total_bytes = 0
        max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
        with open(video_path, "wb") as f:
            while chunk := await video.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    shutil.rmtree(work_dir, ignore_errors=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (max {MAX_UPLOAD_SIZE_MB} MB)",
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc))

    logger.info("Received %s (%.1f MB) → params job %s", video.filename, total_bytes / 1e6, job_id)

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "processing",
            "frames_done": 0,
            "total_frames": 0,
            "work_dir": work_dir,
            "error": None,
            "mode": "params",
            # Streaming buffers – appended as frames become ready.
            "params_buffer": bytearray(),  # N × 79 × 4 bytes
            "valid_buffer":  bytearray(),  # N × 1 byte
            "params_frames_ready": 0,
            "fps": 0.0,
            "focal_length": 0.0,
        }

    def _run():
        try:
            def _progress(frames_done: int, total_frames: int):
                with _jobs_lock:
                    _jobs[job_id]["frames_done"] = frames_done
                    _jobs[job_id]["total_frames"] = total_frames

            def _frame_ready(frame_idx: int, params_row, valid: int, fps: float, focal_length: float = 0.0):
                with _jobs_lock:
                    job = _jobs[job_id]
                    job["params_buffer"] += params_row.astype("float32").tobytes()
                    job["valid_buffer"]  += bytes([valid])
                    job["params_frames_ready"] += 1
                    if job["fps"] == 0.0:
                        job["fps"] = fps
                    if job["focal_length"] == 0.0 and focal_length > 0.0:
                        job["focal_length"] = focal_length

            t0 = time.time()
            if USE_SAM3D:
                bin_path, n_frames, fps = generate_sam3d_params(
                    video_path, work_dir,
                    progress_callback=_progress,
                    frame_ready_callback=_frame_ready,
                )
            else:
                # Stub: emit rest-pose T-pose frames one by one.
                import cv2 as _cv2, numpy as _np
                cap = _cv2.VideoCapture(video_path)
                fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
                n_frames = max(1, int(cap.get(_cv2.CAP_PROP_FRAME_COUNT)))
                cap.release()
                PARAMS_PER_FRAME = 79
                zero_row = _np.zeros(PARAMS_PER_FRAME, dtype=_np.float32)
                for i in range(n_frames):
                    _frame_ready(i, zero_row, 1, fps, focal_length=0.0)
                    _progress(i + 1, n_frames)
                bin_path = None  # not used; result reconstructed from buffer

            elapsed = time.time() - t0
            logger.info("Params job %s: %d frames in %.2fs", job_id, n_frames, elapsed)

            with _jobs_lock:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["frames_done"] = n_frames
                _jobs[job_id]["total_frames"] = n_frames
        except Exception as exc:
            logger.exception("Params job %s failed", job_id)
            with _jobs_lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = str(exc)
            shutil.rmtree(work_dir, ignore_errors=True)

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"job_id": job_id})


# ---------------------------------------------------------------------------
# GET /progress/{job_id}  –  poll for processing progress
# ---------------------------------------------------------------------------
@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return {
        "status": job["status"],          # "processing" | "done" | "error"
        "frames_done": job["frames_done"],
        "total_frames": job["total_frames"],
        "error": job["error"],
    }


# ---------------------------------------------------------------------------
# GET /result/{job_id}  –  fetch the ZIP once done
# ---------------------------------------------------------------------------
@app.get("/result/{job_id}")
async def get_result(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job["error"] or "Processing failed")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="Job not finished yet")

    zip_path = job["zip_path"]
    work_dir = job["work_dir"]

    def _cleanup():
        with _jobs_lock:
            _jobs.pop(job_id, None)
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.debug("Cleaned up job %s", job_id)

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename="meshes.zip",
        background=_cleanup_task(work_dir, job_id),
    )


# ---------------------------------------------------------------------------
# GET /result_params/{job_id}  –  fetch the full SMPL binary once done
# ---------------------------------------------------------------------------
@app.get("/result_params/{job_id}")
async def get_result_params(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job["error"] or "Processing failed")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="Job not finished yet")

    body = _build_params_binary(job, from_frame=0)
    work_dir = job["work_dir"]
    return Response(
        content=bytes(body),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=smpl_params.bin"},
        background=_cleanup_task(work_dir, job_id),
    )


# ---------------------------------------------------------------------------
# GET /result_params_partial/{job_id}  –  fetch frames ready so far
# ---------------------------------------------------------------------------
@app.get("/result_params_partial/{job_id}")
async def get_result_params_partial(job_id: str, from_frame: int = 0):
    """
    Return however many SMPL-param frames have been computed so far,
    starting from *from_frame*.  Safe to call while the job is still
    processing.  Returns an empty-frame binary (frame_count=0) when
    nothing new is available yet.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job["error"] or "Processing failed")

    body = _build_params_binary(job, from_frame=from_frame)
    return Response(
        content=bytes(body),
        media_type="application/octet-stream",
    )


# ---------------------------------------------------------------------------
# Helper: build SMPL binary from the in-memory streaming buffers
# ---------------------------------------------------------------------------
import struct as _struct

_PARAMS_PER_FRAME = 79
_PARAMS_BYTES     = _PARAMS_PER_FRAME * 4  # float32
_MAGIC            = 0x534D504C  # 'SMPL'


def _build_params_binary(job: dict, from_frame: int) -> bytearray:
    """
    Slice job['params_buffer'] / job['valid_buffer'] from *from_frame* to
    however many frames are ready, and wrap with the standard binary header.

    Caller must hold *no* lock – this function takes _jobs_lock internally
    to copy the buffers atomically.
    """
    with _jobs_lock:
        frames_ready = job["params_frames_ready"]
        fps          = job["fps"]
        focal_length = job["focal_length"]
        # Take a snapshot of the bytes we need (avoids holding lock during Response).
        start_byte   = from_frame * _PARAMS_BYTES
        end_byte     = frames_ready * _PARAMS_BYTES
        params_slice = bytes(job["params_buffer"][start_byte:end_byte])
        valid_slice  = bytes(job["valid_buffer"][from_frame:frames_ready])

    count = max(0, frames_ready - from_frame)
    out = bytearray()
    out += _struct.pack("<I", _MAGIC)
    out += _struct.pack("<I", count)
    out += _struct.pack("<I", _PARAMS_PER_FRAME)
    out += _struct.pack("<f", float(fps))
    out += _struct.pack("<f", float(focal_length))
    out += params_slice
    out += valid_slice
    return out


# ---------------------------------------------------------------------------
# Background cleanup
# ---------------------------------------------------------------------------
from starlette.background import BackgroundTask


def _cleanup_task(work_dir: str, job_id: str | None = None) -> BackgroundTask:
    def _rm():
        if job_id is not None:
            with _jobs_lock:
                _jobs.pop(job_id, None)
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.debug("Cleaned up %s", work_dir)
    return BackgroundTask(_rm)


# ---------------------------------------------------------------------------
# Entry point for `python main.py`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import socket
    import uvicorn
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        if _s.connect_ex(("127.0.0.1", PORT)) == 0:
            print(f"\n  Port {PORT} is already in use. Kill it with:\n\n    fuser -k {PORT}/tcp\n")
            raise SystemExit(1)
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False, log_level="info")
