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
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import HOST, MAX_UPLOAD_SIZE_MB, PORT, TEMP_DIR, USE_SAM3D
from mesh_gen import generate_stub_meshes, generate_sam3d_meshes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sam3d_server")


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
# POST /process  –  accept video, return ZIP of OBJ meshes
# ---------------------------------------------------------------------------
@app.post("/process")
async def process_video(video: UploadFile = File(...)):
    # --- validate upload -------------------------------------------------
    if video.content_type and not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail=f"Expected a video file, got {video.content_type}")

    # Create a unique working directory
    work_dir = tempfile.mkdtemp(dir=TEMP_DIR)
    video_path = os.path.join(work_dir, video.filename or "upload.mp4")

    try:
        # --- save uploaded video to disk ---------------------------------
        total_bytes = 0
        max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
        with open(video_path, "wb") as f:
            while chunk := await video.read(1024 * 1024):  # 1 MB chunks
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (max {MAX_UPLOAD_SIZE_MB} MB)",
                    )
                f.write(chunk)

        logger.info("Received %s (%.1f MB)", video.filename, total_bytes / 1e6)

        # --- run mesh generation -----------------------------------------
        t0 = time.time()

        if USE_SAM3D:
            zip_path, n_frames, fps = generate_sam3d_meshes(video_path, work_dir)
        else:
            zip_path, n_frames, fps = generate_stub_meshes(video_path, work_dir)

        elapsed = time.time() - t0
        logger.info("Generated %d frames @ %.1f fps in %.2fs", n_frames, fps, elapsed)

        # --- return ZIP --------------------------------------------------
        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename="meshes.zip",
            # Cleanup the work dir once the response is sent
            background=_cleanup_task(work_dir),
        )

    except HTTPException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.exception("Processing failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Background cleanup
# ---------------------------------------------------------------------------
from starlette.background import BackgroundTask


def _cleanup_task(work_dir: str) -> BackgroundTask:
    def _rm():
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
