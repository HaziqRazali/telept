"""Mocap -> RGB projection tool (PySide6).

Loads a video + C3D recorded in the SAME session as the calibration, applies
the calibration saved by pc_calib_tool (intrinsics + mocap->camera transform)
and projects ALL mocap markers onto the synchronized RGB video, with
frame-exact scrubbing.  It never calibrates.

Run:  .venv/bin/python app.py   (needs a display; no browser, no Gradio)

Tabs (in order):
  0. Data   - set the iPad video + C3D paths and load them
  1. Sync   - LED blink: ROI + threshold (video), 3D box (mocap),
              offset fine-tune (save/load sync.json)
  2. Trim   - shared start/end trim for both streams
  3. Project- load the calibration, project the mocap markers onto the
              video and scrub through the take

The video is preloaded into RAM so scrubbing is instant; the projection
overlay is recomputed on the fly from the mocap frame mapped by the sync
offset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# NOTE: mocap_view imports matplotlib with the "Agg" backend; we deliberately
# do NOT call matplotlib.use() here.  The embedded traces plot uses explicit
# Figure() objects + FigureCanvasQTAgg, which render via Qt directly and work
# fine alongside Agg for the off-screen mocap renders.
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import config
import marker_layout as ml
from calibrate import load_markers_mm, run_calibration, solve_board_pose
from data_loader import load_c3d, pre_extract_frames
from intrinsics import calibrate_intrinsics, detect_board
from mocap_view import render_mocap_2d, render_mocap_3d
from sync import (
    auto_find_led,
    compute_mocap_trace,
    compute_video_trace,
    load_sync,
    mocap_index_for_video_time,
    save_sync,
    threshold_trace,
)
from trim import apply_trim, load_trim, save_trim


# ----------------------------------------------------------------------
# Qt <-> OpenCV plugin-path fix (defensive; only matters with a GUI build of
# opencv-python installed).
# ----------------------------------------------------------------------
# opencv-python (NOT headless) bundles its own Qt build and, at import time,
# sets QT_QPA_PLATFORM_PLUGIN_PATH to cv2/qt/plugins.  That Qt build differs
# from PySide6's, so Qt then fails to load the "xcb" platform plugin and the
# app aborts with "Could not load the Qt platform plugin 'xcb' ... in
# cv2/qt/plugins".  With opencv-python-headless (recommended, see
# requirements.txt) this is a harmless no-op.
# Fix: point Qt back at the binding's own plugins directory, and strip any
# cv2 plugin dirs from QApplication's library paths.
def _fix_qt_plugin_path() -> None:
    import os
    plugins = ""
    try:
        from PySide6.QtCore import QLibraryInfo
        try:  # Qt6 API
            plugins = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
        except Exception:  # older Qt5-style API
            plugins = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    except Exception:
        try:  # fall back to PyQt5 if someone runs this with PyQt5 installed
            from PyQt5.QtCore import QLibraryInfo
            plugins = QLibraryInfo.location(QLibraryInfo.PluginsPath)
        except Exception:
            pass
    if plugins and os.path.isdir(plugins):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugins
        os.environ["QT_PLUGIN_PATH"] = plugins


_fix_qt_plugin_path()  # after cv2 (and friends) imported, before QApplication


def _strip_cv2_plugin_paths(app: "QApplication") -> None:
    """Remove opencv-bundled Qt plugin dirs from the app's library search path."""
    for p in list(app.libraryPaths()):
        if "cv2" in p:
            app.removeLibraryPath(p)

# ----------------------------------------------------------------------
# Data globals -- filled by load_data() (from the "0. Data" tab, or at app
# start if the configured / last-saved paths exist on this machine).
# ----------------------------------------------------------------------
FRAMES: list = []
TIMES: np.ndarray = np.array([])
MOCAP: dict = {}
N_VID = 0
N_MOC = 0
FPS_M = 100.0
IMG_H, IMG_W = 720, 1280
VIDEO_DUR = 0.0
CURRENT_VIDEO_PATH = str(config.VIDEO_PATH)
CURRENT_C3D_PATH = str(config.C3D_PATH)
FRAME_ARR: np.ndarray | None = None  # (N, H, W, 3) uint8 BGR - all frames in RAM


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _rgb_to_pixmap(rgb: np.ndarray) -> QPixmap:
    """Contiguous RGB numpy array -> QPixmap (fast, copies once into Qt)."""
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _bgr_to_pixmap(bgr: np.ndarray) -> QPixmap:
    return _rgb_to_pixmap(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ----------------------------------------------------------------------
# Widgets
# ----------------------------------------------------------------------
class ClickableLabel(QLabel):
    """QLabel that shows an image scaled to FIT the widget (aspect kept) and
    reports clicks / wheel events in SOURCE-image pixel coordinates.

    setSource()/setArray() store the full-res image; it is re-scaled to fit
    the widget on every show/resize, so the whole image is always visible
    (never clipped) and clicks map back to exact source pixels.
    """

    clicked = Signal(int, int)             # (x, y) in source-image pixels
    wheelZoomed = Signal(float, int, int)  # (factor, src_x, src_y) on wheel

    def __init__(self, parent=None):
        super().__init__(parent)
        self._src_w = 1
        self._src_h = 1
        self._sx = 1.0
        self._sy = 1.0
        self._pix = None                   # full-res pixmap (source)
        self.setMinimumSize(200, 120)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1c1c1c; border: 1px solid #3c3c3c;")
        # fill the space the layout gives us (sizeHint is irrelevant: we fit)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

    # -- image input ---------------------------------------------------
    def setSource(self, rgb: np.ndarray, src_w: int, src_h: int) -> None:
        """Store an RGB image; display scaled to fit the widget."""
        if rgb is None or rgb.size == 0:
            self.clear()
            return
        self._src_w, self._src_h = src_w, src_h
        self._pix = _rgb_to_pixmap(rgb)
        self._refit()

    def setArray(self, rgb: np.ndarray) -> None:
        """Display an RGB array at its native aspect, scaled to fit."""
        if rgb is None or rgb.size == 0:
            self.clear()
            return
        self.setSource(rgb, rgb.shape[1], rgb.shape[0])

    def clear(self) -> None:
        self._pix = None
        super().clear()

    # -- fit to widget --------------------------------------------------
    def _refit(self) -> None:
        if self._pix is None or self._pix.isNull():
            return
        ww, wh = self.width(), self.height()
        if ww <= 1 or wh <= 1:
            return
        scaled = self._pix.scaled(ww, wh, Qt.KeepAspectRatio,
                                  Qt.FastTransformation)
        self._sx = self._src_w / max(scaled.width(), 1)
        self._sy = self._src_h / max(scaled.height(), 1)
        self.setPixmap(scaled)

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._refit()

    # -- interaction ----------------------------------------------------
    def _widget_to_source(self, pos) -> tuple[int, int]:
        """Map a widget-coordinate point to source-image pixels."""
        pix = self.pixmap()
        if pix is None or pix.isNull():
            return 0, 0
        # The (scaled) pixmap is drawn CENTERED in the widget; go
        # widget coords -> pixmap coords -> source pixels.
        ox = (self.width() - pix.width()) // 2
        oy = (self.height() - pix.height()) // 2
        return int((pos.x() - ox) * self._sx), int((pos.y() - oy) * self._sy)

    def mousePressEvent(self, ev) -> None:
        x, y = self._widget_to_source(ev.pos())
        self.clicked.emit(x, y)

    def wheelEvent(self, ev) -> None:
        delta = ev.angleDelta().y()
        if delta:
            # Qt6: ev.position() -> QPointF (widget coords)
            x, y = self._widget_to_source(ev.position())
            factor = 1.2 if delta > 0 else 1.0 / 1.2
            self.wheelZoomed.emit(factor, x, y)
        ev.accept()


class TracesCanvas(FigureCanvas):
    """Embedded matplotlib canvas for the sync traces plot."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(9, 3.2), dpi=100)
        super().__init__(self.fig)
        self.setMinimumHeight(230)

    def update_traces(self, state, times: np.ndarray, fps: float,
                      video_cursor_t: float | None = None,
                      mocap_cursor_t: float | None = None,
                      xlim: tuple[float, float] | None = None,
                      ylim: tuple[float, float] = (-0.1, 2.6)) -> None:
        """Plot the traces; ``video_cursor_t``/``mocap_cursor_t`` (video s)
        draw dashed vertical bars for the video and mocap scrub sliders.

        Axes are FIXED to the passed ranges (default: full y span of all
        traces) so moving the bars never rescales the plot.
        """
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        any_line = False
        if state["video_trace"] is not None:
            t = np.array(state["video_trace"])
            norm = (t - t.min()) / max(t.max() - t.min(), 1e-9)
            ax.plot(times, norm, color="tab:blue", lw=1,
                    label="video intensity (norm)")
            any_line = True
        if state["video_bin"] is not None:
            ax.step(times, np.array(state["video_bin"]), color="tab:green",
                    lw=1.2, where="post", label="video binary")
            any_line = True
        if state["mocap_bin"] is not None:
            m = np.array(state["mocap_bin"])
            t_m = np.arange(len(m)) / fps
            ax.step(t_m + state["offset"], m + 1.5, color="tab:red", lw=1.2,
                    where="post", label="mocap binary (+1.5)")
            any_line = True
        if video_cursor_t is not None:
            ax.axvline(video_cursor_t, color="tab:blue", ls="--", lw=1.2,
                       alpha=0.7, label="video scrub")
            any_line = True
        if mocap_cursor_t is not None:
            ax.axvline(mocap_cursor_t, color="tab:purple", ls="--", lw=1.2,
                       alpha=0.7, label="mocap scrub")
            any_line = True
        ax.set_xlabel("video time (s)")
        ax.set_ylabel("signal")
        ax.set_title("move the video + mocap scrub sliders to line their "
                     "bars up on the pulses, then 'Set offset from bars'")
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if any_line:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        self.fig.tight_layout()
        self.draw_idle()


# ----------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mocap -> RGB Projection (PC)")
        self.resize(1380, 940)

        self.state = {
            "placed": [],          # list of [x_mm, y_mm]
            "video_roi": None,     # [x0, y0, x1, y1] full-frame px
            "roi_click": None,     # first of two ROI clicks (full-frame px)
            "box_lo": None,
            "box_hi": None,
            "video_trace": None,
            "video_bin": None,
            "mocap_bin": None,
            "threshold": 128.0,
            "offset": 0.0,
            "zoom": 1.0,
        }
        self._crop_ox = 0           # zoom-crop origin (full-frame px)
        self._crop_oy = 0
        self._mocap2d_map = None    # pixel->mm map of the last 2D mocap render
        self._box_draw = []         # clicked corners (X, Y mm) while box-drawing
        self._sync_last = 0         # last sync-jog slider value (centi-seconds)
        self._calib = None          # loaded calibration (R, t, K, dist)

        self._build_ui()

        # pre-fill saved / default paths, then auto-load if they exist here
        saved = config.load_settings()
        self.vid_edit.setText(saved.get("video_path", str(config.VIDEO_PATH)))
        self.c3d_edit.setText(saved.get("c3d_path", str(config.C3D_PATH)))
        if Path(self.vid_edit.text()).is_file() and \
                Path(self.c3d_edit.text()).is_file():
            self._load_data(self.vid_edit.text(), self.c3d_edit.text())
        else:
            self.status.setText(
                "Set the video + C3D paths (or Browse), then click 'Load data'.\n"
                "The default paths point at the server "
                "(/data/haziq/telept/data/NUS/...) and are not present here.")

    # ==================================================================
    # UI construction
    # ==================================================================
    def _build_ui(self):
        tabs = QTabWidget()
        tabs.addTab(self._build_tab0(), "0. Data")
        tabs.addTab(self._build_tab2(), "1. Sync")
        tabs.addTab(self._build_tab3(), "2. Trim")
        tabs.addTab(self._build_tab4(), "3. Project")
        self.setCentralWidget(tabs)

    # ---- Tab 0: data -------------------------------------------------
    def _build_tab0(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        form = QFormLayout()
        self.vid_edit = QLineEdit()
        self.c3d_edit = QLineEdit()
        vid_row = QHBoxLayout()
        vid_row.addWidget(self.vid_edit)
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: self._browse(self.vid_edit, "iPad video (*.mp4 *.mov *.avi *.mkv)"))
        vid_row.addWidget(b)
        c3d_row = QHBoxLayout()
        c3d_row.addWidget(self.c3d_edit)
        b2 = QPushButton("Browse…")
        b2.clicked.connect(lambda: self._browse(self.c3d_edit, "Mocap C3D (*.c3d)"))
        c3d_row.addWidget(b2)
        form.addRow("iPad video (.mp4)", vid_row)
        form.addRow("Mocap C3D (.c3d)", c3d_row)
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.0, 1000.0)
        self.fps_spin.setDecimals(1)
        self.fps_spin.setSingleStep(1.0)
        self.fps_spin.setValue(
            float(config.load_settings().get("mocap_fps_override", 0.0) or 0.0))
        self.fps_spin.setToolTip(
            "True capture rate of the C3D if the rate stored in the file is "
            "wrong (0 = use the file's rate). If mocap should be LONGER than "
            "the video but isn't, the file rate is too high - lower it here "
            "and reload.")
        form.addRow("Mocap FPS override (0 = from C3D)", self.fps_spin)
        lay.addLayout(form)

        load_btn = QPushButton("Load data")
        load_btn.setStyleSheet("font-weight: bold;")
        load_btn.clicked.connect(self._on_load)
        lay.addWidget(load_btn)

        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setMaximumHeight(220)
        lay.addWidget(self.status)
        lay.addStretch(1)
        return w

    def _browse(self, edit: QLineEdit, filt: str):
        fn, _ = QFileDialog.getOpenFileName(self, "Choose file", edit.text(), filt)
        if fn:
            edit.setText(fn)

    def _on_load(self):
        self._load_data(self.vid_edit.text().strip(),
                        self.c3d_edit.text().strip())

    def _load_data(self, video_path: str, c3d_path: str):
        global FRAMES, TIMES, MOCAP, N_VID, N_MOC, FPS_M, VIDEO_DUR
        global CURRENT_VIDEO_PATH, CURRENT_C3D_PATH, FRAME_ARR, IMG_H, IMG_W
        try:
            self.status.setText(f"Loading…\n  video: {video_path}\n  c3d: {c3d_path}")
            frames, times = pre_extract_frames(video_path)
            mocap = load_c3d(c3d_path)
        except Exception as e:
            self.status.setText(f"ERROR: {e}")
            return

        FRAMES, TIMES, MOCAP = frames, times, mocap
        N_VID = len(FRAMES)
        N_MOC = MOCAP["n_frames"]
        fps_override = self.fps_spin.value()
        if fps_override and fps_override > 0:
            MOCAP["fps"] = float(fps_override)
        FPS_M = MOCAP["fps"]
        VIDEO_DUR = float(TIMES[-1])
        CURRENT_VIDEO_PATH = video_path
        CURRENT_C3D_PATH = c3d_path
        config.save_settings(
            video_path, c3d_path,
            mocap_fps_override=(fps_override if fps_override else None))

        self.status.setText("Preloading frames into RAM…")
        FRAME_ARR = self._preload_frames(FRAMES)
        IMG_H, IMG_W = FRAME_ARR.shape[1], FRAME_ARR.shape[2]

        # update widget ranges
        self.video_scrub.setRange(0, max(N_VID - 1, 0))
        self.video_scrub.setValue(0)
        self.mocap_scrub.setRange(0, max(N_MOC - 1, 0))
        self.mocap_scrub.setValue(0)
        self.sync_scrub.setValue(0)
        self.trim_start.setRange(0.0, max(VIDEO_DUR, 0.01))
        self.trim_end.setRange(0.0, max(VIDEO_DUR, 0.01))
        self.trim_end.setValue(max(VIDEO_DUR, 0.01))
        self.offset_spin.setRange(-60.0, 60.0)

        # resume convenience: pre-fill the trim boxes from a saved trim.json
        _t = load_trim()
        if _t is not None:
            self.trim_start.setValue(_t["start_s"])
            self.trim_end.setValue(_t["end_s"])

        # reset per-session state but keep saved markers/sync if any
        self.state.update({
            "placed": [], "video_roi": None, "roi_click": None,
            "box_lo": None, "box_hi": None,
            "video_trace": None, "video_bin": None, "mocap_bin": None,
            "threshold": 128.0, "offset": 0.0, "zoom": 1.0,
        })
        self.state.pop("zoom_focus", None)
        self._box_draw = []
        self._mocap2d_map = None
        self._sync_last = 0
        self.sync_scrub_label.setText("0.00s")

        self._update_video_views()
        self._update_mocap_views()
        self._refresh_traces()
        self._set_proj_range()
        self._update_proj_view()

        self.status.setText(
            f"Loaded:\n  video: {Path(video_path).name}  "
            f"({N_VID} frames, {VIDEO_DUR:.1f} s, "
            f"{FRAME_ARR.nbytes / 1e6:.0f} MB in RAM)\n"
            f"  mocap: {Path(c3d_path).name}  "
            f"({N_MOC} frames @ {FPS_M:.0f} Hz)\n\n"
            "Proceed to tabs 1-3.  For frame-exact video scrubbing: click the\n"
            "video scrub slider, then use arrow keys (PageUp/Down = ±10).")

    @staticmethod
    def _preload_frames(frames: list) -> np.ndarray:
        """Decode all cached JPEGs into one (N, H, W, 3) uint8 BGR array."""
        first = cv2.imread(str(frames[0]))
        H, W = first.shape[:2]
        arr = np.empty((len(frames), H, W, 3), np.uint8)
        arr[0] = first
        for i in range(1, len(frames)):
            arr[i] = cv2.imread(str(frames[i]))
        return arr

    # ---- Tab 1: marker layout ----------------------------------------
    def _build_tab1(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)

        self.canvas_view = ClickableLabel()
        self.canvas_view.setMinimumSize(720, 520)
        self.canvas_view.clicked.connect(self._on_canvas_click)
        lay.addWidget(self.canvas_view, 1)

        right = QVBoxLayout()
        self.ml_info = QTextEdit()
        self.ml_info.setReadOnly(True)
        self.ml_info.setMaximumHeight(180)
        right.addWidget(QLabel("Click the orange dots where the 6 reflective "
                               "markers sit (grid = 40 mm):"))
        right.addWidget(self.ml_info)
        grid = QGridLayout()
        undo = QPushButton("Undo last"); clear = QPushButton("Clear all")
        verify = QPushButton("Verify vs C3D"); save = QPushButton("Save markers")
        undo.clicked.connect(self._on_ml_undo)
        clear.clicked.connect(self._on_ml_clear)
        verify.clicked.connect(self._on_ml_verify)
        save.clicked.connect(self._on_ml_save)
        grid.addWidget(undo, 0, 0); grid.addWidget(clear, 0, 1)
        grid.addWidget(verify, 1, 0); grid.addWidget(save, 1, 1)
        loadm = QPushButton("Load markers (layout + board config)")
        loadm.clicked.connect(self._on_ml_load)
        grid.addWidget(loadm, 2, 0, 1, 2)
        right.addLayout(grid)
        self.ml_result = QTextEdit()
        self.ml_result.setReadOnly(True)
        self.ml_result.setMaximumHeight(140)
        right.addWidget(self.ml_result)
        right.addStretch(1)
        lay.addLayout(right, 0)
        return w

    def _render_canvas(self):
        named = [(f"{i + 1}", p[0], p[1]) for i, p in enumerate(self.state["placed"])]
        self.canvas_view.setArray(cv2.cvtColor(ml.render_canvas(named),
                                               cv2.COLOR_BGR2RGB))
        msg = "Placed: " + ", ".join(
            f"{i + 1}@({x:.0f},{y:.0f})mm"
            for i, (x, y) in enumerate(self.state["placed"])) or "No markers yet"
        self.ml_info.setText(msg)

    def _on_canvas_click(self, x: int, y: int):
        x_mm, y_mm = ml.px_to_mm(x, y)
        x_mm, y_mm = ml.snap_to_grid(x_mm, y_mm)
        if len(self.state["placed"]) >= config.NUM_MARKERS:
            self.ml_result.setText(
                f"Already have {config.NUM_MARKERS} markers - use Undo to remove one.")
            return
        self.state["placed"].append([x_mm, y_mm])
        self._render_canvas()

    def _on_ml_undo(self):
        if self.state["placed"]:
            self.state["placed"].pop()
        self._render_canvas()

    def _on_ml_clear(self):
        self.state["placed"] = []
        self._render_canvas()

    def _on_ml_verify(self):
        if not MOCAP:
            self.ml_result.setText("Load data first (tab 0).")
            return
        if len(self.state["placed"]) != config.NUM_MARKERS:
            self.ml_result.setText(
                f"Need exactly {config.NUM_MARKERS} markers, have "
                f"{len(self.state['placed'])}.")
            return
        markers_mm = np.array([[x, y, 0.0] for x, y in self.state["placed"]], float)
        v = ml.verify_markers(markers_mm, MOCAP)
        if not v.get("ok"):
            self.ml_result.setText(f"Verification failed: {v.get('error')}")
            return
        self.ml_result.setText(
            f"OK. Click order maps to C3D labels: {', '.join(v['assigned'])}\n"
            f"mean distance error: {v['mean_dist_err_mm']:.1f} mm, "
            f"max: {v['max_dist_err_mm']:.1f} mm")

    def _on_ml_save(self):
        if not MOCAP:
            self.ml_result.setText("Load data first (tab 0).")
            return
        if len(self.state["placed"]) != config.NUM_MARKERS:
            self.ml_result.setText(
                f"Need exactly {config.NUM_MARKERS} markers, have "
                f"{len(self.state['placed'])}.")
            return
        markers_mm = np.array([[x, y, 0.0] for x, y in self.state["placed"]], float)
        v = ml.verify_markers(markers_mm, MOCAP)
        ml.save_markers(markers_mm, v)
        self.ml_result.setText(
            "Saved output/markers.json" +
            (f" (labels: {', '.join(v['assigned'])})" if v.get("ok")
             else " (unverified order)"))

    def _apply_board_config(self, inner_corners, square_mm, margin):
        """Override the chessboard geometry at runtime (from a loaded file).

        The geometry modules imported these values at import time, so we
        propagate the new values into their namespaces too.
        """
        import intrinsics as _intr
        config.BOARD_INNER_CORNERS = tuple(int(v) for v in inner_corners)
        config.SQUARE_SIZE_MM = float(square_mm)
        config.BOARD_MARGIN_SQUARES = int(margin)
        for _mod in (ml, _intr):
            _mod.BOARD_INNER_CORNERS = config.BOARD_INNER_CORNERS
            _mod.SQUARE_SIZE_MM = config.SQUARE_SIZE_MM
        ml.BOARD_MARGIN_SQUARES = config.BOARD_MARGIN_SQUARES

    def _on_ml_load(self):
        """Load a saved marker layout + board config from one JSON file."""
        fn, _ = QFileDialog.getOpenFileName(
            self, "Load marker layout", str(config.MARKERS_FILE),
            "Marker layout (*.json)")
        if fn:
            self._load_markers_file(fn)

    def _load_markers_file(self, fn: str):
        try:
            data = json.loads(Path(fn).read_text())
        except Exception as e:
            self.ml_result.setText(f"ERROR reading {fn}: {e}")
            return
        markers = data.get("markers")
        if not isinstance(markers, list) or not markers:
            self.ml_result.setText(
                f"'{Path(fn).name}' is not a marker layout file "
                "(no 'markers' list).\n"
                "Pick the markers.json saved by 'Save markers', or place "
                "the markers again in this tab and click 'Save markers'.")
            return
        bc = data.get("board_inner_corners")
        sq = data.get("square_size_mm")
        mg = data.get("margin_squares")
        has_board = bc and sq and mg is not None
        if has_board:
            self._apply_board_config(bc, sq, mg)
        placed = [[float(m["x_mm"]), float(m["y_mm"])] for m in markers]
        self.state["placed"] = placed
        self._render_canvas()
        names = ", ".join(m.get("name", "?") for m in markers)
        board_info = (f"board: {bc[0]}x{bc[1]} squares, {sq:.0f} mm, "
                      f"{mg} margin squares\n" if has_board
                      else "board config: not in file (keeping current)\n")
        self.ml_result.setText(
            f"Loaded {len(placed)} markers from {Path(fn).name}\n"
            f"{board_info}saved labels: {names}\n"
            "Click 'Verify vs C3D' to confirm.")

    # ---- Tab 2: sync --------------------------------------------------
    def _build_tab2(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)

        top = QHBoxLayout()

        # --- video column ---------------------------------------------
        vcol = QVBoxLayout()
        self.video_view = ClickableLabel()
        self.video_view.setMinimumSize(640, 360)
        self.video_view.clicked.connect(self._on_video_click)
        self.video_view.wheelZoomed.connect(self._on_video_wheel_zoom)
        vcol.addWidget(self.video_view, 1)

        self.zoom_view = QLabel()
        self.zoom_view.setAlignment(Qt.AlignCenter)
        self.zoom_view.setMinimumHeight(130)
        self.zoom_view.setStyleSheet(
            "background-color: #1c1c1c; border: 1px solid #3c3c3c;")
        vcol.addWidget(self.zoom_view)

        scrub_row = QHBoxLayout()
        self.video_scrub = QSlider(Qt.Horizontal)
        self.video_scrub.setRange(0, 0)
        self.video_scrub.valueChanged.connect(self._on_video_scrub_change)
        scrub_row.addWidget(self.video_scrub, 1)
        self.video_frame_info = QLabel("frame -")
        scrub_row.addWidget(self.video_frame_info)
        vcol.addLayout(scrub_row)

        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Video zoom:"))
        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(1.0, 8.0)
        self.zoom_spin.setSingleStep(0.5)
        self.zoom_spin.setValue(1.0)
        self.zoom_spin.valueChanged.connect(self._on_zoom)
        zoom_row.addWidget(self.zoom_spin)
        reset_roi = QPushButton("Reset ROI")
        reset_roi.clicked.connect(self._on_reset_roi)
        zoom_row.addWidget(reset_roi)
        zoom_row.addStretch(1)
        vcol.addLayout(zoom_row)

        self.roi_status = QLabel("Click 2 points on the video to set the LED ROI.")
        self.roi_status.setWordWrap(True)
        vcol.addWidget(self.roi_status)

        # fine-tune buttons
        ft = QGroupBox("Fine-tune ROI corners (±1 px, shift-click = ±5)")
        ft.setCheckable(False)
        fg = QGridLayout(ft)
        self._nudge_buttons = []
        for ci, (corner, clabel) in enumerate([("tl", "TL"), ("tr", "TR"),
                                               ("bl", "BL"), ("br", "BR")]):
            fg.addWidget(QLabel(clabel), 0, ci)
            for ai, axis in enumerate(["x", "y"]):
                for di, delta in enumerate([-1, 1]):
                    b = QPushButton(f"{axis}{'+' if delta > 0 else '-'}")
                    b.clicked.connect(
                        lambda _c=corner, _a=axis, _d=delta:
                        self._on_nudge(_c, _a, _d))
                    fg.addWidget(b, 1 + ai, ci * 2 + di)
        vcol.addWidget(ft)
        top.addLayout(vcol, 3)

        # --- mocap column ---------------------------------------------
        mcol = QVBoxLayout()
        self.mocap3d_view = ClickableLabel()
        self.mocap3d_view.setMinimumSize(320, 280)
        mcol.addWidget(self.mocap3d_view, 1)
        self.mocap2d_view = ClickableLabel()
        self.mocap2d_view.setMinimumSize(320, 220)
        self.mocap2d_view.clicked.connect(self._on_mocap_box_click)
        mcol.addWidget(self.mocap2d_view, 1)
        mcol.addWidget(QLabel("2D view: click 2 corners to draw the LED box "
                              "(Z ± half from the spinboxes)"))

        mscrub_row = QHBoxLayout()
        self.mocap_scrub = QSlider(Qt.Horizontal)
        self.mocap_scrub.setRange(0, 0)
        self.mocap_scrub.valueChanged.connect(self._on_mocap_scrub_change)
        mscrub_row.addWidget(self.mocap_scrub, 1)
        self.mocap_frame_info = QLabel("frame -")
        mscrub_row.addWidget(self.mocap_frame_info)
        mcol.addLayout(mscrub_row)

        auto_led = QPushButton("Auto-find LED (mocap)")
        auto_led.clicked.connect(self._on_auto_led)
        mcol.addWidget(auto_led)

        box = QGroupBox("Mocap 3D box (LED presence)")
        bg = QGridLayout(box)
        self.box_x = self._make_box_spin(-3000, 1000, 0)
        self.box_y = self._make_box_spin(-2000, 1000, 0)
        self.box_z = self._make_box_spin(-1000, 1000, 0)
        self.box_half = self._make_box_spin(10, 200, 50)
        for i, (label, spin) in enumerate([("X (mm)", self.box_x),
                                           ("Y (mm)", self.box_y),
                                           ("Z (mm)", self.box_z),
                                           ("half (mm)", self.box_half)]):
            bg.addWidget(QLabel(label), i // 2, (i % 2) * 2)
            bg.addWidget(spin, i // 2, (i % 2) * 2 + 1)
            spin.valueChanged.connect(self._on_box_change)
            spin.valueChanged.connect(self._on_box_release)
        mcol.addWidget(box)
        top.addLayout(mcol, 2)

        root.addLayout(top, 1)

        # --- traces + controls ----------------------------------------
        self.traces = TracesCanvas()
        root.addWidget(self.traces)

        btns = QHBoxLayout()
        compute = QPushButton("Compute traces")
        compute.setStyleSheet("font-weight: bold;")
        compute.clicked.connect(self._on_compute_traces)
        save_sync = QPushButton("Save sync")
        save_sync.clicked.connect(self._on_save_sync)
        load_sync = QPushButton("Load sync")
        load_sync.clicked.connect(self._on_load_sync)
        btns.addWidget(compute); btns.addWidget(save_sync)
        btns.addWidget(load_sync)
        btns.addStretch(1)
        root.addLayout(btns)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Video threshold:"))
        self.thr_slider = QSlider(Qt.Horizontal)
        self.thr_slider.setRange(0, 255)
        self.thr_slider.setValue(128)
        self.thr_slider.valueChanged.connect(self._on_threshold)
        ctrl.addWidget(self.thr_slider, 1)
        self.thr_value = QLabel("128")
        self.thr_value.setMinimumWidth(34)
        self.thr_value.setAlignment(Qt.AlignCenter)
        ctrl.addWidget(self.thr_value)
        ctrl.addWidget(QLabel("Offset fine-tune (s):"))
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-60.0, 60.0)
        self.offset_spin.setSingleStep(0.01)
        self.offset_spin.setDecimals(3)
        self.offset_spin.valueChanged.connect(self._on_offset)
        ctrl.addWidget(self.offset_spin)
        invert_btn = QPushButton("Invert offset")
        invert_btn.clicked.connect(self._on_invert_offset)
        ctrl.addWidget(invert_btn)
        reset_off = QPushButton("Reset offset")
        reset_off.clicked.connect(lambda: self.offset_spin.setValue(0.0))
        ctrl.addWidget(reset_off)
        bars_btn = QPushButton("Set offset from bars")
        bars_btn.clicked.connect(self._on_offset_from_bars)
        ctrl.addWidget(bars_btn)
        root.addLayout(ctrl)

        jog = QHBoxLayout()
        jog.addWidget(QLabel("Sync scrub (jog both, s):"))
        self.sync_scrub = QSlider(Qt.Horizontal)
        self.sync_scrub.setRange(-6000, 6000)   # centi-seconds; 0 = center
        self.sync_scrub.setValue(0)
        self.sync_scrub.setSingleStep(5)        # 0.05 s
        self.sync_scrub.setPageStep(100)        # 1 s
        self.sync_scrub.valueChanged.connect(self._on_sync_scrub)
        self.sync_scrub.sliderReleased.connect(self._sync_scrub_release)
        jog.addWidget(self.sync_scrub, 1)
        self.sync_scrub_label = QLabel("0.00s")
        self.sync_scrub_label.setMinimumWidth(64)
        self.sync_scrub_label.setAlignment(Qt.AlignCenter)
        jog.addWidget(self.sync_scrub_label)
        root.addLayout(jog)

        self.sync_info = QLabel("Compute both traces first.")
        self.sync_info.setWordWrap(True)
        root.addWidget(self.sync_info)
        return w

    @staticmethod
    def _make_box_spin(lo: float, hi: float, val: float) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(5.0)
        s.setDecimals(1)
        s.setValue(val)
        return s

    # ---- Tab 3: trim --------------------------------------------------
    def _build_tab3(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        self.trim_start = QDoubleSpinBox()
        self.trim_start.setRange(0.0, 1.0)
        self.trim_start.setSingleStep(0.1)
        self.trim_start.setDecimals(3)
        self.trim_end = QDoubleSpinBox()
        self.trim_end.setRange(0.0, 1.0)
        self.trim_end.setSingleStep(0.1)
        self.trim_end.setDecimals(3)
        form.addRow("Trim start (video s)", self.trim_start)
        form.addRow("Trim end (video s)", self.trim_end)
        lay.addLayout(form)
        btns = QHBoxLayout()
        prev = QPushButton("Preview trim")
        prev.clicked.connect(self._on_trim_preview)
        save = QPushButton("Save trim")
        save.clicked.connect(self._on_trim_save)
        load = QPushButton("Load trim")
        load.clicked.connect(self._on_trim_load)
        btns.addWidget(prev); btns.addWidget(save); btns.addWidget(load)
        btns.addStretch(1)
        lay.addLayout(btns)
        self.trim_info = QTextEdit()
        self.trim_info.setReadOnly(True)
        self.trim_info.setMaximumHeight(200)
        lay.addWidget(self.trim_info)
        lay.addStretch(1)
        return w

    # ---- Tab 4: calibrate --------------------------------------------
    def _build_tab4(self) -> QWidget:
        """3. Project - overlay all mocap markers on the synced video."""
        w = QWidget()
        lay = QVBoxLayout(w)

        cal_row = QHBoxLayout()
        load_btn = QPushButton("Load calibration (intrinsics + transform)")
        load_btn.setStyleSheet("font-weight: bold;")
        load_btn.clicked.connect(self._on_load_calib)
        cal_row.addWidget(load_btn)
        self.calib_status = QLabel(
            "No calibration loaded - click the button to read the saved "
            "transform + intrinsics (same-session calibration).")
        self.calib_status.setWordWrap(True)
        cal_row.addWidget(self.calib_status, 1)
        lay.addLayout(cal_row)

        row = QHBoxLayout()
        vcol = QVBoxLayout()
        self.proj_view = ClickableLabel()
        self.proj_view.setMinimumSize(460, 620)
        vcol.addWidget(self.proj_view, 1)
        scrub = QHBoxLayout()
        self.proj_scrub = QSlider(Qt.Horizontal)
        self.proj_scrub.setRange(0, 0)
        self.proj_scrub.valueChanged.connect(self._on_proj_scrub)
        scrub.addWidget(self.proj_scrub, 1)
        self.proj_info = QLabel("frame -")
        scrub.addWidget(self.proj_info)
        vcol.addLayout(scrub)
        opts = QHBoxLayout()
        self.proj_labels_cb = QCheckBox("Show labels")
        self.proj_labels_cb.setChecked(True)
        self.proj_labels_cb.toggled.connect(lambda _: self._update_proj_view())
        self.proj_all_cb = QCheckBox("Show all markers")
        self.proj_all_cb.setChecked(False)
        self.proj_all_cb.toggled.connect(lambda _: self._update_proj_view())
        self.proj_trim_cb = QCheckBox("Limit scrub to trim")
        self.proj_trim_cb.setChecked(True)
        self.proj_trim_cb.toggled.connect(lambda _: self._set_proj_range())
        opts.addWidget(self.proj_labels_cb)
        opts.addWidget(self.proj_all_cb)
        opts.addWidget(self.proj_trim_cb)
        opts.addStretch(1)
        vcol.addLayout(opts)
        row.addLayout(vcol, 3)

        mcol = QVBoxLayout()
        self.proj_mocap3d = ClickableLabel()
        self.proj_mocap3d.setMinimumSize(300, 300)
        mcol.addWidget(self.proj_mocap3d, 1)
        mrow = QHBoxLayout()
        self.proj_mocap_scrub = QSlider(Qt.Horizontal)
        self.proj_mocap_scrub.setRange(0, 0)
        self.proj_mocap_scrub.valueChanged.connect(self._on_proj_mocap_scrub)
        mrow.addWidget(self.proj_mocap_scrub, 1)
        self.proj_mocap_info = QLabel("frame -")
        mrow.addWidget(self.proj_mocap_info)
        mcol.addLayout(mrow)
        row.addLayout(mcol, 2)
        lay.addLayout(row, 1)
        return w

    # ==================================================================
    # Rendering helpers
    # ==================================================================
    def _zoom_crop_region(self) -> tuple[int, int, int, int]:
        """(x0, y0, cw, ch) of the zoom crop in full-frame pixels.

        Centers on the wheel-zoom focus point if set, else the ROI center,
        else the frame center.
        """
        zoom = self.state.get("zoom", 1.0)
        W, H = IMG_W, IMG_H
        if not zoom or zoom <= 1.0:
            return 0, 0, W, H
        focus = self.state.get("zoom_focus")
        if focus is not None:
            cx, cy = int(focus[0]), int(focus[1])
        else:
            cx, cy = W // 2, H // 2
            roi = self.state["video_roi"]
            if roi is not None:
                cx, cy = (roi[0] + roi[2]) // 2, (roi[1] + roi[3]) // 2
        cw = max(int(W / zoom), 1)
        ch = max(int(H / zoom), 1)
        x0 = max(0, min(cx - cw // 2, W - cw))
        y0 = max(0, min(cy - ch // 2, H - ch))
        return x0, y0, cw, ch

    def _render_video_bgr(self, idx: int) -> tuple[np.ndarray, int, int]:
        """Full (or zoom-cropped) BGR frame with ROI rect drawn.

        Returns (img, crop_ox, crop_oy); img may be a view of FRAME_ARR
        (callers must not mutate it in place).
        """
        roi = self.state["video_roi"]
        zoom = self.state.get("zoom", 1.0)
        full = FRAME_ARR[int(idx)]
        x0, y0, cw, ch = self._zoom_crop_region()
        if zoom and zoom > 1.0:
            img = full[y0:y0 + ch, x0:x0 + cw].copy()
        else:
            img = full

        drew = False
        if roi is not None:
            rx0, ry0, rx1, ry1 = (int(v) for v in roi)
            bx0 = max(rx0 - x0, 0); by0 = max(ry0 - y0, 0)
            bx1 = min(rx1 - x0, img.shape[1]); by1 = min(ry1 - y0, img.shape[0])
            if bx1 > bx0 and by1 > by0:
                if img is full:
                    img = img.copy()
                cv2.rectangle(img, (bx0, by0), (bx1, by1), (0, 255, 0), 2)
                cv2.putText(img, "LED ROI", (bx0, max(by0 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                            cv2.LINE_AA)
                drew = True
        if self.state["roi_click"] is not None and not drew:
            # show the pending first click while drawing the ROI box
            cx, cy = self.state["roi_click"]
            if img is full:
                img = img.copy()
            cv2.circle(img, (cx - x0, cy - y0), 6, (0, 200, 255), -1)
        return img, x0, y0

    def _roi_crop_rgb(self, idx: int):
        """Upscaled RGB crop of the ROI box contents (zoom pane), or None."""
        roi = self.state["video_roi"]
        if roi is None:
            return None
        x0, y0, x1, y1 = (int(v) for v in roi)
        if x1 <= x0 or y1 <= y0:
            return None
        img = FRAME_ARR[int(idx)][y0:y1, x0:x1]
        h = img.shape[0]
        if h > 0:
            scale = min(4.0, 220.0 / max(h, 1))
            if scale > 1.0:
                img = cv2.resize(img, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _update_video_views(self):
        if FRAME_ARR is None:
            return
        idx = self.video_scrub.value()
        bgr, ox, oy = self._render_video_bgr(idx)
        self._crop_ox, self._crop_oy = ox, oy
        src_h, src_w = bgr.shape[:2]
        self.video_view.setSource(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                  src_w, src_h)
        zoom = self._roi_crop_rgb(idx)
        if zoom is not None:
            self.zoom_view.setPixmap(_rgb_to_pixmap(zoom))
        else:
            self.zoom_view.clear()
        self.video_frame_info.setText(
            f"frame {idx}/{max(N_VID - 1, 0)}  t={TIMES[idx]:.3f}s")

    def _update_mocap_views(self):
        if not MOCAP:
            return
        midx = self.mocap_scrub.value()
        box = self._box_arrays()
        self.mocap3d_view.setArray(render_mocap_3d(MOCAP, midx, box))
        img2, info = render_mocap_2d(MOCAP, midx, "top", box,
                                     return_transform=True)
        self.mocap2d_view.setArray(img2)
        self._mocap2d_map = info
        self.mocap_frame_info.setText(
            f"frame {midx}/{max(N_MOC - 1, 0)}  t={midx / FPS_M:.3f}s")

    def _box_arrays(self):
        if self.state["box_lo"] is None:
            return None
        return np.array(self.state["box_lo"]), np.array(self.state["box_hi"])

    def _mocap2d_px_to_data(self, sx: int, sy: int):
        """Map a pixel on the 2D mocap view to (X, Y) data coords (mm)."""
        info = self._mocap2d_map
        if info is None:
            return None
        bx0, by0, bw, bh = info["bbox_img"]
        xl0, xl1 = info["xlim"]
        yl0, yl1 = info["ylim"]
        if bw <= 0 or bh <= 0:
            return None
        fx = (sx - bx0) / bw
        fy = (sy - by0) / bh
        if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
            return None
        dx = xl0 + fx * (xl1 - xl0)
        dy = yl1 - fy * (yl1 - yl0)  # image y grows downward, data y upward
        return dx, dy

    def _on_mocap_box_click(self, sx: int, sy: int):
        """Two clicks on the 2D top view draw the LED box (X-Y rect; Z ± half
        come from the box spinboxes)."""
        if not MOCAP:
            self.sync_info.setText("Load data first (tab 0).")
            return
        xy = self._mocap2d_px_to_data(sx, sy)
        if xy is None:
            self.sync_info.setText("Click inside the 2D plot area.")
            return
        self._box_draw.append(xy)
        if len(self._box_draw) == 1:
            self.sync_info.setText(
                f"Box corner 1: X={xy[0]:.0f}, Y={xy[1]:.0f} mm - "
                "click the opposite corner.")
            return
        (x1, y1), (x2, y2) = self._box_draw
        self._box_draw = []
        zc = self.box_z.value()
        half = self.box_half.value()
        self.state["box_lo"] = [min(x1, x2), min(y1, y2), zc - half]
        self.state["box_hi"] = [max(x1, x2), max(y1, y2), zc + half]
        # sync the spinboxes (block signals so they don't recompute the box)
        for s, v in ((self.box_x, (x1 + x2) / 2), (self.box_y, (y1 + y2) / 2),
                     (self.box_z, zc), (self.box_half, half)):
            s.blockSignals(True)
            s.setValue(v)
            s.blockSignals(False)
        self.sync_info.setText(
            f"Box drawn: X [{min(x1, x2):.0f}, {max(x1, x2):.0f}], "
            f"Y [{min(y1, y2):.0f}, {max(y1, y2):.0f}], "
            f"Z {zc:.0f} +/- {half:.0f} mm -> Compute traces.")
        self._update_mocap_views()

    # ==================================================================
    # Tab 2 handlers
    # ==================================================================
    def _on_video_click(self, x: int, y: int):
        if FRAME_ARR is None:
            self.roi_status.setText("Load data first (tab 0).")
            return
        W, H = IMG_W, IMG_H
        fx = min(x + self._crop_ox, W - 1)
        fy = min(y + self._crop_oy, H - 1)
        if self.state["roi_click"] is None:
            self.state["roi_click"] = (fx, fy)
            self.roi_status.setText("ROI: click the opposite corner")
        else:
            (x0, y0), (x1, y1) = self.state["roi_click"], (fx, fy)
            self.state["video_roi"] = [min(x0, x1), min(y0, y1),
                                       max(x0, x1), max(y0, y1)]
            self.state["roi_click"] = None
            self.roi_status.setText(f"ROI set: {self.state['video_roi']}  "
                                    "-> press 'Compute traces'.")
        self._update_video_views()

    def _on_zoom(self, z: float):
        self.state["zoom"] = float(z)
        self.state.pop("zoom_focus", None)  # spinbox zoom recenters
        self._update_video_views()

    def _on_video_wheel_zoom(self, factor: float, x_in_view: int, y_in_view: int):
        """Middle-mouse scrollwheel zoom around the cursor position."""
        if FRAME_ARR is None:
            return
        z = float(self.state.get("zoom", 1.0)) * factor
        z = max(1.0, min(8.0, z))
        self.state["zoom"] = z
        # keep the point under the cursor fixed while zooming
        self.state["zoom_focus"] = (
            min(self._crop_ox + x_in_view, IMG_W - 1),
            min(self._crop_oy + y_in_view, IMG_H - 1),
        )
        self.zoom_spin.blockSignals(True)
        self.zoom_spin.setValue(z)
        self.zoom_spin.blockSignals(False)
        self._update_video_views()

    def _on_reset_roi(self):
        self.state["video_roi"] = None
        self.state["roi_click"] = None
        self.state.pop("zoom_focus", None)
        self.roi_status.setText("ROI cleared. Click 2 points on the video to "
                                "set a new one.")
        self._update_video_views()

    def _on_nudge(self, corner: str, axis: str, delta: int):
        roi = self.state["video_roi"]
        if roi is None:
            self.roi_status.setText("Draw the ROI box first (2 clicks on the video).")
            return
        x0, y0, x1, y1 = (int(v) for v in roi)
        if corner == "tl":
            if axis == "x":
                x0 += delta
            else:
                y0 += delta
        elif corner == "tr":
            if axis == "x":
                x1 += delta
            else:
                y0 += delta
        elif corner == "bl":
            if axis == "x":
                x0 += delta
            else:
                y1 += delta
        else:  # br
            if axis == "x":
                x1 += delta
            else:
                y1 += delta
        W, H = IMG_W, IMG_H
        x0 = max(0, min(x0, W)); x1 = max(0, min(x1, W))
        y0 = max(0, min(y0, H)); y1 = max(0, min(y1, H))
        if x0 == x1 or y0 == y1:
            self.roi_status.setText("Box collapsed - keep a positive size.")
            return
        self.state["video_roi"] = [min(x0, x1), min(y0, y1),
                                   max(x0, x1), max(y0, y1)]
        self.state["roi_click"] = None
        self.roi_status.setText(f"ROI = {self.state['video_roi']}")
        self._update_video_views()

    def _on_auto_led(self):
        if not MOCAP:
            self.sync_info.setText("Load data first (tab 0).")
            return
        self._box_draw = []
        led = auto_find_led(MOCAP)
        if led is None:
            self.sync_info.setText("No stationary blinking marker found - "
                                   "set the box manually.")
            return
        pos = led["position_mm"]
        half = 50.0
        self.state["box_lo"] = (pos - half).tolist()
        self.state["box_hi"] = (pos + half).tolist()
        self.box_x.setValue(pos[0]); self.box_y.setValue(pos[1])
        self.box_z.setValue(pos[2]); self.box_half.setValue(half)
        self._update_mocap_views()
        self.sync_info.setText(
            f"Auto box around '{led['name']}' at ({pos[0]:.0f}, {pos[1]:.0f}, "
            f"{pos[2]:.0f}) mm, size +/-{half:.0f} mm")

    def _on_box_change(self):
        self._box_draw = []   # switching to manual/spin control cancels drawing
        x, y, z, h = (self.box_x.value(), self.box_y.value(),
                      self.box_z.value(), self.box_half.value())
        self.state["box_lo"] = [x - h, y - h, z - h]
        self.state["box_hi"] = [x + h, y + h, z + h]

    def _on_box_release(self):
        # re-render mocap only on slider release (3D render is the slow part)
        if MOCAP:
            self._update_mocap_views()

    def _on_compute_traces(self):
        if FRAME_ARR is None:
            self.sync_info.setText("Load data first (tab 0).")
            return
        if self.state["video_roi"] is None:
            self.sync_info.setText("Set the video ROI first (2 clicks).")
            return
        trace, _ = compute_video_trace(tuple(self.state["video_roi"]),
                                       video_path=CURRENT_VIDEO_PATH)
        self.state["video_trace"] = trace.tolist()
        self.state["video_bin"] = threshold_trace(
            trace, self.state["threshold"]).tolist()
        box = self._box_arrays()
        if box is None:
            self.state["mocap_bin"] = None
            self.sync_info.setText(
                f"Video trace done: ROI intensity {trace.min():.0f}.."
                f"{trace.max():.0f} (0-255), threshold "
                f"{self.state['threshold']:.0f}. Set the mocap 3D box to "
                f"compute its trace.")
        else:
            self.state["mocap_bin"] = compute_mocap_trace(
                MOCAP, box[0], box[1]).tolist()
            msg = (f"Both traces computed (video ROI intensity "
                   f"{trace.min():.0f}..{trace.max():.0f}, threshold "
                   f"{self.state['threshold']:.0f}). Align the pulses with "
                   f"'Offset fine-tune', then Save.")
            extra = self._trace_warnings()
            if extra:
                msg += "\n\n" + "\n".join("  ! " + w for w in extra)
            self.sync_info.setText(msg)
        self._refresh_traces()

    def _trace_warnings(self) -> list[str]:
        """Flag degenerate binary traces (a blink needs both traces to
        actually toggle, otherwise manual pulse alignment is impossible)."""
        warns = []
        vb = self.state.get("video_bin")
        if vb is not None:
            frac = float(np.asarray(vb).mean())
            if frac == 0.0:
                warns.append("video binary is ALL 0 - the LED never crosses the "
                             "threshold (ROI off-target or threshold too high)")
            elif frac == 1.0:
                warns.append("video binary is ALL 1 - the ROI is always above the "
                             "threshold; RAISE 'Video threshold' until the LED's "
                             "OFF state drops below it")
        mb = self.state.get("mocap_bin")
        if mb is not None:
            frac = float(np.asarray(mb).mean())
            if frac == 0.0:
                warns.append("mocap binary is ALL 0 - no marker inside the 3D box; "
                             "move/resize the box around the blinking LED")
            elif frac == 1.0:
                warns.append("mocap binary is ALL 1 - a marker is ALWAYS inside the "
                             "3D box; shrink it around just the blinking LED")
        return warns

    def _on_threshold(self, v: int):
        self.state["threshold"] = float(v)
        self.thr_value.setText(f"{v}")
        if self.state["video_trace"] is not None:
            self.state["video_bin"] = threshold_trace(
                np.array(self.state["video_trace"]), float(v)).tolist()
        self._refresh_traces()
        if self.state.get("video_bin") is not None or \
                self.state.get("mocap_bin") is not None:
            extra = self._trace_warnings()
            if extra:
                msg = "Threshold updated.\n\n" + \
                    "\n".join("  ! " + w for w in extra)
            else:
                msg = (f"Threshold updated to {v}. Video binary now toggles "
                       f"(LED on/off detected) - align pulses with 'Offset "
                       f"fine-tune', then Save.")
            self.sync_info.setText(msg)

    def _on_offset(self, v: float):
        self.state["offset"] = float(v)
        self._refresh_traces()

    def _on_invert_offset(self):
        """Flip the sign of the offset (useful when the pulses align on the
        opposite side of the shared axis)."""
        self.offset_spin.setValue(-self.offset_spin.value())

    def _on_save_sync(self):
        if self.state["video_bin"] is None or self.state["mocap_bin"] is None:
            self.sync_info.setText("Compute both traces first.")
            return
        save_sync(self.state["offset"], 0.0, self.state["video_roi"],
                  self.state["box_lo"], self.state["box_hi"],
                  self.state["threshold"])
        self.sync_info.setText(
            f"Saved output/sync.json  offset={self.state['offset']:.3f} s")

    def _on_load_sync(self):
        """Restore the saved sync (offset, ROI, mocap box, threshold) from
        output/sync.json and recompute the traces - resume a session."""
        s = load_sync()
        if s is None:
            self.sync_info.setText("No output/sync.json yet - do the sync "
                                   "once and click 'Save sync' first.")
            return
        off = float(s.get("offset_s", 0.0))
        roi = s.get("video_roi")
        blo = s.get("mocap_box_lo_mm")
        bhi = s.get("mocap_box_hi_mm")
        thr = float(s.get("video_threshold", 128.0))
        self.state["offset"] = off
        self.state["video_roi"] = list(roi) if roi else None
        self.state["box_lo"] = list(blo) if blo else None
        self.state["box_hi"] = list(bhi) if bhi else None
        self.state["threshold"] = thr
        # widgets (signals blocked so nothing recomputes mid-restore)
        for wdg, v in ((self.offset_spin, off), (self.thr_slider, int(thr))):
            wdg.blockSignals(True)
            wdg.setValue(v)
            wdg.blockSignals(False)
        self.thr_value.setText(str(int(thr)))
        if blo and bhi:
            lo = np.array(blo)
            hi = np.array(bhi)
            for wdg, v in ((self.box_x, (lo[0] + hi[0]) / 2.0),
                           (self.box_y, (lo[1] + hi[1]) / 2.0),
                           (self.box_z, (lo[2] + hi[2]) / 2.0),
                           (self.box_half, (hi[0] - lo[0]) / 2.0)):
                wdg.blockSignals(True)
                wdg.setValue(v)
                wdg.blockSignals(False)
        # re-render views, then recompute traces from the restored settings
        if FRAME_ARR is not None:
            self._update_video_views()
        if MOCAP:
            self._update_mocap_views()
        self.state["video_trace"] = None
        self.state["video_bin"] = None
        self.state["mocap_bin"] = None
        self._on_compute_traces()
        self.sync_info.setText(
            f"Loaded output/sync.json  offset={off:.3f}s, "
            f"threshold={thr:.0f}, ROI={roi}, box={blo}..{bhi}\n"
            "Traces recomputed - tweak if needed, then 'Save sync' to keep.")
        self._update_proj_view()   # projection follows the restored offset

    def _on_video_scrub_change(self, _v: int):
        """Video scrub slider moved: update its view + bar."""
        self._update_video_views()
        self._refresh_traces()

    def _on_mocap_scrub_change(self, _v: int):
        """Mocap scrub slider moved: update its view + bar."""
        self._update_mocap_views()
        self._refresh_traces()

    def _refresh_traces(self):
        """Redraw the traces plot with the two scrub cursor bars (they track
        the video + mocap scrub sliders).

        The x-axis is FIXED to the union of both recordings' time spans
        (video [0, dur]; mocap [offset, dur + offset]) so moving the bars
        never rescales the plot - only the bars move.
        """
        vt = mt = None
        xlo, xhi = 0.0, 1.0
        if FRAME_ARR is not None and len(TIMES):
            vt = float(TIMES[self.video_scrub.value()])
            xlo, xhi = 0.0, float(TIMES[-1])
        if MOCAP:
            off = float(self.state["offset"])
            mt = self.mocap_scrub.value() / FPS_M + off
            m_end = N_MOC / FPS_M + off
            xlo = min(xlo, off)
            xhi = max(xhi, m_end)
        pad = max((xhi - xlo) * 0.02, 0.1)   # small margin so edge bars are visible
        xlo -= pad
        xhi += pad
        self.traces.update_traces(self.state, TIMES, FPS_M,
                                  video_cursor_t=vt, mocap_cursor_t=mt,
                                  xlim=(xlo, xhi))

    def _on_offset_from_bars(self):
        """Set offset = (video bar time) - (mocap bar time): line the two
        bars up on the same blink, click this, and the sync is locked."""
        if FRAME_ARR is None or not MOCAP:
            self.sync_info.setText("Load data first (tab 0).")
            return
        tv = float(TIMES[self.video_scrub.value()])
        tm = self.mocap_scrub.value() / FPS_M
        self.offset_spin.setValue(tv - tm)
        self.sync_info.setText(
            f"Offset set from bars: video {tv:.3f}s - mocap {tm:.3f}s = "
            f"{tv - tm:.3f} s. The mocap (purple) bar should now sit on the "
            f"video (blue) bar - use the sync scrub to verify.")

    def _on_sync_scrub(self, v: int):
        """Jog BOTH scrub sliders by the sync-slider delta (centi-seconds).

        The slider is center-zero (0 = no shift): dragging right moves both
        forward, left moves both backward.  Movement is blocked when either
        slider would leave its recording (end-frame clamp).  On release the
        jog recentres (see _sync_scrub_release).
        """
        if FRAME_ARR is None or not MOCAP:
            return
        delta = (v - self._sync_last) / 100.0        # seconds
        self._sync_last = v
        self.sync_scrub_label.setText(f"{v / 100.0:+.2f}s")
        if delta == 0.0:
            return
        tv = float(TIMES[self.video_scrub.value()]) + delta
        tm = self.mocap_scrub.value() / FPS_M + delta
        if tv < 0.0 or tv > TIMES[-1] or tm < 0.0 or tm > N_MOC / FPS_M:
            # either would leave its recording -> don't move either
            self.sync_info.setText(
                f"Sync jog {delta:+.2f}s blocked: video would reach "
                f"{tv:.2f}s / mocap {tm:.2f}s (out of range).")
            return
        vi = int(np.argmin(np.abs(TIMES - tv)))
        mi = int(round(tm * FPS_M))
        self.video_scrub.setValue(vi)    # -> _on_video_scrub_change (view+bar)
        self.mocap_scrub.setValue(mi)    # -> _on_mocap_scrub_change (view+bar)
        self.sync_info.setText(
            f"Sync jog {delta:+.2f}s -> video {tv:.2f}s, mocap {tm:.2f}s")

    def _sync_scrub_release(self):
        """Recentre the jog slider after a drag so the next drag can go in
        either direction (blocked so it doesn't re-apply the delta)."""
        self.sync_scrub.blockSignals(True)
        self.sync_scrub.setValue(0)
        self.sync_scrub.blockSignals(False)
        self._sync_last = 0
        self.sync_scrub_label.setText("0.00s")

    # ==================================================================
    # Tab 3 handlers
    # ==================================================================
    def _on_trim_preview(self):
        if not MOCAP:
            self.trim_info.setText("Load data first (tab 0).")
            return
        res = apply_trim(TIMES, MOCAP, self.state["offset"],
                         self.trim_start.value(), self.trim_end.value())
        self.trim_info.setText(
            f"start {res['start_s']:.2f}s  end {res['end_s']:.2f}s  "
            f"duration {res['duration_s']:.2f}s\n"
            f"video frames {res['video_first']}..{res['video_last']} "
            f"({res['video_frames']})\n"
            f"mocap frames {res['mocap_first']}..{res['mocap_last']} "
            f"({res['mocap_frames']})\nvalid: {res['valid']}")

    def _on_trim_save(self):
        save_trim(self.trim_start.value(), self.trim_end.value())
        self.trim_info.setText(
            f"Saved output/trim.json  {self.trim_start.value():.2f}s .. "
            f"{self.trim_end.value():.2f}s")

    def _on_trim_load(self):
        t = load_trim()
        if t is None:
            self.trim_info.setText("No output/trim.json yet - set start/end "
                                   "and click 'Save trim' first.")
            return
        self.trim_start.setValue(t["start_s"])
        self.trim_end.setValue(t["end_s"])
        self.trim_info.setText(
            f"Loaded output/trim.json  {t['start_s']:.2f}s .. {t['end_s']:.2f}s")

    # ==================================================================
    # Tab 4 handler
    # ==================================================================
    def _cal_progress(self, done: int, total: int, msg: str = "") -> None:
        """Progress callback for the long calibration loops."""
        if total > 0:
            self.cal_progress.setRange(0, total)
            self.cal_progress.setValue(int(done))
        if msg:
            self.cal_out.setText(msg)
        QApplication.processEvents()

    def _on_calibrate_intrinsics(self):
        """Stage A: self-calibrate the camera matrix from the chessboard video."""
        if FRAME_ARR is None:
            self.cal_out.setText("Load data first (tab 0).")
            return
        try:
            self.cal_progress.setRange(0, 1)
            self.cal_progress.setValue(0)
            self.cal_out.setText("Detecting chessboard and calibrating "
                                 "intrinsics - this may take a minute...")
            r = calibrate_intrinsics(video_path=CURRENT_VIDEO_PATH,
                                     progress_cb=self._cal_progress)
            self.cal_progress.setValue(self.cal_progress.maximum())
            lines = [
                f"Intrinsics from {r['n_detected_frames']}/"
                f"{r['n_total_frames']} chessboard frames",
                f"resolution: {r['resolution'][0]}x{r['resolution'][1]}",
                f"RMS (cv2): {r['rms']:.4f} px",
                f"mean reprojection: {r['mean_reproj_px']:.3f} px   "
                f"max: {r['max_reproj_px']:.3f} px",
                "camera matrix (K):",
                np.array2string(np.round(np.array(r["camera_matrix"]), 2)),
                "distortion: " + np.array2string(
                    np.round(np.array(r["dist_coeffs"]), 5)),
                "Saved output/intrinsics.json",
            ]
            self.cal_out.setText("\n".join(lines))
        except Exception as e:
            self.cal_out.setText(f"ERROR: {e}")

    def _render_preview_frame(self, vi, R, t, K, dist, offset):
        """Overlay the chessboard corners (green dots), the board-pose
        markers (green rings) and the mocap markers projected through
        transform.json (red rings) on video frame ``vi``.

        Returns (img BGR, caption_line) or (None, error_line).
        """
        img = (FRAME_ARR[vi] if FRAME_ARR is not None else
               cv2.imread(str(config.FRAME_DIR / f"{vi:06d}.jpg")))
        if img is None:
            return None, f"frame {vi}: no image"
        img = img.copy()
        tv = float(TIMES[vi])
        mi = mocap_index_for_video_time(tv, offset, FPS_M)
        if mi < 0 or mi >= N_MOC:
            return None, f"frame {vi}: mocap frame {mi} out of range"
        label_idx = {n: i for i, n in enumerate(MOCAP["labels"])}
        try:
            mi_list = [label_idx[n] for n in config.MARKER_NAMES]
        except KeyError as e:
            return None, f"frame {vi}: marker {e} not in mocap labels"
        if not all(MOCAP["presence"][j, mi] for j in mi_list):
            return None, f"frame {vi}: markers not all tracked at mocap {mi}"

        # chessboard corners (small green dots) for context
        ok, corners = detect_board(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if not ok:
            return None, f"frame {vi}: no chessboard detected"
        for c in corners.reshape(-1, 2):
            cv2.circle(img, (int(c[0]), int(c[1])), 3, (0, 220, 0), -1)

        # green rings = board-pose markers (solvePnP + marker layout)
        Rb, tb = solve_board_pose(corners, K, dist)
        mm = load_markers_mm()
        proj_b, _ = cv2.projectPoints(mm, cv2.Rodrigues(Rb)[0], tb, K, dist)
        for c in proj_b[:, 0, :]:
            cv2.circle(img, (int(c[0]), int(c[1])), 12, (0, 220, 0), 2)

        # red rings = mocap markers projected via transform.json
        Pm = MOCAP["xyz"][mi_list, :, mi]  # (6,3) mocap frame
        Pc = (R @ Pm.T + t[:, None]).T     # (6,3) camera frame
        proj, _ = cv2.projectPoints(Pc.astype(np.float64),
                                    np.zeros(3), np.zeros(3), K, dist)
        for k, c in enumerate(proj[:, 0, :]):
            x, y = int(c[0]), int(c[1])
            cv2.circle(img, (x, y), 12, (0, 0, 255), 2)
            cv2.putText(img, config.MARKER_NAMES[k], (x + 14, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                        cv2.LINE_AA)
        return img, f"frame {vi}  video {tv:.2f}s  mocap {mi} ({mi / FPS_M:.2f}s)"

    def _on_preview_verify(self):
        """Project the mocap Board markers onto a few random trim-window
        frames using transform.json - a visual sanity check.

        Red rings = Board1..6 projected from mocap via transform.json.
        Green rings = where the board-pose (solvePnP) puts them.
        With a good calibration + correct sync they coincide on the board.
        """
        if not MOCAP:
            self.cal_out.setText("Load data first (tab 0).")
            return
        if not config.TRANSFORM_FILE.exists():
            self.cal_out.setText("output/transform.json missing - run "
                                 "'2) Run calibration' first.")
            return
        if not config.INTRINSICS_FILE.exists():
            self.cal_out.setText("output/intrinsics.json missing - run "
                                 "'1) Calibrate intrinsics' first.")
            return
        try:
            tf = json.loads(config.TRANSFORM_FILE.read_text())
            intr = json.loads(config.INTRINSICS_FILE.read_text())
            sync = json.loads(config.SYNC_FILE.read_text())
            trim = json.loads(config.TRIM_FILE.read_text())
        except Exception as e:
            self.cal_out.setText(f"ERROR reading saved files: {e}")
            return
        R = np.array(tf["rotation"])
        t = np.array(tf["translation_mm"])
        K = np.array(intr["camera_matrix"])
        dist = np.array(intr["dist_coeffs"])
        offset = float(sync.get("offset_s", 0.0))

        # all trim-window frames where the chessboard is detected
        lo = max(0, int(np.searchsorted(TIMES, trim["start_s"])))
        hi = min(len(TIMES) - 1, int(np.searchsorted(TIMES, trim["end_s"])))
        good = []
        for cand in range(lo, hi + 1):
            img0 = FRAME_ARR[cand] if FRAME_ARR is not None else \
                cv2.imread(str(config.FRAME_DIR / f"{cand:06d}.jpg"))
            if img0 is None:
                continue
            ok, _ = detect_board(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY))
            if ok:
                good.append(cand)
        if not good:
            self.cal_out.setText("No chessboard found in the trim window - "
                                 "nothing to preview.")
            return

        # sample up to 3 random frames (sorted so panels run left->right)
        if len(good) <= 3:
            picks = list(good)
        else:
            rng = np.random.default_rng()
            picks = sorted(good[i] for i in
                           rng.choice(len(good), size=3, replace=False))

        shown = []
        for i, lbl in enumerate(self.cal_previews):
            if i < len(picks):
                img, line = self._render_preview_frame(
                    picks[i], R, t, K, dist, offset)
                if img is not None:
                    lbl.setArray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    shown.append(line)
                else:
                    lbl.clear()
                    shown.append(line)
            else:
                lbl.clear()

        self.cal_preview_cap.setText(
            "\n".join(shown) + "\n"
            "green dots = chessboard; green rings = board-pose markers; "
            "red rings = mocap markers via transform.json\n"
            "Red and green should coincide on the chessboard border.")

    def _on_calibrate(self):
        # Stage A prerequisite: intrinsics must exist (self-calibrate if not)
        self.cal_progress.setRange(0, 1)
        self.cal_progress.setValue(0)
        if not config.INTRINSICS_FILE.exists():
            self.cal_out.setText("output/intrinsics.json missing - "
                                 "calibrating from the chessboard video first...")
            try:
                calibrate_intrinsics(video_path=CURRENT_VIDEO_PATH,
                                     progress_cb=self._cal_progress)
            except Exception as e:
                self.cal_out.setText(f"ERROR calibrating intrinsics: {e}")
                return
        missing = [p.name for p in (config.MARKERS_FILE, config.SYNC_FILE,
                                    config.TRIM_FILE)
                   if not p.exists()]
        if missing:
            how = {
                "markers.json": "tab 1: place all 6 markers, then click "
                                "'Save markers'",
                "sync.json": "tab 2: align the bars / set the offset, then "
                              "click 'Save sync'",
                "trim.json": "tab 3: set start/end, then click 'Save trim'",
            }
            self.cal_out.setText(
                "Missing required files: " + ", ".join(missing) + "\n" +
                "\n".join(f"  {n}: {how[n]}" for n in missing))
            return
        try:
            r = run_calibration(video_path=CURRENT_VIDEO_PATH,
                                c3d_path=CURRENT_C3D_PATH,
                                progress_cb=self._cal_progress)
            self.cal_progress.setValue(self.cal_progress.maximum())
            lines = [
                f"Used {r['n_frames_used']} frames, "
                f"{r['n_correspondences']} marker correspondences",
                f"mean residual: {r['mean_residual_mm']:.2f} mm",
                f"median: {r['median_residual_mm']:.2f} mm   "
                f"p95: {r['p95_residual_mm']:.2f} mm   "
                f"max: {r['max_residual_mm']:.2f} mm",
                "R =",
                np.array2string(np.round(np.array(r["rotation"]), 5),
                                precision=5),
                "t (mm) = " + np.array2string(
                    np.round(np.array(r["translation_mm"]), 2)),
                "Saved output/transform.json",
            ]
            self.cal_out.setText("\n".join(lines))
        except Exception as e:
            self.cal_out.setText(f"ERROR: {e}")

    # ==================================================================
    # Tab 3: Project
    # ==================================================================
    def _on_load_calib(self):
        """Load the calibration (intrinsics + mocap->camera transform) saved
        by pc_calib_tool - applies to this same-session take."""
        missing = [str(f) for f in (config.INTRINSICS_FILE,
                                    config.TRANSFORM_FILE)
                   if not f.exists()]
        if missing:
            self._calib = None
            self.calib_status.setText(
                "Missing calibration file(s):\n" + "\n".join(missing) +
                "\nRun the calibration tool's '2) Run calibration' first.")
            self._update_proj_view()
            return
        try:
            intr = json.loads(config.INTRINSICS_FILE.read_text())
            tf = json.loads(config.TRANSFORM_FILE.read_text())
        except Exception as e:
            self._calib = None
            self.calib_status.setText(f"ERROR reading calibration: {e}")
            self._update_proj_view()
            return
        self._calib = {
            "R": np.array(tf["rotation"], float),
            "t": np.array(tf["translation_mm"], float),
            "K": np.array(intr["camera_matrix"], float),
            "dist": np.array(intr["dist_coeffs"], float),
        }
        med = tf.get("median_residual_mm")
        self.calib_status.setText(
            f"Calibration loaded ✓  (median residual "
            f"{med:.2f} mm if available)\n"
            f"t (mm) = {np.round(self._calib['t'], 1).tolist()}")
        self._update_proj_view()

    def _marker_color(self, name: str) -> tuple[int, int, int]:
        """Stable BGR color per marker (auto/LED markers get yellow)."""
        if name.startswith("*"):
            return (0, 200, 255)      # yellow-ish
        i = sum(ord(ch) for ch in name)
        pal = [(60, 60, 255), (60, 200, 60), (255, 120, 60),
               (200, 60, 255), (60, 220, 220), (255, 60, 120),
               (255, 255, 120), (200, 200, 200)]
        return pal[i % len(pal)]

    def _render_proj_overlay(self, vi: int):
        """Full-res BGR frame at vi with every mocap marker projected."""
        img = (FRAME_ARR[vi] if FRAME_ARR is not None else
               cv2.imread(str(config.FRAME_DIR / f"{vi:06d}.jpg")))
        if img is None:
            return None
        img = img.copy()
        if self._calib is None or not MOCAP:
            return img
        c = self._calib
        mi = mocap_index_for_video_time(float(TIMES[vi]), self.state["offset"],
                                        FPS_M)
        if not (0 <= mi < N_MOC):
            return img
        H, W = img.shape[:2]
        for j, name in enumerate(MOCAP["labels"]):
            if not MOCAP["presence"][j, mi]:
                if not self.proj_all_cb.isChecked():
                    continue
            P = MOCAP["xyz"][j, :, mi]
            if not np.isfinite(P).all():
                continue
            Pc = (c["R"] @ P + c["t"]).reshape(1, 1, 3).astype(np.float64)
            proj, _ = cv2.projectPoints(Pc, np.zeros(3), np.zeros(3),
                                        c["K"], c["dist"])
            x, y = int(round(proj[0, 0, 0])), int(round(proj[0, 0, 1]))
            if not (0 <= x < W and 0 <= y < H):
                continue
            col = self._marker_color(name)
            cv2.circle(img, (x, y), 10, col, 2)
            if self.proj_labels_cb.isChecked():
                cv2.putText(img, name, (x + 13, y - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        return img

    def _set_proj_range(self):
        """Scrub range: full video, or just the trim window."""
        if FRAME_ARR is None:
            return
        if self.proj_trim_cb.isChecked():
            t = load_trim()
            if t is not None:
                lo = max(0, int(np.searchsorted(TIMES, t["start_s"])))
                hi = min(N_VID - 1, int(np.searchsorted(TIMES, t["end_s"])))
                self.proj_scrub.setRange(lo, hi)
                self.proj_scrub.setValue(lo)
                return
        self.proj_scrub.setRange(0, max(N_VID - 1, 0))

    def _on_proj_scrub(self, _v: int):
        self._update_proj_view()

    def _on_proj_mocap_scrub(self, _v: int):
        if MOCAP:
            midx = self.proj_mocap_scrub.value()
            self.proj_mocap3d.setArray(render_mocap_3d(MOCAP, midx, None))
            self.proj_mocap_info.setText(
                f"mocap {midx}/{max(N_MOC - 1, 0)}  t={midx / FPS_M:.2f}s")

    def _update_proj_view(self):
        if FRAME_ARR is None:
            return
        vi = self.proj_scrub.value()
        img = self._render_proj_overlay(vi)
        if img is not None:
            self.proj_view.setArray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        mi = mocap_index_for_video_time(float(TIMES[vi]), self.state["offset"],
                                        FPS_M)
        cal_txt = "cal: ✓" if self._calib is not None else "cal: -"
        self.proj_info.setText(
            f"frame {vi}/{max(N_VID - 1, 0)}  t={TIMES[vi]:.3f}s  "
            f"mocap {mi} ({mi / FPS_M:.2f}s)  off={self.state['offset']:.3f}  "
            f"{cal_txt}")
        if MOCAP and 0 <= mi < N_MOC:
            self.proj_mocap_scrub.blockSignals(True)
            self.proj_mocap_scrub.setValue(mi)
            self.proj_mocap_scrub.blockSignals(False)
            self._on_proj_mocap_scrub(0)


def main():
    _fix_qt_plugin_path()
    app = QApplication(sys.argv)
    _strip_cv2_plugin_paths(app)
    app.setApplicationName("project_tool")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
