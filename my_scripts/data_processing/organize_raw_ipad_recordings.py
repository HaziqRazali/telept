#!/usr/bin/env python3
"""
organize_raw_ipad_recordings.py

Organize raw TelePT iPad session dumps into the fit3d-compatible directory layout
under <out_root>/<split>/ (default: data/NUS/val). This script only COPIES
from the raw repository -- it never deletes, moves, or modifies anything under
the raw root.

Expected raw layout (one directory per recording session; rgbd + mocap dumped
together into the same session dir):
    <raw_root>/session_<id>/
        rgb.mp4 | *.mp4          # ipad rgb video
        depth.zip | depth/       # ipad depth (archive or already-extracted frames)
        calibration.json         # ipad calibration (intrinsics + distortion LUTs)
        manifest.json            # ipad recording manifest (optional)
        recording_stats.json     # (optional)
        *.c3d  *.vsk  *.mp ...   # mocap trial(s) + Vicon sidecars

Output layout (mirrors fit3d: fit3d/train/<subject>/...):
    <out_root>/<split>/<subject>/
        videos/<session_id>/<trial>.mp4              # run.sh consumes this
        camera_parameters/<session_id>/<trial>.json  # fit3d-style (best effort)
        camera_parameters/<session_id>/calibration.json|manifest.json  # provenance
        depth/<session_id>/depth.zip                 # additive (not in fit3d)
        mocap/<trial>.c3d ...                        # additive (extraction later)
        session_info.json                            # provenance + mapping

Naming rules:
    subject : --subject, else the common prefix of the .c3d stems
              (e.g. "haziq_upperlimb_right_24082026 dynamic 03.c3d" -> "haziq_upperlimb_right_24082026"),
              else the session id.
    trial   : the non-"Cal*" .c3d stem (spaces -> underscores), else the video
              stem, else "rgb".  The video is renamed to <trial>.mp4 so the
              sam3d outputs come out as <trial>_mhr_outputs.npz (pairing with the mocap).

Usage:
    python organize_raw_ipad_recordings.py \
        --raw-root data/NUS/raw \
        --out-root data/NUS \
        --split val

    python organize_raw_ipad_recordings.py \
            --raw-root data/NUS/raw \
            --out-root data/NUS \
            --split val

    # optional: --subject haziq_upperlimb  --force  --with-intermediates
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

# Vicon sidecar files that are regenerable intermediates -> skipped unless
# --with-intermediates. Everything else under the session dir is copied.
INTERMEDIATE_EXTS = (".x1d", ".x2d")

# mocap files that carry meaning (kept in val/mocap/); everything else with a
# mocap-ish extension is also copied as provenance.
MOCAP_EXTS = (".c3d", ".vsk", ".mp", ".system", ".xcp", ".history",
             ".Trial.enf", ".x1d", ".x2d")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def is_mocap(f: str) -> bool:
    return any(f.lower().endswith(e) for e in MOCAP_EXTS)


def is_intermediate(f: str) -> bool:
    return any(f.lower().endswith(e) for e in INTERMEDIATE_EXTS)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def natural_key(s: str):
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else -1


def _stem(name: str) -> str:
    return os.path.splitext(name)[0]


def sanitize(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip()).strip("_")


def derive_subject(c3d_stems: list[str], session_id: str) -> str:
    """Common prefix of c3d stems -> subject name (fit3d uses <subj>)."""
    if not c3d_stems:
        return session_id
    # e.g. "haziq_upperlimb_right_24082026 dynamic 03" -> "haziq_upperlimb_right_24082026"
    subj = c3d_stems[0]
    for tok in (" dynamic ", " Cal ", "_dynamic ", "_Cal "):
        if tok in subj:
            subj = subj.split(tok)[0]
            break
    return sanitize(subj) or session_id


def derive_trial(c3d_stems: list[str], video_stem: str | None) -> str:
    """Pick the motion trial name (prefer the 'dynamic' c3d, then any non-Cal)."""
    for s in c3d_stems:
        m = re.search(r"\s+(dynamic\s+\S+)\s*$", s, re.I)
        if m:
            return sanitize(m.group(1))
    for s in c3d_stems:
        if " cal " not in s.lower():
            return sanitize(s)
    if c3d_stems:
        return sanitize(c3d_stems[0])
    return sanitize(video_stem) if video_stem else "rgb"


def _copy(src: str, dst: str, force: bool) -> bool:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and not force:
        return False
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return True


def write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# --------------------------------------------------------------------------
# fit3d-style camera parameters (best effort from the ipad calibration.json)
# --------------------------------------------------------------------------

def to_fit3d_cam(calib: dict) -> dict | None:
    """Map ipad calibration.json -> fit3d {extrinsics, intrinsics_w_distortion, ...}.

    The ipad distortion is stored as lookup tables, not polynomial coefficients,
    so k/p are set to 0 and the reference dimensions are recorded. Intrinsics
    are the *reference* intrinsics (full sensor) -- scaling to the actual video
    resolution is left to the consumer.
    """
    K = calib.get("intrinsic_matrix")
    if not K or len(K) < 3:
        return None
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]
    E = calib.get("extrinsic_matrix")
    if E and len(E) >= 3 and len(E[0]) >= 4:
        # ipad stores a 3x4 homogeneous extrinsic (R|t)
        R = [row[:3] for row in E[:3]]
        T = [[row[3] for row in E[:3]]]
    else:
        R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        T = [[0.0, 0.0, 0.0]]
    ref = calib.get("intrinsic_reference_dimensions", {})
    return {
        "extrinsics": {"R": R, "T": T},
        "intrinsics_w_distortion": {
            "f": [[fx, fy]],
            "c": [[cx, cy]],
            "k": [[0.0, 0.0, 0.0]],
            "p": [[0.0, 0.0]],
        },
        "intrinsics_wo_distortion": {"c": [cx, cy], "f": [fx, fy]},
        "note": "best-effort from ipad calibration.json; distortion is LUT-based so k/p=0; "
                "intrinsics are reference-dimension (not scaled to the video); "
                "see calibration.json for full LUTs",
        "intrinsic_reference_dimensions": ref,
        "source": "calibration.json",
    }


# --------------------------------------------------------------------------
# per-session organizer
# --------------------------------------------------------------------------

def organize_session(session_dir: str, session_id: str, out_root: str,
                     split: str, subject: str | None, force: bool,
                     with_intermediates: bool) -> dict:
    files = [f for f in os.listdir(session_dir) if os.path.isfile(os.path.join(session_dir, f))]
    files.sort(key=natural_key)

    videos = [f for f in files if os.path.splitext(f)[1].lower() in VIDEO_EXTS]
    c3ds = [f for f in files if f.lower().endswith(".c3d")]
    mocap_files = [f for f in files if is_mocap(f)]
    depth_zips = [f for f in files if f.lower().endswith(".zip") and "depth" in f.lower()]

    subj = subject or derive_subject([_stem(f) for f in c3ds], session_id)
    trial = derive_trial([_stem(f) for f in c3ds], _stem(videos[0]) if videos else None)

    base = os.path.join(out_root, split, subj)
    cam = session_id
    copied, skipped = [], []

    def put(rel_dst: str, src: str) -> None:
        dst = os.path.join(base, rel_dst)
        if _copy(src, dst, force):
            copied.append(os.path.relpath(dst, out_root))
        else:
            skipped.append(os.path.relpath(dst, out_root))

    # --- videos -----------------------------------------------------------
    if videos:
        src_vid = os.path.join(session_dir, videos[0])
        put(os.path.join("videos", cam, f"{trial}.mp4"), src_vid)

    # --- camera parameters --------------------------------------------------
    calib = None
    calib_path = os.path.join(session_dir, "calibration.json")
    if os.path.exists(calib_path):
        put(os.path.join("camera_parameters", cam, "calibration.json"), calib_path)
        try:
            with open(calib_path) as f:
                calib = json.load(f)
        except Exception as e:
            print(f"[warn] could not parse {calib_path}: {e}")
    for prov in ("manifest.json", "recording_stats.json"):
        p = os.path.join(session_dir, prov)
        if os.path.exists(p):
            put(os.path.join("camera_parameters", cam, prov), p)
    if calib is not None:
        fit3d_cam = to_fit3d_cam(calib)
        if fit3d_cam is not None:
            write_json(os.path.join(base, "camera_parameters", cam, f"{trial}.json"), fit3d_cam)

    # --- depth --------------------------------------------------------------
    for z in depth_zips:
        put(os.path.join("depth", cam, z), os.path.join(session_dir, z))
    depth_dir = os.path.join(session_dir, "depth")
    if os.path.isdir(depth_dir):
        put(os.path.join("depth", cam, "frames"), depth_dir)

    # --- mocap ---------------------------------------------------------------
    for f in mocap_files:
        if is_intermediate(f) and not with_intermediates:
            print(f"[skip-intermediate] {f}")
            continue
        put(os.path.join("mocap", f), os.path.join(session_dir, f))

    # --- provenance ----------------------------------------------------------
    info = {
        "source_session_dir": session_dir,
        "session_id": session_id,
        "subject": subj,
        "camera_name": cam,
        "trial": trial,
        "video_source": videos[0] if videos else None,
        "mocap_c3d_sources": c3ds,
        "notes": "organized by organize_raw_ipad_recordings.py (copy-only)",
    }
    write_json(os.path.join(base, "session_info.json"), info)
    return {"subject": subj, "trial": trial, "copied": copied, "skipped": skipped}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", required=True, help="raw repository root, e.g. data/NUS/raw")
    ap.add_argument("--out-root", default="data/NUS", help="output root (default data/NUS)")
    ap.add_argument("--split", default="val", help="split folder name (default val)")
    ap.add_argument("--subject", default=None, help="override subject name (default: derived from c3d)")
    ap.add_argument("--session", default=None, help="only organize this session id")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    ap.add_argument("--with-intermediates", action="store_true",
                    help="also copy regenerable Vicon intermediates (.x1d/.x2d)")
    args = ap.parse_args()

    if not os.path.isdir(args.raw_root):
        sys.exit(f"raw root not found: {args.raw_root}")

    sessions = sorted(
        (d for d in os.listdir(args.raw_root)
         if os.path.isdir(os.path.join(args.raw_root, d)) and not d.startswith(".")),
        key=natural_key,
    )
    if args.session:
        sessions = [s for s in sessions if s == args.session]
    if not sessions:
        print(f"[warn] no session directories under {args.raw_root}")
        return

    print(f"[info] raw-root={args.raw_root}  out-root={args.out_root}  split={args.split}  "
          f"subject={args.subject or '(derived)'}")
    for s in sessions:
        session_dir = os.path.join(args.raw_root, s)
        print(f"\n=== organizing {s} ===")
        res = organize_session(session_dir, s, args.out_root, args.split,
                               args.subject, args.force, args.with_intermediates)
        print(f"  subject={res['subject']}  trial={res['trial']}")
        print(f"  copied {len(res['copied'])} files, skipped {len(res['skipped'])} existing")
        for rel in res["copied"]:
            print(f"    + {rel}")
    print("\n[done] copy-paste the output to the GPU server, e.g.:")
    print(f"  rsync -av {os.path.abspath(os.path.join(args.out_root, args.split))}/ "
          f"haziq@100.83.137.120:/data/haziq/telept/data/NUS/")


if __name__ == "__main__":
    main()
