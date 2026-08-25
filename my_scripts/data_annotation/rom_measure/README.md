# rom_measure — 2D joint ROM annotation webapp

Upload/place RGB image frames in `images/`, click to place points, name them,
and compute 2D joint angles (e.g. elbow flexion). 2D only — RGBD/3D is future work.

## UX notes

- Placing your **3rd point auto-creates an angle** (middle/2nd point = vertex).
- A big status banner shows **"✅ Saved" + the exact saved file path** so users
  without shell access can confirm persistence (path like
  `annotations/<user>/<image>.json`).
- Annotations auto-save ~1s after any change and are flushed immediately when
  switching images (no data loss).

## Multi-user

Each username has its own annotation namespace at `annotations/<username>/`.
Logging back in with the same username restores that user's annotations.

- **Shared password** (any username + one password):
  `ROM_PASSWORD=changeme`
- **Per-user passwords** (overrides shared):
  `ROM_USERS="alice:pw1,bob:pw2"`
- **No password** (any username, no password): set neither env var.

## Run

```bash
pip install flask
ROM_PASSWORD=changeme python3 server.py --host 0.0.0.0 --port 8080
```

- Open `http://localhost:8080` locally.
- Images uploaded via the webapp go to `images/` (also drop files there manually;
  click Refresh).
- Annotations auto-save to `annotations/<image>.json`; export all as CSV.

## Expose to the internet

```bash
cloudflared tunnel --url http://localhost:8080     # free, no account
# or:  ngrok http 8080
```

- Always set `ROM_PASSWORD` — the tool is publicly reachable.
- For extra protection add a Cloudflare Access policy in front of the tunnel.

## Notes / limitations (2D)

- A single photo projects the 3D joint angle onto the image plane; accuracy
  depends on the motion plane being roughly parallel to the camera.
- "180−θ" checkbox reports flexion-from-full-extension (extended = 0°) instead
  of the raw interior angle.
- Point coordinates are stored in natural image pixels (resolution-independent).
