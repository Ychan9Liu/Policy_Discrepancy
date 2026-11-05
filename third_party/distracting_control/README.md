# DCS background provenance

Upstream: Google Research, `distracting_control`.

Pinned revision: `5b09c22d73a9d35eb6c5d2a99b95677a45053466`.

- [background.py](https://github.com/google-research/google-research/blob/5b09c22d73a9d35eb6c5d2a99b95677a45053466/distracting_control/background.py)
  is preserved byte-for-byte as `background_reference.py`.
  SHA256 `1a5f95fa3b53dabe7111549a5fec38fabab17256bab35f9f5e28cf756119e1d9`.
- `suite_utils_reference.py` preserves the upstream ground-opacity rules.
- `LICENSE` is the upstream Apache-2.0 license. Reference files are evidence,
  not runtime imports; TensorFlow is not required by this port.

The runtime implementation is `embodied/envs/dcs_background.py`. The source
copyright/license notice is retained. Local changes are deterministic sorted
frame enumeration, local files only, Pillow bilinear resizing instead of TF
resize, bounded lazy texture caching, strict input/layout checks, an independent
named RNG stream, and JSON diagnostic state. Original sky pixels are retained
for alpha blending across resets to avoid repeated-blending drift. This is a
documented background-only adaptation, **not an exact reproduction of the
original combined DCS benchmark or its pixel values**. No color/camera wrapper,
action modification, reward modification, or dynamics noise is introduced.

The DAVIS dataset has its own terms and is not included in Git. Select and
record concrete video/frame identities and content hashes before experiments.
