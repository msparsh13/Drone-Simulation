"""Image arrays -> (mask, Formation). Shared by CLI and web server."""
from __future__ import annotations
from .imaging import extract_mask, crop_to_content, resize_keep_aspect
from .sampling import sample_targets


def formation_from_arrays(bgr, alpha, n=500, threshold=127, invert=False, scale=1.0,
                          extent=100.0, max_side=400, iterations=30, seed=0):
    bgr = resize_keep_aspect(bgr, max_side)
    alpha = resize_keep_aspect(alpha, max_side) if alpha is not None else None
    mask = extract_mask(bgr, alpha, threshold, invert)
    mask, bgr = crop_to_content(mask, bgr)
    return mask, sample_targets(mask, n, bgr, extent=extent, scale=scale,
                                iterations=iterations, seed=seed)
