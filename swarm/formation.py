from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Formation:
    """A set of target positions for a swarm.

    positions: (N, 2) world coords, centered at origin, +y up.
    colors:    (N, 3) uint8 RGB.
    ids:       (N,)   stable particle/target IDs.
    meta:      pixel->world transform info (scale, image width/height).
    """

    positions: np.ndarray
    colors: np.ndarray
    ids: np.ndarray
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)
