"""Rolling feature window built from live telemetry.

    window = LiveFeatureWindow(feature_names, window_steps)
    window.push(**telemetry_state)
    if window.ready:
        pred_delta = predict_delta_t1(model, checkpoint, window.array(), device)
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np


class LiveFeatureWindow:
    """Accumulates real telemetry samples into a model-ready feature window."""

    def __init__(self, feature_names: Sequence[str], window_steps: int) -> None:
        self.feature_names = list(feature_names)
        self.window_steps = int(window_steps)
        self._rows: deque[np.ndarray] = deque(maxlen=self.window_steps)

    @property
    def ready(self) -> bool:
        return len(self._rows) >= self.window_steps

    @property
    def n_samples(self) -> int:
        return len(self._rows)

    def push(
        self,
        *,
        temp: float,
        temp_ref: float,
        kp: float,
        ki: float,
        kd: float,
        temp_u: float = 0.0,
        temp_u_p: float = 0.0,
        temp_u_i: float = 0.0,
        temp_u_d: float = 0.0,
        **extra: float,
    ) -> None:
        values = {
            "temp": float(temp),
            "temp_ref": float(temp_ref),
            "error": float(temp_ref) - float(temp),
            "temp_u": float(temp_u),
            "temp_u_p": float(temp_u_p),
            "temp_u_i": float(temp_u_i),
            "temp_u_d": float(temp_u_d),
            "kp": float(kp),
            "ki": float(ki),
            "kd": float(kd),
            **{k: float(v) for k, v in extra.items()},
        }
        row = np.asarray(
            [values.get(name, 0.0) for name in self.feature_names], dtype=np.float32
        )
        self._rows.append(row)

    def array(self) -> np.ndarray:
        """The window as (window_steps, n_features), ready for predict_delta_t1."""
        if not self.ready:
            raise RuntimeError(f"Only {len(self._rows)}/{self.window_steps} real samples collected.")
        return np.stack(self._rows, axis=0)

    def reset(self) -> None:
        self._rows.clear()
