"""Deteccion de plateau en scores para reiniciar exploracion de forma uniforme."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np


class PlateauAction(str, Enum):
    """Accion sugerida tras observar un score."""

    NONE = "none"
    RESTART = "restart"


@dataclass
class PlateauDetector:
    """
    Ventana deslizante de scores: si la desviacion tipica cae bajo umbral, sugiere RESTART.

    Args:
        window_size: tamano maximo de la ventana (deque maxlen).
        std_max: si std(window) < std_max, hay plateau.
        restart_callback: opcional; no usado en observe (compat reservada).
    """

    window_size: int
    std_max: float
    restart_callback: Optional[Callable[[], None]] = None

    def __post_init__(self) -> None:
        self._window: deque[float] = deque(maxlen=max(2, int(self.window_size)))

    def reset(self) -> None:
        """Limpia la ventana (p.ej. tras un reinicio de exploracion)."""
        self._window.clear()

    def observe(self, score: float) -> PlateauAction:
        """
        Registra un score y devuelve RESTART si la ventana esta llena y es plateau.

        Args:
            score: score del candidato evaluado (menor es mejor).

        Returns:
            PlateauAction.RESTART o NONE.
        """
        self._window.append(float(score))
        if len(self._window) < self._window.maxlen:
            return PlateauAction.NONE
        arr = np.asarray(self._window, dtype=np.float64)
        st = float(np.std(arr, ddof=0))
        if st < float(self.std_max):
            if self.restart_callback is not None:
                self.restart_callback()
            self._window.clear()
            return PlateauAction.RESTART
        return PlateauAction.NONE
