from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import pandas as pd

Closed = Literal["left", "right", "both", "neither"]
Label = Literal["left", "right"]


@dataclass(frozen=True, slots=True)
class WindowConfig:
    window_size: pd.Timedelta
    step_size: pd.Timedelta

    # como alinhar o grid das janelas (ex: "h" para hora cheia)
    align_to: Optional[str] = "h"

    # convenções de intervalo
    closed: Closed = "left"
    label: Label = "left"

    # Quantidade mínima de eventos por janela
    min_events: int = 1
