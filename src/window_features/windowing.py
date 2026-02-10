from __future__ import annotations

from dataclasses import replace
from typing import Iterator, Optional, Tuple

import pandas as pd

from .schemas import WindowConfig

def compute_time_bounds(
    df: pd.DataFrame,
    time_col: str = "time",
    align_to: str | None = "h",
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Calcula os limites [start, end] do período coberto pelo df, com alinhamento opcional
    (ex.: "h" para hora cheia). Esses bounds são derivados do dado (não fazem parte do schema).
    """
    if time_col not in df.columns:
        raise KeyError(f"Coluna '{time_col}' não encontrada no DataFrame.")

    t0 = df[time_col].min()
    t1 = df[time_col].max()

    if pd.isna(t0) or pd.isna(t1):
        raise ValueError("time bounds inválidos: df sem timestamps válidos.")

    # Mantém timezone, se houver
    start = t0.floor(align_to) if align_to else t0
    end = t1.ceil(align_to) if align_to else t1
    return start, end


def _validate_config(cfg: WindowConfig) -> None:
    if cfg.window_size <= pd.Timedelta(0):
        raise ValueError("window_size deve ser > 0.")
    if cfg.step_size <= pd.Timedelta(0):
        raise ValueError("step_size deve ser > 0.")
    if cfg.step_size > cfg.window_size:
        raise ValueError("step_size não deve ser maior que window_size (janela ficaria 'pulando').")
    if cfg.min_events < 0:
        raise ValueError("min_events deve ser >= 0.")


def _ensure_datetime_series(df: pd.DataFrame, time_col: str) -> pd.Series:
    s = df[time_col]
    if not pd.api.types.is_datetime64_any_dtype(s):
        raise TypeError(
            f"'{time_col}' precisa ser datetime64. "
            "Converta antes (ex.: pd.to_datetime(..., utc=True))."
        )
    return s


def _make_mask(
    t: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    closed: str,
) -> pd.Series:
    """
    Cria máscara booleana para selecionar eventos no intervalo conforme 'closed':
    - left    : [start, end)
    - right   : (start, end]
    - both    : [start, end]
    - neither : (start, end)
    """
    if closed == "left":
        return (t >= start) & (t < end)
    if closed == "right":
        return (t > start) & (t <= end)
    if closed == "both":
        return (t >= start) & (t <= end)
    if closed == "neither":
        return (t > start) & (t < end)
    raise ValueError(f"closed inválido: {closed!r}")


def iter_windows(
    df: pd.DataFrame,
    cfg: WindowConfig,
    *,
    time_col: str = "time",
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
    sort: bool = True,
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    """
    Itera janelas deslizantes (rolling) sobre o df.

    Yields:
        (win_start, win_end, win_label, df_window)

    Convenções:
      - Os limites (win_start, win_end) seguem cfg.closed via máscara:
        left -> [start, end), right -> (start, end], etc.
      - win_label segue cfg.label:
        left  -> win_start
        right -> win_end

    Observação:
      - start_time/end_time se omitidos são derivados via compute_time_bounds(df, cfg.align_to).
      - Janelas com n < cfg.min_events são ignoradas (não yield).
    """
    _validate_config(cfg)
    if time_col not in df.columns:
        raise KeyError(f"Coluna '{time_col}' não encontrada no DataFrame.")

    # Garante datetime
    t = _ensure_datetime_series(df, time_col)

    # Ordenação (recomendado para consistência)
    if sort:
        df = df.sort_values(time_col).reset_index(drop=True)
        t = df[time_col]

    # Bounds
    if start_time is None or end_time is None:
        derived_start, derived_end = compute_time_bounds(df, time_col=time_col, align_to=cfg.align_to)
        start_time = derived_start if start_time is None else start_time
        end_time = derived_end if end_time is None else end_time

    if start_time >= end_time:
        return  # nada a iterar

    # Constrói grade: starts = start_time, start_time+step, ... até caber uma janela inteira
    last_start = end_time - cfg.window_size
    if last_start < start_time:
        return  # período menor que a janela

    starts = pd.date_range(start=start_time, end=last_start, freq=cfg.step_size)

    for win_start in starts:
        win_end = win_start + cfg.window_size

        mask = _make_mask(t, win_start, win_end, cfg.closed)
        dfw = df.loc[mask]

        if cfg.min_events and len(dfw) < cfg.min_events:
            continue

        win_label = win_start if cfg.label == "left" else win_end
        yield (win_start, win_end, win_label, dfw)


def make_window_index(
    df: pd.DataFrame,
    cfg: WindowConfig,
    *,
    time_col: str = "time",
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Retorna um DataFrame com o grid de janelas (sem materializar df_window),
    útil para debug/auditoria.

    Columns: ["window_start", "window_end", "window_label"]
    """
    if start_time is None or end_time is None:
        start_time_, end_time_ = compute_time_bounds(df, time_col=time_col, align_to=cfg.align_to)
        start_time = start_time_ if start_time is None else start_time
        end_time = end_time_ if end_time is None else end_time

    rows = []
    for ws, we, wl, _ in iter_windows(
        df, cfg, time_col=time_col, start_time=start_time, end_time=end_time, sort=True
    ):
        rows.append((ws, we, wl))

    return pd.DataFrame(rows, columns=["window_start", "window_end", "window_label"])