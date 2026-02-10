# src/window_features/builder.py
import pandas as pd

from .windowing import iter_windows, compute_time_bounds
from .features import extract_features
from .schemas import WindowConfig


def build_features_df(
    df: pd.DataFrame,
    cfg: WindowConfig,
    *,
    time_col: str = "time",
) -> pd.DataFrame:
    """
    Constrói o features_df a partir do catálogo sísmico.
    """

    start_time, end_time = compute_time_bounds(
        df, time_col=time_col, align_to=cfg.align_to
    )

    rows = []

    for win_start, win_end, win_label, dfw in iter_windows(
        df,
        cfg,
        time_col=time_col,
        start_time=start_time,
        end_time=end_time,
    ):
        feats = extract_features(dfw, win_start, win_end, time_col=time_col)

        row = {
            "window_start": win_start,
            "window_end": win_end,
            "window_label": win_label,
            **feats,
        }
        rows.append(row)

    df_feat =  pd.DataFrame(rows)

    # Rounding para floats
    float_cols = df_feat.select_dtypes(include="float").columns
    df_feat[float_cols] = df_feat[float_cols].round(3)

    return df_feat
