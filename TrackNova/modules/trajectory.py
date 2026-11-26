"""
Trajectory Generation and Path Geometry Module.

Extracts, stores, and summarizes 2D image-plane trajectory coordinates
for the selected tracking ID.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def build_trajectory_data(
    movement_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """
    Generate structured trajectory data from a selected object's movement DataFrame.

    Returns start/end points, ordered (x, y) coordinates, bounding extents,
    net straight-line displacement, total path length, and path linearity ratio.
    """
    if movement_df is None or movement_df.empty:
        return {
            "has_trajectory": False,
            "points": [],
            "start_point": None,
            "end_point": None,
            "path_length_px": 0.0,
            "net_displacement_px": 0.0,
            "linearity_ratio": None,
            "x_range": (0.0, 0.0),
            "y_range": (0.0, 0.0),
        }

    points: List[Tuple[float, float]] = [
        (float(r["center_x"]), float(r["center_y"]))
        for _, r in movement_df.iterrows()
    ]
    start_pt = points[0]
    end_pt = points[-1]
    path_length = float(movement_df["cumulative_displacement_px"].iloc[-1])
    net_disp = math.hypot(end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])
    linearity = (net_disp / path_length) if path_length > 0.0 else 1.0

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    return {
        "has_trajectory": True,
        "points": points,
        "start_point": start_pt,
        "end_point": end_pt,
        "path_length_px": path_length,
        "net_displacement_px": net_disp,
        "linearity_ratio": linearity,
        "x_range": (min(xs), max(xs)),
        "y_range": (min(ys), max(ys)),
        "point_count": len(points),
    }
