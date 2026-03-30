from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
import numpy as np
import pandas as pd


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
HALF_LENGTH = PITCH_LENGTH / 2.0
HALF_WIDTH = PITCH_WIDTH / 2.0
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PlayerPoint:
    x: float
    y: float
    team_id: int
    slot_suffix: str
    player_id: int | None


def _cluster_complete_linkage_1d(values: np.ndarray, k: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.array([], dtype=float)

    clusters: List[List[float]] = [[value] for value in values]

    def cluster_distance(cluster_a: List[float], cluster_b: List[float]) -> float:
        return max(abs(a - b) for a in cluster_a for b in cluster_b)

    while len(clusters) > k:
        best_i, best_j, best_dist = 0, 1, float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = cluster_distance(clusters[i], clusters[j])
                if dist < best_dist:
                    best_i, best_j, best_dist = i, j, dist

        merged = clusters[best_i] + clusters[best_j]
        clusters = [clusters[idx] for idx in range(len(clusters)) if idx not in (best_i, best_j)]
        clusters.append(merged)

    centroids = np.asarray([np.mean(cluster) for cluster in clusters], dtype=float)
    centroids.sort()
    return centroids


def _default_lines(axis_min: float, axis_max: float, k: int = 3) -> np.ndarray:
    return np.linspace(axis_min + 0.2 * (axis_max - axis_min), axis_min + 0.8 * (axis_max - axis_min), k)


def _resolve_dynamic_lines(values: np.ndarray, axis_min: float, axis_max: float, k: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size >= k:
        lines = _cluster_complete_linkage_1d(values, k=k)
    elif values.size > 0:
        partial = _cluster_complete_linkage_1d(values, k=int(values.size))
        lines = _default_lines(axis_min, axis_max, k=k)
        lines[: partial.size] = partial
        lines.sort()
    else:
        lines = _default_lines(axis_min, axis_max, k=k)

    if lines.size != k:
        fixed = _default_lines(axis_min, axis_max, k=k)
        keep = min(k, lines.size)
        fixed[:keep] = np.sort(lines)[:keep]
        fixed.sort()
        lines = fixed

    return lines


def _drop_goalkeeper_like_outlier(points: Sequence[PlayerPoint], min_players: int = 8, min_gap_m: float = 6.0) -> List[PlayerPoint]:
    """Remove a single extreme outlier when it forms a lone pressure line.

    This keeps dynamic pressure lines focused on outfield team shape while preserving
    all players for regular plotting/other features.
    """
    if len(points) < min_players:
        return list(points)

    x_values = np.asarray([point.x for point in points], dtype=float)
    centroids = _cluster_complete_linkage_1d(x_values, k=3)
    if centroids.size != 3:
        return list(points)

    nearest_cluster = np.argmin(np.abs(x_values[:, None] - centroids[None, :]), axis=1)
    cluster_sizes = np.bincount(nearest_cluster, minlength=3)
    singleton_cluster = int(np.argmin(cluster_sizes))

    if cluster_sizes[singleton_cluster] != 1 or singleton_cluster not in (0, 2):
        return list(points)

    singleton_idx = int(np.where(nearest_cluster == singleton_cluster)[0][0])
    sorted_idx = np.argsort(x_values)
    position = int(np.where(sorted_idx == singleton_idx)[0][0])

    if position == 0:
        separation_gap = float(x_values[sorted_idx[1]] - x_values[sorted_idx[0]])
    elif position == len(x_values) - 1:
        separation_gap = float(x_values[sorted_idx[-1]] - x_values[sorted_idx[-2]])
    else:
        return list(points)

    if separation_gap < min_gap_m:
        return list(points)

    return [point for idx, point in enumerate(points) if idx != singleton_idx]


def _extract_player_points(row: pd.Series) -> List[PlayerPoint]:
    points: List[PlayerPoint] = []
    for column in row.index:
        if not column.startswith("x_player_"):
            continue

        suffix = column.replace("x_player_", "")
        y_col = f"y_player_{suffix}"
        team_col = f"team_id_player_{suffix}"
        pid_col = f"original_pId_player_{suffix}"

        if y_col not in row.index or team_col not in row.index:
            continue

        x_val = row[column]
        y_val = row[y_col]
        team_val = row[team_col]
        if pd.isna(x_val) or pd.isna(y_val) or pd.isna(team_val):
            continue

        pid_val = row[pid_col] if pid_col in row.index else np.nan
        pid = int(pid_val) if pd.notna(pid_val) else None

        points.append(
            PlayerPoint(
                x=float(x_val),
                y=float(y_val),
                team_id=int(team_val),
                slot_suffix=suffix,
                player_id=pid,
            )
        )

    return points


def _extract_carrier_position(row: pd.Series) -> Tuple[float, float] | None:
    if "player_id" not in row.index or pd.isna(row["player_id"]):
        return None

    carrier_id = int(row["player_id"])
    for column in row.index:
        if not column.startswith("original_pId_player_"):
            continue
        raw_value = row[column]
        if pd.isna(raw_value) or int(raw_value) != carrier_id:
            continue

        suffix = column.replace("original_pId_player_", "")
        x_col = f"x_player_{suffix}"
        y_col = f"y_player_{suffix}"
        if x_col in row.index and y_col in row.index and pd.notna(row[x_col]) and pd.notna(row[y_col]):
            return float(row[x_col]), float(row[y_col])

    return None


def _gaussian_influence_map(
    points: Sequence[PlayerPoint],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    sigma: float = 4.8,
) -> np.ndarray:
    if not points:
        return np.zeros_like(x_grid)

    influence = np.zeros_like(x_grid, dtype=float)
    for point in points:
        dx = x_grid - point.x
        dy = y_grid - point.y
        influence += np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))

    max_val = float(np.max(influence))
    if max_val > 0:
        influence = influence / max_val
    return influence


def _draw_pitch(ax: plt.Axes) -> None:
    ax.set_xlim(-HALF_LENGTH - 2.0, HALF_LENGTH + 2.0)
    ax.set_ylim(-HALF_WIDTH - 2.0, HALF_WIDTH + 2.0)
    ax.set_aspect("equal")
    ax.set_facecolor("#f9f9f9")

    # Outer boundaries and half-way line.
    ax.add_patch(Rectangle((-HALF_LENGTH, -HALF_WIDTH), PITCH_LENGTH, PITCH_WIDTH, fill=False, ec="#777777", lw=1.4))
    ax.plot([0, 0], [-HALF_WIDTH, HALF_WIDTH], color="#aaaaaa", lw=1.0)

    # Center circle and spot.
    ax.add_patch(Circle((0, 0), radius=9.15, fill=False, ec="#aaaaaa", lw=1.0))
    ax.add_patch(Circle((0, 0), radius=0.35, color="#aaaaaa"))

    # Penalty and six-yard boxes.
    for sign in (-1, 1):
        x0 = sign * HALF_LENGTH
        penalty_x = x0 - sign * 16.5
        six_x = x0 - sign * 5.5

        ax.add_patch(
            Rectangle(
                (min(x0, penalty_x), -20.16),
                abs(penalty_x - x0),
                40.32,
                fill=False,
                ec="#c0c0c0",
                lw=1.0,
            )
        )
        ax.add_patch(
            Rectangle(
                (min(x0, six_x), -9.16),
                abs(six_x - x0),
                18.32,
                fill=False,
                ec="#d0d0d0",
                lw=1.0,
            )
        )

    ax.set_xticks([])
    ax.set_yticks([])


def _draw_group_polygons(
    ax: plt.Axes,
    points: Sequence[PlayerPoint],
    vertical_lines: np.ndarray,
    color: str = "#d9b44a",
) -> None:
    if not points or vertical_lines.size == 0:
        return

    groups: Dict[int, List[PlayerPoint]] = {i: [] for i in range(len(vertical_lines))}
    for point in points:
        nearest_idx = int(np.argmin(np.abs(vertical_lines - point.x)))
        groups[nearest_idx].append(point)

    for line_idx, cluster in groups.items():
        if not cluster:
            continue

        xs = np.asarray([p.x for p in cluster], dtype=float)
        ys = np.asarray([p.y for p in cluster], dtype=float)
        padding_x = 1.2
        padding_y = 1.2
        x0, x1 = float(xs.min() - padding_x), float(xs.max() + padding_x)
        y0, y1 = float(ys.min() - padding_y), float(ys.max() + padding_y)

        polygon = Polygon(
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            fill=False,
            ec=color,
            lw=1.8,
            linestyle="-",
            zorder=8,
        )
        ax.add_patch(polygon)
        ax.text((x0 + x1) / 2.0, y1 + 0.7, f"L{line_idx + 1}", ha="center", va="bottom", fontsize=8, color=color)


def build_pressure_line_figure(
    csv_path: Path,
    row_index: int | None,
    output_path: Path,
) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["pass_outcome_type"].notna()].copy()
    if df.empty:
        raise RuntimeError("No rows with pass_outcome_type found in the provided CSV.")

    if row_index is None:
        row = df.iloc[len(df) // 2]
        selected_index = int(row.name)
    else:
        if row_index not in df.index:
            raise ValueError(f"row_index={row_index} is not in the filtered dataframe index.")
        row = df.loc[row_index]
        selected_index = int(row_index)

    attacking_team_id = int(row["team_id"])
    points = _extract_player_points(row)
    attacking_players = [point for point in points if point.team_id == attacking_team_id]
    defending_players = [point for point in points if point.team_id != attacking_team_id]
    defending_line_players = _drop_goalkeeper_like_outlier(defending_players)

    def_x = np.asarray([point.x for point in defending_line_players], dtype=float)
    def_y = np.asarray([point.y for point in defending_line_players], dtype=float)

    vertical_lines = _resolve_dynamic_lines(def_x, -HALF_LENGTH, HALF_LENGTH, k=3)
    horizontal_lines = _resolve_dynamic_lines(def_y, -HALF_WIDTH, HALF_WIDTH, k=3)

    gx = np.linspace(-HALF_LENGTH, HALF_LENGTH, 240)
    gy = np.linspace(-HALF_WIDTH, HALF_WIDTH, 160)
    xx, yy = np.meshgrid(gx, gy)
    attacking_influence = _gaussian_influence_map(attacking_players, xx, yy, sigma=4.8)
    defending_influence = _gaussian_influence_map(defending_players, xx, yy, sigma=4.8)

    fig, ax = plt.subplots(figsize=(14, 9))
    _draw_pitch(ax)

    # Soft influence surfaces (blue attack / red defense).
    ax.contourf(xx, yy, attacking_influence, levels=12, cmap="Blues", alpha=0.28, zorder=1)
    ax.contourf(xx, yy, defending_influence, levels=12, cmap="Reds", alpha=0.22, zorder=2)

    # Team formation block (defending team).
    block_points = defending_line_players if defending_line_players else defending_players
    if block_points:
        block_x = np.asarray([point.x for point in block_points], dtype=float)
        block_y = np.asarray([point.y for point in block_points], dtype=float)
        x0, x1 = float(block_x.min()), float(block_x.max())
        y0, y1 = float(block_y.min()), float(block_y.max())
        formation_block = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            ec="#6f6f6f",
            lw=2.0,
            zorder=6,
        )
        ax.add_patch(formation_block)

        # Dotted pressure-line blocks between vertical lines.
        yb0, yb1 = y0, y1
        line_sequence = np.sort(vertical_lines)
        for i in range(len(line_sequence) - 1):
            xa, xb = float(line_sequence[i]), float(line_sequence[i + 1])
            dotted_block = Rectangle(
                (xa, yb0),
                xb - xa,
                yb1 - yb0,
                fill=False,
                ec="#404040",
                lw=1.2,
                linestyle=":",
                zorder=5,
            )
            ax.add_patch(dotted_block)

    # Players.
    if attacking_players:
        ax.scatter(
            [point.x for point in attacking_players],
            [point.y for point in attacking_players],
            c="#74b9ff",
            edgecolors="#2b4f6d",
            s=52,
            linewidths=1.0,
            zorder=9,
            label="Attacking team",
        )
    if defending_players:
        ax.scatter(
            [point.x for point in defending_players],
            [point.y for point in defending_players],
            c="#e17055",
            edgecolors="#7f2f1d",
            s=52,
            linewidths=1.0,
            zorder=9,
            label="Defending team",
        )

    # Ball and event direction.
    ball_start = (float(row["ball_x_start"]), float(row["ball_y_start"]))
    ball_end = (float(row["ball_x_end"]), float(row["ball_y_end"]))
    ax.scatter([ball_start[0]], [ball_start[1]], c="#00b894", s=95, edgecolors="#0b5345", zorder=12, label="Ball")
    ax.annotate(
        "",
        xy=ball_end,
        xytext=ball_start,
        arrowprops=dict(arrowstyle="->", color="#f5b041", lw=1.8, linestyle="--"),
        zorder=10,
    )

    # Ball carrier marker.
    carrier = _extract_carrier_position(row)
    if carrier is not None:
        ax.scatter([carrier[0]], [carrier[1]], s=170, facecolors="none", edgecolors="#16a085", linewidths=2.0, zorder=13)

    # Defending pressure lines.
    sorted_lines = np.sort(vertical_lines)
    for idx, line_x in enumerate(sorted_lines):
        ax.plot([line_x, line_x], [-HALF_WIDTH, HALF_WIDTH], color="#2ecc71", linestyle="--", lw=1.8, zorder=7)
        ax.text(line_x, HALF_WIDTH + 0.8, f"{idx + 1}st pressure line" if idx == 0 else f"{idx + 1}nd pressure line" if idx == 1 else f"{idx + 1}rd pressure line", ha="center", va="bottom", fontsize=8, color="#2ecc71")

    # Horizontal line hints (light).
    for line_y in np.sort(horizontal_lines):
        ax.plot([-HALF_LENGTH, HALF_LENGTH], [line_y, line_y], color="#a3e4d7", linestyle=":", lw=0.8, zorder=4)

    _draw_group_polygons(ax, block_points, sorted_lines, color="#d4ac0d")

    ax.set_title(
        "Pressure Line Visualization (Pass Snapshot)\n"
        f"file={csv_path.name} | row_index={selected_index} | pass_outcome={row['pass_outcome_type']}",
        fontsize=12,
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize dynamic pressure lines from a pass tracking snapshot.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="passes/final_pass_track_4616.csv",
        help="Path to one pass tracking CSV file.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=None,
        help="Exact dataframe index to visualize. If omitted, the middle valid row is used.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/heatmaps/pressure_lines_visualization/pressure_lines_snapshot.png",
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = (ROOT / csv_path).resolve()

    output_path = Path(args.output_path)
    if not output_path.is_absolute():
        output_path = (ROOT / output_path).resolve()

    build_pressure_line_figure(
        csv_path=csv_path,
        row_index=args.row_index,
        output_path=output_path,
    )
    print("PRESSURE_LINE_VISUALIZATION_OK")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()