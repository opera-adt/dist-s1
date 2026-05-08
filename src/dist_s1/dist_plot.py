import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure

from dist_s1.constants import (DIST_STATUS_CMAP, DISTVAL2CBARLABEL,
                               DISTVAL2CBARLABEL_FULL)


def get_dist_s1_mpl_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    values = sorted(DIST_STATUS_CMAP.keys())
    colors = np.array([DIST_STATUS_CMAP[v] for v in values]) / 255.0
    mpl_cmap = ListedColormap(colors)
    bounds = values + [values[-1] + 1]
    norm = BoundaryNorm(bounds, mpl_cmap.N)
    return mpl_cmap, norm


def get_colorbar_label(label: int, include_numerical_vals: bool = True, use_full_labels: bool = True) -> str:
    color_label_map = DISTVAL2CBARLABEL.copy() if not use_full_labels else DISTVAL2CBARLABEL_FULL.copy()
    if include_numerical_vals:
        color_label_map = {k: f'{v} ({k})' for k, v in color_label_map.items()}
    return color_label_map[label]


def add_dist_s1_colorbar(
    cax: Axes,
    include_numerical_vals: bool = True,
    use_full_labels: bool = True,
    label_wrap_width: int | None = None,
) -> Colorbar:
    mpl_cmap, norm = get_dist_s1_mpl_cmap()
    values = sorted(DIST_STATUS_CMAP.keys())

    cb = Colorbar(cax, cmap=mpl_cmap, norm=norm)

    # Set tick positions at the center of each color bin
    tick_positions = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
    tick_positions.append(values[-1] + 0.5)
    cb.set_ticks(tick_positions)

    labels = [
        get_colorbar_label(val, include_numerical_vals=include_numerical_vals, use_full_labels=use_full_labels)
        for val in values
    ]
    if label_wrap_width is not None:
        labels = [textwrap.fill(label, width=label_wrap_width) for label in labels]
    cb.set_ticklabels(labels)

    return cb


def calculate_scalebar_size(profile: dict, width_proportion: float = 0.3) -> tuple[float, str]:
    transform = profile["transform"]
    pixel_size = abs(transform[0])
    width_pixels = profile["width"]
    width_meters = width_pixels * pixel_size
    target_length = width_meters * width_proportion

    if target_length >= 1000:
        target = target_length / 1000
        units = "km"
    else:
        target = target_length
        units = "m"

    magnitude = 10 ** np.floor(np.log10(target))
    normalized = target / magnitude

    if normalized < 1.5:
        nice = 1
    elif normalized < 3.5:
        nice = 2
    elif normalized < 7.5:
        nice = 5
    else:
        nice = 10

    size = nice * magnitude
    return float(size), units


def add_scalebar(
    ax: Axes,
    profile: dict,
    position: str = "lower left",
    color: str = "white",
    width_proportion: float = 0.3,
    fontsize: int | float = 10,
) -> None:
    position_map = {
        "upper left": (0.05, 0.95),
        "top left": (0.05, 0.95),
        "upper right": (0.95, 0.95),
        "top right": (0.95, 0.95),
        "lower left": (0.05, 0.05),
        "bottom left": (0.05, 0.05),
        "lower right": (0.95, 0.05),
        "bottom right": (0.95, 0.05),
    }

    x_frac, y_frac = position_map.get(position.lower(), (0.05, 0.05))

    size, units = calculate_scalebar_size(profile, width_proportion=width_proportion)

    if units == "km":
        size_meters = size * 1000
    else:
        size_meters = size

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    width = xlim[1] - xlim[0]
    height = ylim[1] - ylim[0]

    if x_frac > 0.5:
        x_end = xlim[0] + x_frac * width
        x_start = x_end - size_meters
    else:
        x_start = xlim[0] + x_frac * width
        x_end = x_start + size_meters

    y_pos = ylim[0] + y_frac * height

    ax.plot([x_start, x_end], [y_pos, y_pos], color=color, linewidth=3)

    text_x = (x_start + x_end) / 2
    if y_frac > 0.5:
        text_y = y_pos - 0.02 * height
        va = "top"
    else:
        text_y = y_pos + 0.02 * height
        va = "bottom"

    ax.text(
        text_x, text_y, f"{size:.0f} {units}", color=color, ha="center", va=va, fontsize=fontsize, weight="bold"
    )


def plot_scalebar(
    profile: dict,
    position: str = "lower left",
    figsize: tuple[float, float] = (2, 0.5),
    color: str = "white",
    width_proportion: float = 0.3,
    fontsize: int | float = 10,
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize)

    transform = profile["transform"]
    width = profile["width"]
    height = profile["height"]

    left = transform[2]
    right = transform[2] + width * transform[0]
    top = transform[5]
    bottom = transform[5] + height * transform[4]

    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.axis("off")

    add_scalebar(ax, profile, position=position, color=color, width_proportion=width_proportion, fontsize=fontsize)

    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    return fig, ax
