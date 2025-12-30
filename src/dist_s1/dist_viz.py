import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from dist_s1.constants import DIST_STATUS_CMAP


def get_dist_s1_mpl_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    values = sorted(DIST_STATUS_CMAP.keys())
    colors = np.array([DIST_STATUS_CMAP[v] for v in values]) / 255.0
    mpl_cmap = ListedColormap(colors)
    bounds = values + [values[-1] + 1]
    norm = BoundaryNorm(bounds, mpl_cmap.N)
    return mpl_cmap, norm
