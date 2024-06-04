import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from LoopStructural.utils import rng


class StratigraphicColumnView:
    def __init__(self, model, ax=None, cmap=None, labels=None):
        self.model = model
        self.ax = ax
        self.cmap = cmap
        self.labels = labels

    def plot(self):
        n_units = 0  # count how many discrete colours
        xmin = 0
        ymin = 0
        ymax = 1
        xmax = 1
        fig = None
        if self.ax is None:
            fig, self.ax = plt.subplots(figsize=(2, 10))
        patches = []
        prev_coords = [0, 0]
        for g in self.model.stratigraphic_column.keys():
            if g == "faults":
                continue
            for u in self.model.stratigraphic_column[g].keys():
                n_units += 1
                ymin = -self.model.stratigraphic_column[g][u]["min"]
                if not np.isfinite(ymin):
                    ymin = 0
                ymax = -self.model.stratigraphic_column[g][u]["max"]
                if not np.isfinite(ymax):
                    ymax = prev_coords[1] + (prev_coords[1] - prev_coords[0]) * (1 + rng.random())

                prev_coords = (ymin, ymax)
                polygon_points = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
                patches.append(Polygon(polygon_points))
                xy = (0, ymin + (ymax - ymin) / 2)
                if self.labels:
                    self.ax.annotate(self.labels[u], xy)
                else:
                    self.ax.annotate(u, xy)
        if self.cmap is None:
            import matplotlib.colors as colors

            colours = []
            boundaries = []
            data = []
            for g in self.model.stratigraphic_column.keys():
                if g == "faults":
                    continue
                for v in self.model.stratigraphic_column[g].values():
                    data.append((v["id"], v["colour"]))
                    colours.append(v["colour"])
                    boundaries.append(v["id"])  # print(u,v)
            cmap = colors.ListedColormap(colours)
        else:
            cmap = cm.get_cmap(self.cmap, n_units - 1)
        p = PatchCollection(patches, cmap=cmap)

        colors = np.arange(len(patches))
        p.set_array(np.array(colors))

        self.ax.add_collection(p)
        self.ax.set_ylim(ymax + (ymax - ymin) * -2, 0)  # ax.set_ylim(0,ymax)
        self.ax.axis("off")

        return fig
