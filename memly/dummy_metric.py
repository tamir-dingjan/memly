import pandas as pd
import numpy as np
import os

import self as self

from memly.metrics import Metric


class Dummy(Metric):
    def __init__(self, membrane, title="dummy", units="nm"):
        Metric.__init__(self, membrane, title, units)
        self.z_locations_upper = []
        self.z_locations_lower = []
        self.calc_max_z_positions()

        # store results:
        self.add_results(lipid="All", value=self.z_locations, leaflet="Both")

    def calc_max_z_positions(self):
        """
        Finds the lipids with highest z position in each frame and adds it to a list

        Methods:
        1. go through frames in head group centroids
        2. find using np.max the largest item in every frame's z position
        """
        i = 0
        for frame in self.membrane.hg_centroids:
            i += 1
            max_z = np.max(frame[:, 2])
            self.z_locations.append({i: [max_z, np.where(frame[:, 2] == max_z)[0][0]]})
        print(self.z_locations)
        print(self.membrane.hg_centroids[0][:, 2])

