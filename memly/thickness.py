"""
thickness.py
A metric to measure thickness of the bilayer.
"""
import math
import numpy as np
import logging
import numba as nb

from memly.metrics import Metric


def euclidean_distance(v1, v2):
    dist = [(i - j)**2 for i, j in zip(v1, v2)]
    return math.sqrt(sum(dist))


class Thickness(Metric):
    def __init__(self, membrane, title="Thickness", units="nm"):
        """
        Measures the thickness of the bilayer.

        :param membrane: memly.membrane.Membrane object.
        :param title: str, optional. The title of the metric. Default is "Thickness".
        :param units: str, optional. Units of the metric. Default is "angstrom".
        """

        # Run the initialisation of the parent Metric class
        Metric.__init__(self, membrane, title, units)

        self.threshold_home_leaflet = 2
        self.threshold_opposite_leaflet = 5
        self.threshold_normal_align = 10
        self.membrane_heights = []

        # Matrix subtraction during neighbour searching is much faster when the output matrix is pre-allocated
        self.hg_distance = np.zeros_like(self.membrane.hg_centroids[0])

        self.calculate_thickness()

        self.add_results(lipid="All", value=self.membrane_heights, leaflet="Both")

    def calculate_thickness(self):
        """
        Calculate the membrane thickness, using a method adapted from that used in FATSLIM (Buchoux, S. Bioinformatics 33, 133–134 (2017)).

        Method:
        1. Select a pre-calculated membrane normal
        2. Select nearest head groups in home leaflet
        3. Calculate average position in home leaflet
        4. Select nearest head groups from opposing leaflet
        5. Select from nearest head groups those with aligned normals
        6. Calculate average position in opposing leaflet
        7. Find the Euclidean distance between the two average positions.

        Repeat this process for each membrane normal to find the average membrane thickness.
        """

        for frame_i in range(0, len(self.membrane.sim)):
            logging.debug("Thickness calculation frame: %s" % frame_i)
            frame_thickness = []
            for resid, centroid in enumerate(self.membrane.hg_centroids[frame_i]):
                logging.debug("Thickness - residue: %s" % resid)
                thickness = self.thickness_at_lipid_faster(frame_i, resid)
                if not np.isnan(thickness):
                    frame_thickness.append(thickness)
            self.membrane_heights.append(frame_thickness)

    def thickness_at_lipid_faster(self, frame_i, resid):
        """
        1. Select a head group centroid
        2. Select nearest head groups from opposing leaflet.
        3. Calculate distance vectors from opposing leaflet head groups to home head group centroid
        4. Select from opposing leaflet head groups ones for which the distance vector is within 10 deg of head group normal
        5. Find averaged position of selected opposite head groups
        6. Calculate distance between home head group centroid and averaged opposite head groups.

        :param frame_i:
        :param resid:
        :return:
        """

        source = self.membrane.hg_centroids[frame_i, resid, :]
        home_leaflet = self.membrane.leaflet_occupancy_by_resid[resid][frame_i]

        if home_leaflet == "aggregate":
            return np.nan
        elif home_leaflet == "upper":
            opposite_leaflet = "lower"
        elif home_leaflet == "lower":
            opposite_leaflet = "upper"

        nbors_opposite = self.select_nbors(source, frame_i, opposite_leaflet, self.threshold_opposite_leaflet)

        # Build distance vectors, from opposite leaflet nbors to source. This is a small subset of all centroids.
        nbor_centroids = self.membrane.hg_centroids[frame_i, nbors_opposite, :]

        # Tile the source vector to make an array of the same length
        source_expanded = np.tile(source, (len(nbor_centroids), 1))

        # Get vectors from nbors to source
        dist_vectors = np.subtract(np.asarray(source_expanded), nbor_centroids)

        # Compare the normal vector for the selected lipid to the distance vectors of the opposite leaflet nbors
        reference_expanded = np.tile(self.membrane.normals[frame_i, resid, :], (len(dist_vectors), 1))
        vector_alignment = vector_angle_faster(reference_expanded, dist_vectors) < self.threshold_normal_align

        if np.sum(vector_alignment) == 0:
            return np.nan

        selected_nbors = nbor_centroids[vector_alignment]
        avg_nbors = np.mean(selected_nbors, axis=0)
        lipid_thickness = euclidean_distance(source, avg_nbors)
        return lipid_thickness

    def select_nbors(self, source, frame, target_leaflet, threshold):
        """
        Returns the head groups within distance threshold to the source coordinates, sampled from within the target
        leaflet.

        :param source: A NumPy array of shape (1, 3) containing coordinates from which to search for neighbors.
        :param frame: int, frame index for neighbor searching.
        :param target_leaflet: str, Label of the desired leaflet to find neighbors in. E.g., "upper", "lower".
        :param threshold: float, Distance threshold (nm) around the source coordinate for retrieving neighbors.
        :return: Boolean Numpy array of shape (num_lipids, ) with True marking the selected neighbors. Can be used for
                 indexing membrane.detected_lipids, membrane.hg_centroids, and membrane.normals (for example).
        """

        # What list of head group particles can we use for searching in?
        # self.membrane.hg_centroids - shape (frames, hg, 3). One centroid for each residue.
        # self.membrane.detected_lipids - list of lipid residue indices

        # This following line is the most expensive, and it applies to every residue in every frame. So it needs to be fast.


        # nbor_distances = np.asarray([distance.euclidean(centroid, source) for centroid in self.membrane.hg_centroids[frame]]) < threshold
        # nbor_distances = np.asarray([np.linalg.norm(centroid - source) for centroid in self.membrane.hg_centroids[frame]]) < threshold
        # nbor_distances = np.asarray([euclidean_distance(centroid, source) for centroid in self.membrane.hg_centroids[frame]]) < threshold

        # The basic idea is to find lipid residues that are close to the query lipid. We don't actually need to calculate
        # the distance, just find lipids that are close.
        # A simple way to do this is to fine lipids whose XY coordinates are within threshold from the reference
        source_coords = [source]*len(self.membrane.hg_centroids[frame])

        #dist_from_source = np.asarray(source_coords - self.membrane.hg_centroids[0])
        dist_from_source = np.subtract(source_coords, self.membrane.hg_centroids[0], out=self.hg_distance)

        nbor_distances = np.zeros(len(self.membrane.hg_centroids[frame]), dtype=bool)
        nbor_distances[np.where((dist_from_source[:, 0] + dist_from_source[:, 1]) < threshold)[0]] = True

        # leaflet_filter = np.asarray([res in set(self.membrane.leaflets[frame][target_leaflet]) for res in self.membrane.detected_lipids])

        leaflet_filter = np.zeros(len(self.membrane.hg_centroids[frame]), dtype=bool)
        leaflet_filter[self.membrane.leaflets[frame][target_leaflet]] = True

        # Combine the two masks to select head group particles which meet both
        selected_neighbors = nbor_distances & leaflet_filter
        return selected_neighbors


@nb.njit(fastmath=True, error_model="numpy", parallel=True)
def vector_angle_faster(v1, v2):
    """
    Takes two Numpy arrays of vectors and returns the angles between v1[0] & v2[0], v1[1] & v2[1], etc.

    :param v1: numpy.ndarray
    :param v2: numpy.ndarray
    :return:
    """
    # Confirm same number of vectors in both arrays, and vectors are the same shape
    assert v1.shape == v2.shape

    result = np.empty(v1.shape[0])

    for i in nb.prange(v1.shape[0]):
        dot = 0.
        a = 0.
        b = 0.

        for j in range(v1.shape[1]):
            dot += v1[i, j] * v2[i, j]
            a += v1[i, j] ** 2
            b += v2[i, j] ** 2

        result[i] = np.degrees(np.arccos(dot/(np.sqrt(a * b))))
    return result
