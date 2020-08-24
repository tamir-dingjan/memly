"""
sa_analyses.py
Contains the methods for surface area-based analysis metrics: the surface area per lipid and the area compressibility
modulus.
"""
from shapely.geometry import Polygon, box
from geovoronoi import voronoi_regions_from_coords, calculate_polygon_areas
import numpy as np

from memly.metrics import Metric


class SurfaceArea(Metric):
    def __init__(self, membrane, title="Surface area", units="angstrom^2"):
        """
        Measures the surface area of each leaflet in the simulation.
        Uses a 2D (i.e., X-Y plane) Voronoi tesselation on the lipid head group centroid positions.

        :param membrane: memly.membrane.Membrane object.
        :param title: str, optional. The title of the metric. Default is "Surface area".
        :param units: str, optional. Units of the metric. Default is "angstrom^2".
        """

        # Run the initialisation of the parent Metric class
        Metric.__init__(self, membrane, title, units)

        # Setup containers for results
        self.projection = {}
        self.voronoi = {}
        self.apl = {}
        self.apl_mean = {}

        # Get the bounds of the simulation box
        self.box_vertices = [[[0, 0],
                              [0, i],
                              [i, i],
                              [i, 0]] for i in self.membrane.sim.unitcell_lengths[:, 0]]
        self.box_bounds = [Polygon(x) for x in self.box_vertices]

        # The box boundaries may cut off points for head groups which lie over the boundary
        # Build bounds from the extents of the data
        self.data_vertices = [[np.min(i[:, 0]), np.min(i[:, 1]), np.max(i[:, 0]), np.max(i[:, 1])] for i in
                              self.membrane.hg_centroids]
        self.data_bounds = [box(x[0], x[1], x[2], x[3]) for x in self.data_vertices]

        for leaflet in ["upper", "lower"]:
            self.calculate_apl(leaflet)

        # TODO - Store the area per lipid type

    def get_projection(self, leaflet="upper"):
        """
        Construct the 2D projection for the given leaflet.

        :param leaflet: str, optional. The leaflet to generate a projection for. Default value is "upper".
        :return: lst. Collection of lipid head group centroid X and Y coordinates to use as a 2D projection
        for Voronoi tesselation.
        """
        # Check if the leaflet is a valid leaflet label
        if leaflet in self.membrane.leaflets[0].keys():
            # Get the set of head group centroids to be used for this projection
            leaflet_slice = [x[leaflet] for x in self.membrane.leaflets]

            # Select frame-by-frame the head group centroids for this
            # Broadcast first by the leaflet slice and then by the X and Y axes.
            projection_raw = [self.membrane.hg_centroids[x_i, x, :] for x_i, x in enumerate(leaflet_slice)]
            projection = [x[:, [0, 1]] for x in projection_raw]

            self.projection[leaflet] = projection
        else:
            raise ValueError("Leaflet label not found.")

    def calculate_apl(self, leaflet="upper"):
        """
        Calculate the area per lipid for the chosen leaflet.
        :param leaflet: str. The label of the leaflet to calculate APL for. Defaults to "upper".
        :return:
        """
        # Construct the 2D projection for this leaflet
        self.get_projection(leaflet=leaflet)

        # Perform Voronoi tesselation for this projection, bounded by the simulation box.
        self.voronoi[leaflet] = [voronoi_regions_from_coords(points, bound) for points, bound in
                                 zip(self.projection[leaflet], self.data_bounds)]

        # Calculate the area of each lipid in this leaflet
        self.apl[leaflet] = [calculate_polygon_areas(i[0]) for i in self.voronoi[leaflet]]

        # Find the average area per lipid for this leaflet in each frame
        self.apl_mean[leaflet] = [np.mean(x) for x in self.apl[leaflet]]

        # Log the result
        self.add_results(lipid="All", value=self.apl_mean[leaflet], leaflet=leaflet)
