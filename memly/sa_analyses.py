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
        self.acm = 0
        self.acm_feller = {}

        # Values of constants used in analyses
        self.boltzmann = 1.380649E-23  # J/K
        self.temp = 310  # K

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

        self.calculate_acm()
        # TODO - evaluate the area compression modulus over each leaflet seperately

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

    def calculate_acm(self):
        """
        Calculate the area compressibility modulus for the chosen leaflet. This is the energetic
        cost associated with stretching or compressing the membrane area.

        This implementation is drawn from: Wang, E. & Klauda, J. B. J. Phys. Chem. B 123, 2525–2535 (2019):
        Eq (2):
        K(A) = ( k(B) * T * <A> ) / ( N * sigma^2(<A>) )
        where:  k(B)    Boltzmann's constant
                T       Absolute temperature
                <A>     Ensemble average surface area of membrane
                N       Number of lipids in membrane
                sigma^2(<A>)    Variance of the surface area

        The area used here is the area of the entire simulation box, and so is not leaflet-specific.

        A slightly different implementation is presented in: Doktorova, M., et al. Biophys. J. 116, 487–502 (2019):
        Eq (2):
        K(A) = ( k(B) * T * a0 ) / ( <(a - a0)^2> )
        where:  k(B)    Boltzmann's constant
                T       Absolute temperature
                a0      Equilibrium membrane area
                a       Membrane area

        Note that the denominator is just the variance of the membrane area (ensemble average of the squared deviations
        from the mean area). So this implementation simplifies to the same as Wang's, without scaling by the number of
        lipids present.

        Finally, Feller and Pastor present another implementation.
        Feller, S. E. & Pastor, R. W. J. Chem. Phys. 111, 1281–1287 (1999):
        Eq (3):
        K(A) = ( k(B) * T * A0 ) / ( N * <sigma(A0)^2> )
        where:  k(B)    Boltzmann's constant
                T       Absolute temperature
                A0      Average area per molecule
                N       Number of molecules
                sigma(A0)^2     Variance of the area per molecule



        :return:
        """

        # Get the average surface area of the whole bilayer, over time. This can be approximated by taking the
        # XY area of the simulation box.
        boxarea = [i.area for i in self.box_bounds]
        area = np.mean(boxarea)
        variance = np.var(boxarea)

        # Calculate the ACM
        # The area scale here is in nm^2, so apply the 10^-18 conversion to reach m^2, and square the units for variance

        # Doktorova uses the area of the whole simulation box
        self.acm = (self.boltzmann * self.temp * area * 1E-18) / (variance * 1E-36)

        # Feller and Pastor use the area per lipid - need to do separately per leaflet
        for leaflet in ["upper", "lower"]:
            # Get the number of lipids in each leaflet over time
            lipidcount = [len(self.membrane.leaflets[frame][leaflet]) for frame in range(0,len(self.membrane.sim))]
            self.acm_feller[leaflet] = [(self.boltzmann * self.temp * np.mean(apl) * 1E-18) / (nlipids * np.var(apl) * 1E-36)
                                        for apl, nlipids in zip(self.apl[leaflet], lipidcount)]

        # Log the results
        self.add_results(title="Area compression modulus (Doktorova2019)", lipid="All", value=self.acm, units="N/m", leaflet="Both")
        for leaflet in ["upper", "lower"]:
            self.add_results(title="Area compression modulus (Feller1999)", lipid="All", value=self.acm_feller[leaflet], units="N/m", leaflet=leaflet)
