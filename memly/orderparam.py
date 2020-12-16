"""
orderparam.py
Contains the methods for order parameter calculation and anlyses.
"""
import numpy as np

from memly.metrics import Metric
from memly.membrane import unit_vector, angle_between

class OrderParam(Metric):
    def __init__(self, membrane, title="Order parameter", units="unitless"):
        """
        Calculates order parameters for each lipid species in the simulation.

        :param membrane: memly.membrane.Membrane object
        :param title: str, optional. The title of the metric. Default is "Order parameter".
        :param units: str, optional. Units of the metric. Default is "unitless".
        """

        # Run the initialisation of the parent Metric class
        Metric.__init__(self, membrane, title, units)

        # Set up variables
        self.vectors = []
        self.normals = []
        self.angle = []
        self.p2 = []

        #self.calculate_orderparam(self)

        self.add_results(lipid="Test", value=self.p2, leaflet="Test")

    def calculate_orderparam(self, particle_i, particle_j):
        """
        Calculate the second-rank order parameter for a pair of particles:
        P2 = 0.5 * (3 * <(cos(theta))**2> - 1)
        where:  theta   angle between the selected bond and the bilayer normal

        The values of P2 range between:
        1      perfect alignment between bond and bilayer normal
        0      random alignment
        -0.5   anti-alignment between bond and bilayer normal

        The bilayer normal is fetched from the memly.membrane object, using the pre-calculated normals
        for the lipid residue that contains both of the given particles.
        :return:
        """

        # Get vector from particle_i to particle_j
        self.vectors = self.membrane.sim.xyz[:, particle_j, :] - self.membrane.sim.xyz[:, particle_i, :]

        # Get bilayer normals
        # Normals are not consistently +ve or -ve Z-direction... don't give a good indication of
        # leaflet side
        self.normals = self.membrane.normals[:, self.membrane.lipid_residues_by_particle(particle_i), :]

        # Calculate ensemble average angle between vectors
        self.angle = np.radians(np.mean([angle_between(i, j) for i, j in zip(self.vectors, self.normals)]))

        # Compute order parameter
        self.p2 = 0.5 * (3 * (np.cos(self.angle))**2 - 1)
        return self.p2

