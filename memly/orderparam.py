"""
orderparam.py
Contains the methods for order parameter calculation and anlyses.
"""
from collections import defaultdict

import numpy as np

from memly.metrics import Metric
from memly.membrane import unit_vector, angle_between


def calculate_orderparam(angle):
    """
    Calculate the order parameter for the provided angle
    :param angle:
    :return:
    """
    return 0.5 * (3 * (np.cos(np.radians(angle))) ** 2 - 1)


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

        # Collect bonded pairs, grouped together by residue name
        self.bonded_catalogue = defaultdict(list)
        for bond in self.membrane.sim.topology.bonds:
            catalogue_name = bond.atom1.residue.name + "-" + bond.atom1.name + "-" + bond.atom2.name
            self.bonded_catalogue[catalogue_name].append((bond.atom1.index, bond.atom2.index))

        # Calculate order parameters
        self.orderparams = {catalogue_name: calculate_orderparam(self.get_ensemble_average_angle(catalogue_name))
                            for catalogue_name in self.bonded_catalogue.keys()}

        # Store results
        for catalogue_name, orderp in self.orderparams.items():
            self.add_results(lipid=catalogue_name, value=orderp, leaflet="Both")

        # TODO - order parameter calculation seperately for upper and lower leaflets

    def get_ensemble_average_angle(self, catalogue_name):
        """
        Returns the ensemble-averaged angle between all the bonded pairs and the bilayer normal
        :param catalogue_name:
        :return:
        """
        angles = []
        for particle_pair in self.bonded_catalogue[catalogue_name]:
            angles.append(np.asarray(self.calculate_angles_to_bilayer_normal(particle_pair)))
        return np.mean(np.asarray(angles))

    def calculate_angles_to_bilayer_normal(self, particle_pair):
        """
        Calculate the angle between the given particle pair and the bilayer normal at the residue
        containing the first particle.
        :param particle_pair:
        :return:
        """
        particle_i = particle_pair[0]
        particle_j = particle_pair[1]

        # Get vector from particle_i to particle_j
        vectors = self.membrane.sim.xyz[:, particle_j, :] - self.membrane.sim.xyz[:, particle_i, :]

        # Get bilayer normals
        normals = self.membrane.normals[:, self.membrane.lipid_residues_by_particle[particle_i], :]

        # Calculate each frame's angle between the vectors
        angles = [angle_between(i, j) for i, j in zip(vectors, normals)]
        return angles

    def calculate_orderparam_particle_pair(self, particle_i, particle_j):
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
        vectors = self.membrane.sim.xyz[:, particle_j, :] - self.membrane.sim.xyz[:, particle_i, :]

        # Get bilayer normals
        normals = self.membrane.normals[:, self.membrane.lipid_residues_by_particle[particle_i], :]

        # Calculate ensemble average angle between vectors
        angle_raw = [angle_between(i, j) for i, j in zip(self.vectors, self.normals)]
        angle = np.radians(np.mean([angle_between(i, j) for i, j in zip(self.vectors, self.normals)]))

        # Compute order parameter
        p2 = 0.5 * (3 * (np.cos(self.angle))**2 - 1)
        return p2

