"""
analysis.py.
The main object containing the simulation data and analysis results.

Handles the primary functions
"""
import pandas as pd
import numpy as np


from memly import loader
from memly import headgroups
from memly.countlipids import CountLipids


class Analysis:
    def __init__(self, traj, top, load=True):
        """
        Populate the analysis object with loaded trajectory.

        Parameters
        ----------
        traj : str
            Full filepath specifying the location of simulation trajectory.
        top : str
            Full filepath specifying the location of topology file.

        Returns
        -------
        None.

        """

        self.load = load
        self.traj_file = traj
        self.top_file = top
        if self.load:
            self.sim = loader.load(self.traj_file, self.top_file)
            # Check if simulation loaded
            if self.sim is None:
                print("ERROR: No simulation data loaded.")
                raise FileNotFoundError

        # TODO Add leaflet splitting
        self.leaflet = "All"
        # self.split_leaflets()

        self.results = []

    def analyse(self):
        """
        Run all the analysis functions. The results are stored in self.results,
        and concatenated in the final step.

        Returns
        -------
        None.

        """
        self.results.append(CountLipids(sim=self.sim, leaflet=self.leaflet).results)

        # Collect all the results into one dataframe
        self.results = pd.concat(self.results)

    def split_leaflets(self):
        """Separate the upper and lower leaflets of the simulation.

        The separated leaflets are stored as lists of residue numbers, which can be applied as masks
        by analysis functions.

        Leaflet assignment is done by the following process:
        1. Represent lipids as vectors, constructed from centroid of lipid head group to centroid of whole-lipid
        2. Assign lipids to leaflets based on comparison of lipid vectors:
            2.1. Origins must be within a cutoff
            2.2. Orientation must be roughly colinear
        3. Leaflets are stored in descending order of average vector origin Z-coordinate values, so the upper leaflet
           is the first one.

        To accommodate lipid flip-flop, leaflets are assigned in each frame.

        :return:
        """
        print("Leaflet splitter")

        # Construct lipid vectors

        # Assign lipid vectors to leaflets

        # Store leaflet mask


def get_lipid_vectors(sim, headgroups):
    """Construct the lipid vectors for the provided simulation.

    Lipid vectors start at the centroid of the head group coordinates, and terminate at the centroid of the
    coordinates for the whole lipid molecule. To define which particles to use for the head group, use a
    premade list of suitable particle names.

    The lipid vectors are returned as a matrix of shape <n_frames>, <n_lipids>. Each lipid vector is formatted as:
    [[head_X, head_Y, head_Z],
     [magn_X, magn_Y, magn_Z]],
    where head_X/Y/Z are the centroid coordinates of the head group, and magn_X/Y/Z are the vector magnitudes.


    :param sim:
    :param headgroups:
    :return:
    """


    # Get the head group centroids for each residue


    # Get the whole lipid centroids for each residue

    # Calculate the vector magnitude from head group to whole lipid centroid


def get_head_group_particles(sim):
    """Returns the coordinates of the named head group particles present in the simulation.
    This has to be done per-residue.

    :param sim:
    :param headgroups:
    :return:
    """
    hg_indices = {}

    for residue in sim.topology.residues:
        # Check residue is not water or ion
        if residue.name == 'ION' or residue.name == 'W':
            continue
        else:
            # Get the atom indices of the head group particles
            hg_indices[residue.index] = [atom.index for atom in residue.atoms if atom.name in headgroups.names]
    return hg_indices


def get_centroid(coords):
    """Returns the centroid of the provided coordinates.

    Coordinates should be a matrix in the shape <n_frames>, <n_particles>, 3.
    The returned centroids are a matrix of shape <n_frames>, <n_centroids>, 3.

    :type coords: object
    :param coords:
    :return:
    """

    return np.mean(coords)

