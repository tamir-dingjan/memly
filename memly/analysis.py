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
        self.lipid_vectors = get_lipid_vectors(self.sim)

        # Create empty mask
        self.leaflet_mask = np.zeros((self.sim.n_frames, len(self.lipid_vectors.keys())))

        # Populate leaflet mask
        #for residue in self.lipid_vectors.keys():





        # Label leaflet mask with upper, lower, and aggregate


def compare_lipid_vectors(a, b, origin_cutoff, collinear_cutoff):
    # Check if vector origins within distance cutoff
    origin_distance = np.linalg.norm(a[0]-b[0], axis=1)

    # Check if vectors are roughly co-linear
    collinear = []
    for a_magnitude, b_magnitude in zip(a[1], b[1]):
        collinear.append(np.degrees(angle_between(a_magnitude, b_magnitude)))
    collinear = np.asarray(collinear)

    # Check that both comparisons completed on all data
    if not (len(origin_distance) == len(collinear)):
        raise Exception("Lipid vector comparison failed.")

    # Create comparison mask
    mask = (origin_distance < origin_cutoff) & (collinear < collinear_cutoff)
    return mask




def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors v1 and v2."""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



def get_lipid_vectors(sim):
    """Construct the lipid vectors for the provided simulation.

    Lipid vectors start at the centroid of the head group coordinates, and terminate at the centroid of the
    coordinates for the whole lipid molecule. To define which particles to use for the head group, use a
    premade list of suitable particle names.

    The lipid vectors are returned as:
    [[head_X, head_Y, head_Z],
     [magn_X, magn_Y, magn_Z]],
    where head_X/Y/Z are the centroid coordinates of the head group, and magn_X/Y/Z are the vector magnitudes.

    These are stored by residue indices.


    :param sim:
    :return:
    """
    lipid_vectors = {}

    # Get the head group centroids for each residue
    hg_centroids = get_centroids(sim, selection="head")

    # Get the whole lipid centroids for each residue
    whole_centroids = get_centroids(sim, selection="all")

    # Check if both centroids have the same number of entries
    if not (len(hg_centroids.keys()) == len(whole_centroids.keys())):
        raise Exception("Centroid lists are different lengths.")

    # Calculate the vector magnitude from head group to whole lipid centroid
    for residue in hg_centroids.keys():
        magnitude = whole_centroids[residue] - hg_centroids[residue]
        lipid_vectors[residue] = [hg_centroids[residue], magnitude]

    return lipid_vectors

def get_centroids(sim, selection="all"):
    """Returns the coordinates of the selected particles present in the simulation.
    This has to be done per-residue.

    Use the "selection" argument to specify which particles to return centroids for.
    "all" - get centroid for each residue
    "head" - get centroid for headgroup particles only

    :param sim:
    :param selection:
    :return:
    """
    centroids = {}

    for residue in sim.topology.residues:
        # Check residue is not water or ion
        if residue.name == 'ION' or residue.name == 'W':
            continue
        else:
            # Get the atom indices of the selected particles
            if selection == "head":
                indices = [atom.index for atom in residue.atoms if atom.name in headgroups.names]
            elif selection == "all":
                indices = [atom.index for atom in residue.atoms]
            if indices is None:
                raise ValueError("No atom indices found.")
            # Get the coordinates of the head group particles
            coordinates = sim.xyz[:, indices, :]
            # Get the centroid of the coordinates
            centroids[residue.index] = np.mean(coordinates, axis=1)
    return centroids

