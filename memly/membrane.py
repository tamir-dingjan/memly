"""
membrane.py.
This object contains the simulation data and methods for detecting aggregates, and assigning leaflets.

"""

from collections import defaultdict
import logging
import os
from string import ascii_uppercase
from numbers import Number

import numpy as np
import point_cloud_utils as pcu
import networkx as nx

from memly import loader
from memly import particle_naming


class Membrane:
    def __init__(self, traj, top, load=True):
        """
        Instantiate the membrane object with loaded trajectory.

        :param str traj: Full filepath location to MDTraj-readable trajectory file (e.g., trr, xtc, dcd).
        :param str top: Full filepath location to MDTraj-readable topology file (e.g., pdb, gro).
        :param bool load: Flag to load trajectory.
        """

        self.load = load
        self.traj_file = traj
        self.top_file = top
        self.raw_leaflets = []
        self.leaflets = []
        self.min_leaflet_size = 10

        if self.load:
            self.sim = loader.load(self.traj_file, self.top_file)
            self.topol = self.sim.topology.to_dataframe()[0]
            # Check if simulation loaded
            if self.sim is None:
                print("ERROR: No simulation data loaded.")
                raise FileNotFoundError

        # Populate useful one-off lookups
        # Construct set for head group filter membership testing
        self.hg_set = set(self.sim.topology.select("name " + " or name ".join([x for x in particle_naming.headgroup_names])))

        # Collect lipid residue indices and lipid particle indices
        self.detected_lipids = [x.index for x in self.sim.topology.residues if (particle_naming.ion_names.isdisjoint({x.name}) and particle_naming.water_names.isdisjoint({x.name}))]
        logging.debug("Number of lipids detected: %s" % len(self.detected_lipids))
        self.residue_names = sorted(set([self.sim.topology.residue(index).name for index in self.detected_lipids]))

        # Collect all lipid particle indices
        self.lipid_particles = self.topol[~np.isin(self.topol["resName"].values, list(particle_naming.ion_names)+list(particle_naming.water_names))].index

        # OPTIONAL Collect head group particle indices - not yet sure if this is helpful
        # self.hg_particles = self.hg_set.intersection(self.lipid_particles)

        # Construct lookups linking lipid residues and particles
        self.lipid_particles_by_res = defaultdict(list)
        for index in self.topol.index:
            self.lipid_particles_by_res[self.sim.topology.atom(index).residue.index].append(index)

        self.lipid_residues_by_particle = {index: self.sim.topology.atom(index).residue.index for index in self.topol.index}

        self.hg_particles_by_res = {resid: tuple(self.hg_set.intersection(self.lipid_particles_by_res[resid])) for resid in self.detected_lipids}

        # Pre-calculate all lipid vectors
        # NOTE: This requires the original trajectory file to be PBC-corrected, so that all molecules are whole in every frame.
        # Use numpy stack to allow indexing using different numbers of head group particles in each lipid
        self.hg_centroids = np.stack([get_centroid_of_particles(self.sim, self.hg_particles_by_res[resid]) for resid in self.detected_lipids], axis=1)

        # Use numpy stack to allow indexing using different numbers of particles found in each lipid
        self.com_centroids = np.stack([get_centroid_of_particles(self.sim, self.lipid_particles_by_res[resid]) for resid in self.detected_lipids], axis=1)

        self.vectors = self.com_centroids - self.hg_centroids

        # Precalculate local neighborhood lipid vector normals
        # This is not done in a PBC-aware fashion. It therefore tends to mess up in the corners if too many nearest
        # neighbors are selected for the normal estimation, by dragging the vectors towards the box COM.
        self.normals = np.asarray([pcu.estimate_normals(frame, k=20) for frame in self.hg_centroids])

        # The normal detection from the point cloud doesn't correctly estimate the Z-direction. The leaflets are
        # assigned using the lipid vectors rather than the normal vectors. Here, the lipid vectors can also be
        # used to correct the polarity of the normals.

        # Correct "positive" normals - these are the normals which point up from the upper leaflet towards
        # +ve Z-direction.
        # Where the lipid vectors and normals are negative (i.e., pointing down-Z) make the normals positive (i.e., pointing up-Z)
        # Likewise, correct "negative" normals - those pointing down from the lower leaflet towards -ve Z
        # Where the lipid vectors and normals are positive (i.e., pointing up-Z) make the normals negative (i.e., pointing down-Z)
        pos_inversion = (self.vectors[:, :, 2] < 0) & (self.normals[:, :, 2] < 0)
        neg_inversion = (self.vectors[:, :, 2] > 0) & (self.normals[:, :, 2] > 0)

        inversion_slice = 1 + (np.logical_or(pos_inversion, neg_inversion) * -2)
        self.normals[:, :, 2] = self.normals[:, :, 2] * inversion_slice

        # Detect leaflets using vector alignment
        self.detect_leaflets()

        # Store lookup for residue leaflet occupancy
        self.leaflet_occupancy_by_resid = defaultdict(list)
        # Shape: X[resid] = [upper upper lower upper ... ]
        for frame_leaflets in self.leaflets:
            for resid in self.detected_lipids:
                if resid in set(frame_leaflets["upper"]):
                    self.leaflet_occupancy_by_resid[resid].append("upper")
                elif resid in set(frame_leaflets["lower"]):
                    self.leaflet_occupancy_by_resid[resid].append("lower")
                elif resid in set(frame_leaflets["aggregate"]):
                    self.leaflet_occupancy_by_resid[resid].append("aggregate")
                else:
                    self.leaflet_occupancy_by_resid[resid].append("none")


    def detect_leaflets(self):
        """
        Detect the raw leaflets, built from neighbor connectedness (criteria: head group distance and
        normal vector alignment).
        :return:
        """

        for frame_index, frame in enumerate(self.sim):
            logging.debug("Leaflet detection frame: %s" % frame_index)

            # Instantiate the network graph with all lipids
            g = nx.Graph()
            g.add_nodes_from(self.detected_lipids)

            # How to add all graph edges at once?
            # Need to package up the edges as an iterable of tuples
            # map(g.add_edges_from, self.detected_lipids,

            # For each unconnected lipid in the network
            for lipid in self.detected_lipids:
                if len(g[lipid]) == 0:
                    # Connect it to all its neighbors (multiple connections per node to prevent dead ends in linkage)
                    for nbor in self.get_lipid_nbors(frame_index, lipid):
                        # Ignore the self-matched neighbor
                        if nbor == lipid:
                            continue
                        g.add_edge(lipid, nbor)

            # Merge self-contained networks to give leaflets and aggregates
            self.raw_leaflets.append(list(nx.connected_components(g)))

        # Categorise the leaflets
        for frame_index, leaflets in enumerate(self.raw_leaflets):
            categorised = defaultdict(list)

            # Classify leaflets below minimum size as aggregate
            for leaflet in leaflets:
                # Assign small aggregates
                if len(leaflet) < self.min_leaflet_size:
                    categorised["aggregate"] += list(leaflet)
                    continue

                # Remaining leaflets are large enough to be considered a membrane
                # Check the average lipid vector orientation to assign upper vs lower
                avg_leaflet_vector = np.mean(self.vectors[frame_index, list(leaflet)], axis=0)
                if avg_leaflet_vector[2] < 0:
                    categorised["upper"] += list(leaflet)
                else:
                    categorised["lower"] += list(leaflet)

            self.leaflets.append(categorised)

    def get_lipid_nbors(self, frame, query_lipid):
        """
        Get the neighboring lipids for the provided query lipid residue index.
        The requirements for a nearby lipid to be considered a neighbor are:
            1. Head group centroid is within the (Euclidean) distance threshold of the query lipid headgroup centroid.
            2. Normal vector is within the tilt threshold of the query lipid normal vector.
        :param int frame: Simulation frame to check in (0-indexed).
        :param int query_lipid: Residue index of the query lipid.
        :return: NDArray of neighboring lipid residue indices
        """

        threshold = 1
        tilt = 30

        # First check Euclidean-near lipids to the query head group
        # Compare the distances to every other head group centroid - could do this as a mask operation in init?
        nbors_distance = np.flatnonzero(np.all(np.abs(self.hg_centroids[frame] - self.hg_centroids[frame][query_lipid]) < threshold, axis=1))

        # Then check the normal vector alignment
        nbors_tilt = np.flatnonzero(np.asarray([angle_between(self.normals[frame][i], self.normals[frame][query_lipid]) for i in nbors_distance]) < tilt)

        # Final neighbors meet both requirements
        nbors = nbors_distance[nbors_tilt]

        return nbors


def export_frame_with_normals(frame, hg_centroids, normals, output_path):
    frame.save_pdb(output_path)

    # Normal vectors are specified in unit circles - need to extend these the greater magnitude to make
    # them more obvious
    normals = 5 * normals

    # Calculate the spatial end positions of normal vectors
    termini = hg_centroids + normals

    # Add centroids and termini to the frame
    converted_path = os.path.splitext(output_path)[0] + \
                     "_converted" + \
                     os.path.splitext(output_path)[1]
    with open(output_path, 'r') as fin, open(converted_path, 'w') as fout:
        atom_num = 0
        conect_records = []
        for line in fin:
            # Transfer over existing coordinate lines
            if line[:4] == 'ATOM':
                atom_num = int(line[6:11].strip())
                fout.write(line)
            elif line[:3] == 'TER':
                # Insert new particles before TER
                for centroid, terminus in zip(hg_centroids, termini):
                    atom_num += 1
                    # Write the start and end coordinates for the normal vector
                    # Note: MDTraj uses nm coordinates, while PDB exports use Angstrom, so convert by 10.
                    fout.write("ATOM  " +
                               '{:>5}'.format(atom_num) +
                               " VNC  VEC V   1    " +
                               '{:8.3f}'.format(10 * centroid[0]) +
                               '{:8.3f}'.format(10 * centroid[1]) +
                               '{:8.3f}'.format(10 * centroid[2]) +
                               "  1.00  0.00          VP\n")
                    atom_num += 1
                    fout.write("ATOM  " +
                               '{:>5}'.format(atom_num) +
                               " VNT  VEC V   1    " +
                               '{:8.3f}'.format(10 * terminus[0]) +
                               '{:8.3f}'.format(10 * terminus[1]) +
                               '{:8.3f}'.format(10 * terminus[2]) +
                               "  1.00  0.00          VP\n")
                    # Save CONECT record
                    conect_records.append("CONECT" + '{:>5}'.format(atom_num-1) + '{:>5}'.format(atom_num) + "\n")
                fout.write("TER   " + '{:>5}'.format(atom_num) + "\n")
            elif line[:3] == 'END':
                # Write CONECT records before END
                for rec in conect_records:
                    fout.write(rec)
            else:
                fout.write(line)


def get_centroid_of_particles(sim, particles):
    return np.mean(sim.xyz[:, particles, :], axis=1)


def get_lipid_vector(frame, hg_particles_by_res, lipid_particles_by_res, residue_index):
    """
    Return the lipid vector for the chosen residue in the given frame.

    :param frame:
    :param hg_particles_by_res:
    :param lipid_particles_by_res:
    :param residue_index:
    :return:
    """

    # Get the coordinates of the head group particles
    hg_centroid = np.mean(frame.xyz[:, np.asarray(hg_particles_by_res[residue_index]).flatten(), :], axis=1)
    com = np.mean(frame.xyz[:, np.asarray(lipid_particles_by_res[residue_index]).flatten(), :], axis=1)

    # Assemble vector
    vector = com - hg_centroid

    return vector.flatten()


def export_labelled_snapshot(frame, labels, output_path):
    # MDTraj has no way to set the chain ID for different residues.
    # So, this routine saves out a snapshot, then edits the chainID field
    # to labels chains from A-Z for each of the labels.
    # Labels should be a dictionary of format: {label: [resid, resid, resid...]}

    # Save the snapshot file.
    frame.save_pdb(output_path)

    # Assign labels to chains
    chains = {}
    for label in labels.keys():
        # If the label is not a number, cast a new label that is.
        if not isinstance(label, Number):
            new_label = sorted(list(labels.keys())).index(label)
        else:
            new_label = label

        for resid in labels[label]:
            chains[int(resid)] = ascii_uppercase[int(new_label) % len(ascii_uppercase)]
            logging.debug("Assigned chain %s to resid %s." % (chains[int(resid)], int(resid)))

    logging.debug("Total number of residues with assigned chains: %s" % (len(chains.keys())))
    logging.debug("Residue indices with assigned chains: %s" % chains.keys())
    # Edit the snapshot chain IDs
    converted_path = os.path.splitext(output_path)[0] + \
                     "_converted" + \
                     os.path.splitext(output_path)[1]
    with open(output_path, 'r') as fin, open(converted_path, 'w') as fout:
        for line in fin:
            # Apply conversion on ATOM lines
            if line[:4] == 'ATOM':
                pre_chain = line[:21]
                post_chain = line[22:]
                # Decrement the resid to move to 0-indexing
                resid = int(line[22:26].strip()) - 1
                if resid in chains.keys():
                    logging.debug("Assigned new chain for resid %s: %s" % (resid, chains[resid]))
                    fout.write(pre_chain + chains[resid] + post_chain)
                else:
                    fout.write(line)
            else:
                fout.write(line)


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors v1 and v2."""
    # TODO: Add an accelerated implementation using Numba, as in thickness.py
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
