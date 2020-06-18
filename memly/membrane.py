"""
membrane.py.
This object contains the simulation data and methods for detecting aggregates, and assigning leaflets.

"""

from collections import defaultdict
import logging
import os
from string import ascii_uppercase

import numpy as np
import mdtraj as md

from memly import loader
from memly import particle_naming

logging.basicConfig(level=logging.INFO)


class Membrane:
    def __init__(self, traj, top, load=True):
        """
        Instantiate the membrane object with loaded trajectory.

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

        self.raw_leaflets = [detect_aggregates(frame) for frame in self.sim]
        self.categorise_leaflets(min_leaflet_size=10)

    def categorise_leaflets(self, min_leaflet_size=10):
        """
        Sort the detected leaflets to apply the labels used in metric analysis.

        Process:
            1. Leaflets with fewer than min_leaflet_size lipids are labelled "aggregate"
            2. Find the average lipid vector orientation of large leaflets
            3. Define "upper" as vectors pointing towards negative Z axis

        :return:
        """
        self.leaflets = []

        for frame, leaflets in zip(self.sim, self.raw_leaflets):
            categorised = defaultdict(list)

            # Remaining leaflets are large enough to be considered a membrane
            for leaflet_id in leaflets.keys():
                # Assign small aggregates
                if len(leaflets[leaflet_id]) < min_leaflet_size:
                    categorised["aggregate"] += leaflets[leaflet_id]
                    continue
                # Get lipid vectors for all lipids in the leaflet
                leaflet_vectors = np.asarray([get_lipid_vector(frame,resid) for resid in leaflets[leaflet_id]])
                avg_leaflet_vector = np.mean(leaflet_vectors, axis=1)
                # Get orientation of leaflet w.r.t Z axis
                if avg_leaflet_vector[2] < 0:
                    categorised["upper"] += leaflets[leaflet_id]
                else:
                    categorised["lower"] += leaflets[leaflet_id]
            self.leaflets.append(categorised)


def detect_aggregates(frame, neighbor_cutoff=10, merge_cutoff=15):
    """
    Detect aggregates in a single simulation frame.
    :param frame:
    :param neighbor_cutoff:
    :param merge_cutoff:
    :return:
    """

    # Initialise
    processed = {}
    aggregates = defaultdict(list)
    agg_id = -1

    # Construct list of lipid residues
    detected_lipids = []
    for residue in frame.topology.residues:
        if residue.name == 'ION' or residue.name == 'W':
            continue
        else:
            detected_lipids.append(residue.index)
    logging.debug("Number of lipids detected: %s" % len(detected_lipids))

    # Assign each lipid residue to an aggregate
    for lipid in detected_lipids:
        # Proceed if residue has not been processed
        if processed.get(lipid, False):
            continue
        # Add lipid to a new aggregate
        agg_id += 1
        aggregates[agg_id].append(lipid)
        # Mark lipid as processed
        processed[lipid] = True
        # Get the lipid neighbourhood
        query = frame.topology.select("resid " + str(lipid))
        excluded_names = " and not resname ".join(particle_naming.water_names) + \
                         " and not resname " + \
                         " and not resname ".join(particle_naming.ion_names)
        haystack = frame.topology.select("not resname " + excluded_names)
        nhood_particles = md.compute_neighbors(traj=frame,
                                               cutoff=neighbor_cutoff,
                                               haystack_indices=haystack,
                                               query_indices=query)[0]
        nhood = sorted(set([frame.topology.atom(particle).residue.index for particle in nhood_particles]))
        # For each neighbor
        for nbor in nhood:
            # Proceed if residue has not been processed
            if processed.get(nbor, False):
                continue
            # Proceed if vectors are co-linear
            if not (angle_between(get_lipid_vector(frame, lipid), get_lipid_vector(frame, nbor)) < 90):
                continue
            # Add neighbor to aggregate
            aggregates[agg_id].append(nbor)
            # Mark neighbor as processed
            processed[nbor] = True

    logging.debug("Collected %s aggregates" % len(aggregates.keys()))

    # Merge aggregates
    # Where aggregate lipids have head groups close to the head groups of other lipids, those
    # aggregates should be considered as the same. However, the proximity should be smaller than the bilayer width.
    merged = {}
    leaflets = defaultdict(list)
    leaflet_id = -1
    for agg_id in aggregates.keys():
        # Proceed if not already merged
        if merged.get(agg_id, False):
            continue
        leaflet_id += 1
        leaflets[leaflet_id] += aggregates[agg_id]
        logging.debug("New leaflet: %s (%s lipids)" % (leaflet_id, len(leaflets[leaflet_id])))
        merged[agg_id] = True
        for agg_to_join in aggregates.keys():
            # Proceed if not already merged
            if merged.get(agg_to_join, False):
                continue
            # If the agg_to_join lipid head groups are close to those in the current leaflet, add to leaflet
            # MDTraj doesn't include a library for finding the minimum distance between two groups,
            # so evaluate this using compute_neighbors with the decision threshold as the cutoff.
            query_string = " or resid ".join(str(x) for x in aggregates[agg_id])
            all_particles_query = frame.topology.select("resid " + query_string)
            hg_query = [particle for particle in all_particles_query if
                        frame.topology.atom(particle).name in particle_naming.headgroup_names]

            haystack_string = " or resid ".join(str(x) for x in aggregates[agg_to_join])
            all_particles_haystack = frame.topology.select("resid " + haystack_string)
            hg_haystack = [particle for particle in all_particles_haystack if
                           frame.topology.atom(particle).name in particle_naming.headgroup_names]

            logging.debug("Searching for mergeables: leaflet %s (%s lipids), haystack agg %s (%s lipids)" %
                          (leaflet_id, len(leaflets[leaflet_id]), agg_to_join, len(aggregates[agg_to_join])))
            contacts = md.compute_neighbors(traj=frame,
                                            cutoff=merge_cutoff,
                                            haystack_indices=hg_haystack,
                                            query_indices=hg_query)[0]
            # Check if any contact between the two aggregates
            logging.debug(("Found %s particles within merge_cutoff of leaflet %s" %
                           (len(contacts), leaflet_id)))
            if len(contacts) == 0:
                # If this aggregate has no contacts, it may only be in proximity to aggregates that have already
                # been processed and are now allocated to leaflets. So, this aggregate may still belong to a
                # leaflet. How to handle this aggregate?
                # 1. Check at end when we have more leaflets?
                # 2. Check against all currently existing leaflets?
                # 3. Something else?

                continue
            leaflets[leaflet_id] += aggregates[agg_to_join]
            merged[agg_to_join] = True
            logging.debug("Merged aggregates %s and %s" % (agg_id, agg_to_join))
    logging.debug("Found %s leaflets." % len(leaflets.keys()))
    return leaflets


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
        for resid in labels[label]:
            chains[int(resid)] = ascii_uppercase[int(label) % len(ascii_uppercase)]
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
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def get_lipid_vector(frame, residue_index):
    """
    Return the lipid vector for the chosen residue in the given frame.

    :param frame:
    :param residue_index:
    :return:
    """
    # Get the atom indices of the selected particles
    hg_indices = [atom.index for atom in frame.topology.residue(residue_index).atoms if
                  atom.name in particle_naming.headgroup_names]
    all_indices = [atom.index for atom in frame.topology.residue(residue_index).atoms]

    if len(hg_indices) == 0:
        raise Exception("Couldn't find head group indices for residue %i." % residue_index)
    elif len(all_indices) == 0:
        raise Exception("Couldn't find atom indices for residue %i." % residue_index)

    # Get the coordinates of the head group particles
    hg_centroid = np.mean(frame.xyz[:, hg_indices, :], axis=1)
    com = np.mean(frame.xyz[:, all_indices, :], axis=1)

    # Assemble vector
    vector = com - hg_centroid

    return vector.flatten()
