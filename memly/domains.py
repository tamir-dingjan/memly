"""
domnains.py

A metric to detect domains of lipids and measure their contents relative to the whole bilayer.
"""

from scipy.spatial import Delaunay
from collections import defaultdict
from itertools import combinations
import numpy as np

from memly.metrics import Metric


class Domains(Metric):
    def __init__(self, membrane, title="Domains", units="unitless"):
        """
        Detects lipid domains using Delaunay triangulation.
        Labels domains according to their contents relative to the whole bilayer.

        :param membrane: memly.membrane.Membrane objects
        :param title: str, optional.
        :param units: str, optional.
        """

        # Rune parent init
        Metric.__init__(self, membrane, title, units)

        # Setup containers
        # The lipid type lookup gives scores for each lipid that contributes to "domain likeness"
        self.lipid_type_lookup = {'CHOL': 1,
                                  'CEPP': 1,
                                  'CEPX': 1,
                                  'DSLB': 1,
                                  'DSLP': 1,
                                  'GCPP': 1,
                                  'PAPP': 1,
                                  'PCLL': 1,
                                  'PCPB': 1,
                                  'PCPP': 1,
                                  'PEPB': 1,
                                  'PEPP': 1,
                                  'PSPP': 1,
                                  'SMPB': 1,
                                  'SMPP': 1,
                                  'SMPX': 1,
                                  'PGPP': 1,
                                  'PSBB': 1}

        self.projection = {}
        self.delaunay_tri = {}
        self.neighbors = {}
        self.domain_positions = {}

        for leaflet in ["upper", "lower"]:
            self.get_projection(leaflet)
            self.triangulate(leaflet)
            self.get_tri_neighbors(leaflet)

        self.label_domains()

        # Log the results
        for leaflet in ["upper", "lower"]:
            self.add_results(lipid="All", value=self.domain_positions[leaflet], leaflet=leaflet)


    def get_projection(self, leaflet="upper"):
        """
        Construct the 2D projection for the given leaflet.

        The resulting projection is a list of length n_frames, where each element is
        a numpy.ndarray of shape (n_head_groups, 2) containing the XY projection
        for each head group particle in the frame.

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

    def triangulate(self, leaflet="upper"):
        """
        Perform the Delaunay triangulation for the specified leaflet.

        :param leaflet:
        :return:
        """
        # Run the triangulation on each frame's projection.
        self.delaunay_tri[leaflet] = [Delaunay(x) for x in self.projection[leaflet]]

    def get_tri_neighbors(self, leaflet="upper"):
        """
        Identify the neighboring lipids from the Delaunay triangulation vertices.

        :return:
        """

        self.neighbors[leaflet] = []

        for frame_i, tri in enumerate(self.delaunay_tri[leaflet]):
            nbors = defaultdict(set)

            # Each vertex in the triangulation is a point on the projection.
            # The index of points in the projection corresponds to lipid residue numbers in self.membrane.leaflets
            # So, want to use the triangulation to find neighboring lipid residues.

            for p in tri.vertices:
                for i, j in combinations(p, 2):
                    nbors[self.membrane.leaflets[frame_i][leaflet][i]].add(self.membrane.leaflets[frame_i][leaflet][j])
                    nbors[self.membrane.leaflets[frame_i][leaflet][j]].add(self.membrane.leaflets[frame_i][leaflet][i])

            self.neighbors[leaflet].append(nbors)

    def get_domain_score(self, lipid_resids):
        """
        Return the score of the domain containing the supplied lipid residues.

        The score is calculated by adding a point for certain lipid types, and averaging the sum by the number of
        lipids. Thus, the score ranges from 0 to 1. Depending on which lipids are scored, domains can be selected for
        different lipid species.

        :param lipid_resids:
        :return:
        """

        # Translate lipid residues into residue names/types
        lipid_names = [self.membrane.sim.topology.residue(x).name for x in lipid_resids]

        # Get "domain score" for these lipids
        lipid_types = [self.lipid_type_lookup.get(x, 0) for x in lipid_names]
        domain_score = np.sum(lipid_types) / len(lipid_types)

        return domain_score

    def label_domains(self):
        """
        The domains, defined by a lipid with its neighbors, can contain a different makeup of lipids relative to the
        whole leaflet composition. In particular, domains enriched in cholesterol and saturated phospholipids can be
        considered to be lipid rafts, according to the line of argument put forward by Kusumi:
            Kusumi, A. et al. Defining raft domains in the plasma membrane. Traffic vol. 21 106–137 (2020).

        This approach requires computing the domain and leaflet composition, then comparing the two with regards to
        cholesterol content and saturated acyl chain content. Domains which meet the raft identification criteria
        can then be saved - in particular, the 2D domain coordinates in the projection can be collected for later
        graphing.

        :return:
        """

        # TODO: Make this faster using matrices - could calculate the domain scores all in one go.

        for leaflet in ["upper", "lower"]:
            self.domain_positions[leaflet] = {}

            for frame_i, frame_nbors in enumerate(self.neighbors[leaflet]):
                self.domain_positions[leaflet][frame_i] = []
                domain_positions_raw = []

                # Get the leaflet composition - only need to do this once per frame
                leaflet_score = self.get_domain_score(self.membrane.leaflets[frame_i][leaflet])

                for core, nbors in frame_nbors.items():
                    # The domain cores and nbors are lipid residue numbers.
                    # Get the composition of this domain from the lipid residue numbers
                    domain_composition = list(nbors)
                    domain_composition.append(int(core))
                    domain_score = self.get_domain_score(domain_composition)

                    # Compare domain composition to leaflet composition
                    # The domain score ranges from 0 to 1. How to set a threshold for saving a domain?
                    # If the domain score is greater than the leaflet average, that selects for too many lipids.
                    # This threshold represents the % of the domain lipids that must be in the desired types.
                    if (domain_score > 0.8):
                        # Apply label / store domain position
                        # The order of head group centroids corresponds with the residue indices in membrane.detected_lipids
                        # So, can find the positions of the lipids in the domain by first finding which index
                        # each lipid is in membrane.detected_lipids, and selecting head group centroids with those indices.
                        # Note that the positions of residue indices in membrane.detected_lipids will only match the
                        # residue index number if the topology begins with lipids, and is uninterrupted.

                        # Save the position of the domain core only
                        domain_positions_raw.append(self.membrane.hg_centroids[frame_i, self.membrane.detected_lipids.index(core), :2])

                        # Saving the entire domain composition results in overlap, since each lipid can be a neighbor to multiple others
                        #domain_positions_raw.append(self.membrane.hg_centroids[frame_i, [self.membrane.detected_lipids.index(x) for x in domain_composition], :2])

                # Concat the domain positions
                self.domain_positions[leaflet][frame_i] = np.stack(domain_positions_raw)
