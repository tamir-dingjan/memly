"""
countlipids.py
Contains the method for counting how many lipids are present in the simulation.
"""
from collections import defaultdict, Counter
from memly.metrics import Metric


class CountLipids(Metric):
    def __init__(self, membrane, title="Number of lipids", units="count"):
        """
        Creates the CountLipids metric, and populates the results DataFrame
        with the number of residues in the simulation.

        Parameters
        ----------
        membrane : memly.membrane.Membrane object.
            The Membrane object to be analysed.
        title : str, optional. Default is "Number of lipids".
            The title of the metric.
        units : str, optional. Default is "count".
            Units of the metric.


        Returns
        -------
        None.

        """
        # Run the initialisation of the parent Metric class
        Metric.__init__(self, membrane, title, units)

        # Get the values of this metric from the membrane simulation
        self.value = {}
        self.leaflet_population = {}

        for leaflet in self.membrane.leaflets[0].keys():
            self.value[leaflet] = [len(i[leaflet]) for i in self.membrane.leaflets]

            # Use the parent class method to append this metric's results
            self.add_results(self.title, lipid="All", value=self.value[leaflet], leaflet=leaflet)

        self.count_leaflet_populations()

        for leaflet in self.leaflet_population.keys():
            for lipid_species in self.leaflet_population[leaflet]:
                self.add_results(title="Number of lipids", lipid=lipid_species,
                                 value=self.leaflet_population[leaflet][lipid_species],
                                 units="count",
                                 leaflet=leaflet)


    def count_leaflet_populations(self):
        """
        Count the number of lipids of each residue type in all leaflets, over time.

        :return:
        """
        # Convert the leaflet inventory dictionary in membrane.leaflets to replace residue index with resname
        for frame_i, frame_leaflets in enumerate(self.membrane.leaflets):
            # Replace resids with resnames and count up the occurance of each resname
            self.raw_leaflet_population = {leaflet: Counter(map(self.resid_to_resname, resids)) for leaflet, resids in frame_leaflets.items()}

            # Store as counts over time for each residue-leaflet pair
            for leaflet in self.raw_leaflet_population.keys():
                if leaflet not in self.leaflet_population.keys():
                    self.leaflet_population[leaflet] = defaultdict(list)
                # For each lipid species, store the leaflet count or 0. Counter returns a 0 for non-found items
                for lipid_species in self.membrane.residue_names:
                    self.leaflet_population[leaflet][lipid_species].append(self.raw_leaflet_population[leaflet][lipid_species])

    def resid_to_resname(self, resid):
        """
        Gives the corresponding residue name for the provided residue index (0-indexed).

        :param resid:
        :return:
        """
        return self.membrane.sim.topology.residue(resid).name
