"""
countlipids.py
Contains the method for counting how many lipids are present in the simulation.
"""

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

        for leaflet in self.membrane.leaflets[0].keys():
            self.value[leaflet] = [len(i[leaflet]) for i in self.membrane.leaflets]

            # Use the parent class method to append this metric's results
            self.add_results(lipid="All", value=self.value[leaflet], leaflet=leaflet)
