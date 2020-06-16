"""
countlipids.py
Contains the method for counting how many lipids are present in the simulation.
"""

from memly.metrics import Metric


class CountLipids(Metric):
    def __init__(self, sim, leaflet, title="Number of lipids", units="#"):
        """
        Creates the CountLipids metric, and populates the results DataFrame
        with the number of residues in the simulation.

        Parameters
        ----------
        sim : Trajectory.
            An mdtraj trajectory.
        title : str, optional. Default is "Number of lipids".
            The title of the metric.
        units : str, optional. Default is "#".
            Units of the metric.
        leaflet : str
            Label for the area of the simulation used to calculate the metric.

        Returns
        -------
        None.

        """
        
        Metric.__init__(self, sim, title, units, leaflet)
        self.add_results(lipid="All", value=self.sim.topology.n_residues)

