"""
metrics.py
Contains the base class for an analysis metric.

Handles the wrapping of metric data.
"""

import pandas as pd

class Metric():
    """
    Generic container for an analysis metric.
    """
    def __init__(self, sim, title, units, leaflet):
        """
        Initialise the metric with basic information.

        Parameters
        ----------
        sim : Trajectory.
            An mdtraj trajectory.
        title : str
            The title of the metric.
        units : str
            Units of the metric.
        leaflet : str
            Label for the area of the simulation used to calculate the metric.

        Returns
        -------
        None.

        """
        self.sim = sim
        self.title = title
        self.units = units
        self.leaflet = leaflet
        self.columns = ["metric_title", "lipid", "value", "units", "leaflet"]
        self.results = pd.DataFrame(data=None, index=None, columns=self.columns)
        
        
    def add_results(self, lipid=None, value=None):
        """
        Populates the self.results dataframe, called by children metric classes.

        Parameters
        ----------
        lipid : str, optional
            Label of the lipid species pertinent to the value. The default is None.
        value : str or int or float, optional
            Metric value. The default is None.

        Returns
        -------
        None.

        """
        self.results = self.results.append({"metric_title": self.title,
                                            "lipid": lipid,
                                            "value": value,
                                            "units": self.units,
                                            "leaflet": self.leaflet}, ignore_index=True)
        
        
    