"""
metrics.py
Contains the base class for an analysis metric.

Handles the wrapping of metric data.
"""

import pandas as pd


class Metric:
    """
    Generic container for an analysis metric.
    """
    def __init__(self, membrane, title, units):
        """
        Initialise the metric with basic information.

        Parameters
        ----------
        membrane : memly.membrane.Membrane object.
            The Membrane object to be analysed.
        title : str
            The title of the metric.
        units : str
            Units of the metric.


        Returns
        -------
        None.

        """
        self.membrane = membrane
        self.title = title
        self.units = units
        self.columns = ["metric_title", "lipid", "value", "units", "leaflet"]
        self.results = pd.DataFrame(data=None, index=None, columns=self.columns)

    def add_results(self, lipid="All", value=None, leaflet="Both"):
        """
        Populates the self.results dataframe.
        This function is called by children metric classes.

        Parameters
        ----------
        lipid : str, optional
            Label of the lipid species pertinent to the value. The default is "All".
        value : str or int or float, optional
            Metric value. The default is None.
        leaflet : str, optional
            Label of the leaflet this result corresponds to. The default is "Both".

        Returns
        -------
        None.

        """
        self.results = self.results.append({"metric_title": self.title,
                                            "lipid": lipid,
                                            "value": value,
                                            "units": self.units,
                                            "leaflet": leaflet}, ignore_index=True)

