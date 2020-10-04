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

    def add_results(self, title=None, lipid="All", value=None, units=None, leaflet="Both"):
        """
        Populates the self.results dataframe.
        This function is called by children metric classes.

        The title and units have been given None defaults to allow them to be replaced by commands in metric
        objects. This is useful when including multiple kinds of closely related metric in a single object.

        Parameters
        ----------
        title : str, optional
            Title of the metric. Defaults to None, which is replaced below with the self.title of the metric object.
        lipid : str, optional
            Label of the lipid species pertinent to the value. The default is "All".
        value : str or int or float, optional
            Metric value. The default is None.
        units : str, optional
            Units of the metric. Defaults to None, which is replaced below with the self.units of the metric object.
        leaflet : str, optional
            Label of the leaflet this result corresponds to. The default is "Both".

        Returns
        -------
        None.

        """
        if title is None:
            title = self.title

        if units is None:
            units = self.units

        self.results = self.results.append({"metric_title": title,
                                            "lipid": lipid,
                                            "value": value,
                                            "units": units,
                                            "leaflet": leaflet}, ignore_index=True)

