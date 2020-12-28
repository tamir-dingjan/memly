"""
analysis.py.
The main object containing the simulation data and analysis results.

Handles the primary functions
"""
import pandas as pd
import numpy as np

from .membrane import Membrane
from .countlipids import CountLipids
from .sa_analyses import SurfaceArea
from .orderparam import OrderParam
from .thickness import Thickness


class Analysis:
    def __init__(self, traj, top, load=True):
        """
        Populate the analysis object with a Membrane object.

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

        self.membrane = Membrane(traj=traj, top=top, load=load)
        self.results = []

    def run_all_analyses(self):
        """
        Run all the analysis functions. The results are stored in self.results,
        and concatenated in the final step.

        Returns
        -------
        None.

        """
        # List all analysis functions to be run here
        self.results.append(CountLipids(membrane=self.membrane).results)
        self.results.append(SurfaceArea(membrane=self.membrane).results)
        self.results.append(OrderParam(membrane=self.membrane).results)
        self.results.append(Thickness(membrane=self.membrane).results)

        # Collect all the results into one dataframe
        self.results = pd.concat(self.results)
