"""
analysis.py
The main object containing the simulation data and analysis results.

Handles the primary functions
"""
import pandas as pd

from memly import loader
from memly.countlipids import CountLipids

class Analysis():    
    def __init__(self, traj, top):
        """
        Populate the analysis object with loaded trajectory.
    
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
        
        self.traj_file = traj
        self.top_file = top
        self.sim = loader.load(self.traj_file, self.top_file)
       
        # TODO Add leaflet splitting
        self.leaflet = "All"
        
        self.results = []
        
    def analyse(self):
        """
        Run all the analysis functions. The results are stored in self.results,
        and concatenated in the final step.

        Returns
        -------
        None.

        """
        
        self.results.append(CountLipids(sim=self.sim, leaflet=self.leaflet).results)        
        
        # Collect all the results into one dataframe
        self.results = pd.concat(self.results)
        