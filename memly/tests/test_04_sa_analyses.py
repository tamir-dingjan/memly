"""
Test the surface area analysis functions.
"""

import os
import numpy as np
import memly

from memly import sa_analyses

# Setup access to datafiles
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
traj = os.path.join(THIS_DIR, "data/2.xtc")
top = os.path.join(THIS_DIR, "data/2.pdb")

analyser = memly.Analysis(traj, top, load=True)

sa_obj = sa_analyses.SurfaceArea(membrane=analyser.membrane)

