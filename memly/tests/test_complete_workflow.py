#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:46:05 2020

@author: tamir
"""
import os

import memly

# Setup access to datafiles
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
traj = os.path.join(THIS_DIR, "data/1.xtc")
top = os.path.join(THIS_DIR, "data/1.pdb")


x = memly.Analysis(traj, top, load=True)

lipid_vectors = memly.analysis.get_lipid_vectors(x.sim)

assert list(lipid_vectors.keys()) == [i for i in range(0,336)]



#x.split_leaflets()

