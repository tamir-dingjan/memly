#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:46:05 2020

@author: tamir
"""
import os
import numpy as np
import memly

# Setup access to datafiles
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
traj = os.path.join(THIS_DIR, "data/1.xtc")
top = os.path.join(THIS_DIR, "data/1.pdb")

x = memly.Membrane(traj, top, load=True)

lipid_vector = memly.membrane.get_lipid_vector(x.sim[0], 0)
np.testing.assert_allclose(np.round(lipid_vector,3), np.asarray([0.003, 0.486, -1.064]))

leaflets = memly.membrane.detect_aggregates(x.sim[0], neighbor_cutoff=1, merge_cutoff=1)
print(" or resi ".join(str(i) for i in leaflets[0]))