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
traj = os.path.join(THIS_DIR, "data/1.pdb")
top = os.path.join(THIS_DIR, "data/1.pdb")

x = memly.Membrane(traj, top, load=True)
assert len(x.leaflets[0]["upper"]) == 168
assert np.max(x.leaflets[0]["lower"]) == 335

lipid_vector = memly.membrane.get_lipid_vector(x.sim[0], 0)
np.testing.assert_allclose(np.round(lipid_vector,3), np.asarray([-0.476, 0.743, -0.687]))

leaflets = memly.membrane.detect_aggregates(x.sim[0], neighbor_cutoff=3, merge_cutoff=1)
assert len(leaflets) == 2
assert len(leaflets[0]) == 168
assert np.max(leaflets[0]) == 199
