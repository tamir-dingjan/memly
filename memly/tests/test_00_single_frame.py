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
# assert len(x.leaflets[0]["upper"]) == 168
# assert np.max(x.leaflets[0]["lower"]) == 335

np.testing.assert_allclose(np.round(memly.membrane.get_lipid_vector(x.sim[0], x.hg_particles_by_res, x.lipid_particles_by_res, 0),3), np.asarray([-0.476, 0.743, -0.687]))

assert len(x.raw_leaflets) == len(x.sim)
assert len(x.raw_leaflets[0][0]) == 168
assert np.max(x.raw_leaflets[0][0]) == 199
assert np.median(x.raw_leaflets[0][0]) == 83.5
assert np.min(x.raw_leaflets[0][0]) == 0

assert len(x.leaflets) == len(x.sim)
assert len(x.leaflets[0]["upper"]) == 168
assert np.max(x.leaflets[0]["upper"]) == 199
assert np.median(x.leaflets[0]["upper"]) == 83.5
assert np.min(x.leaflets[0]["upper"]) == 0