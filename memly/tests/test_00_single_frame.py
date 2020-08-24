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

np.testing.assert_allclose(np.round(memly.membrane.get_lipid_vector(x.sim[0],
                                                                    x.hg_particles_by_res,
                                                                    x.lipid_particles_by_res,
                                                                    0),
                                    3),
                           np.asarray([-0.476,  0.743, -0.687]))

assert len(x.leaflets[0]["upper"]) == 168
assert len(x.leaflets[0]["lower"]) == 162
assert len(x.leaflets[0]["aggregate"]) == 6

analyser = memly.Analysis(traj, top, load=True)

analyser.run_all_analyses()

