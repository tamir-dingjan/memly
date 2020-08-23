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
traj = os.path.join(THIS_DIR, "data/2.xtc")
top = os.path.join(THIS_DIR, "data/2.pdb")

x = memly.Membrane(traj, top, load=True)

# frame_id = 1
# memly.membrane.export_frame_with_normals(x.sim[frame_id],
#                                          x.hg_centroids[frame_id],
#                                          x.normals[frame_id],
#                                          os.path.join(THIS_DIR, "data/normals.pdb"))

# for frame_i, frame in enumerate(x.sim):
#     memly.membrane.export_labelled_snapshot(frame, x.leaflets[frame_i], os.path.join(THIS_DIR, "data/leaflet_id/"+str(frame_i)+".pdb"))

np.testing.assert_array_equal(len(x.leaflets[10]['upper']), 1460)
np.testing.assert_array_equal(len(x.leaflets[10]['lower']), 1404)
np.testing.assert_array_equal(len(x.leaflets[10]['aggregate']), 92)
