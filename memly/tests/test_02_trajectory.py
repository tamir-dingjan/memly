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

# memly.membrane.export_labelled_snapshot(x.sim[0], x.leaflets[0], os.path.join(THIS_DIR, "data/labelled.pdb"))