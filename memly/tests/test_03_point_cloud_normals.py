#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:21:05 2020

@author: tamir
"""
import os
import numpy as np
import point_cloud_utils as pcu


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
points = os.path.join(THIS_DIR, "data/point_cloud.obj")

# v is a nv by 3 NumPy array of vertices
v = pcu.load_mesh_v(points)

# Estimate a normal at each point (row of v) using its 5 nearest neighbors
n = pcu.estimate_point_cloud_normals_knn(v, 5)

np.testing.assert_allclose(n[1][5], np.asarray([0, 0, 1]))
