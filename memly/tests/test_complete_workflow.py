#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:46:05 2020

@author: tamir
"""
import os
import numpy as np
import pytest

import memly

def test_complete_workflow():
    # Setup access to datafiles
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    traj = os.path.join(THIS_DIR, "data/1.xtc")
    top = os.path.join(THIS_DIR, "data/1.pdb")

    x = memly.Analysis(traj, top, load=True)

    lipid_vectors = memly.analysis.get_lipid_vectors(x.sim)
    assert list(lipid_vectors.keys()) == [i for i in range(0, 336)]

    origin_cutoff = 50
    collinear_cutoff = 90

    pairmask = memly.analysis.compare_lipid_vectors(lipid_vectors[0], lipid_vectors[1], origin_cutoff, collinear_cutoff)
    #assert np.count_nonzero(pairmask) == 13

    allmask = memly.analysis.cluster_lipid_vectors(lipid_vectors, origin_cutoff, collinear_cutoff)

    return allmask
    # x.split_leaflets()

result = test_complete_workflow()