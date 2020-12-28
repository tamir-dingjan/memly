#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

import memly
from memly import thickness


def test_thickness():
    # Setup access to datafiles
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    traj = os.path.join(THIS_DIR, "data/2.xtc")
    top = os.path.join(THIS_DIR, "data/2.pdb")
    x = memly.Analysis(traj, top, load=True)

    metric = thickness.Thickness(membrane=x.membrane)
    return metric


metric = test_thickness()
