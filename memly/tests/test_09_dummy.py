#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:46:05 2020

@author: hila
"""
import os
import numpy as np
import pytest

import memly
from memly import dummy_metric


def test_dummy():
    # Setup access to datafiles
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    traj = os.path.join(THIS_DIR, "data/1.xtc")
    top = os.path.join(THIS_DIR, "data/1.pdb")
    x = memly.Analysis(traj, top, load=True)

    metric = dummy_metric.Dummy(membrane=x.membrane)
    return metric, metric.results

#
# # Setup access to datafiles
# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# traj = os.path.join(THIS_DIR, "data/2.xtc")
# top = os.path.join(THIS_DIR, "data/2.pdb")
# x = memly.Analysis(traj, top, load=True)
#
# metric = countlipids.CountLipids(membrane=x.membrane)


metric, results = test_dummy()
