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
from memly import countlipids


def test_countlipids():
    # Setup access to datafiles
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    traj = os.path.join(THIS_DIR, "data/2.xtc")
    top = os.path.join(THIS_DIR, "data/2.pdb")
    x = memly.Analysis(traj, top, load=True)

    metric = countlipids.CountLipids(membrane=x.membrane)
    return metric, metric.results

#
# # Setup access to datafiles
# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# traj = os.path.join(THIS_DIR, "data/2.xtc")
# top = os.path.join(THIS_DIR, "data/2.pdb")
# x = memly.Analysis(traj, top, load=True)
#
# metric = countlipids.CountLipids(membrane=x.membrane)


metric, results = test_countlipids()
