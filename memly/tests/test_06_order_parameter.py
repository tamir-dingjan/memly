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
from memly import orderparam


def test_orderparam():
    # Setup access to datafiles
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    traj = os.path.join(THIS_DIR, "data/2.xtc")
    top = os.path.join(THIS_DIR, "data/2.pdb")
    x = memly.Analysis(traj, top, load=True)

    metric = orderparam.OrderParam(membrane=x.membrane)

    np.testing.assert_equal(orderparam.calculate_orderparam(0), 1.0,
                            "Order parameter calculation is broken (0 deg) !")
    np.testing.assert_equal(orderparam.calculate_orderparam(90), -0.5,
                            "Order parameter calculation is broken (90 deg)")

    return metric, metric.results

#
# # Setup access to datafiles
# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# traj = os.path.join(THIS_DIR, "data/2.xtc")
# top = os.path.join(THIS_DIR, "data/2.pdb")
# x = memly.Analysis(traj, top, load=True)
#
# metric = countlipids.CountLipids(membrane=x.membrane)


metric, results = test_orderparam()

