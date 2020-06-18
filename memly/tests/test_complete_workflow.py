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

    # x.split_leaflets()

result = test_complete_workflow()