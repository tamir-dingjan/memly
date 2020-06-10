#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:46:05 2020

@author: tamir
"""


import memly

traj = "/home/tamir/Documents/memly_project/memly/memly/tests/data/1.xtc"
top = "/home/tamir/Documents/memly_project/memly/memly/tests/data/1.pdb"

x = memly.Analysis(traj, top)

x.analyse()