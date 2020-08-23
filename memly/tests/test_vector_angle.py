#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:46:05 2020

@author: tamir
"""

import memly

i = [1, 1, 0]
j = [0, 1, 0]
k = [1, 0, 0]

assert (memly.membrane.unit_vector(k) == [1, 0, 0]).all()

assert (round(memly.membrane.angle_between(i, k), 1) == round(45.0, 1))
