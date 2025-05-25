"""
    Copyright (c) 2018-present. Ben Athiwaratkun
    All rights reserved.
    
    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree.
"""
import subprocess
import numpy as np

# BenA:
# the constant values here reflect the values in original FastText implementation

BOW = "<"
EOW = ">"

# In Python 3, integers automatically handle arbitrary size, so we don't need 'L' or long()
# M32 is still used for bitwise operations to simulate 32-bit unsigned integers.
M32 = 0xFFFFFFFF 

def m32(n):
    return n & M32

def mmul(a, b):
    # In Python 3, integer multiplication handles large numbers automatically.
    # We apply m32 to keep it within the 32-bit unsigned range as per the original logic.
    return m32(a * b)

def hash(s): # Renamed 'str' to 's' to avoid conflict with built-in str()
    h = m32(2166136261) # Removed 'L'
    for c in s:
        cc = m32(ord(c)) # Removed 'long()'
        h = m32(h ^ cc)
        h = mmul(h, 16777619) # Removed 'L'
    return h