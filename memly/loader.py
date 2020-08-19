"""
loader.py
Loader wrapper surrounding file reading commands.

Handles fetching files from the disk and loading them using MDTraj
"""

import os
import mdtraj as md

# Loading function


def load(traj, top):
    """
    Load the specified trajectory and topology files.

    Parameters
    ----------
    traj : str
        Full filepath specifying the location of simulation trajectory.
    top : str
        Full filepath specifying the location of topology file.

    Returns
    -------
    MDTraj trajectory object
        The loaded trajectory.
        :rtype:

    """
    # Check if the specified files exist
    if not os.path.exists(traj):
        print("ERROR: Couldn't find file ("+str(traj)+").")
    elif not os.path.exists(top):
        print("ERROR: Couldn't find file ("+str(top)+").")
    else:
        try:
            return md.load(traj, top=top)
        except FileNotFoundError:
            print("ERROR: Couldn't load from disk.")
