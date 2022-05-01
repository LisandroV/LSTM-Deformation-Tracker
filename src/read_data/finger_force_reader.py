import numpy as np


def read_finger_forces_file(forces_file: str):
    """
    Reads the finger forces file and returns the Fz column.
    Arguments:
        forces_file: path of the forces file to read
    Returns:
        np.float64 numpy ndarray of the forces in Fz.
    """
    forces = np.loadtxt(
        forces_file,
        dtype={
            "names": ("time", "fx", "fy", "fz", "tx", "ty", "tz", "d"),
            "formats": ("f", "f", "f", "f", "f", "f", "f", "f"),
        },
    )
    # only_forces = forces[["fx", "fy", "fz", "tx", "ty", "tz"]]
    return forces[["fz"][0]]
