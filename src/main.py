# READ FORCE FILE -------------------------------------------------
import numpy as np
import numpy.typing as npt
import os

DATA_DIR = "data"


def read_forces(forces_file: str) -> npt.NDArray[np.float64]:
    """
    Reads forces file and returns the relevant forces.
    Arguments:
        forces_file: path of the forces file to read
    Returns:
        float64 array of the forces in Fz.
        [ 0.02, -0.02, ...]
    """
    forces = np.loadtxt(
        forces_file,
        dtype={
            "names": ("time", "fx", "fy", "fz", "tx", "ty", "tz", "d"),
            "formats": ("f", "f", "f", "f", "f", "f", "f", "f"),
        },
    )
    # only_forces = forces[["fx", "fy", "fz", "tx", "ty", "tz"]]
    only_forces = forces[["fz"]]
    force_array = np.zeros(len(only_forces), dtype=np.float64)
    for i, row in enumerate(only_forces):
        force_array[i] = row
    return force_array


force_file_path = os.path.join(DATA_DIR, "sponge_centre_100.txt")
forces = read_forces(force_file_path)
print(forces)
print(np.shape(forces))

# READ FINGER POSITION FILE -------------------------------------------------

# crear transformers
