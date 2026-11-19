import numpy as np


P0_ROBOT = np.array([0.30, -0.25, 0.12])
PX_ROBOT = np.array([0.37, -0.25, 0.12])
PY_ROBOT = np.array([0.30, -0.205, 0.12])


def pcb_a_robot(x_mm, y_mm):
    vector_x = (PX_ROBOT - P0_ROBOT) / 70.0
    vector_y = (PY_ROBOT - P0_ROBOT) / 45.0

    posicion_robot = (
        P0_ROBOT
        + vector_x * x_mm
        + vector_y * y_mm
    )

    return posicion_robot