import numpy as np

X_MIN = -0.50
X_MAX = 0.50

Y_MIN = -0.50
Y_MAX = 0.50

Z_MIN = 0.05
Z_MAX = 0.50

MAX_STEP = 0.15


def posicion_segura(pose):
    x = pose[0]
    y = pose[1]
    z = pose[2]

    if x < X_MIN or x > X_MAX:
        return False

    if y < Y_MIN or y > Y_MAX:
        return False

    if z < Z_MIN or z > Z_MAX:
        return False

    return True


def distancia_cartesiana(pose_actual, pose_objetivo):
    dx = pose_objetivo[0] - pose_actual[0]
    dy = pose_objetivo[1] - pose_actual[1]
    dz = pose_objetivo[2] - pose_actual[2]

    distancia = (dx**2 + dy**2 + dz**2) ** 0.5

    return distancia


def movimiento_seguro(pose_actual, pose_objetivo):
    if not posicion_segura(pose_objetivo):
        return False

    distancia = distancia_cartesiana(pose_actual, pose_objetivo)

    if distancia > MAX_STEP:
        return False

    return True


LIMITE_ARTICULAR_MIN = np.array([-6.28, -6.28, -3.14, -6.28, -6.28, -6.28])
LIMITE_ARTICULAR_MAX = np.array([ 6.28,  6.28,  3.14,  6.28,  6.28,  6.28])


def articulaciones_seguras(q):
    q = np.array(q)

    if np.any(q < LIMITE_ARTICULAR_MIN):
        return False

    if np.any(q > LIMITE_ARTICULAR_MAX):
        return False

    return True