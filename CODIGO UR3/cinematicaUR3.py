import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

PI = np.pi


def matriz_dh(a, alpha, d, theta):
    return np.array([
        [
            np.cos(theta),
            -np.sin(theta) * np.cos(alpha),
            np.sin(theta) * np.sin(alpha),
            a * np.cos(theta)
        ],
        [
            np.sin(theta),
            np.cos(theta) * np.cos(alpha),
            -np.cos(theta) * np.sin(alpha),
            a * np.sin(theta)
        ],
        [
            0,
            np.sin(alpha),
            np.cos(alpha),
            d
        ],
        [
            0,
            0,
            0,
            1
        ]
    ])


def parametros_dh_ur3(q):
    return [
        [0,        PI / 2,  0.1519,  q[0]],
        [-0.24365,    0,      0,     q[1]],
        [-0.21325,    0,      0,     q[2]],
        [0,        PI / 2,  0.11235, q[3]],
        [0,       -PI / 2,  0.08535, q[4]],
        [0,           0,    0.0819,  q[5]]
    ]


def cinematica_directa_ur3(q, tcp_offset=None):
    """
    Calcula la matriz homogénea T_0^TCP del UR3.
    Si tcp_offset no es None, suma la transformación de la herramienta activa.
    """
    dh = parametros_dh_ur3(q)
    T = np.identity(4)

    for fila in dh:
        a, alpha, d, theta = fila[0], fila[1], fila[2], fila[3]
        A = matriz_dh(a, alpha, d, theta)
        T = T @ A

    if tcp_offset is not None:
        tcp_offset = np.asarray(tcp_offset, dtype=float)
        if tcp_offset.shape == (6,):
            rot_tcp = Rotation.from_rotvec(tcp_offset[3:]).as_matrix()
            T_tcp = np.identity(4)
            T_tcp[:3, :3] = rot_tcp
            T_tcp[:3, 3] = tcp_offset[:3]
            T = T @ T_tcp

    return T


def posicion_tcp(T):
    return [T[0, 3], T[1, 3], T[2, 3]]


def cinematica_inversa_pose_ur3(pose_objetivo, q_inicial, tcp_offset=None):
    """
    Calcula las articulaciones q para alcanzar pose_objetivo considerando el tcp_offset.
    """
    pose_objetivo = np.asarray(pose_objetivo, dtype=float)
    q_inicial = np.asarray(q_inicial, dtype=float)

    if pose_objetivo.shape != (6,) or q_inicial.shape != (6,):
        raise ValueError("La pose y las articulaciones deben tener seis valores")

    if not np.all(np.isfinite(pose_objetivo)) or not np.all(np.isfinite(q_inicial)):
        raise ValueError("Los datos deben ser numeros finitos")

    posicion_objetivo = pose_objetivo[:3]
    rotacion_objetivo = Rotation.from_rotvec(pose_objetivo[3:]).as_matrix()

    escala_posicion = 0.001
    escala_orientacion = np.deg2rad(1.0)

    def errores_pose(articulaciones):
        transformacion = cinematica_directa_ur3(articulaciones, tcp_offset)
        error_posicion = transformacion[:3, 3] - posicion_objetivo
        rotacion_relativa = rotacion_objetivo.T @ transformacion[:3, :3]
        error_orientacion = Rotation.from_matrix(rotacion_relativa).as_rotvec()
        return error_posicion, error_orientacion

    def residuo(articulaciones):
        error_posicion, error_orientacion = errores_pose(articulaciones)
        return np.concatenate((
            error_posicion / escala_posicion,
            error_orientacion / escala_orientacion
        ))

    limite_local = 0.50

    resultado = least_squares(
        residuo,
        q_inicial,
        bounds=(q_inicial - limite_local, q_inicial + limite_local),
        max_nfev=300
    )

    error_posicion, error_orientacion = errores_pose(resultado.x)

    return {
        "q": resultado.x,
        "convergio": bool(resultado.success),
        "error_posicion_m": float(np.linalg.norm(error_posicion)),
        "error_orientacion_rad": float(np.linalg.norm(error_orientacion)),
        "mensaje": resultado.message
    }
    