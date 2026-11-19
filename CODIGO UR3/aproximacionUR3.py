import numpy as np

from cinematicaUR3 import (
    cinematica_directa_ur3,
    cinematica_inversa_pose_ur3
)

from seguroUR3 import (
    movimiento_seguro,
    articulaciones_seguras,
    posicion_segura
)


PASO_APROXIMACION_M = 0.005
ALTURA_SOBRE_PCB_M = 0.020


def validar_vector(datos, cantidad, nombre):
    vector = np.asarray(datos, dtype=float)
    if vector.shape != (cantidad,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{nombre}: datos no validos")
    return vector.copy()


def preparar_aproximacion(robot, posicion_primer_punto):
    posicion_primer_punto = validar_vector(
        posicion_primer_punto, 3, "Primer punto"
    )

    q_inicial = validar_vector(
        robot.angAct(), 6, "Articulaciones iniciales"
    )

    pose_inicial = validar_vector(
        robot.poseActTCP(), 6, "Pose inicial"
    )

    offset = validar_vector(
        robot.rtde_c.getTCPOffset(), 6, "Offset TCP"
    )

    print(f"TCP activo detectado en el robot: {offset.tolist()}")

    if not articulaciones_seguras(q_inicial):
        raise ValueError("La configuracion inicial supera tus limites")

    if not posicion_segura(pose_inicial):
        raise ValueError("La posicion inicial esta fuera de tu zona")

    orientacion = pose_inicial[3:].copy()

    altura_transporte = max(
        pose_inicial[2],
        posicion_primer_punto[2] + ALTURA_SOBRE_PCB_M
    )

    sobre_posicion_actual = pose_inicial[:3].copy()
    sobre_posicion_actual[2] = altura_transporte

    sobre_primer_punto = posicion_primer_punto.copy()
    sobre_primer_punto[2] = altura_transporte

    destinos = [
        sobre_posicion_actual,
        sobre_primer_punto,
        posicion_primer_punto
    ]

    plan = []
    q_anterior = q_inicial.copy()
    posicion_anterior = pose_inicial[:3].copy()

    for destino in destinos:
        inicio_segmento = posicion_anterior.copy()
        distancia = np.linalg.norm(destino - inicio_segmento)

        if distancia < 1e-9:
            continue

        pasos = max(
            1,
            int(np.ceil(distancia / PASO_APROXIMACION_M))
        )

        for numero in range(1, pasos + 1):
            posicion_objetivo = (
                inicio_segmento
                + (numero / pasos) * (destino - inicio_segmento)
            )

            pose_objetivo = np.concatenate((
                posicion_objetivo,
                orientacion
            ))

            if not movimiento_seguro(
                posicion_anterior,
                posicion_objetivo
            ):
                raise ValueError(
                    "Aproximacion fuera de zona o distancia permitida"
                )

            resultado = cinematica_inversa_pose_ur3(
                pose_objetivo,
                q_anterior,
                tcp_offset=offset
            )

            q_objetivo = validar_vector(
                resultado["q"], 6, "Resultado IK"
            )

            errores = np.asarray([
                resultado["error_posicion_m"],
                resultado["error_orientacion_rad"]
            ], dtype=float)

            if not np.all(np.isfinite(errores)):
                raise ValueError("La IK devolvio errores no validos")

            if (
                not resultado["convergio"]
                or errores[0] > 0.001
                or errores[1] > np.deg2rad(1.0)
            ):
                raise ValueError(
                    f"No hay solucion IK aceptable en paso {len(plan) + 1}. "
                    f"Error posicion: {errores[0] * 1000:.3f} mm; "
                    f"orientacion: {np.rad2deg(errores[1]):.3f} grados"
                )

            if not articulaciones_seguras(q_objetivo):
                raise ValueError(
                    "La aproximacion supera tus limites articulares"
                )

            if np.linalg.norm(q_objetivo - q_anterior) > 0.30:
                raise ValueError(
                    "La aproximacion supera el salto articular permitido"
                )

            plan.append(pose_objetivo.tolist())
            q_anterior = q_objetivo
            posicion_anterior = posicion_objetivo.copy()

    return q_inicial, plan


def aproximar_al_primer_punto(robot, posicion_primer_punto):
    print("\nPREPARANDO APROXIMACION AUTOMATICA")
    print("Todavia no se enviaron movimientos.")

    try:
        q_inicial, plan = preparar_aproximacion(
            robot,
            posicion_primer_punto
        )

        q_actual = validar_vector(
            robot.angAct(), 6, "Articulaciones actuales"
        )

        if np.max(np.abs(q_actual - q_inicial)) > 0.0001:
            print("Aproximacion cancelada: URSim cambio durante el calculo")
            return False

        print("Aproximacion calculada:", len(plan), "objetivos")
        print("Comenzando ejecucion automatica.")

        for numero, pose in enumerate(plan, 1):
            if not robot.moverPoseJ_DH(pose):
                print("Aproximacion detenida en objetivo", numero)
                return False

        print("Aproximacion terminada y llegada al primer punto verificada")
        return True

    except ValueError as error:
        print("Aproximacion rechazada:", error)
        return False