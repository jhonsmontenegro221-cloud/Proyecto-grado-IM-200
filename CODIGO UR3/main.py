import csv
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from rtde_receive import RTDEReceiveInterface

from aproximacionUR3 import aproximar_al_primer_punto
from calibracionUR3 import pcb_a_robot
from conexionUR import brazo


ROBOT_IP = "192.168.116.128"
CSV_ENTRADA = "puntos_soldadura.csv"

PCB_ANCHO_MM = 70.0
PCB_ALTO_MM = 45.0


def leer_puntos(ruta):
    puntos = []

    with open(ruta, newline="", encoding="utf-8-sig") as archivo:
        lector = csv.DictReader(archivo)

        columnas = {"Punto", "X_mm", "Y_mm"}

        if not columnas.issubset(lector.fieldnames or []):
            raise ValueError(
                "El CSV debe contener las columnas Punto, X_mm y Y_mm"
            )

        for numero_fila, fila in enumerate(lector, start=2):
            x_mm = float(fila["X_mm"])
            y_mm = float(fila["Y_mm"])

            if not np.all(np.isfinite([x_mm, y_mm])):
                raise ValueError(
                    f"Coordenadas no validas en la fila {numero_fila}"
                )

            if not (
                0 <= x_mm <= PCB_ANCHO_MM
                and 0 <= y_mm <= PCB_ALTO_MM
            ):
                raise ValueError(
                    f"Punto fuera de la PCB en la fila {numero_fila}"
                )

            posicion = np.asarray(
                pcb_a_robot(x_mm, y_mm),
                dtype=float
            )

            if (
                posicion.shape != (3,)
                or not np.all(np.isfinite(posicion))
            ):
                raise ValueError(
                    f"Conversion a robot no valida en la fila {numero_fila}"
                )

            puntos.append({
                "id": fila["Punto"],
                "x_mm": x_mm,
                "y_mm": y_mm,
                "robot": posicion
            })

    if not puntos:
        raise ValueError("El CSV no contiene puntos")

    return puntos


def ordenar_puntos_por_cercania(puntos):
    puntos_ordenados = []
    copia = puntos.copy()
    actual = copia.pop(0)
    puntos_ordenados.append(actual)

    while copia:
        siguiente = min(
            copia,
            key=lambda p: np.linalg.norm(p["robot"] - actual["robot"])
        )
        puntos_ordenados.append(siguiente)
        copia.remove(siguiente)
        actual = siguiente

    return puntos_ordenados


def registrar_tcp(receptor, detener, muestras, errores):
    inicio = time.monotonic()

    try:
        while not detener.is_set():
            pose = np.asarray(
                receptor.getActualTCPPose(),
                dtype=float
            )

            if (
                pose.shape != (6,)
                or not np.all(np.isfinite(pose))
            ):
                raise ValueError(
                    "Lectura TCP no valida durante el registro"
                )

            muestras.append((
                time.monotonic() - inicio,
                float(pose[0]),
                float(pose[1]),
                float(pose[2])
            ))

            detener.wait(0.02)

    except Exception as error:
        errores.append(str(error))


def guardar_trayectoria(ruta, muestras):
    with open(ruta, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)

        escritor.writerow([
            "tiempo_s",
            "x_m",
            "y_m",
            "z_m"
        ])

        escritor.writerows(muestras)


def graficar_trayectoria_3d(puntos, muestras, completados, estado):
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Esquema rectangular de la PCB en el espacio del robot
    esquinas = np.array([
        pcb_a_robot(0, 0),
        pcb_a_robot(PCB_ANCHO_MM, 0),
        pcb_a_robot(PCB_ANCHO_MM, PCB_ALTO_MM),
        pcb_a_robot(0, PCB_ALTO_MM),
        pcb_a_robot(0, 0)
    ], dtype=float)

    ax.plot(
        esquinas[:, 0], esquinas[:, 1], esquinas[:, 2],
        'b--', label='Borde PCB', linewidth=1.8
    )

    # Puntos de soldadura objetivo
    pts = np.array([p["robot"] for p in puntos])
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        color='red', s=45, label='Puntos Soldadura', depthshade=False
    )

    for i, p in enumerate(pts):
        ax.text(p[0], p[1] + 0.001, p[2], f'{i+1}', fontsize=8, color='darkred')

    # Recorrido del TCP registrado por RTDE
    if len(muestras) > 0:
        trayectoria = np.array([m[1:] for m in muestras])
        
        # Filtro visual para evitar que lecturas lejanas deformen el encuadre
        x_min, x_max = esquinas[:, 0].min() - 0.03, esquinas[:, 0].max() + 0.03
        y_min, y_max = esquinas[:, 1].min() - 0.03, esquinas[:, 1].max() + 0.03
        
        mascara_valida = (
            (trayectoria[:, 0] >= x_min) & (trayectoria[:, 0] <= x_max) &
            (trayectoria[:, 1] >= y_min) & (trayectoria[:, 1] <= y_max)
        )
        tray_filtrada = trayectoria[mascara_valida]

        if len(tray_filtrada) > 0:
            ax.plot(
                tray_filtrada[:, 0], tray_filtrada[:, 1], tray_filtrada[:, 2],
                color='green', linewidth=1.2, label='Recorrido TCP'
            )

            ax.scatter(
                tray_filtrada[-1, 0], tray_filtrada[-1, 1], tray_filtrada[-1, 2],
                color='magenta', s=70, marker='^', label='Posicion Final TCP'
            )

    # Vista cenital idéntica a la segunda imagen (desde arriba)
    ax.view_init(elev=85, azim=-90)

    ax.set_xlim(esquinas[:, 0].min() - 0.01, esquinas[:, 0].max() + 0.01)
    ax.set_ylim(esquinas[:, 1].min() - 0.01, esquinas[:, 1].max() + 0.01)

    ax.set_xlabel('X (m)', labelpad=10)
    ax.set_ylabel('Y (m)', labelpad=10)
    ax.set_zlabel('Z (m)', labelpad=10)
    ax.set_title(f'Visualizacion Recorrido UR3\nEstado: {estado} | Puntos completados: {completados}/{len(puntos)}', fontsize=11)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()


def main():
    carpeta = Path(__file__).resolve().parent

    ruta_puntos = carpeta / CSV_ENTRADA
    ruta_trayectoria = carpeta / "trayectoria_ursim.csv"

    if not ruta_puntos.is_file():
        print("No se encontro el CSV:", ruta_puntos)
        return

    try:
        puntos = leer_puntos(ruta_puntos)
        puntos = ordenar_puntos_por_cercania(puntos)

    except (ValueError, KeyError, TypeError, OSError) as error:
        print("No se pudo cargar el CSV:", error)
        return

    print("CSV cargado y ordenado por cercania:", ruta_puntos)
    print("Total de puntos:", len(puntos))
    print("Aproximacion automatica al primer punto.")

    robot = None
    receptor_registro = None
    hilo = None

    detener = threading.Event()

    muestras = []
    errores = []

    completados = 0
    seleccionado = None
    posicion_inicial = None
    estado = "Secuencia no iniciada"

    try:
        robot = brazo(ROBOT_IP)

        pose_inicial = np.asarray(
            robot.poseActTCP(),
            dtype=float
        )

        if (
            pose_inicial.shape != (6,)
            or not np.all(np.isfinite(pose_inicial))
        ):
            raise ValueError("Pose inicial no valida")

        posicion_inicial = pose_inicial[:3].tolist()
        orientacion_objetivo = pose_inicial[3:].tolist()

        print("Pose inicial:", pose_inicial.tolist())
        print("Orientacion objetivo:", orientacion_objetivo)

        receptor_registro = RTDEReceiveInterface(ROBOT_IP)

        hilo = threading.Thread(
            target=registrar_tcp,
            args=(
                receptor_registro,
                detener,
                muestras,
                errores
            )
        )

        hilo.start()

        seleccionado = 0
        estado = "Preparando aproximacion automatica"

        aproximacion_completa = aproximar_al_primer_punto(
            robot,
            puntos[0]["robot"]
        )

        if not aproximacion_completa:
            estado = "Aproximacion rechazada o no completada"
            raise ValueError(
                "No se iniciara el recorrido del CSV porque "
                "la aproximacion no termino correctamente."
            )

        if errores:
            estado = "Secuencia detenida: fallo de registro TCP"
            raise ValueError(
                "El registro TCP fallo durante la aproximacion."
            )

        print("\nAPROXIMACION COMPLETADA")
        print("Comenzando comprobacion y recorrido de los puntos del CSV.")

        HEIGHT_SEGURA_Z = 0.010  # Elevacion de seguridad de 10 mm sobre el punto

        for seleccionado, punto in enumerate(puntos):
            if errores:
                estado = "Secuencia detenida: fallo de registro TCP"
                break

            print(
                f'\nPunto {seleccionado + 1}/{len(puntos)} '
                f'| ID CSV: {punto["id"]}'
            )
            print(
                f'PCB: X={punto["x_mm"]:.2f} mm, '
                f'Y={punto["y_mm"]:.2f} mm'
            )

            # 1. Pose sobre el punto (Aproximacion alta)
            pose_arriba = punto["robot"].copy()
            pose_arriba[2] += HEIGHT_SEGURA_Z
            verificado_arriba = robot.moverPoseJ_DH(
                pose_arriba.tolist() + orientacion_objetivo
            )

            if not verificado_arriba:
                estado = (
                    f"Detenido en punto {seleccionado + 1}: "
                    "rechazo en aproximacion Z"
                )
                print(estado)
                break

            # 2. Descenso al punto de soldadura real
            pose_soldadura = punto["robot"].tolist() + orientacion_objetivo
            verificado_soldadura = robot.moverPoseJ_DH(pose_soldadura)

            if not verificado_soldadura:
                estado = (
                    f"Detenido en punto {seleccionado + 1}: "
                    "rechazo en descenso a soldadura"
                )
                print(estado)
                break

            # 3. Simulacion de tiempo de soldadura (Cautin sobre la pista)
            time.sleep(1.0)

            # 4. Retractado vertical (Elevacion de seguridad)
            verificado_retracto = robot.moverPoseJ_DH(
                pose_arriba.tolist() + orientacion_objetivo
            )

            if not verificado_retracto:
                estado = (
                    f"Detenido en punto {seleccionado + 1}: "
                    "rechazo en retractado Z"
                )
                print(estado)
                break

            completados += 1

            print(
                f"Puntos completados: "
                f"{completados}/{len(puntos)}"
            )

        else:
            estado = "Todos los puntos tienen llegada verificada"

    except KeyboardInterrupt:
        estado = "Programa interrumpido desde la consola"
        print(estado)

    except Exception as error:
        if estado not in (
            "Aproximacion rechazada o no completada",
            "Secuencia detenida: fallo de registro TCP"
        ):
            estado = "Secuencia interrumpida por un error"

        print(estado, ":", error)

    finally:
        detener.set()

        if hilo is not None and hilo.ident is not None:
            hilo.join()

        if receptor_registro is not None:
            try:
                receptor_registro.disconnect()
            except Exception as error:
                print("Error al cerrar el registro RTDE:", error)

        if robot is not None:
            try:
                robot.stop()
            except Exception as error:
                print("Error al cerrar el script RTDE:", error)

    if errores:
        estado += " | Registro incompleto"
        print("Errores de registro:", errores)

    print("\nResultado:", estado)
    print(f"Puntos completados: {completados}/{len(puntos)}")

    if posicion_inicial is None:
        return

    try:
        guardar_trayectoria(ruta_trayectoria, muestras)
        print("Trayectoria guardada:", ruta_trayectoria)
    except OSError as error:
        print("No se pudo guardar la trayectoria:", error)

    # Renderizado 3D de la trayectoria final
    graficar_trayectoria_3d(puntos, muestras, completados, estado)


if __name__ == "__main__":
    main()