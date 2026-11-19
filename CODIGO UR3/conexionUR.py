import time
import numpy as np
from scipy.spatial.transform import Rotation
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from seguroUR3 import (
    movimiento_seguro,
    articulaciones_seguras,
    posicion_segura
)

from cinematicaUR3 import (
    cinematica_inversa_pose_ur3,
    cinematica_directa_ur3
)


class brazo:

    def __init__(self, ip):
        self.ip = ip
        self.rtde_c = RTDEControlInterface(self.ip)
        self.rtde_r = RTDEReceiveInterface(self.ip)

    def poseActTCP(self):
        return self.rtde_r.getActualTCPPose()

    def angAct(self):
        return self.rtde_r.getActualQ()

    def movimientoL(self, pose, speed=0.03, acceleration=0.03):
        self.rtde_c.moveL(pose, speed, acceleration)

    def moverSeguroL(self, pose_objetivo, speed=0.03, acceleration=0.03):
        pose_actual = self.poseActTCP()

        if not movimiento_seguro(pose_actual, pose_objetivo):
            print("Movimiento lineal bloqueado por seguridad")
            return False

        self.movimientoL(pose_objetivo, speed, acceleration)
        return True

    def validarPoseDH(self, pose_objetivo):
        pose_objetivo = np.asarray(pose_objetivo, dtype=float)

        if pose_objetivo.shape != (6,) or not np.all(np.isfinite(pose_objetivo)):
            print("Validacion rechazada: la pose debe tener seis numeros finitos")
            return None

        pose_actual = np.asarray(self.poseActTCP(), dtype=float)
        q_actual = np.asarray(self.angAct(), dtype=float)
        offset_tcp = np.asarray(self.rtde_c.getTCPOffset(), dtype=float)

        for datos in (pose_actual, q_actual, offset_tcp):
            if datos.shape != (6,) or not np.all(np.isfinite(datos)):
                print("Validacion rechazada: lectura de URSim no valida")
                return None

        if not movimiento_seguro(pose_actual, pose_objetivo):
            print("Validacion rechazada: zona o distancia cartesiana")
            return None

        # Resolver la cinemática inversa con el TCP Offset activo del robot
        resultado = cinematica_inversa_pose_ur3(
            pose_objetivo,
            q_actual,
            tcp_offset=offset_tcp
        )

        q_objetivo = np.asarray(resultado["q"], dtype=float)
        error_posicion = float(resultado["error_posicion_m"])
        error_orientacion = float(resultado["error_orientacion_rad"])

        if q_objetivo.shape != (6,) or not np.all(np.isfinite(q_objetivo)):
            print("Validacion rechazada: articulaciones calculadas no validas")
            return None

        if not np.all(np.isfinite([error_posicion, error_orientacion])):
            print("Validacion rechazada: errores numericos no validos")
            return None

        if not resultado["convergio"]:
            print("Validacion rechazada: el optimizador no convergio")
            return None

        if error_posicion > 0.001 or error_orientacion > np.deg2rad(1.0):
            print("Validacion rechazada: error de posicion u orientacion")
            return None

        salto_articular = np.linalg.norm(q_objetivo - q_actual)

        if salto_articular > 0.30:
            print("Validacion rechazada: salto articular demasiado grande")
            return None

        if not articulaciones_seguras(q_actual) or not articulaciones_seguras(q_objetivo):
            print("Validacion rechazada: articulaciones fuera de los limites configurados")
            return None

        return q_objetivo.tolist()

    def validarRecorridoDH(self, pose_objetivo):
        q_objetivo = self.validarPoseDH(pose_objetivo)

        if q_objetivo is None:
            print("No se reviso el recorrido: objetivo rechazado")
            return None

        q_inicial = np.asarray(self.angAct(), dtype=float)
        q_objetivo = np.asarray(q_objetivo, dtype=float)
        offset_tcp = np.asarray(self.rtde_c.getTCPOffset(), dtype=float)

        if q_inicial.shape != (6,) or not np.all(np.isfinite(q_inicial)):
            print("Recorrido rechazado: lectura articular no valida")
            return None

        salto_articular = np.linalg.norm(q_objetivo - q_inicial)

        if salto_articular > 0.30:
            print("Recorrido rechazado: cambio de estado o salto excesivo")
            return None

        paso_articular = 0.005
        cambio_maximo = np.max(np.abs(q_objetivo - q_inicial))
        intervalos = max(1, int(np.ceil(cambio_maximo / paso_articular)))

        for indice in range(intervalos + 1):
            fraccion = indice / intervalos
            q_intermedio = q_inicial + fraccion * (q_objetivo - q_inicial)

            if not articulaciones_seguras(q_intermedio):
                print("Recorrido rechazado: limites articulares en muestra", indice)
                return None

            transformacion = cinematica_directa_ur3(q_intermedio, tcp_offset=offset_tcp)

            if not np.all(np.isfinite(transformacion)):
                print("Recorrido rechazado: calculo DH no valido")
                return None

            posicion = transformacion[:3, 3]

            if not posicion_segura(posicion):
                print("Recorrido rechazado: posicion fuera de zona en muestra", indice)
                return None

        return q_objetivo.tolist()

    def moverPoseJ_DH(self, pose_objetivo):
        pose_objetivo = np.asarray(pose_objetivo, dtype=float).copy()
        q_antes = np.asarray(self.angAct(), dtype=float)

        if q_antes.shape != (6,) or not np.all(np.isfinite(q_antes)):
            print("Movimiento bloqueado: lectura inicial no valida")
            return False

        q_objetivo = self.validarRecorridoDH(pose_objetivo)

        if q_objetivo is None:
            print("No se envio ninguna orden de movimiento")
            return False

        q_despues = np.asarray(self.angAct(), dtype=float)

        if q_despues.shape != (6,) or not np.all(np.isfinite(q_despues)):
            print("Movimiento bloqueado: lectura posterior no valida")
            return False

        cambio_estado = np.max(np.abs(q_despues - q_antes))

        if cambio_estado > 0.0001:
            print("Movimiento bloqueado: el robot cambio durante la revision")
            return False

        ejecutado = self.rtde_c.moveJ(q_objetivo, 0.03, 0.03, False)

        if not ejecutado:
            print("RTDE no confirmo la ejecucion del movimiento")
            return False

        rotacion_objetivo = Rotation.from_rotvec(pose_objetivo[3:])
        tiempo_limite = time.monotonic() + 8.0

        while True:
            pose_final = np.asarray(self.poseActTCP(), dtype=float)

            if pose_final.shape != (6,) or not np.all(np.isfinite(pose_final)):
                print("No se pudo verificar la pose final")
                return False

            error_posicion = np.linalg.norm(pose_final[:3] - pose_objetivo[:3])
            rotacion_final = Rotation.from_rotvec(pose_final[3:])
            error_orientacion = (rotacion_objetivo.inv() * rotacion_final).magnitude()

            llegada_verificada = (
                error_posicion <= 0.001
                and error_orientacion <= np.deg2rad(1.0)
            )

            if llegada_verificada or time.monotonic() >= tiempo_limite:
                break

            time.sleep(0.05)

        if not llegada_verificada:
            print("La pose final no cumple las tolerancias")
            return False

        return True

    def stop(self):
        try:
            self.rtde_c.stopScript()
            self.rtde_c.disconnect()
            self.rtde_r.disconnect()
        except Exception:
            pass
        print(self.rtde_r.getActualTCPPose())