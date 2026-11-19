import cv2
import csv
import math
import os
import numpy as np

# CONFIGURACIÓN GENERAL
CAMARA = 0
FRAME_W, FRAME_H = 1920, 1080

PCB_W_MM, PCB_H_MM = 70.0, 45.0
PX_MM = 10

W = int(PCB_W_MM * PX_MM)
H = int(PCB_H_MM * PX_MM)

AREA_PCB = 25000

MARGEN = 25

DIST_MIN = 9
DIST_TRACK = 18

FRAMES_ESTABLE = 4
FRAMES_PERDIDOS = 6

CSV_SALIDA = "puntos_soldadura_valid001.csv"

# Suavizado de la posicion de la PCB
SUAVIZADO_PCB = 0.90


def dist(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


def ordenar(p):

    p = np.float32(p)

    s = p.sum(axis=1)
    d = np.diff(p, axis=1).ravel()

    return np.float32([
        p[np.argmin(s)],
        p[np.argmin(d)],
        p[np.argmax(s)],
        p[np.argmax(d)]
    ])


def detectar_pcb(frame):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    # PCB marron/cobre sobre fondo claro
    mask = cv2.inRange(
        hsv,
        (5, 30, 25),
        (40, 255, 255)
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (25, 25)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contornos, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    mejor = None
    mejor_score = 0

    for c in contornos:

        area = cv2.contourArea(c)

        if area < AREA_PCB:
            continue

        rect = cv2.minAreaRect(c)

        rw, rh = rect[1]

        if min(rw, rh) <= 0:
            continue

        aspecto = (
            max(rw, rh) /
            min(rw, rh)
        )

        if 1.2 <= aspecto <= 2.2:

            if area > mejor_score:

                mejor_score = area

                mejor = ordenar(
                    cv2.boxPoints(rect)
                )

    return mejor

# RECTIFICACIÓN DE PERSPECTIVA
def rectificar(frame, caja):

    caja = ordenar(caja)

    ancho = np.linalg.norm(
        caja[1] - caja[0]
    )

    alto = np.linalg.norm(
        caja[3] - caja[0]
    )

    if alto > ancho:

        caja = np.float32([
            caja[3],
            caja[0],
            caja[1],
            caja[2]
        ])

    destino = np.float32([
        [0, 0],
        [W - 1, 0],
        [W - 1, H - 1],
        [0, H - 1]
    ])

    M = cv2.getPerspectiveTransform(
        caja,
        destino
    )

    Mi = cv2.getPerspectiveTransform(
        destino,
        caja
    )

    warp = cv2.warpPerspective(
        frame,
        M,
        (W, H)
    )

    return warp, Mi, caja

# DETECCIÓN DE HUECOS
def detectar_puntos(warp):

    gris = cv2.cvtColor(
        warp,
        cv2.COLOR_BGR2GRAY
    )

    gris = cv2.GaussianBlur(
        gris,
        (5, 5),
        0
    )

    # Buscar zonas realmente oscuras
    _, mask = cv2.threshold(
        gris,
        75,
        255,
        cv2.THRESH_BINARY_INV
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Ignorar bordes
    mask[:MARGEN, :] = 0
    mask[-MARGEN:, :] = 0
    mask[:, :MARGEN] = 0
    mask[:, -MARGEN:] = 0

    contornos, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidatos = []

    for c in contornos:

        area = cv2.contourArea(c)

        peri = cv2.arcLength(
            c,
            True
        )

        if area < 8 or area > 350:
            continue

        if peri <= 0:
            continue

        circularidad = (
            4 * math.pi * area /
            (peri * peri)
        )

        # El hueco debe ser aproximadamente circular
        if circularidad < 0.60:
            continue

        (x, y), r = cv2.minEnclosingCircle(c)

        x = int(round(x))
        y = int(round(y))
        r = int(round(r))

        # Tamaño esperado del hueco
        if r < 2 or r > 10:
            continue

        bx, by, bw, bh = cv2.boundingRect(c)

        if min(bw, bh) == 0:
            continue

        relacion = (
            max(bw, bh) /
            min(bw, bh)
        )

        # Rechazar pistas alargadas
        if relacion > 1.35:
            continue

        radio_centro = max(
            2,
            int(r * 0.65)
        )

        x1 = max(
            0,
            x - radio_centro
        )

        x2 = min(
            W,
            x + radio_centro + 1
        )

        y1 = max(
            0,
            y - radio_centro
        )

        y2 = min(
            H,
            y + radio_centro + 1
        )

        centro = gris[
            y1:y2,
            x1:x2
        ]

        if centro.size == 0:
            continue

        intensidad = np.mean(
            centro
        )

        # El centro debe ser oscuro
        if intensidad > 80:
            continue

        candidatos.append(
            (
                x,
                y,
                r,
                intensidad
            )
        )

    # Priorizar puntos mas oscuros
    candidatos.sort(
        key=lambda p: p[3]
    )

    puntos = []

    for x, y, r, _ in candidatos:

        if all(
            dist(
                (x, y),
                (px, py)
            ) >= DIST_MIN
            for px, py, _ in puntos
        ):

            puntos.append(
                (x, y, r)
            )

    return puntos, mask

#SEGUIMIENTO DE LOS PUNTOS ENTRE FRAMES
def actualizar_tracks(
    puntos,
    tracks,
    next_id
):

    usados = set()
    nuevos = []

    for t in tracks:

        mejor = None
        mejor_d = 99999

        for i, (x, y, r) in enumerate(puntos):

            if i in usados:
                continue

            d = dist(
                (t["x"], t["y"]),
                (x, y)
            )

            if d < mejor_d:

                mejor_d = d
                mejor = i

        if (
            mejor is not None
            and mejor_d <= DIST_TRACK
        ):

            x, y, r = puntos[mejor]

            usados.add(mejor)

            # Suavizado de los puntos
            t["x"] = int(
                0.8 * t["x"]
                + 0.2 * x
            )

            t["y"] = int(
                0.8 * t["y"]
                + 0.2 * y
            )

            t["r"] = r

            t["vistos"] += 1
            t["perdidos"] = 0

            nuevos.append(t)

        else:

            t["perdidos"] += 1

            if (
                t["perdidos"]
                <= FRAMES_PERDIDOS
            ):
                nuevos.append(t)

    for i, (x, y, r) in enumerate(puntos):

        if i not in usados:

            nuevos.append({
                "id": next_id,
                "x": x,
                "y": y,
                "r": r,
                "vistos": 1,
                "perdidos": 0
            })

            next_id += 1

    return nuevos, next_id

# GUARDAR COORDENADAS EN CSV
def guardar(estables):

    with open(
        CSV_SALIDA,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Punto",
            "X_mm",
            "Y_mm",
            "Distancia_mm"
        ])

        for n, t in enumerate(
            estables,
            1
        ):

            x_mm = (
                t["x"] /
                PX_MM
            )

            y_mm = (
                H - 1 - t["y"]
            ) / PX_MM

            distancia = math.hypot(
                x_mm,
                y_mm
            )

            writer.writerow([
                n,
                f"{x_mm:.2f}",
                f"{y_mm:.2f}",
                f"{distancia:.2f}"
            ])

    print(
        f"Guardados {len(estables)} puntos"
    )

    print(
        "CSV guardado en:"
    )

    print(
        os.path.abspath(CSV_SALIDA)
    )

# PROGRAMA PRINCIPAL
def main():

    cam = cv2.VideoCapture(
        CAMARA
    )

    cam.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_W
    )

    cam.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_H
    )

    cam.set(
        cv2.CAP_PROP_AUTOFOCUS,
        1
    )

    if not cam.isOpened():

        print(
            "ERROR: No se pudo abrir la camara"
        )

        return

    tracks = []
    next_id = 1
    estables = []

    # Caja suavizada de la PCB
    caja_suave = None

    print("PCB Vision")

    print(
        "g = guardar | "
        "c = limpiar | "
        "q = salir"
    )

    while True:

        ok, frame = cam.read()

        if not ok:
            continue

        caja = detectar_pcb(
            frame
        )

        if caja is not None:

            caja = ordenar(caja)

            # --------------------------
            # ESTABILIZAR LA PCB
            # --------------------------

            if caja_suave is None:

                caja_suave = caja.copy()

            else:

                caja_suave = (
                    SUAVIZADO_PCB
                    * caja_suave
                    + (1 - SUAVIZADO_PCB)
                    * caja
                )

            warp, inversa, caja_rect = rectificar(
                frame,
                caja_suave
            )

            puntos, mask = detectar_puntos(
                warp
            )

            tracks, next_id = actualizar_tracks(
                puntos,
                tracks,
                next_id
            )

            estables = [
                t
                for t in tracks
                if t["vistos"] >= FRAMES_ESTABLE
                and t["perdidos"] == 0
            ]

            estables.sort(
                key=lambda t: (
                    t["y"],
                    t["x"]
                )
            )

            visual = warp.copy()

            # Verde = deteccion actual
            for x, y, r in puntos:

                cv2.circle(
                    visual,
                    (x, y),
                    max(4, r),
                    (0, 255, 0),
                    1
                )

            # Amarillo = deteccion estable
            for n, t in enumerate(
                estables,
                1
            ):

                cv2.circle(
                    visual,
                    (
                        t["x"],
                        t["y"]
                    ),
                    max(
                        6,
                        t["r"] + 2
                    ),
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    visual,
                    str(n),
                    (
                        t["x"] + 6,
                        t["y"] - 6
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1
                )

            cv2.imshow(
                "PCB rectificada",
                visual
            )

            cv2.imshow(
                "Mascara agujeros",
                mask
            )

            cv2.polylines(
                frame,
                [
                    np.int32(
                        caja_rect
                    )
                ],
                True,
                (255, 0, 0),
                2
            )

            if estables:

                pts = np.float32([
                    [
                        [
                            t["x"],
                            t["y"]
                        ]
                    ]
                    for t in estables
                ])

                pts_img = cv2.perspectiveTransform(
                    pts,
                    inversa
                ).reshape(
                    -1,
                    2
                )

                for n, p in enumerate(
                    pts_img,
                    1
                ):

                    x = int(p[0])
                    y = int(p[1])

                    cv2.circle(
                        frame,
                        (x, y),
                        6,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        str(n),
                        (
                            x + 7,
                            y - 7
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1
                    )

        else:

            tracks = []
            estables = []

        cv2.putText(
            frame,
            f"Puntos estables: {len(estables)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "g=guardar  c=limpiar  q=salir",
            (20, FRAME_H - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "PCB Vision",
            frame
        )

        tecla = (
            cv2.waitKey(1)
            & 0xFF
        )

        if tecla == ord("g"):

            if estables:

                guardar(
                    estables
                )

            else:

                print(
                    "No hay puntos estables"
                )

        elif tecla == ord("c"):

            tracks = []
            next_id = 1
            caja_suave = None

            print(
                "Detecciones limpiadas"
            )

        elif tecla == ord("q"):

            break

    cam.release()

    cv2.destroyAllWindows()

# INICIO DEL PROGRAMA
if __name__ == "__main__":
    main()