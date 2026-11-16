
from robodk.robolink import *
from robodk.robomath import *
from math import pi

RDK = Robolink()

robot = RDK.Item('', ITEM_TYPE_ROBOT)

if not robot.Valid():
    raise Exception("No se encontro el UR10e")

caja = RDK.Item(
    'Box 12x10in (Open)',
    ITEM_TYPE_OBJECT
)

if not caja.Valid():
    raise Exception("No se encontro la caja")

pinza = robot.getLink(ITEM_TYPE_TOOL)

if not pinza.Valid():
    raise Exception("No se encontro la pinza")

ref = robot.Parent()

# Guardar configuracion inicial
juntas_inicio = robot.Joints()
frame_anterior = robot.PoseFrame()
tcp_anterior = robot.PoseTool()

robot.setPoseFrame(ref)
robot.setPoseTool(pinza)


ALTURA_AGARRE = 75.0
Z_SEGURA = 350.0
ALTURA_PISO_CAJA = 5.0
SEPARACION = 110.0

VELOCIDAD = 80.0
VELOCIDAD_JUNTAS = 30.0

# True: repetir indefinidamente
# False: ejecutar una ida y vuelta
BUCLE_CONTINUO = False

# Se utiliza cuando el bucle no es continuo
NUMERO_CICLOS = 1

robot.setSpeed(
    VELOCIDAD,
    VELOCIDAD_JUNTAS
)

robot.setRounding(0)



orden_ida = [
    'Cube4',
    'Cube3',
    'Cube1',
    'Cube2'
]

# Orden contrario para restaurar la pila
orden_vuelta = list(reversed(orden_ida))

mitad = SEPARACION / 2.0

desplazamientos = [
    (-mitad, -mitad),
    ( mitad, -mitad),
    (-mitad,  mitad),
    ( mitad,  mitad)
]

orientacion = roty(pi)


cubos = {}
poses_iniciales = {}
padres_originales = {}
poses_caja = {}

for nombre in orden_ida:

    cubo = RDK.Item(
        nombre,
        ITEM_TYPE_OBJECT
    )

    if not cubo.Valid():
        raise Exception(
            "No se encontro " + nombre
        )

    cubos[nombre] = cubo

    poses_iniciales[nombre] = (
        cubo.PoseAbs()
    )

    padres_originales[nombre] = (
        cubo.Parent()
    )

print("Posiciones iniciales guardadas")


# Si existe Centro_Caja, se utiliza.
# Si no existe, se mantiene el calculo
# original basado en el origen de la caja.

centro = RDK.Item(
    'Centro_Caja',
    ITEM_TYPE_TARGET
)

if centro.Valid():
    pose_centro = centro.PoseAbs()

    def calcular_destino(dx, dy):
        return (
            pose_centro
            * transl(
                dx,
                dy,
                ALTURA_PISO_CAJA
            )
        )

else:
    print(
        "AVISO: No existe Centro_Caja. "
        "Se usara el origen del objeto caja."
    )

    def calcular_destino(dx, dy):
        return (
            caja.PoseAbs()
            * transl(
                dx,
                dy,
                ALTURA_PISO_CAJA
            )
        )

destinos = {}

for i, nombre in enumerate(orden_ida):

    dx, dy = desplazamientos[i]

    destinos[nombre] = calcular_destino(
        dx, dy
    )



def pose_tcp(x, y, z):
    return (
        transl(x, y, z)
        * orientacion
    )


def mover_juntas(pose_absoluta):

    destino = (
        ref.PoseAbs().inv()
        * pose_absoluta
    )

    robot.MoveJ(destino)


def mover_lineal(pose_absoluta):

    destino = (
        ref.PoseAbs().inv()
        * pose_absoluta
    )

    robot.MoveL(destino)


def posicion_segura(pose):

    p = pose.Pos()

    return pose_tcp(
        p[0],
        p[1],
        max(Z_SEGURA, p[2] + 120)
    )


def pose_para_agarrar(pose_objeto):

    p = pose_objeto.Pos()

    return pose_tcp(
        p[0],
        p[1],
        p[2] + ALTURA_AGARRE
    )



def mover_cubo(
    nombre,
    pose_origen,
    pose_destino
):

    cubo = cubos[nombre]

    # Posicion de recogida
    pose_agarrar = pose_para_agarrar(
        pose_origen
    )

    pose_arriba_origen = posicion_segura(
        pose_agarrar
    )

    # Posicion de entrega
    pose_dejar = pose_para_agarrar(
        pose_destino
    )

    pose_arriba_destino = posicion_segura(
        pose_dejar
    )

    print("Moviendo:", nombre)

    # 1. Ir sobre el cubo
    mover_juntas(pose_arriba_origen)

    # 2. Bajar
    mover_lineal(pose_agarrar)

    # 3. Agarrar el cubo
    cubo.setParentStatic(pinza)

    print("Agarrado:", nombre)

    # 4. Levantar
    mover_lineal(pose_arriba_origen)

    # 5. Ir al destino
    mover_juntas(pose_arriba_destino)

    # 6. Bajar
    mover_lineal(pose_dejar)

    # 7. Soltar el cubo
    cubo.setParentStatic(
        padres_originales[nombre]
    )

    print("Soltado:", nombre)

    # 8. Retirar pinza
    mover_lineal(pose_arriba_destino)



def llevar_a_caja():

    print("")
    print("===== FASE DE IDA =====")

    for nombre in orden_ida:

        mover_cubo(
            nombre,
            cubos[nombre].PoseAbs(),
            destinos[nombre]
        )

        # Guardar posicion real
        # despues de soltar el cubo
        poses_caja[nombre] = (
            cubos[nombre].PoseAbs()
        )

    print("4 CUBOS EN LA CAJA")




def devolver_a_inicio():

    print("")
    print("===== FASE DE VUELTA =====")

    for nombre in orden_vuelta:

        mover_cubo(
            nombre,
            cubos[nombre].PoseAbs(),
            poses_iniciales[nombre]
        )

    print("4 CUBOS DEVUELTOS")




def ciclo_completo():

    # Llevar los cubos a la caja
    llevar_a_caja()

    # Devolverlos a su posicion original
    devolver_a_inicio()

    # Regresar el UR10e a su posicion
    print("Regresando UR10e al inicio")

    robot.MoveJ(juntas_inicio)

    print("CICLO COMPLETADO")




contador = 0

try:

    while True:

        contador += 1

        print("")
        print("==========================")
        print("INICIANDO CICLO", contador)
        print("==========================")

        ciclo_completo()

        if not BUCLE_CONTINUO:
            if contador >= NUMERO_CICLOS:
                break

    print("")
    print("SIMULACION FINALIZADA")

finally:

    # Restaurar las configuraciones activas
    robot.setPoseFrame(frame_anterior)
    robot.setPoseTool(tcp_anterior)
