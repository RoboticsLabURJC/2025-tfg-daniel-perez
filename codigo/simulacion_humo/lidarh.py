import carla
import numpy as np
import open3d as o3d
from matplotlib import colormaps as cm
import math



VIRIDIS = np.array(cm.get_cmap('inferno').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

def set_lidar(world, ego_vehicle, delta):
    # Crear LiDAR
    lidar_transform = carla.Transform(carla.Location(z=2.5))
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    
    # Ajustar parámetros de degradación
    lidar_bp.set_attribute('atmosphere_attenuation_rate', '0.02')  # Ajustar degradación atmosférica
    lidar_bp.set_attribute('dropoff_general_rate', '0.4')  # Simular bloqueos por humo
    
    # Otros parámetros
    lidar_bp.set_attribute('channels', '128')        # Número de rayos
    lidar_bp.set_attribute('range', '100')           # Alcance en metros
    lidar_bp.set_attribute('rotation_frequency', str(1 / delta))  # Hz
    lidar_bp.set_attribute('points_per_second', '1200000') 

    # Spawnear sensor y aplicarlo al vehículo
    sensor_lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
    return sensor_lidar

    
# Función para calcular la matriz de rotación
def euler_to_rotation_matrix(pitch, yaw, roll):
    # Convertir los ángulos de grados a radianes
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)
    roll = math.radians(roll)

    # Crear la matriz de rotación para cada eje
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Multiplicar en el orden Yaw -> Pitch -> Roll
    rotation_matrix = R_z @ R_y @ R_x
    return rotation_matrix

# Función de callback del sensor LiDAR

def lidar_callback(lidar_data, point_cloud, vehicle_transform, absolute_view):

    # Leer los datos del LiDAR y reorganizarlos en forma de puntos e intensidad
    data = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    # Reflejar los datos en el eje X para alinearlos con el sistema de CARLA
    data[:, 0] = -data[:, 0]

    if absolute_view:
        # Obtener ubicación y rotación del vehículo
        vehicle_location = np.array([-vehicle_transform.location.x,
                                     vehicle_transform.location.y,
                                     vehicle_transform.location.z])
        
        pitch = vehicle_transform.rotation.pitch
        yaw = vehicle_transform.rotation.yaw
        roll = vehicle_transform.rotation.roll

        # Crear la matriz de rotación inversa usando la orientación del vehículo
        rotation_matrix = euler_to_rotation_matrix(pitch, yaw, roll).T  # Usamos la transpuesta como la inversa

        # Aplicar la rotación inversa y luego posicionar en el sistema global
        points_in_world = []
        for point in data[:, :3]:  # 'data' contiene los puntos LiDAR relativos
            # Aplicar la rotación inversa al punto
            rotated_point = rotation_matrix @ point
            # Posicionar el punto en el sistema global usando la posición inicial del vehículo
            global_point = rotated_point + vehicle_location
            points_in_world.append(global_point)

        # Convertir a formato numpy para visualización en Open3D
        points_to_display = np.array(points_in_world)

    else:
        # Usar coordenadas relativas sin transformación
        points_to_display = data[:, :3]

    # Mapear la intensidad de cada punto a un color para visualización
    intensity = data[:, -1]
    int_color = np.c_[
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 2])
    ]

    # Actualizar el point cloud en Open3D
    point_cloud.points = o3d.utility.Vector3dVector(points_to_display)
    point_cloud.colors = o3d.utility.Vector3dVector(int_color)
