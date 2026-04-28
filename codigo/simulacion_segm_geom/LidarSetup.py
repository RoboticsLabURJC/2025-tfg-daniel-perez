import carla
import numpy as np
from matplotlib import colormaps as cm


VIRIDIS = np.array(cm.get_cmap('inferno').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
def setup_scenario_simple(world):
    """Configuración simple usando el mapa existente"""
    for actor in world.get_actors().filter('*vehicle*'):
        actor.destroy()
    
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        carla.Location(x=15, y=10, z=2.0), 
        carla.Rotation(pitch=-60)
    ))
    
    print("Usando estructuras existentes del mapa")
    return []

def set_lidar(world, location):
    """Configura el sensor LiDAR"""
    lidar_transform = carla.Transform(location)
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    
    lidar_bp.set_attribute('rotation_frequency', '5')
    lidar_bp.set_attribute('channels', '128')           # Más resolución
    lidar_bp.set_attribute('range', '100')              # Mayor alcance
    lidar_bp.set_attribute('points_per_second', '120000') # Más puntos
    
    sensor_lidar = world.spawn_actor(lidar_bp, lidar_transform)
    return sensor_lidar

def set_rgb_camera(world, location):
    """Configura el sensor RGB (cámara)"""
    camera_transform = carla.Transform(location)  # Mirando hacia la izquierda
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

    camera_bp.set_attribute('image_size_x', '800')  # Resolución horizontal
    camera_bp.set_attribute('image_size_y', '600')  # Resolución vertical
    camera_bp.set_attribute('fov', '90')           # Campo de visión

    sensor_camera = world.spawn_actor(camera_bp, camera_transform)
    return sensor_camera

def rgb_camera_callback(image, visualizer):
    """Callback para la cámara RGB"""
    try:
        # Convertir imagen a un arreglo de NumPy
        image_array = np.frombuffer(image.raw_data, dtype=np.uint8)
        image_array = np.reshape(image_array, (image.height, image.width, 4))  # RGBA
        image_array = image_array[:, :, :3]  # Eliminar el canal alfa
        
        # Asegurarse de que los datos sean contiguos
        image_array = np.ascontiguousarray(image_array)

        # Visualizar la imagen
        visualizer.update_camera_image(image_array)

    except Exception as e:
        print(f"Error en callback de la cámara RGB: {e}")

def lidar_callback(lidar_data, visualizer, segmenter=None):
    """Callback para el LiDAR"""
    try:
        data = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        
        #data[:, 0] = -data[:, 0]  # Reflejar eje X
        
        points = data[:, :3]
        intensity = data[:, -1]
        
        # Filtrar puntos muy cercanos
        distances = np.linalg.norm(points, axis=1)
        mask = distances > 2.0
        points = points[mask]
        intensity = intensity[mask]
        
        if len(points) > 0:
            # Mapear intensidad a colores
            int_color = np.c_[ 
                np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]),
                np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]),
                np.interp(intensity, VID_RANGE, VIRIDIS[:, 2])
            ]
            
            visualizer.update_point_cloud(points, int_color)
            
            # Segmentación en tiempo real (opcional)
            if segmenter and len(points) > 100:
                planes = segmenter.segment_planes_improved(points, max_planes=3)
                if planes and len(planes) > 0:
                    print(f"Detectados {len(planes)} planos en tiempo real")
            
    except Exception as e:
        print(f"Error en callback LiDAR: {e}")
    return