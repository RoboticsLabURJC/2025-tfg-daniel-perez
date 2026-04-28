import carla
from lidar_semantich import set_lidar
from lidar_semantich import lidar_callback
from vehicle import control
from camera import set_camera
from camera import camera_callback
import pygame
import open3d as o3d
from datetime import datetime
import sys
import time


actor_list = []         # Lista de actores de Carla
absolute_view = False   # Vista del LiDAR
current_view_index = 0  # Indice de la vista actual

# Tipos de vistas de visualización del sensor LiDAR
camera_views = [
    {"zoom": 0.3, "front": [0, 0, 1], "lookat": [0, 0, 0], "up": [-1, 0, 0], "view":"Primera persona"},  # Primera persona
    {"zoom": 0.06, "front": [1.0, 0.0, 0.3], "lookat": [0, 0, 0], "up": [0, 0, 1], "view":"Tercera persona"},  # Tercera persona
    {"zoom": 0.2, "front": [0, 0, -1], "lookat": [0, 0, 0], "up": [0, 1, 0], "view":"Vista cenital"}  # Vista cenital
]

# Crear una esfera en el origen: donde está el sensor/vehículo
def create_origin_sphere():
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3)
    sphere.translate([0, 0, 0])  # Posicionar en el origen
    sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Color rojo
    return sphere

# Ajustar la vista seleccionada del diccionario "camera_views"
def set_camera_view(viz, view_index):
    ctr = viz.get_view_control()
    view = camera_views[view_index]
    ctr.set_zoom(view["zoom"])
    ctr.set_front(view["front"])
    ctr.set_lookat(view["lookat"])
    ctr.set_up(view["up"])
    print(f'Cámara ajustada: {view["view"]}')

# Cerebro / Función principal
def brain():

    pygame.init()

    # Configurar Pygame sin usar OpenGL explícitamente
    pygame.display.gl_set_attribute(pygame.GL_ACCELERATED_VISUAL, 0)
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height), pygame.SRCALPHA)
    pygame.display.set_caption("CARLA Vehículo Control")

    # Conexión con servidor de Carla
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    client.load_world('Town01')  # Establecer escenario determinado

    # Crear escenario
    world = client.get_world()

    # Configuración de la niebla (Clima)
    weather = carla.WeatherParameters(
        fog_density=18.8,  
        cloudiness=40.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=0.0,
        sun_azimuth_angle=180.0,
        sun_altitude_angle=70.0
    )
    world.set_weather(weather)  # Aplicar los cambios en el clima

    # Librería del simulador
    blueprint_library = world.get_blueprint_library()
    traffic_manager = client.get_trafficmanager(8000)               # Control del tráfico de vehículos automáticos
    traffic_manager.set_synchronous_mode(True)                      # Modo síncrono: avanza en cada "world.tick()"
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)     # Distancia de seguridad entre vehículos

    # Configuración del simulador
    settings = world.get_settings()                # Obtener configuración actual              
    delta = 0.033                                   # 0.05 s por cada tick: 20 FPS
    settings.fixed_delta_seconds = delta           # la simulación avanza cada 0.05 s
    settings.synchronous_mode = True               # Activa el modo síncrono
    world.apply_settings(settings)                 # Aplica los ajustes modificados

    # Librería de vehículos
    vehicle_blueprints = blueprint_library.filter('*vehicle*')

    # Cogemos todos los puntos de aparición predefinidos del escenario
    spawn_points = world.get_map().get_spawn_points()

    # Seleccionamos un vehículo determinado de la librería
    vehicle_bp = vehicle_blueprints.find('vehicle.audi.tt')

    # Generar vehículo con el que se va a trabajar
    ego_vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    if ego_vehicle is not None:
        print('Vehículo spawneado correctamente')

    # Establecer sensor LiDAR al vehículo
    lidar = set_lidar(world, ego_vehicle, delta)

    # Establecer cámara rgb al vehículo
    camera = set_camera(world, ego_vehicle)

    global actor_list, current_view_index, origin_sphere_added, absolute_view

    actor_list.append(ego_vehicle)
    actor_list.append(camera)
    actor_list.append(lidar)

    camera.listen(lambda image: camera_callback(image, screen))

    # Crear nube de puntos y llamar al callback
    point_cloud = o3d.geometry.PointCloud()
    lidar.listen(lambda data: lidar_callback(data, point_cloud, lidar.get_transform(), absolute_view))

    # Visualizador de Open3D y configuraciones iniciales
    viz = o3d.visualization.VisualizerWithKeyCallback()
    viz.create_window(window_name='Sensor LiDAR', width=960, height=540, left=480, top=270)
    viz.get_render_option().background_color = [0.05, 0.05, 0.05]
    viz.get_render_option().point_size = 1.35
    viz.get_render_option().show_coordinate_frame = True

    # Configurar vista
    set_camera_view(viz, current_view_index)

    # Inicializar esfera en el origen
    origin_sphere = create_origin_sphere()
    origin_sphere_added = False

    def toggle_camera_view(_):
        global current_view_index, absolute_view, origin_sphere_added
        # Alternar a la siguiente vista en el array de cámaras
        current_view_index = (current_view_index + 1) % len(camera_views)
        set_camera_view(viz, current_view_index)
        
        # Activar o desactivar vista absoluta y la esfera en el origen
        absolute_view = (current_view_index == len(camera_views) - 1)  # Vista cenital como absoluta
        if absolute_view and not origin_sphere_added:
            viz.add_geometry(origin_sphere)
            origin_sphere_added = True
        elif not absolute_view and origin_sphere_added:
            viz.remove_geometry(origin_sphere)
            origin_sphere_added = False
        
        print(f"Cambiando a vista {current_view_index + 1}")
        return True

    # Callback para alternar vistas
    viz.register_key_callback(ord("V"), toggle_camera_view)

    # Ciclo principal de visualización
    lidar_data_received = False
    dt0 = datetime.now()
    frame = 0
    try:
        while True:

            # Procesar control del vehículo
            control(ego_vehicle)

            # Procesar eventos de Pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\n Cierre de ventana detectado.")
                    raise KeyboardInterrupt

            # Añadir nube de puntos al visualizador una sola vez
            if frame == 5 and not lidar_data_received:
                viz.add_geometry(point_cloud)
                lidar_data_received = True
                print("Nube de puntos añadida a Open3D")

            # Actualizar visualización
            viz.update_geometry(point_cloud)    # actualizar nube de puntos
            viz.poll_events()                   # procesar eventos del sistema de ventanas
            viz.update_renderer()               # actualizar el contenido de la ventana
            time.sleep(0.03)
            world.tick()                        # avanzar la simulación
            o3d.io.write_point_cloud("nube_de_puntos.ply", point_cloud)  # Guardar como PLY



            # Mostrar FPS
            process_time = datetime.now() - dt0
            if process_time.total_seconds() > 0:
                fps = 1.0 / process_time.total_seconds()
                sys.stdout.write(f'\rFPS: {fps:.2f} ')
                sys.stdout.flush()
            dt0 = datetime.now()
            frame += 1

            # Salir si Open3D se cierra
            if not viz.poll_events():
                break

    except KeyboardInterrupt:
        pass
    finally:
        cleanup()

# Función de limpieza de actores del simulador (vehículos, sensores,etc)
def cleanup():
    global actor_list
    print("\nLimpiando actores...")
    for actor in actor_list:
        if actor is not None:
            actor.destroy()
    actor_list = []
    print("Actores eliminados.")


brain()
