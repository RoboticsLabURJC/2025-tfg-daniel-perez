import carla
import numpy as np
import pygame

# Función para establecer cámara en el vehículo
def set_camera(world, vehicle):
    
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(
        carla.Location(x=1.5, z=1.2),       # 1.5 m adelante, 1.2 m de altura
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))  # mirando hacia adelante
    
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    return camera

# Función para procesar imágenes de la cámara
def camera_callback(image, display):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Quitar el canal alfa
    array = array[:, :, ::-1]  # Convertir de BGRA a RGB
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display.blit(surface, (0, 0))
    # Actualizar solo esta superficie en vez de toda la pantalla
    pygame.display.update(display.get_rect())
