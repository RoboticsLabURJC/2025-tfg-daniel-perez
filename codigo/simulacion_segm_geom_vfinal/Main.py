import carla
import time
from LidarVisualizer import LidarVisualizer
from ImprovedPlaneSegmenter import *
from LidarSetup import *
from DataProcessing4 import process_saved_data_improved

def main_improved():
    actor_list = []
    visualizer = LidarVisualizer(save_data=True)
    segmenter = ImprovedPlaneSegmenter()
    
    try:
        print("Conectando con servidor CARLA...")
        
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        client.load_world('Town03')  # Establecer escenario determinado

        
        world = client.get_world()
        print("✓ Conectado al mundo CARLA")
        
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        
        print("Configurando escenario...")
        
        actors = setup_scenario_simple(world)
        actor_list.extend(actors)

        # Posicionar LiDAR
        lidar_location = carla.Location(x=38, y=0, z=1.5)
        lidar = set_lidar(world, lidar_location)
        actor_list.append(lidar)

        # Colocar la vista del simulador cerca del LiDAR
        spectator = world.get_spectator()

        spectator_location = carla.Location(
            x=lidar_location.x - 35,
            y=lidar_location.y,
            z=lidar_location.z + 8.0
        )

        spectator_rotation = carla.Rotation(
            pitch=-10,
            yaw=0,
            roll=0
        )

        spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))

        # Añadir la cámara RGB al escenario
        camera_location = carla.Location(x=30, y=0, z=1.5)
        camera = set_rgb_camera(world, camera_location)
        actor_list.append(camera)
        
        visualizer.start_visualization()
        time.sleep(2)
        
        lidar.listen(lambda data: lidar_callback(data, visualizer, segmenter))
        # Configurar el callback para la cámara RGB
        camera.listen(lambda image: rgb_camera_callback(image, visualizer))
        
        print("=" * 60)
        print("ESCENARIO MEJORADO CONFIGURADO")
        print("LiDAR en posición elevada (Z=4.0)")
        print("Segmentador con parámetros optimizados")
        print("Presiona Ctrl+C para salir")
        print("=" * 60)
        
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nInterrupción por usuario")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Limpiando...")
        visualizer.stop()
        time.sleep(1)
        
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()
        print("Limpieza completada")
        
        # Procesar datos con el método mejorado
        response = input("\n¿Procesar datos con segmentador mejorado? (s/n): ")
        if response.lower() == 's':
            process_saved_data_improved()

if __name__ == "__main__":
    main_improved()
