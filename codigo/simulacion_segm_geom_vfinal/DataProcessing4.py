import os
import pickle
from ImprovedPlaneSegmenter import ImprovedPlaneSegmenter
from WallIdentification import identify_walls
import open3d as o3d
import numpy as np

def process_saved_data_improved(data_folder="lidar_data"):
    """Procesar datos con el segmentador mejorado"""
    if not os.path.exists(data_folder):
        print(f"No se encontró la carpeta {data_folder}")
        return
    
    pkl_files = [f for f in os.listdir(data_folder) if f.endswith('.pkl')]
    
    if not pkl_files:
        print("No se encontraron archivos de datos")
        return
    
    #segmenter = ImprovedPlaneSegmenter(min_points=15, distance_threshold=0.2)
    total_planes_detected = 0
    frames_with_planes = 0
    
    all_points = []  # Aquí acumularemos todos los puntos LiDAR
    
    for pkl_file in sorted(pkl_files):
        filepath = os.path.join(data_folder, pkl_file)
        print(f"\nProcesando {pkl_file}...")

        try:
            with open(filepath, 'rb') as f:
                point_clouds_data = pickle.load(f)
            
            for frame_data in point_clouds_data:
                points = frame_data['points']
                
                # Filtrar puntos que estén muy cerca de los demás 
                distances = np.linalg.norm(points, axis=1)  # Calcula las distancias de cada punto al origen
                mask = distances > 0.1  # Mantener solo los puntos a más de 0.1 metros de distancia
                points = points[mask]  # Aplica el filtro
                
                all_points.append(points)  # Acumula los puntos de cada frame
                print(f"Frame {frame_data['frame_id']}: {len(points)} puntos después de filtrado")

        except Exception as e:
            print(f"Error procesando {pkl_file}: {e}")
    
    # Convertir la lista de listas de puntos en un solo array de puntos
    all_points = np.vstack(all_points)  # Combina todos los puntos en un solo arreglo

    planes = segment_walls_and_floor_open3d(
        all_points,
        max_planes=20,
        distance_threshold=0.20,
        min_points=50,
        wall_angle_threshold=75,
        floor_angle_threshold=25
    )

    if planes:
        frames_with_planes += 1
        total_planes_detected += len(planes)
        print(f"  ✓ Detectadas {len(planes)} posibles paredes!")

        identify_walls(planes, lidar_position=np.array([0, 0, 2.5]))

        for i, plane in enumerate(planes):
            normal = plane['normal']
            angle_deg = np.degrees(np.arccos(np.abs(normal[2])))
            print(f"    Pared {i+1}: {plane['size']} puntos, ángulo: {angle_deg:.1f}°")

        if any(plane['size'] > 20 for plane in planes):
            original_cloud = o3d.geometry.PointCloud()
            original_cloud.points = o3d.utility.Vector3dVector(all_points)
            original_cloud.paint_uniform_color([0.3, 0.3, 0.3])

            visualize_planes_with_bounding_boxes(planes)

            response = input("\n¿Continuar? (s/n): ")
            if response.lower() != 's':
                return
    else:
        print("  ✗ No se detectaron paredes verticales")
    
    print(f"\n=== RESUMEN ===")
    print(f"Frames procesados: {sum(len(pickle.load(open(os.path.join(data_folder, f), 'rb'))) for f in pkl_files)}")
    print(f"Frames con planos: {frames_with_planes}")
    print(f"Total planos detectados: {total_planes_detected}")

def visualize_planes_with_bounding_boxes(planes):
    """Visualización de los planos con cajas delimitadoras (bounding boxes)"""
    if not planes:
        print("No hay planos para visualizar")
        return None
    
    # Crear una lista de geometrías (nubes de puntos y bounding boxes)
    geometries = []

    # Colores para diferentes planos
    colors = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1],
    ]
    
    for i, plane in enumerate(planes):
        if len(plane['points']) < 10:  # Solo visualizar planos significativos
            continue
        
        color = colors[i % len(colors)]
        
        # Crear nube de puntos del plano
        plane_cloud = o3d.geometry.PointCloud()
        plane_cloud.points = o3d.utility.Vector3dVector(plane['points'])
        plane_cloud.paint_uniform_color(color)
        geometries.append(plane_cloud)
        
        # Crear la caja delimitadora orientada (Oriented Bounding Box)
        try:
            if len(plane['points']) > 20:  # Solo crear cajas para planos con suficientes puntos
                obb = plane_cloud.get_oriented_bounding_box()  # Caja delimitadora orientada
                obb.color = [color[0] * 0.7, color[1] * 0.7, color[2] * 0.7]  # Ajustar color de la caja
                geometries.append(obb)  # Añadir la caja a las geometrías

        except Exception as e:
            print(f"  Nota: No se pudo crear bounding box para plano {i+1}: {e}")
    
    # Si no hay geometrías, salimos
    if not geometries:
        print("No hay geometrías válidas para visualizar")
        return None
    
    # Crear visualizador para los planos y las cajas delimitadoras
    viz = o3d.visualization.Visualizer()
    viz.create_window(
        window_name="Planos y Bounding Boxes",
        width=1000,
        height=800,
        left=100,
        top=100
    )
    
    # Añadir todas las geometrías (planos y cajas delimitadoras)
    for geometry in geometries:
        viz.add_geometry(geometry)
    
    # Configurar vista
    viz.get_render_option().background_color = [0.1, 0.1, 0.1]  # Fondo oscuro
    viz.get_render_option().point_size = 3.0  # Tamaño de los puntos
    viz.get_render_option().show_coordinate_frame = True  # Mostrar el sistema de coordenadas
    
    # Configurar cámara
    view_control = viz.get_view_control()
    view_control.set_front([0, -1, -0.5])  # Mirar hacia adelante
    view_control.set_up([0, 0, 1])         # Eje Z arriba
    view_control.set_lookat([0, 0, 0])     # Centro de la vista
    view_control.set_zoom(0.8)             # Ajuste de zoom
    
    print("Mostrando Planos y Bounding Boxes...")
    viz.run()  # Iniciar el visualizador
    viz.destroy_window()  # Cerrar la ventana de visualización después de terminar


def visualize_lidar_points(original_cloud=None):
    """Visualización de la nube de puntos LiDAR sin transformaciones"""
    
    if original_cloud is not None:
        # Asegúrate de que no haya transformaciones no deseadas. Esto mantiene las coordenadas tal como se capturan
        original_cloud.paint_uniform_color([0.5, 0.5, 0.5])  # Nube de puntos en gris
        
        # Crear el visualizador
        viz = o3d.visualization.Visualizer()
        viz.create_window(
            window_name="Nube de Puntos LiDAR",
            width=1000,
            height=800,
            left=100,
            top=100
        )
        
        # Añadir la nube de puntos al visualizador
        viz.add_geometry(original_cloud)
        
        # Configurar la vista
        viz.get_render_option().background_color = [0.1, 0.1, 0.1]  # Fondo oscuro
        viz.get_render_option().point_size = 3.0  # Tamaño de los puntos
        viz.get_render_option().show_coordinate_frame = True  # Mostrar el sistema de coordenadas
        
        # Configurar la cámara
        view_control = viz.get_view_control()
        view_control.set_front([0, -1, -0.5])  # Mirar hacia adelante
        view_control.set_up([0, 0, 1])         # Eje Z arriba
        view_control.set_lookat([0, 0, 0])     # Centro de la vista
        view_control.set_zoom(0.8)             # Zoom de la cámara
        
        print("  Mostrando Nube de Puntos LiDAR...")
        viz.run()  # Iniciar el visualizador
        viz.destroy_window()  # Cerrar la ventana de visualización después de terminar
def segment_walls_and_floor_open3d(
    points,
    max_planes=20,
    distance_threshold=0.20,
    min_points=50,
    wall_angle_threshold=75,
    floor_angle_threshold=25
):
    """
    Detecta planos con Open3D y devuelve:
    - las paredes verticales
    - el suelo más grande
    """

    wall_planes = []
    floor_candidates = []
    remaining_points = points.copy()

    print("\n=== DETECCIÓN DE PAREDES Y SUELO CON OPEN3D ===")

    for i in range(max_planes):

        if len(remaining_points) < min_points:
            print("No quedan suficientes puntos para seguir buscando planos.")
            break

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(remaining_points)

        plane_model, inliers = cloud.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=2000
        )

        if len(inliers) < min_points:
            print("Plano descartado: tiene pocos puntos.")
            break

        a, b, c, d = plane_model

        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)

        plane_points = remaining_points[inliers]
        center = np.mean(plane_points, axis=0)
        angle_deg = np.degrees(np.arccos(np.abs(normal[2])))

        plane_data = {
            "points": plane_points,
            "normal": normal,
            "d": d,
            "center": center,
            "size": len(plane_points),
            "angle_deg": angle_deg
        }

        print(f"Plano {i+1}: {len(plane_points)} puntos, ángulo {angle_deg:.1f}º")

        if angle_deg > wall_angle_threshold:
            plane_data["tipo"] = "pared"
            wall_planes.append(plane_data)
            print("  ✓ Guardado como posible pared")

        elif angle_deg < floor_angle_threshold:
            plane_data["tipo"] = "suelo"
            floor_candidates.append(plane_data)
            print("Candidato a suelo")

        else:
            plane_data["tipo"] = "inclinado"
            print("Descartado porque parece superficie inclinada")

        # Eliminar los puntos del plano detectado
        # aunque no se guarde, para poder buscar otros planos
        mask = np.ones(len(remaining_points), dtype=bool)
        mask[inliers] = False
        remaining_points = remaining_points[mask]

    selected_planes = []

    # Mantener el suelo más grande
    if floor_candidates:
        largest_floor = max(floor_candidates, key=lambda p: p["size"])
        selected_planes.append(largest_floor)
        print(f"\nSuelo seleccionado: {largest_floor['size']} puntos, ángulo {largest_floor['angle_deg']:.1f}º")

    # Añadir todas las paredes
    selected_planes.extend(wall_planes)

    print(f"Total planos seleccionados: {len(selected_planes)}")
    print(f"  - Suelos: {1 if floor_candidates else 0}")
    print(f"  - Paredes: {len(wall_planes)}")

    return selected_planes
