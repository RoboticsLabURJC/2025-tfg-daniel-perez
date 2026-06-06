import open3d as o3d
import numpy as np
import time
import os
import pickle
import threading
# Configuración de colores
from matplotlib import colormaps as cm
VIRIDIS = np.array(cm.get_cmap('inferno').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

class LidarVisualizer:
    def __init__(self, save_data=True, data_folder="lidar_data"):
        self.point_cloud = o3d.geometry.PointCloud()
        self.viz = None
        self.lidar_data_received = False
        self.last_update_time = 0
        self.update_interval = 0.2
        self.is_running = True
        self.points_queue = []
        self.colors_queue = []
        
        # Configuración para guardar datos
        self.save_data = save_data
        self.data_folder = data_folder
        self.point_clouds_data = []  # Almacenar todas las nubes de puntos
        self.frame_count = 0
        
        # Crear carpeta para guardar datos
        if self.save_data and not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
    
    def update_camera_image(self, image_array):
        img = o3d.geometry.Image(image_array)
        self.camera_image = img
        # Mostrar la imagen (opcional, se puede visualizar en OpenCV si es necesario)
        import cv2
        cv2.imshow('RGB Camera', image_array)
        cv2.waitKey(1)  # Mostrar la imagen una vez por frame
    
    def start_visualization(self):
        """Iniciar visualización en un hilo separado"""
        def visualization_thread():
            self.viz = o3d.visualization.Visualizer()
            self.viz.create_window(
                window_name='LiDAR - Detección de Paredes y Segmentación',
                width=1200, height=800,
                left=50, top=50
            )
            self.viz.get_render_option().background_color = [0.05, 0.05, 0.05]
            self.viz.get_render_option().point_size = 2.0
            self.viz.get_render_option().show_coordinate_frame = True
            
            print("Visualizador Open3D iniciado")
            
            while self.is_running:
                if self.points_queue and self.colors_queue:
                    points = self.points_queue.pop(0)
                    colors = self.colors_queue.pop(0)
                    
                    # Guardar datos si está habilitado
                    if self.save_data:
                        self.save_point_cloud_data(points, colors)
                    
                    # Acumular puntos
                    if self.lidar_data_received and len(np.asarray(self.point_cloud.points)) > 0:
                        all_points = np.vstack([np.asarray(self.point_cloud.points), points])
                        all_colors = np.vstack([np.asarray(self.point_cloud.colors), colors])
                    else:
                        all_points = points
                        all_colors = colors
                    
                    self.point_cloud.points = o3d.utility.Vector3dVector(all_points)
                    self.point_cloud.colors = o3d.utility.Vector3dVector(all_colors)
                    
                    if not self.lidar_data_received:
                        self.viz.add_geometry(self.point_cloud)
                        self.lidar_data_received = True
                        print("Nube de puntos añadida al visualizador")
                    else:
                        self.viz.update_geometry(self.point_cloud)
                    
                    self.viz.poll_events()
                    self.viz.update_renderer()
                
                time.sleep(0.05)
            
            if self.viz:
                self.viz.destroy_window()
        
        self.vis_thread = threading.Thread(target=visualization_thread)
        self.vis_thread.daemon = True
        self.vis_thread.start()
    
    def save_point_cloud_data(self, points, colors):
        """Guardar datos de nube de puntos"""
        timestamp = time.time()
        frame_data = {
            'frame_id': self.frame_count,
            'timestamp': timestamp,
            'points': points.copy(),
            'colors': colors.copy()
        }
        self.point_clouds_data.append(frame_data)
        self.frame_count += 1
        
        # Guardar cada 10 frames
        if self.frame_count % 50 == 0:
            self.save_to_disk()
    
    def save_to_disk(self):
        """Guardar datos en disco"""
        if not self.point_clouds_data:
            return
        
        filename = f"{self.data_folder}/pointclouds_{int(time.time())}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self.point_clouds_data, f)
        
        print(f"Datos guardados en {filename}")
        
        # Limpiar lista después de guardar
        self.point_clouds_data = []
    
    def update_point_cloud(self, points, colors):
        """Añadir datos a la cola para procesar en el hilo de visualización"""
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.points_queue.append(points)
            self.colors_queue.append(colors)
            self.last_update_time = current_time
    
    def stop(self):
        """Detener visualización y guardar datos finales"""
        self.is_running = False
        if self.save_data and self.point_clouds_data:
            self.save_to_disk()