import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree

class ImprovedPlaneSegmenter:
    def __init__(self, min_points=10, distance_threshold=0.15, normal_threshold=0.7):
        self.min_points = min_points
        self.distance_threshold = distance_threshold
        self.normal_threshold = normal_threshold
        
    def segment_planes_improved(self, points, max_planes=5):
        """
        Segmentación mejorada de planos con preprocesamiento
        """
        if len(points) < self.min_points:
            return []
        
        # PREPROCESAMIENTO: Filtrar puntos muy dispersos
        filtered_points = self._preprocess_points(points)
        
        if len(filtered_points) < self.min_points:
            return []
        
        planes = []
        remaining_points = filtered_points.copy()
        
        for i in range(max_planes):
            if len(remaining_points) < self.min_points:
                break
                
            # Método 1: Intentar con RANSAC mejorado
            plane_points, plane_normal, plane_d = self._fit_plane_ransac(remaining_points)
            
            if plane_points is not None and len(plane_points) >= self.min_points:
                plane_data = {
                    'points': plane_points,
                    'normal': plane_normal,
                    'd': plane_d,
                    'center': np.mean(plane_points, axis=0),
                    'size': len(plane_points)
                }
                planes.append(plane_data)
                
                # Remover inliers para siguiente iteración
                mask = self._create_exclusion_mask(remaining_points, plane_points)
                remaining_points = remaining_points[mask]
            else:
                # Método 2: Si RANSAC falla, usar agrupamiento por densidad
                cluster_planes = self._segment_by_clustering(remaining_points)
                planes.extend(cluster_planes)
                break
        
        return planes
    
    def _preprocess_points(self, points):
        """Preprocesamiento para mejorar la detección de planos"""
        if len(points) == 0:
            return points
        
        # 1. Filtrar puntos basados en densidad local
        if len(points) > 50:
            try:
                tree = KDTree(points)
                # Calcular densidad de puntos
                distances, _ = tree.query(points, k=5)
                avg_distances = np.mean(distances, axis=1)
                
                # Mantener puntos en áreas densas (umbral adaptativo)
                density_threshold = np.percentile(avg_distances, 70)
                dense_mask = avg_distances < density_threshold
                points = points[dense_mask]
            except:
                pass
        
        # 2. Filtrar outliers basados en distancia al centroide
        if len(points) > 30:
            centroid = np.mean(points, axis=0)
            distances_to_centroid = np.linalg.norm(points - centroid, axis=1)
            distance_threshold = np.percentile(distances_to_centroid, 85)
            inlier_mask = distances_to_centroid < distance_threshold
            points = points[inlier_mask]
        
        return points
    
    def _fit_plane_ransac(self, points):
        """Ajustar plano usando RANSAC con parámetros adaptativos"""
        if len(points) < 10:
            return None, None, None
            
        try:
            # Parámetros adaptativos basados en el número de puntos
            adaptive_threshold = max(0.1, self.distance_threshold * (100 / len(points)))
            
            # Usar diferentes estrategias según la orientación esperada
            strategies = [
                {'use_xy': True, 'predict_z': True},   # Planos horizontales
                {'use_xy': False, 'predict_z': False}, # Planos verticales
            ]
            
            best_plane = None
            best_inliers = None
            best_score = 0
            
            for strategy in strategies:
                if strategy['use_xy']:
                    # Para planos cercanos a horizontales (z = ax + by + c)
                    X = points[:, :2]  # x, y
                    y = points[:, 2]   # z
                else:
                    # Para planos cercanos a verticales (y = ax + bz + c)
                    X = np.column_stack([points[:, 0], points[:, 2]])  # x, z
                    y = points[:, 1]   # y
                
                ransac = RANSACRegressor(
                    residual_threshold=adaptive_threshold,
                    max_trials=200,
                    min_samples=max(10, len(points) // 5)
                )
                
                try:
                    ransac.fit(X, y)
                    inlier_mask = ransac.inlier_mask_
                    
                    if np.sum(inlier_mask) > best_score:
                        best_score = np.sum(inlier_mask)
                        best_inliers = points[inlier_mask]
                        
                        # Calcular normal del plano
                        if strategy['use_xy']:
                            # Plano: z = a*x + b*y + c -> a*x + b*y - z + c = 0
                            normal = np.array([ransac.estimator_.coef_[0], 
                                             ransac.estimator_.coef_[1], -1])
                            d = ransac.estimator_.intercept_
                        else:
                            # Plano: y = a*x + b*z + c -> a*x - y + b*z + c = 0
                            normal = np.array([ransac.estimator_.coef_[0], -1, 
                                             ransac.estimator_.coef_[1]])
                            d = ransac.estimator_.intercept_
                        
                        normal = normal / np.linalg.norm(normal)
                        best_plane = (best_inliers, normal, d)
                        
                except Exception as e:
                    continue
            
            return best_plane if best_plane else (None, None, None)
            
        except Exception as e:
            return None, None, None
    
    def _segment_by_clustering(self, points):
        """Segmentación usando agrupamiento por densidad"""
        if len(points) < 30:
            return []
        
        try:
            # DBSCAN con parámetros adaptativos
            eps = 0.8
            min_samples = max(5, len(points) // 20)
            
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
            labels = clustering.labels_
            
            planes = []
            
            for label in set(labels):
                if label == -1:  # Ruido
                    continue
                    
                cluster_points = points[labels == label]
                
                if len(cluster_points) >= self.min_points:
                    # Calcular planitud del cluster
                    flatness = self._calculate_flatness(cluster_points)
                    
                    if flatness > 0.7:  # Umbral de planitud
                        # Calcular plano aproximado
                        centroid = np.mean(cluster_points, axis=0)
                        cov_matrix = np.cov(cluster_points.T)
                        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                        normal = eigenvectors[:, 0]  # Autovector de menor autovalor
                        
                        plane_data = {
                            'points': cluster_points,
                            'normal': normal,
                            'd': -np.dot(normal, centroid),
                            'center': centroid,
                            'size': len(cluster_points),
                            'flatness': flatness
                        }
                        planes.append(plane_data)
            
            return planes
            
        except Exception as e:
            return []
    
    def _calculate_flatness(self, points):
        """Calcular qué tan plano es un conjunto de puntos"""
        if len(points) < 10:
            return 0.0
            
        try:
            # Análisis de componentes principales
            centered = points - np.mean(points, axis=0)
            cov_matrix = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            eigenvalues = np.sort(eigenvalues)  # Orden ascendente
            
            # La planitud es la relación entre el autovalor más pequeño y la suma
            flatness = eigenvalues[0] / (np.sum(eigenvalues) + 1e-8)
            return flatness
        except:
            return 0.0
    
    def _create_exclusion_mask(self, all_points, plane_points):
        """Crear máscara para excluir puntos del plano detectado"""
        if len(plane_points) == 0:
            return np.ones(len(all_points), dtype=bool)
        
        try:
            tree = KDTree(plane_points)
            distances, _ = tree.query(all_points, k=1)
            mask = distances > self.distance_threshold * 2
            return mask
        except:
            return np.ones(len(all_points), dtype=bool)
