import numpy as np


def identify_walls(planes, lidar_position=np.array([0, 0, 0])):
    """Identifica qué tipo de pared está detectando cada plano"""
    
    print(f"\n=== IDENTIFICACIÓN DE PAREDES ===")
    
    for i, plane in enumerate(planes):
        normal = plane['normal']
        center = plane['center']
        size = plane['size']
        
        # Calcular dirección relativa al LiDAR
        direction_to_plane = center - lidar_position
        direction_to_plane = direction_to_plane / np.linalg.norm(direction_to_plane)
        
        # Ángulo con la horizontal
        horizontal_angle = np.degrees(np.arccos(np.abs(normal[2])))
        
        # Determinar orientación de la pared
        wall_orientation = "DESCONOCIDA"
        
        if horizontal_angle > 75:  # Pared vertical
            # Verificar dirección basada en la normal
            if abs(normal[0]) > abs(normal[1]):
                if normal[0] > 0:
                    wall_orientation = "PARED ESTE (X+)"
                else:
                    wall_orientation = "PARED OESTE (X-)"
            else:
                if normal[1] > 0:
                    wall_orientation = "PARED NORTE (Y+)"
                else:
                    wall_orientation = "PARED SUR (Y-)"
        
        elif horizontal_angle < 25:  # Suelo/techo
            if normal[2] > 0:
                wall_orientation = "TECHO"
            else:
                wall_orientation = "SUELO"
        else:  # Inclinado
            wall_orientation = "SUPERFICIE INCLINADA"
        
        # Distancia al LiDAR
        distance = np.linalg.norm(center -lidar_position)#CENTER
        
        print(f"\nPlano {i+1}:")
        print(f"   Puntos: {size}")
        print(f"   Centro: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}]")
        print(f"   Normal: [{normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f}]")
        print(f"   Ángulo: {horizontal_angle:.1f}°")
        print(f"   Tipo: {wall_orientation}")
        print(f"   Distancia: {distance:.1f}m")
        print(f"   Dirección: [{direction_to_plane[0]:.2f}, {direction_to_plane[1]:.2f}, {direction_to_plane[2]:.2f}]")
