import carla
import pygame

# Función que permite el control del vehículo por teclado
def control(vehicle):

    control = carla.VehicleControl()

    # Leer teclas
    keys = pygame.key.get_pressed()

    # valores iniciales
    control.throttle = 0.0
    control.steer = 0.0
    control.brake = 0.0
    control.hand_brake = False
    control.reverse = False

    # W acelerar
    if keys[pygame.K_w]:
        control.throttle = 1.0

    # S retroceso/stop
    if keys[pygame.K_s]:
        if vehicle.get_velocity().length() < 0.1:
            # Si está casi parado: poner marcha atrás
            control.reverse = True
            control.throttle = 1.0
        else:
            # Si va hacia adelante: frenar
            control.brake = 1.0
    
    # A girar a la izquierda
    if keys[pygame.K_a]:
        control.steer = -0.5  # negativo = izquierda

    # D : girar a la derecha
    if keys[pygame.K_d]:
        control.steer = 0.5   # positivo = derecha

    # Aplicar al vehículo
    vehicle.apply_control(control)
    