# sign_spawner.py
from qvl.stop_sign import QLabsStopSign
from qvl.traffic_light import QLabsTrafficLight

def spawn_signs(qlabs_instance):

    stop_sign = QLabsStopSign(qlabs_instance)
    traffic_light = QLabsTrafficLight(qlabs_instance)

    stop_sign_location = [-2.038, -3.421, 0.2]
    stop_sign_rotation = [0.0, 0.0, 1.535]
    stop_sign_scale = [1.0, 1.0, 1.0]
    stop_sign_configuration = 0 # Default configuration

    traffic_light_location = [7.313, 5.868, 0.215]
    traffic_light_rotation = [0.0, 0.0, 4.712]
    traffic_light_scale = [1.0, 1.0, 1.0]
    traffic_light_configuration = 0 # Default configuration

    try:
        # Attempt to spawn the stop sign
        stop_sign.spawn(stop_sign_location, stop_sign_rotation, stop_sign_scale, stop_sign_configuration)
        print("Stop sign spawned successfully.")
        
        # Attempt to spawn the traffic light
        traffic_light.spawn(traffic_light_location, traffic_light_rotation, traffic_light_scale, traffic_light_configuration)
        print("Traffic light spawned successfully.")
    except Exception as e:
        print(f"Failed to spawn signs: {e}. They might already exist or there's a connection issue with QLabs.")
