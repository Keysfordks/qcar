# main.py

import time
import numpy as np
import cv2
import threading
from qvl.qlabs import QuanserInteractiveLabs as QLabs
from qvl.qcar import QLabsQCar

from lanenet_follower import LaneNetLaneFollower
from yolo_detector import YOLODetector
from sign_spawner import spawn_signs

# Global variables for communication between threads
current_traffic_signal = "GO"
qcar_lock = threading.Lock()
latest_yolo_image = None
base_forward_speed = 1.0 # meters/second
current_forward_speed = base_forward_speed
global was_stopped, stop_timer, stop_hold_time

def update_traffic_signal(signal):
    global current_traffic_signal
    current_traffic_signal = signal

def receive_yolo_image(image):
    """Callback function to receive and store the latest image from the YOLO detector."""
    global latest_yolo_image
    latest_yolo_image = image

def main():
    qlabs = QLabs()
    try:
        qlabs.open("localhost")
    except Exception as e:
        print(f"Failed to connect to QLabs: {e}")
        return

    # Instantiate QCar and set up the camera
    qcar = QLabsQCar(qlabs)
    qcar.spawn(location=[8.404, 44.949, 0.005], rotation=[0, 0, 3.139])
    qcar.possess(qcar.CAMERA_CSI_FRONT)
    
    # Spawn traffic signs for the YOLO detector
    spawn_signs(qlabs)

    # Initialize the LaneNet-based lane follower and YOLO detector
    lane_follower = LaneNetLaneFollower()
    yolo_detector = YOLODetector(
        qcar,
        update_callback=update_traffic_signal,
        image_callback=receive_yolo_image,
        qcar_lock=qcar_lock
    )
    yolo_detector.start()

    # Create windows for debugging
    cv2.namedWindow('Lane Debug', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Lane Debug', 1280, 480)
    cv2.namedWindow('YOLO Debug', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLO Debug', 640, 480)

    # Simulation parameters
    forward_speed = 3.0
    dt = 0.05  # Time step for the control loop

    # PID controller parameters (these can be tuned for better performance)
    Kp = 0.1
    Ki = 0.001
    Kd = 0.03
    integral_error = 0.0
    prev_error = 0.0

    try:
        while True:
            start_time = time.time()

            # Acquire an image from the QCar's front camera
            with qcar_lock:
                status, rgbd = qcar.get_image(qcar.CAMERA_CSI_FRONT)

            if not status or rgbd is None:
                time.sleep(0.01)
                continue

            # Convert BGRA image to BGR for OpenCV processing
            image_rgb = cv2.cvtColor(rgbd, cv2.COLOR_BGRA2BGR) if rgbd.shape[2] == 4 else rgbd
            
            # Use the new LaneNetLaneFollower to get steering and a debug image
            steering, error, lane_x, annotated_image = lane_follower.compute_steering(image_rgb, forward_speed)

            # PID control for smoother steering
            integral_error += error * dt
            derivative_error = (error - prev_error) / dt
            pid_steering = Kp * error + Ki * integral_error + Kd * derivative_error
            filtered_steering = np.clip(pid_steering, -30, 30)
            prev_error = error

            # Adjust speed based on the steering angle to prevent skidding on turns
            mod_speed = forward_speed * np.cos(np.deg2rad(filtered_steering))
            mod_speed = np.clip(mod_speed, 0, forward_speed)

            # Adjust speed based on traffic signals from YOLO detector
            if current_traffic_signal == "STOP":
                modulated_forward_speed = 0.0

            # Send the velocity and steering commands to the QCar
            with qcar_lock:
                qcar.set_velocity_and_request_state_degrees(
                    forward=mod_speed,
                    turn=filtered_steering,
                    headlights=False,
                    leftTurnSignal=False,
                    rightTurnSignal=False,
                    brakeSignal=False,
                    reverseSignal=False
                )

            # Debug visualization
            for i, text in enumerate([
                f"Error: {error:.2f}",
                f"Steering: {filtered_steering:.2f}",
                f"Speed: {mod_speed:.2f}",
                f"Signal: {current_traffic_signal}"
            ]):
                cv2.putText(annotated_image, text, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Lane Debug', annotated_image)

            if latest_yolo_image is not None:
                cv2.imshow('YOLO Debug', latest_yolo_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Ensure the loop runs at a consistent rate (dt)
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)

    finally:
        print("Closing QLabs connection and cleaning up...")
        yolo_detector.stop()
        qlabs.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
