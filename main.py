# main.py
import time
import numpy as np
import cv2
import torch
import threading

from qvl.qlabs import QuanserInteractiveLabs as QLabs
from qvl.qcar import QLabsQCar

from pit.LaneNet.nets import LaneNet 
from lanenet_utils import LaneNetUtils
from yolo_detector import YOLODetector 
from sign_spawner import spawn_signs 

# --- Shared Variables for Thread Communication ---
# Use a lock to protect access to shared resources if needed, though for these specific callbacks
# simple assignments should be mostly safe given the consumers read frequently.
# If more complex shared state (e.g., lists, dictionaries) were involved, a lock would be essential.
latest_yolo_image = None # This will store the image processed by YOLO (with detections)
current_traffic_signal = "GO"
qcar_lock = threading.Lock() # For thread-safe QCar access across threads

# --- Callback Functions for YOLODetector ---
def update_traffic_signal(signal):
    """Callback function to update the current traffic signal."""
    global current_traffic_signal
    current_traffic_signal = signal
    # print(f"Traffic Signal Updated: {current_traffic_signal}") # For debug

def receive_yolo_image(image):
    """Callback function to receive the processed image from YOLODetector."""
    global latest_yolo_image
    latest_yolo_image = image

# --- Main Function ---
def main():
    qlabs = QLabs()
    try:
        qlabs.open("localhost")
    except Exception as e:
        print(f"Failed to connect to QLabs: {e}")
        print("Please ensure QLabs is running and accessible.")
        return

    qcar = QLabsQCar(qlabs)
    

    qcar.spawn(location=[1.855, -7.611, 0.005], rotation=[0, 0, -0.717], scale=[1, 1, 1])

    # Possess the CSI Front camera for LaneNet
    qcar.possess(qcar.CAMERA_CSI_FRONT)

    # Spawn signs using the imported function
    spawn_signs(qlabs) # Call the function to spawn signs

    # Initialize LaneNetUtils
    lanenet_utils = LaneNetUtils(lanenet_processing_width=640, lanenet_processing_height=480)
    
    # Initialize YOLODetector and start its thread
    # YOLODetector will use CAMERA_RGBD
    yolo_detector = YOLODetector(qcar, 
                                 update_callback=update_traffic_signal, 
                                 image_callback=receive_yolo_image,
                                 qcar_lock=qcar_lock) # Pass the lock
    yolo_detector.start() # Start the YOLO detection thread

    # Define camera resolution for CSI Front camera (used by LaneNet)
    CSI_CAMERA_WIDTH = 820
    CSI_CAMERA_HEIGHT = 410

    # Base forward speed before modulation
    base_forward_speed = 1.0 # meters/second
    current_forward_speed = base_forward_speed

    # Set up the debug window size based on the LaneNet processing input size
    cv2.namedWindow('LaneNet Debug', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('LaneNet Debug', lanenet_utils.LANENET_PROCESSING_WIDTH * 2, lanenet_utils.LANENET_PROCESSING_HEIGHT)
    
    cv2.namedWindow('YOLO Debug', cv2.WINDOW_NORMAL) # New window for YOLO output
    cv2.resizeWindow('YOLO Debug', CSI_CAMERA_WIDTH, CSI_CAMERA_HEIGHT)


    filtered_steering_angle = 0.0
    filter_coeff = 0.2

    # Initialize variables returned by compute_steering to default safe values
    steering_angle_raw = 0.0
    error = 0.0
    lane_center = 0.0
    look_ahead_y = 0 
    left_fit = None
    right_fit = None
    left_lane_inds = []
    right_lane_inds = []

    try:
        while True:
            # --- Image Acquisition for LaneNet (CSI Front) ---
            # Acquire lock for thread-safe access to QCar's image acquisition
            with qcar_lock:
                status_rgb, image_rgb_lanenet = qcar.get_image(qcar.CAMERA_CSI_FRONT)
            
            if not status_rgb or image_rgb_lanenet is None:
                print("Warning: Could not get RGB image from CSI_FRONT or image is None for LaneNet.")
                time.sleep(0.01)
                continue 
            
            # --- LaneNet Processing ---
            final_binary_mask = lanenet_utils.process_image_for_lanenet(image_rgb_lanenet)

            current_height, current_width = final_binary_mask.shape

            # Compute steering angle using LaneNetUtils
            steering_angle_raw, error, lane_center, look_ahead_y, left_fit, right_fit, left_lane_inds, right_lane_inds = \
                lanenet_utils.compute_steering(final_binary_mask, current_forward_speed) 

            filtered_steering_angle = filter_coeff * steering_angle_raw + (1 - filter_coeff) * filtered_steering_angle

            # --- Speed Modulation based on Steering and Traffic Signal ---
            modulated_forward_speed = base_forward_speed * np.cos(np.deg2rad(filtered_steering_angle))
            modulated_forward_speed = np.clip(modulated_forward_speed, 0, base_forward_speed)

            # Adjust speed based on traffic signal
            if current_traffic_signal == "STOP":
                modulated_forward_speed = 0.0
                print("--- STOP SIGN DETECTED: CAR STOPPED ---")
            elif current_traffic_signal == "SLOW":
                modulated_forward_speed = min(modulated_forward_speed, base_forward_speed * 0.3) # Slow down to 30% of base
                print("--- YIELD/SLOW SIGN DETECTED: CAR SLOWING DOWN ---")
            
            current_forward_speed = modulated_forward_speed # Update for next iteration's look-ahead calculation


            # --- Debug Visualization for LaneNet ---
            visual_input_img = (lanenet_utils.imgTensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            visual_input_img = cv2.cvtColor(visual_input_img, cv2.COLOR_RGB2BGR) 

            binary_display = final_binary_mask.astype(np.uint8)
            binary_display = cv2.cvtColor(binary_display, cv2.COLOR_GRAY2BGR) 

            midline_x = binary_display.shape[1] // 2
            cv2.line(binary_display, (midline_x, 0), (midline_x, binary_display.shape[0]), (0, 255, 0), 2)

            lane_center_x_display = int(lane_center)
            cv2.circle(binary_display, (lane_center_x_display, look_ahead_y), 5, (0, 0, 255), -1)

            ploty = np.linspace(0, current_height - 1, current_height)
            if left_fit is not None:
                left_fitx = left_fit[0]*ploty + left_fit[1]
                left_points = np.array([np.transpose(np.vstack([np.clip(left_fitx, 0, current_width-1), ploty]))], np.int32)
                cv2.polylines(binary_display, left_points, False, (255, 0, 0), 3)
            if right_fit is not None:
                right_fitx = right_fit[0]*ploty + right_fit[1]
                right_points = np.array([np.transpose(np.vstack([np.clip(right_fitx, 0, current_width-1), ploty]))], np.int32)
                cv2.polylines(binary_display, right_points, False, (255, 0, 0), 3)

            text_y_offset = 30
            cv2.putText(binary_display, f"Error: {error:.2f}", (10, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(binary_display, f"Steering Raw: {steering_angle_raw:.2f}", (10, text_y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(binary_display, f"Steering Flt: {filtered_steering_angle:.2f}", (10, text_y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(binary_display, f"Lane Center: {lane_center:.2f}", (10, text_y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(binary_display, f"Look Ahead Y: {look_ahead_y}", (10, text_y_offset + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(binary_display, f"Speed Mod: {modulated_forward_speed:.2f}", (10, text_y_offset + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(binary_display, f"Signal: {current_traffic_signal}", (10, text_y_offset + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


            debug_image = np.hstack((visual_input_img, binary_display))
            cv2.imshow('LaneNet Debug', debug_image)

            # Display YOLO Debug Image if available
            if latest_yolo_image is not None:
                cv2.imshow('YOLO Debug', latest_yolo_image)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Send drive command using the filtered steering angle and modulated speed
            with qcar_lock: # Use the lock for setting velocity too
                qcar.set_velocity_and_request_state_degrees(
                    forward=modulated_forward_speed,
                    turn=filtered_steering_angle,
                    headlights=False,
                    leftTurnSignal=False,
                    rightTurnSignal=False,
                    brakeSignal=False,
                    reverseSignal=False
                )

            print(f"Driving... Steering: {filtered_steering_angle:.2f} deg (Raw: {steering_angle_raw:.2f}), Error: {error:.2f} px, Speed: {modulated_forward_speed:.2f} m/s, Signal: {current_traffic_signal}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Stopping YOLO detector...")
        yolo_detector.stop() # Stop the YOLO thread gracefully
        yolo_detector.join() # Wait for the YOLO thread to finish
        print("Closing QLabs...")
        qlabs.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()