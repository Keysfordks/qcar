import numpy as np
import cv2
import torch

class LaneNetLaneFollower:
    def __init__(self, width=640, height=480, row_upper_bound=228):
        self.width = width
        self.height = height
        self.row_upper_bound = row_upper_bound
        from pit.LaneNet.nets import LaneNet
        self.lanenet = LaneNet(imageHeight=height, imageWidth=width, rowUpperBound=row_upper_bound)
        self.prev_lane_x = None
        self.max_jump = 50  # max allowed pixel jump between frames for lane center smoothing
        self.steering_history = []
        self.history_size = 5

    def compute_steering(self, image_rgb, current_speed):
        cropped = image_rgb[self.row_upper_bound:, :, :]
        resized = cv2.resize(cropped, (512, 256), interpolation=cv2.INTER_LINEAR)

        input_img = resized.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).to(torch.float32)

        binary_mask, instance_mask = self.lanenet.predict(input_tensor)

        annotated_image = image_rgb.copy()
        orig_height, orig_width = image_rgb.shape[:2]

        lane_points = np.where(binary_mask > 128)
        if len(lane_points[1]) == 0:
            error = 0.0
            steering = 0.0
            lane_x = None
        else:
            # Focus on bottom quarter of the image
            bottom_y_threshold = 256 - (256 // 4)  # bottom 25% rows in resized mask
            mask_bottom_indices = lane_points[0] >= bottom_y_threshold
            
            lane_xs_bottom = lane_points[1][mask_bottom_indices]
            lane_ys_bottom = lane_points[0][mask_bottom_indices]

            if len(lane_xs_bottom) == 0:
                # If no lane pixels near bottom, fallback to all pixels
                lane_xs_to_use = lane_points[1]
            else:
                lane_xs_to_use = lane_xs_bottom

            # Define horizontal ROI around previous lane_x or center
            if self.prev_lane_x is None:
                center_roi_x = 512 // 2
            else:
                center_roi_x = self.prev_lane_x

            roi_width = 100  # pixels to each side from center
            left_bound = max(center_roi_x - roi_width, 0)
            right_bound = min(center_roi_x + roi_width, 511)

            # Filter lane_x pixels within horizontal ROI
            valid_indices = np.where((lane_xs_to_use >= left_bound) & (lane_xs_to_use <= right_bound))[0]
            if len(valid_indices) == 0:
                # No pixels in ROI, fallback to center ROI ignoring lane_xs_to_use
                lane_xs_in_roi = lane_xs_to_use
            else:
                lane_xs_in_roi = lane_xs_to_use[valid_indices]

            lane_x = int(np.mean(lane_xs_in_roi))

            # Smooth sudden jumps
            if self.prev_lane_x is not None and abs(lane_x - self.prev_lane_x) > self.max_jump:
                lane_x = self.prev_lane_x
            self.prev_lane_x = lane_x

            lane_x_scaled = int(lane_x * (orig_width / 512))
            center_x = orig_width // 2
            error = lane_x_scaled - center_x

            gain = 0.005
            steering = np.clip(-gain * error, -30, 30)

            self.steering_history.append(steering)
            if len(self.steering_history) > self.history_size:
                self.steering_history.pop(0)
            steering = np.mean(self.steering_history)

            # Draw center line and lane center as before...
            cv2.line(annotated_image, (center_x, 0), (center_x, orig_height), (0, 255, 0), 2)
            cv2.circle(annotated_image, (lane_x_scaled, orig_height - 30), 10, (0, 0, 255), -1)

            binary_mask_color = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
            binary_mask_resized = cv2.resize(binary_mask_color, (orig_width, orig_height - self.row_upper_bound), interpolation=cv2.INTER_NEAREST)
            overlay = annotated_image[self.row_upper_bound:, :, :].copy()
            cv2.addWeighted(binary_mask_resized, 0.4, overlay, 0.6, 0, overlay)
            annotated_image[self.row_upper_bound:, :, :] = overlay

            cv2.putText(annotated_image, f"Error: {error:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_image, f"Steering: {steering:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return steering, error, lane_x, annotated_image
