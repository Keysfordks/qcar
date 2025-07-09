# lanenet_utils.py
import numpy as np
import cv2
import torch
from pit.LaneNet.nets import LaneNet

class LaneNetUtils:
    """
    Encapsulates LaneNet model and associated lane-finding utilities.
    """
    def __init__(self, lanenet_processing_width=640, lanenet_processing_height=480):
        self.LANENET_PROCESSING_WIDTH = lanenet_processing_width
        self.LANENET_PROCESSING_HEIGHT = lanenet_processing_height
        self.lanenet = LaneNet(imageHeight=self.LANENET_PROCESSING_HEIGHT, imageWidth=self.LANENET_PROCESSING_WIDTH)
        self.imgTensor = None # To store the pre-processed image tensor for visualization

    def find_lane_pixels(self, binary_warped, nwindows=10, margin=25, minpix=250):
        """
        Finds lane pixels using a sliding window approach on a binary warped image.
        Adapted from common lane-finding tutorials (e.g., Udacity Self-Driving Car Nanodegree).
        """
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        midpoint = np.int64(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = np.int64(binary_warped.shape[0]//nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = np.int64(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int64(np.mean(nonzerox[good_right_inds]))

        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            pass

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds

    def compute_steering(self, binary_mask, current_speed):
        """
        Computes the steering angle based on the binary lane mask and current speed.
        """
        height, width = binary_mask.shape
        midline = width // 2

        leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds = \
            self.find_lane_pixels(binary_mask)

        left_fit = None
        right_fit = None

        min_poly_points = 100
        if len(leftx) > min_poly_points and len(lefty) > min_poly_points:
            left_fit = np.polyfit(lefty, leftx, 1)
        if len(rightx) > min_poly_points and len(righty) > min_poly_points:
            right_fit = np.polyfit(righty, rightx, 1)

        MIN_LOOK_AHEAD_Y_RATIO = 0.6
        MAX_LOOK_AHEAD_Y_RATIO = 0.9
        MAX_FORWARD_SPEED = 3.0

        clipped_speed = np.clip(current_speed, 0, MAX_FORWARD_SPEED)
        look_ahead_y_ratio = MIN_LOOK_AHEAD_Y_RATIO + \
                             (MAX_LOOK_AHEAD_Y_RATIO - MIN_LOOK_AHEAD_Y_RATIO) * \
                             (clipped_speed / MAX_FORWARD_SPEED)
        look_ahead_y_pixel = int(height * look_ahead_y_ratio)
        look_ahead_y_pixel = np.clip(look_ahead_y_pixel, 0, height - 1)

        lane_center = float(midline)
        error = 0.0

        ESTIMATED_HALF_ROAD_WIDTH_PX = 175

        if left_fit is not None and right_fit is not None:
            left_x_at_y = left_fit[0]*look_ahead_y_pixel + left_fit[1]
            right_x_at_y = right_fit[0]*look_ahead_y_pixel + right_fit[1]
            lane_center = (left_x_at_y + right_x_at_y) / 2
            error = lane_center - midline
        elif left_fit is not None:
            left_x_at_y = left_fit[0]*look_ahead_y_pixel + left_fit[1]
            lane_center = left_x_at_y + ESTIMATED_HALF_ROAD_WIDTH_PX
            error = lane_center - midline
            print(f"DEBUG: Only left lane. Estimated Center: {lane_center:.2f}")
        elif right_fit is not None:
            right_x_at_y = right_fit[0]*look_ahead_y_pixel + right_fit[1]
            lane_center = right_x_at_y - ESTIMATED_HALF_ROAD_WIDTH_PX
            error = lane_center - midline
            print(f"DEBUG: Only right lane. Estimated Center: {lane_center:.2f}")
        else:
            print("Warning: No polynomial lanes detected. Defaulting to midline (no steering).")

        k_p = 2
        steering = k_p * error
        return np.clip(steering, -30, 30), error, lane_center, look_ahead_y_pixel, left_fit, right_fit, left_lane_inds, right_lane_inds

    def process_image_for_lanenet(self, image_rgb):
        """
        Resizes the input image and processes it through LaneNet.
        Stores the pre-processed image tensor for visualization.
        """
        image_resized = cv2.resize(image_rgb, (self.LANENET_PROCESSING_WIDTH, self.LANENET_PROCESSING_HEIGHT))
        input_tensor = self.lanenet.pre_process(image_resized).unsqueeze(0)
        self.imgTensor = self.lanenet.pre_process(image_resized) # Store for visualization
        final_binary_mask, _ = self.lanenet.predict(input_tensor)
        return final_binary_mask