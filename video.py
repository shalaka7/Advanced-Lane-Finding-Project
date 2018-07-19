from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
import pickle
import glob


class tracker():
    def __init__(self, Mywindow_width, Mywindow_height, Mymargin, My_ym=1, My_xm=1, Mysmooth_factor=15):
        self.recent_centers = []

        self.window_width = Mywindow_width

        self.window_height = Mywindow_height

        self.margin = Mymargin

        self.ym_per_pix = My_ym

        self.xm_per_pix = My_xm

        self.smooth_factor = Mysmooth_factor

    # the main tracking function for finding and storing lane segment position
    def find_window_centroids(self, warped):
        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin

        window_centroids = []
        window = np.ones(window_width)

        # sum quarter bottom of image to get slice

        l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
        r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)

        # add what we found in first layer

        window_centroids.append((l_center, r_center))

        # Go through each layer looking for max pixel location

        for level in range(1, (int)(warped.shape[0] / window_height)):
            image_layer = np.sum(
                warped[int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height),
                :], axis=0)

            conv_signal = np.convolve(window, image_layer)

            offset = window_width / 2

            l_min_index = int(max(l_center + offset - margin, 0))
            l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset

            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset

            window_centroids.append((l_center, r_center))

        self.recent_centers.append(window_centroids)

        return np.average(self.recent_centers[-self.smooth_factor:], axis=0)

    # read in the saved objpoints and imgpoints

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

    def add_fit(self, fit):
        if fit is not None:
            self.current_fit.append(fit)

            if len(self.current_fit) > 12:
                self.current_fit = self.current_fit[1:]

            self.best_fit = np.average(self.current_fit, axis=0)
            self.detected = True

        else:
            self.current_fit = self.current_fit[1:]
            self.detected = False
        if len(self.current_fit) < 12:
            print(len(self.current_fit))

    def update_line(self, fit):
        self.add_fit(fit)



left_line = Line()
right_line = Line()
running_diff = None

dist_pickle = pickle.load(open("./wide_dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


# apply sobel function to identify pixels where the gradient of an image falls within a specified threshold range.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    # apply threshold
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    with np.errstate(divide='ignore', invalid='ignore'):
        absgraddir = np.absolute(np.arctan(sobely / sobelx))
        binary_output = np.zero_like(absgraddir)
        # Apply threshold
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output


def color_threshold(image, sthresh=(0, 255), vthresh=(0, 255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hls[:, :, 2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1
    return output


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width)):min(int(center * width), img_ref.shape[1])] = 1
    return output


def process_image(img):

    ##########################################################
    # PREPARE IMAGE FOR LINE DETECTION
    ##########################################################

    # undistort the image
    img = cv2.undistort(img, mtx, dist, None, mtx)

    # preprocess image and genrate binary pixel of intrest
    preprocessImage = np.zeros_like(img[:, :, 0])
    gradx = abs_sobel_thresh(img, orient='x', thresh_max=255, thresh_min=12)
    grady = abs_sobel_thresh(img, orient='y', thresh_max=255, thresh_min=25)
    c_binary = color_threshold(img, sthresh=(100, 255), vthresh=(50, 255))

    preprocessImage[((gradx == 1) & (grady == 1) | (c_binary == 1))] = 255

    # work on defining perspective transformation area
    img_size = (img.shape[1], img.shape[0])
    bot_width = .76
    mid_width = .08
    height_pct = .62
    bottom_trim = .935
    src = np.float32([[490, 482], [810, 482], [1250, 720], [40, 720]])

    offset = img_size[0] * .25
    dst = np.float32([[0, 0], [1280, 0], [1250, 720], [40, 720]])

    # perform the transform
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)

    ##########################################################
    # FIND LANES USING HISTOGRAM
    ##########################################################

    histogram = np.sum(warped[warped.shape[0] // 4 : warped.shape[0] * 3 // 4, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((warped, warped, warped)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(warped.shape[0] // nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    left_fit = None
    right_fit = None

    global running_diff

    if not left_line.detected or not right_line.detected:
        print("IN SLIDING WINDOW.....")
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped.shape[0] - (window + 1) * window_height
            win_y_high = warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    else:
        left_lane_inds = ((nonzerox > (left_line.best_fit[0] * (nonzeroy ** 2) + left_line.best_fit[1] * nonzeroy +
                                       left_line.best_fit[2] - margin)) & (nonzerox < (left_line.best_fit[0] * (nonzeroy ** 2) +
                                                                                       left_line.best_fit[1] * nonzeroy +
                                                                                       left_line.best_fit[2] + margin)))

        right_lane_inds = ((nonzerox > (right_line.best_fit[0] * (nonzeroy ** 2) + right_line.best_fit[1] * nonzeroy +
                                        right_line.best_fit[2] - margin)) & (nonzerox < (right_line.best_fit[0] * (nonzeroy ** 2) +
                                                                                         right_line.best_fit[1] * nonzeroy +
                                                                                         right_line.best_fit[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Find the diff for deviation
    diff = np.mean(right_fitx - left_fitx)
    # print(diff)

    if running_diff is None:
        # left_line.update_line(left_fit)
        # right_line.update_line(right_fit)
        running_diff = diff
    # Check if running diff id deviated a lot
    elif diff < (0.85 * running_diff) or diff > (1.15 * running_diff):
        left_line.update_line(None)
        right_line.update_line(None)
    else:
        left_line.update_line(left_fit)
        right_line.update_line(right_fit)
        running_diff = diff

    if left_line.best_fit is not None and right_line.best_fit is not None:

        ##########################################################
        # VISUALIZATION OF LANES
        ##########################################################

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((warped, warped, warped)) * 255
        window_img = np.zeros_like(out_img)

        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        ##########################################################
        # MEASURING CURVATURE
        ##########################################################

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 3.048/100 # 30 / 720  #meters per pixel in y dimension. Using dashed lines instead
        xm_per_pix = 3.7/378 # 3.7 / 700  # meters per pixel in x dimension. Actual width of lane is not 700 pixels

        # y_eval = np.max(ploty)

        left_y_eval = np.max(lefty)
        right_y_eval = np.max(righty)


        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx* xm_per_pix, 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * left_y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) /\
                        np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * right_y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / \
                         np.absolute(2 * right_fit_cr[0])

        avg_curverad = round((left_curverad +right_curverad) // 2)

        # Using intercepts from best fit to determine midpoint and find the
        # deviation from image centre
        lane_centre = warped.shape[1] / 2
        l_fit_x_int = left_line.best_fit[0] * ploty ** 2 \
                      + left_line.best_fit[1] * ploty \
                      + left_line.best_fit[2]
        r_fit_x_int = right_line.best_fit[0] * ploty ** 2 \
                      + right_line.best_fit[1] * ploty \
                      + right_line.best_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
        centre_deviation = (lane_centre - lane_center_position) * xm_per_pix
        centre_deviation = np.average(centre_deviation)

        left_or_right = None
        if centre_deviation > 0:
            left_or_right = " to right of centre"
        elif centre_deviation < 0:
            left_or_right = " to left of centre"
        else:
            left_or_right = "at centre"

        ##########################################################
        # INVERSE PERSPECTIVE TRANSFORM
        ##########################################################

        left_line_window = np.array(np.transpose(np.vstack([left_fitx, ploty])))

        right_line_window = np.array(np.flipud(np.transpose(np.vstack([right_fitx, ploty]))))

        line_points = np.vstack((left_line_window, right_line_window))

        cv2.fillPoly(out_img, np.int_([line_points]), [0, 255, 0])

        unwarped = cv2.warpPerspective(out_img, Minv, img_size, flags=cv2.INTER_LINEAR)

        result = cv2.addWeighted(img, 1, unwarped, 0.3, 0)

        cv2.putText(result, 'vehicle is ' + str(abs(round(centre_deviation, 3))) + 'm' + left_or_right, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 2555, 255), 2)

        cv2.putText(result, 'Radius of curvature = ' + str(avg_curverad) + '(m)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

        return result

    else:
        return img



output_video = 'project_video_final_2.mp4'
input_video = 'project_video.mp4'

clip1 = VideoFileClip(input_video)
video_clip = clip1.fl_image(process_image)
video_clip.write_videofile(output_video, audio=False)

