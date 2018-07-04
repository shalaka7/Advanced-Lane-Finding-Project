import numpy as np
import cv2
import pickle
import glob
#tracker fuction
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

#read in the saved objpoints and imgpoints
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

# Define a function to return the magnitude of the gradient for a given sobel kernel size and threshold values
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

# Define a function to threshold an image for a given range and Sobel kernel and calculate gradient direction
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(grad_dir)
    #apply threshold
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    return binary_output

#Define a function that color thresholds
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

# Define window mask
def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width)):min(int(center * width), img_ref.shape[1])] = 1
    return output


# make the list of test images

images = glob.glob('./test_images/test*.jpg')

for idx, fname in enumerate(images):
    # read in image
    img = cv2.imread(fname)
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
    src = np.float32([[490, 482],[810, 482],[1250, 720],[40, 720]])
    offset = img_size[0] * .25
    dst = np.float32([[0, 0], [1280, 0],[1250, 720],[40, 720]])

    # perform the transform
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)

    window_width = 25
    window_height = 80

    # set up the overall class to do all the tracking
    curve_centers = tracker(Mywindow_width=window_width, Mywindow_height=window_height, Mymargin=25, My_ym=10 / 720,
                            My_xm=4 / 384, Mysmooth_factor=15)

    window_centroids = curve_centers.find_window_centroids(warped)

    # points used to draw all left and right window
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # points used to find the left and right lane
    rightx = []
    leftx = []

    # go through each level and draw the window

    for level in range(0, len(window_centroids)):
        l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)

        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])

        l_points[(l_points == 255) | ((l_mask == 1))] = 255
        r_points[(r_points == 255) | ((r_mask == 1))] = 255

    # draw the results
    template = np.array(r_points + l_points, np.uint8)
    zero_channel = np.zeros_like(template)
    template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)
    warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8)
    result = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)

    # fit the lane boudries to the left,right,center position found
    yvals = range(0, warped.shape[0])

    res_yvals = np.arange(warped.shape[0] - (window_height / 2), 0, -window_height)

    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0] * yvals * yvals + left_fit[1] * yvals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)

    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0] * yvals * yvals + right_fit[1] * yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)

    left_lane = np.array(list(
        zip(np.concatenate((left_fitx - window_width / 2, left_fitx[::-1] + window_width / 2), axis=0),
            np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    right_lane = np.array(list(
        zip(np.concatenate((right_fitx - window_width / 2, right_fitx[::-1] + window_width / 2), axis=0),
            np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    middle_lane = np.array(list(
        zip(np.concatenate((right_fitx - window_width / 2, right_fitx[::-1] + window_width / 2), axis=0),
            np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

    road = np.zeros_like(img)
    road_bkg = np.zeros_like(img)
    cv2.fillPoly(road, [left_lane], color=[255, 0, 0])
    cv2.fillPoly(road, [right_lane], color=[0, 0, 255])
    cv2.fillPoly(road_bkg, [left_lane], color=[255, 255, 255])
    cv2.fillPoly(road_bkg, [right_lane], color=[255, 255, 255])

    road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
    road_warped_bkg = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)

    base = cv2.addWeighted(img, 1.0, road_warped_bkg, 1.0, 0.0)
    result = cv2.addWeighted(base, 1.0, road_warped, 1.0, 0.0)

    ym_per_pix = curve_centers.ym_per_pix
    xm_per_pix = curve_centers.xm_per_pix

    curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32) * ym_per_pix, np.array(leftx, np.float32) * xm_per_pix, 2)
    curverad = ((1 + (2 * curve_fit_cr[0] * yvals[-1] * ym_per_pix + curve_fit_cr[1] ** 2) ** 1.5) / np.absolute(2 * curve_fit_cr[0]))

    # calculate the offset of car on road

    camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
    center_diff = (camera_center - warped.shape[1] / 2) * xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    # draw the text shoowing curvature ,offset ,and speed

    cv2.putText(result, 'Radius of curvature = ' + str(round(curverad, 3)) + '(m)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    cv2.putText(result, 'vehicle is ' + str(abs(round(center_diff, 3))) + 'm' + side_pos + ' of center', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 2555, 255), 2)

    write_name = './output_images/tracked' + str(idx) + '.jpg'
    cv2.imwrite(write_name, result)
