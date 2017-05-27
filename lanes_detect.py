import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
import os.path

import skvideo.io
slope = -0.65
height = 450
base = 677

def calibrate_camera():
    images = glob.glob('./camera_cal/calibration*.jpg')
    ret, mtx, dist, rvecs, tvecs = 0, 0, 0, 0, 0
    imgpoints = []
    objpoints = []
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    first_image = True
    shape = None
    for image_name in images:
        img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     
        shape = gray.shape[::-1]
        rest, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if rest == True:
            imgpoints.append(corners)
            objpoints.append(objp)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
    pickle.dump(ret, open('ret.pkl', 'wb'))
    pickle.dump(mtx, open('mtx.pkl', 'wb'))
    pickle.dump(dist, open('dist.pkl', 'wb'))
    pickle.dump(rvecs, open('rvecs.pkl', 'wb'))
    pickle.dump(tvecs, open('tvecs.pkl', 'wb'))
    return ret, mtx, dist, rvecs, tvecs

def test_calibration_camera():
    images = glob.glob('./camera_cal/calibration*.jpg')
    for image_name in images:
        img = cv2.imread(image_name)
        undistorted = cv2.undistort(img, mtx, dist, None, mtx)
        f, (ax1, ax2) = plt.subplots(1,2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(undistorted)
        ax2.set_title('Undistorted Image', fontsize=30)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
        plt.pause(0.5)

def threshold_image(img, thresh_x=(20,100), thresh_color=(85,255)):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_x[0]) & (scaled_sobel <= thresh_x[1])] = 255

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh_color[0]) & (s_channel <= thresh_color[1])] = 255

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 255) | (sxbinary == 255)] = 255

    return combined_binary
    # Plotting thresholded images
    #return color_binary
    '''print(color_binary.shape)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.set_title('Stacked thresholds')
    ax1.imshow(color_binary)

    ax2.set_title('Combined S channel and gradient thresholds')
    ax2.imshow(combined_binary, cmap='gray')
    plt.show()'''


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def persperctive_transform(img):
    middle = img.shape[1]/2
    bottom = img.shape[0]
    space_lines = 250
    src_points = np.float32([[250,base], 
                             [int(((height-base)/slope)+244), height], 
                             [int(((height-base)/(slope*-1))+1036),height], 
                             [1030,base]])
    dst_points = np.float32([[middle-space_lines,bottom], 
                             [middle-space_lines,0], 
                             [middle+space_lines,0], 
                             [middle+space_lines,bottom]])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped

def find_lanes(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
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

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
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

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    curvature = get_lines_curvature(leftx, lefty, rightx, righty, ploty)
    return left_fit, right_fit, nonzeroy, nonzerox, left_lane_inds, right_lane_inds, curvature, out_img

def find_lines_margin(binary_warped, left_fit, right_fit):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    curvature = get_lines_curvature(leftx, lefty, rightx, righty, ploty)
    return left_fit, right_fit, nonzeroy, nonzerox, left_lane_inds, right_lane_inds, curvature, out_img

def get_lines_curvature(leftx, lefty, rightx, righty, ploty):
    y_eval = np.max(ploty)
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    return (left_curverad, right_curverad)

def pipeline_process(img, left_fit=None, right_fit=None):
    left_down_corner = 0.28*img.shape[1]
    rigth_down_corner = 0.72*img.shape[1]
    horizont = 0*img.shape[0]
    horizont_margin = 0.35*img.shape[1]
    middle = img.shape[1]/2
    vertices = np.array([[(left_down_corner,img.shape[0]), 
                          (middle-horizont_margin, horizont), 
                          (middle + horizont_margin, horizont), 
                          (rigth_down_corner, img.shape[0]) 
                         ]], dtype=np.int32)
    warped = persperctive_transform(img)
    thresholded_image = threshold_image(warped)
    out_img = thresholded_image
    if(left_fit == None or right_fit == None):
        thresholded_image = region_of_interest(thresholded_image, vertices)
        left_fit, right_fit, nonzeroy, nonzerox, left_lane_inds, right_lane_inds, curvature, out_img = find_lanes(thresholded_image)
    else:
        left_fit, right_fit, nonzeroy, nonzerox, left_lane_inds, right_lane_inds, curvature, out_img = find_lines_margin(thresholded_image, left_fit, right_fit)
    
    ploty = np.linspace(0, thresholded_image.shape[0]-1, thresholded_image.shape[0] )
    #print(curvature)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    warped = cv2.bitwise_or(warped, out_img)
    warped = cv2.putText(warped,'CURVATURE: {0:.1f}m, {0:.1f}m'.format(curvature[0], curvature[1]), 
                         (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, thickness = 2)
    warped = cv2.putText(warped,'LEFTpoly: {0:.6f}, {0:.6f}, {0:.6f}'.format(left_fit[0], left_fit[1], left_fit[2]), 
                         (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, thickness = 2)
    warped = cv2.putText(warped,'RIGHTpoly: {0:.6f}, {0:.6f}, {0:.6f}'.format(right_fit[0], right_fit[1], right_fit[2]), 
                         (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, thickness = 2)
    #img = np.expand_dims(img, axis=-1)
    return warped, out_img, left_fit, right_fit

def process_video(video):
    vid = skvideo.io.vreader(video)
    writer = skvideo.io.FFmpegWriter('{}_out.mp4'.format(video[:-4]), verbosity=1)
    left_fit, right_fit = None, None
    for frame in vid:
        output, lines, left_fit, right_fit = pipeline_process(frame, left_fit, right_fit)
#        plt.imshow(img)
#        plt.pause(0.01)
        writer.writeFrame(output)
    writer.close()

def test_images():
    images = glob.glob('./test_images/test*')
    for f in images:
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img.shape)
        fig = plt.figure()
        img2 = np.copy(img)
        middle = img.shape[1]/2
        bottom = img.shape[0]
        src_points = [(250, base), (int(((height-base)/slope)+250), height), 
                      (int(((height-base)/(slope*-1))+1052), height), (1052, base)]
        cv2.line(img, src_points[0], src_points[1], [255,0,0], 2)
        cv2.line(img, src_points[2], src_points[3], [0,255,0], 2)

        warped, out_img, left_fit, right_fit = pipeline_process(img2)

        a = fig.add_subplot(2,2,1)
        imgplot = plt.imshow(img)
        a = fig.add_subplot(2,2,2)
        imgplot = plt.imshow(out_img)
        a = fig.add_subplot(2,2,3)
        imgplot = plt.imshow(warped)
        plt.show()
        #plt.pause(10)
  

if(os.path.exists('ret.pkl') and os.path.exists('mtx.pkl') and 
    os.path.exists('dist.pkl') and os.path.exists('rvecs.pkl') and os.path.exists('tvecs.pkl')):
    print('Loading calibration camera data')
    ret = pickle.load(open('ret.pkl', 'rb'))
    mtx = pickle.load(open('mtx.pkl', 'rb'))
    dist = pickle.load(open('dist.pkl', 'rb'))
    rvecs = pickle.load(open('rvecs.pkl', 'rb'))
    tvecs = pickle.load(open('tvecs.pkl', 'rb'))
    print('Data loaded')
else:
    print('Calibrating camera...')
    ret, mtx, dist, rvecs, tvecs = calibrate_camera()
    print('Camera calibrated')

#test_calibration_camera()
#test_images()
process_video('harder_challenge_video.mp4')

