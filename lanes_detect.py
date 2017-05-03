import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
import os.path

import skvideo.io

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

def process_image(image, thresh=(85,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    #H = hls[:,:,0]
    #L = hls[:,:,1]
    S = hls[:,:,2]
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    res = binary * 255
    return res

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
def process_video(video):
    vid = skvideo.io.vreader(video)
    writer = skvideo.io.FFmpegWriter('video_output.mp4', verbosity=1)
    for frame in vid:
        img = process_image(frame)
        img = np.expand_dims(img, axis=-1)
#        plt.imshow(img)
#        plt.pause(0.01)
        writer.writeFrame(img)
    writer.close()
 
images = glob.glob('./test_images/*.jpg')

for f in images:
    img = cv2.imread(f)
    #processed = process_image(img)
    print(img.shape)
    left_down_corner = 0.05*img.shape[0]
    rigth_down_corner = 0.95*img.shape[0]
    horizont = 0.6*img.shape[1]
    horizont_margin = 0.07*img.shape[0]
    middle = img.shape[0]/2
    vertices = np.array([[(left_down_corner,img.shape[1]), 
              (middle-horizont_margin, horizont), 
              (middle + horizont_margin, horizont), 
              (rigth_down_corner, img.shape[1]) ]], dtype=np.int32)
    
    src_points = np.float32([[200,720], [625,430], [660,430], [1110,720]])
    dst_points = np.float32([[400,720], [400,0], [910,0], [910,720]])
    cv2.line(img, (200,720), (625,430), [255,0,0], 2)
    cv2.line(img, (1110,720), (660,430), [0,255,0], 2)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    fig = plt.figure()
    a = fig.add_subplot(1,2,1)
    imgplot = plt.imshow(img)
    a = fig.add_subplot(1,2,2)
    imgplot = plt.imshow(warped)
    plt.show()
    #plt.pause(10)