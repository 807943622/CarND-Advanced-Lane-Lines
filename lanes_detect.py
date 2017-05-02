import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
import os.path

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

def process_image():
    pass


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