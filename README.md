## Advanced Lane Finding
![CarND-Advanced-Lane-Lines - SDC](https://youtu.be/Wcl3YkgNxVg)

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Go directly to the [writeup report](https://github.com/sorny92/CarND-Advanced-Lane-Lines/blob/master/writeup.md)

In this project, thegoal is to write a software pipeline to identify the lane boundaries in a video.
Creating a great writeup:


The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Description of files
---

* `dist.pkl, mtx.pkl, ret.pkl, rvecs.pkl, tvecs.pkl` Those are pickle files with the parameters to calibrate the camera we used in the video.
* `lanes_detect.py` It is the files used to detect the lines. Inside of it you will find methods to use the pipeline with individual files or videos.
* `*.mp4` Those are some example videos you can use to test the pipeline.
