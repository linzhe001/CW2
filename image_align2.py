#Import packages
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
from Tkinter import Tk
from tkSimpleDialog import askinteger
from tkSimpleDialog import askfloat
from tkFileDialog import asksaveasfilename

Tk().withdraw()

#ALIGN_IMAGES
def align_images(im1, im2):

	# Convert images to grayscale
	im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
	im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
 
	# Find size of image1
	sz = im1.shape

	# Define the motion model
	warp_mode = cv2.MOTION_HOMOGRAPHY
	
	#Define the warp matrix
	warp_matrix = np.eye(3, 3, dtype=np.float32)

	#Define the number of iterations
	number_of_iterations = askinteger("Iterations", "Enter a number between 5 andd 5000",initialvalue=500,minvalue=5,maxvalue=5000)

	#Define correllation coefficient threshold
	#Specify the threshold of the increment in the correlation coefficient between two iterations
	termination_eps = askfloat("Threshold", "Enter a number between 1e-10 and 1e-50",initialvalue=1e-10,minvalue=1e-50,maxvalue=1e-10)

	#Define termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
 
	#Run the ECC algorithm. The results are stored in warp_matrix.
	(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)

	#Use warpPerspective for Homography 
	im_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
	
	save1 = asksaveasfilename(defaultextension=".jpg", title="Save aligned image")
	cv2.imwrite(save1,im_aligned)