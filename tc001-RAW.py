#!/usr/bin/env python3


import cv2
import numpy as np
import argparse

from scipy import stats 

alpha = 1.0 # Contrast control (1.0-3.0)
colormap = 0
rad = 0
MouseX =0
MouseY = 0
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="Video Device number e.g. 0, use v4l2-ctl --list-devices")
args = parser.parse_args()
	
if args.device:
	dev = args.device
else:
	dev = 1
	

#init video
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()
#cap = cv2.VideoCapture(0)

#we need to set the resolution here why?
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
'''
[0]: 'YUYV' (YUYV 4:2:2)
		Size: Discrete 256x392 - two images, band at the bottom * (green at the top, blend at the bottom)
		Size: Discrete 256x192 - only thermal (gray)
		Size: Discrete 256x196 - only green with band at the bottom
		Size: Discrete 256x400 - two images, band at the botton and side repeated
		
		Size: Discrete 256x200 - one gray image with bands at bottomtop and side
		Size: Discrete 192x520 - two images in landscape mode, band at the bottom
		Size: Discrete 192x400   -- two images, garbled.

'''

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,392)

def draw_circle(event,x,y,flags,param):
	global MouseX, MouseY
	if event == cv2.EVENT_LBUTTONDBLCLK:
		print('x = %d, y = %d'%(x, y))
		MouseX = x
		MouseY = y

cv2.namedWindow('Heatmap',cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback('Heatmap', draw_circle)

# cv2.namedWindow('th1',cv2.WINDOW_GUI_NORMAL)
# cv2.namedWindow('th0',cv2.WINDOW_GUI_NORMAL)
font=cv2.FONT_HERSHEY_SIMPLEX

def printstats(arr):
	return f"min {np.min(arr)} max {np.max(arr)}, shape {arr.shape} Mode: {stats.mode(arr, axis=None)}"



while(cap.isOpened()):
	# Capture frame-by-frame
	ret, frame = cap.read()

	if ret == True:
		# Captured image contains actual raw sensor data and false color image. Split the two.
		t, i = np.array_split(frame, 2)
		# Clear invalid rows with zeros in the upper byte of sensor data.
		invalid_row = np.where(t[:,:,1]==0)[0].min()
		thermal = np.delete(t, range(invalid_row, t.shape[0], 1), axis=0)
		imgdata = np.delete(i, range(invalid_row, t.shape[0], 1), axis=0)
		
		# Compute temperature from the sensor data.
		temp = np.round((256 * (thermal[..., 1] - 17).astype(np.int32) + thermal[..., 0])/25).astype(np.int8)
		
		# Find locations with min max temperature. 
		# argmin/argmax return index in flattened array, so we need to divmod to extract actual indeces.
		minpos = divmod(temp.argmin(axis=None), temp.shape[1])
		maxpos = divmod(temp.argmax(axis=None), temp.shape[1])
		# print(minpos, thermal[minpos])
		# print(np.where(thermal[:,:,1]==0)[0].min())
		# print(f"min {temp[minpos[0]][minpos[1]]} max {temp[maxpos[0]][maxpos[1]]}")
		print(f"Selected:{temp[MouseY][MouseX]} ({MouseY, MouseX}) Min:{temp[minpos[0]][minpos[1]]} ({minpos}) Max:{temp[maxpos[0]][maxpos[1]]} ({maxpos})", end='\r', flush=True)
	
		
		
		# Convert the real image to RGB
		bgr = cv2.cvtColor(imgdata,  cv2.COLOR_YUV2BGR_YUYV)
		#Contrast
		bgr = cv2.convertScaleAbs(bgr, alpha=alpha)#Contrast
		#bicubic interpolate, upscale and blur
		# newWidth = ,newHeight
		# bgr = cv2.resize(bgr,(newWidth,newHeight),interpolation=cv2.INTER_CUBIC)#Scale up!
		if rad>0:
			bgr = cv2.blur(bgr,(rad,rad))
		
		heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_JET)
		# Draw circles on the Selected/Min/Max locations
		cv2.circle(heatmap,(MouseX,MouseY),2,(0,0,0),-1)
		cv2.circle(heatmap, (maxpos[1], maxpos[0]), 4, (0,0,0), 1)
		cv2.circle(heatmap, (minpos[1], minpos[0]), 4, (255,255,255), 1)

		cv2.imshow('Heatmap',heatmap)
		keyPress = cv2.waitKey(3)
		if keyPress == ord('q'):
			cap.release()
			cv2.destroyAllWindows()
			break
			

print("Exit")
