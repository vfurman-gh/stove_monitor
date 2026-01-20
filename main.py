import cv2
import numpy as np
from mouse_drag_handler import MouseDragHandler

import os
import json

# Global storage for the caller to use
SAVE_FILE = "thermal_regions.json"
regions = {}
current_key = None # Tracks last pressed key globally

def save_regions():
    """Saves the geometry to a JSON file, converting NumPy types to native Python types."""
    serializable_data = {}
    for slot, data in regions.items():
        # Use .item() or int() to convert NumPy int64 to native Python int
        serializable_data[slot] = {
            'center': [int(data['center'][0]), int(data['center'][1])],
            'radius': int(data['radius'])
        }
    with open(SAVE_FILE, 'w') as f:
        json.dump(serializable_data, f, indent=4)
    print("Regions saved to disk.")

def load_regions():
    """Restores regions from JSON and regenerates masks."""
    global regions, thermal_img
    if not os.path.exists(SAVE_FILE):
        return
    
    try:
        with open(SAVE_FILE, 'r') as f:
            saved_data = json.load(f)
            for slot, data in saved_data.items():
                # We store coordinates. Mask generation requires last_thermal_img shape.
                # If image isn't loaded yet, we store as 'pending'
                regions[slot] = {
                    'center': tuple(data['center']),
                    'radius': data['radius'],
                    'mask': None, # To be generated when first frame arrives
                    'temp': 0.0
                }
        print("Regions loaded from disk.")
    except Exception as e:
        print(f"Error loading regions: {e}")

def get_slot():
    """ 
    Determine slot: if '1'-'4' was pressed, use it; otherwise default to '1'
    """
    return current_key if (current_key is not None and current_key in ['1', '2', '3', '4']) else '1'

def update_region_mask(slot_data, last_thermal_img):
    """
    Unified function to generate a mask.
    Call this whenever center/radius changes or image shape changes.
    """
    h, w = last_thermal_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, slot_data['center'], slot_data['radius'], 255, -1)
    return mask

def mouse_drag_complete(start, end):
    """Callback to update or create a regional mask in a specific slot."""
    global regions, current_key
   
    slot = get_slot()
    
    radius = int(np.linalg.norm(end - start))
    if radius > 5:
        regions[slot] = {
            'mask': None,  # will be created in the unified logic
            'center': tuple(start),
            'radius': radius,
            'temp': 0.0
        }
        print(f"Slot {slot} updated")
        save_regions()


def average_temperature_in_region(image_data, mask_data):
    """
    Computes average based on YUYV thermal formula:
    (256 * (U - 17) + Y) / 25
    """
    # Safety check: ensure mask matches current image shape
    if image_data.shape[:2] != mask_data.shape:
        print("Mismatched mask size")
        return 0.0

    pixels = image_data[mask_data == 255]
    if pixels.size == 0: return 0.0
    
    # Channel 0 = Y, Channel 1 = U
    Y = pixels[:, 0].astype(np.int32)
    U = pixels[:, 1].astype(np.int32)
    #teflon
    results = (256 * U + Y) / 10.0 - 175  
    #stainless: /9 - 150
    results = (256 * U + Y) / 9.8 - 160  
    #low range
    #results = (256 * (U - 17) + Y) / 25.0
    return np.mean(results)   

def cleanup_frame(frame):
    """
    Split frame into two images: one has thermal data in UY channels, another is a pseudo-color image. 
    Remove rows which has invalid thermal data.
    """
    t, i = np.array_split(frame, 2)
	# Clear invalid rows with zeros in the upper byte of sensor data.
    invalid_row = np.where(t[:,:,1]==0)[0].min()
    thermal = np.delete(t, range(invalid_row, t.shape[0], 1), axis=0)
    imgdata = np.delete(i, range(invalid_row, t.shape[0], 1), axis=0)	
		
	# Convert the real image to RGB
    bgr = cv2.cvtColor(imgdata,  cv2.COLOR_YUV2BGR_YUYV)
	#Contrast
    bgr = cv2.convertScaleAbs(bgr, alpha=1.5)#Contrast
	#bicubic interpolate, upscale and blur
	# newWidth = ,newHeight
	# bgr = cv2.resize(bgr,(newWidth,newHeight),interpolation=cv2.INTER_CUBIC)#Scale up!
    # if rad>0:
    #     bgr = cv2.blur(bgr,(rad,rad))
		
    heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_JET)
    return (thermal, heatmap)

def draw_centered_text(img, text, center, font=cv2.FONT_HERSHEY_SIMPLEX, 
                       scale=0.5, color=(255, 255, 255), thickness=2):
    """
    Draws text on an image so that the geometric center of the text 
    is aligned with the provided 'center' coordinate.
    """
    # 1. Get the size of the text box
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)

    # 2. Calculate the bottom-left origin for putText
    # Subtract half width to center horizontally
    # Add half height to center vertically (OpenCV Y-axis goes down)
    origin_x = int(center[0] - text_width / 2)
    origin_y = int(center[1] + text_height / 2)

    # 3. Render the text
    cv2.putText(img, text, (origin_x, origin_y), font, scale, color, thickness, cv2.LINE_AA)
#init video
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

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

load_regions()
# Initialize mouse drag Handler with the callback
handler = MouseDragHandler(on_drag_complete_callback=mouse_drag_complete)

cv2.namedWindow("Main App", cv2.WINDOW_NORMAL)

# Optional: Set an initial size (Width, Height)
cv2.resizeWindow("Main App", 1280, 720)
cv2.setMouseCallback("Main App", handler.handle_mouse)

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        print("Cannot read frame")
        break
	
    thermal_img, imdata = cleanup_frame(frame)
    
    # Capture input for slots and clearing
    key_code = cv2.waitKey(1) & 0xFF
    if key_code != 255:
        current_key = chr(key_code)
        if current_key == 'c':
            regions.clear()
            save_regions()
            print("All regions cleared.")
        if current_key == 'q': break

    # Update and draw all active regions
    for slot, data in regions.items():
            # Real-time re-computation from the latest thermal frame
        if data['mask'] is None or data['mask'].shape != thermal_img.shape[:2]:
            data['mask'] = update_region_mask(data, thermal_img)

        data['temp'] = average_temperature_in_region(thermal_img, data['mask'])
            
            # Rendering
        cv2.circle(imdata, data['center'], data['radius'], (0, 255, 0), 2)
        label = f"{data['temp']:.0f}C"
        draw_centered_text(imdata, label, data['center'])

        # Handle UI for active drag
    if handler.is_dragging:
        r_preview = int(np.linalg.norm(handler.end_point - handler.start_point))
        cv2.circle(imdata, tuple(handler.start_point), r_preview, (255, 255, 255), 1)
        cv2.putText(imdata, f"Active Slot: {get_slot()}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)        
    
    
    cv2.imshow("Main App", imdata)
    
cv2.destroyAllWindows()