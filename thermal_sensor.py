import cv2
import numpy as np
import json
import os

SAVE_FILE = "thermal_regions.json"

class ThermalEngine:
    def __init__(self):
        self.regions = {}
        self.current_slot = '1'
        self.load_regions()

    def load_regions(self):
        if os.path.exists(SAVE_FILE):
            try:
                with open(SAVE_FILE, 'r') as f:
                    data = json.load(f)
                    for slot, val in data.items():
                        self.regions[slot] = {
                            'center': tuple(val['center']),
                            'radius': int(val['radius']),
                            'mask': None, 'temp': 0.0
                        }
            except Exception as e: print(f"Load error: {e}")

    def save_regions(self):
        serializable = {s: {'center': [int(v['center'][0]), int(v['center'][1])], 
                            'radius': int(v['radius'])} for s, v in self.regions.items()}
        with open(SAVE_FILE, 'w') as f:
            json.dump(serializable, f, indent=4)

    def update_region(self, start, end):
        radius = int(np.linalg.norm(end - start))
        if radius > 5:
            self.regions[self.current_slot] = {
                'center': tuple(start.astype(int)),
                'radius': radius,
                'mask': None, 'temp': 0.0
            }
            self.save_regions()

    def process_frame(self, frame, produce_ui_image):
        """
        Split frame into two images: one has thermal data in UY channels, another is a pseudo-color image. 
        Remove rows which has invalid thermal data.
        """
        if frame is None: return frame
        if frame.shape[2] != 2:
            print(f"Frame Shape: {frame.shape}")

        t, i = np.array_split(frame, 2)
        
        # Clear invalid rows with zeros in the upper byte of sensor data.
        invalid_row = np.where(t[:,:,1]==0)[0].min()
        thermal = np.delete(t, range(invalid_row, t.shape[0], 1), axis=0)
        h, w = thermal.shape[:2]
        for slot, data in self.regions.items():
            # Lazy mask creation/regeneration
            if data['mask'] is None or data['mask'].shape[:2] != (h, w):
                m = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(m, data['center'], data['radius'], 255, -1)
                data['mask'] = m

            # Calculate Temperature
            pixels = thermal[data['mask'] == 255]
            if pixels.size > 0:
                Y, U = pixels[:, 0].astype(np.int32), pixels[:, 1].astype(np.int32)
                #teflon
                results = (256 * U + Y) / 10.0 - 175  
                #stainless: /9 - 150
                #results = (256 * U + Y) / 9.8 - 160  
                #low range
                #results = (256 * (U - 17) + Y) / 25.0
                data['temp'] = np.mean(results)
        
        if not produce_ui_image:
            return None

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
        # Draw
        for slot, data in self.regions.items():
            cv2.circle(heatmap, data['center'], data['radius'], (0, 255, 0), 2)
            self.draw_centered_text(heatmap, f"{data['temp']:.1f}C", data['center'])
        
        return heatmap

    def draw_centered_text(self, img, text, center):
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        (w, h), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.putText(img, text, (int(center[0]-w/2), int(center[1]+h/2)), font, scale, (255,255,255), thick)