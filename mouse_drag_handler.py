import cv2
import numpy as np

class MouseDragHandler:
    def __init__(self, on_drag_complete_callback=None):
        self.start_point = None
        self.end_point = None
        self.is_dragging = False
        self.on_drag_complete = on_drag_complete_callback

    def handle_mouse(self, event, x, y, flags, param):
        # Initialize both points to the same spot on click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = np.array([x, y])
            self.end_point = np.array([x, y])  # Initialized as requested
            self.is_dragging = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging:
                self.end_point = np.array([x, y])

        elif event == cv2.EVENT_LBUTTONUP:
            self.end_point = np.array([x, y])
            self.is_dragging = False
            
            # Execute the caller's logic if a callback was provided
            if self.on_drag_complete:
                self.on_drag_complete(self.start_point, self.end_point)