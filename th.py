import cv2

cap = cv2.VideoCapture(0)
for i in range(0, 50):  # Check the first 50 standard properties
    val = cap.get(i)
    if val != -1 and val != 0:
        print(f"Property ID {i} is active. Value: {val}")
        
# Potential HIKMICRO Command IDs for Mode Switching
# These are often sent via the ZOOM property
HIGH_TEMP_MODE = 0x8004  # Example value for High Temp mode (120-550C)
LOW_TEMP_MODE  = 0x8000  # Example value for Low Temp mode (-20-120C)

# To switch to High Temp:
cap.set(cv2.CAP_PROP_ZOOM, HIGH_TEMP_MODE)
print(f"Sent High-Temp switch command, new value is {cap.get(cv2.CAP_PROP_ZOOM)}")


# To switch back to Low Temp:
# cap.set(cv2.CAP_PROP_ZOOM, LOW_TEMP_MODE)