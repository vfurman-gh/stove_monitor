import argparse
from flask import Flask, jsonify, render_template_string
import threading
import cv2
from thermal_sensor import ThermalEngine
from mouse_drag_handler import MouseDragHandler
import time
import logging
from logging.handlers import RotatingFileHandler
import os

# 1. Disable the default Flask/Werkzeug logger
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) 

# 2. Setup Rotating Log
LOG_FILENAME = "/home/vadim/opencv/app.log"
os.makedirs(os.path.dirname(LOG_FILENAME), exist_ok=True)

handler = RotatingFileHandler(LOG_FILENAME, maxBytes=1000000, backupCount=3)
handler.setFormatter(logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
))

# Create a custom logger for your app
logger = logging.getLogger('thermal_app')
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Global variable to track health
last_heartbeat = time.time()

def watchdog_thread_function():
    global last_heartbeat
    logger.info("Watchdog thread started.")
    while True:
        # Check if the camera has checked in within the last 10 seconds
        if time.time() - last_heartbeat > 10:
            logger.error("WATCHDOG: Camera thread is non-responsive! Restarting process...")
            # Trigger a clean exit; systemd will then restart the service
            os._exit(1) 
        time.sleep(5)


app = Flask(__name__)
engine = ThermalEngine()
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<style>
    body { 
        background: #101010; color: white; margin: 0; padding: 0; 
        overflow: hidden; width: 100vw; height: 100vh;
        display: flex; flex-direction: column; font-family: sans-serif;
    }
    h1 { position: absolute; top: 10px; left: 50%; transform: translateX(-50%); z-index: 10; font-size: 1.2rem; opacity: 0.5; pointer-events: none; }
    canvas { display: block; width: 100vw; height: 100vh; }
    
    #controls { 
        position: absolute; top: 10px; right: 20px; z-index: 20;
        background: rgba(0,0,0,0.7); padding: 12px; border-radius: 8px; border: 1px solid #333;
    }
   #status { 
    position: absolute; 
    bottom: 20px; 
    left: 20px; 
    z-index: 100;
    padding: 8px 15px;
    background: rgba(0, 0, 0, 0.8); 
    border-radius: 5px;
    font-size: 14px;
    font-weight: bold;
    border: 1px solid #444;
    transition: all 0.3s ease;
    pointer-events: none; /* Allows clicks to pass through to canvas if needed */
}

/* Status Classes */
.online { 
    color: #00ff00; 
    border-color: #00ff00;
    box-shadow: 0 0 10px rgba(0, 255, 0, 0.2);
}

.offline { 
    color: #ff4444; 
    border-color: #ff4444;
    box-shadow: 0 0 10px rgba(255, 68, 68, 0.4);
}
</style>

<body>
    <h1>THERMAL MONITOR LIVE</h1>
    <div id="controls">
        <label><input type="checkbox" id="unitToggle"> Fahrenheit (°F)</label>
        <button onclick="document.documentElement.requestFullscreen()" style="margin-left:10px; cursor:pointer;">Full Screen</button>
    </div>
    <canvas id="c"></canvas>
    <div id="status">Server Connection: Online</div>

<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const RAW_W = 256;
const RAW_H = 192;

let snapPoints = [];
let scaleX = 1;
let scaleY = 1;

function resize() {
    // 1. Update internal canvas resolution to match browser window
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    // 2. Calculate dynamic scale factors
    scaleX = canvas.width / RAW_W;
    scaleY = canvas.height / RAW_H;

    // 3. Redefine snap points for the new screen size
    snapPoints = [
        { x: canvas.width * 0.25, y: canvas.height * 0.25 }, // Top-Left
        { x: canvas.width * 0.75, y: canvas.height * 0.25 }, // Top-Right
        { x: canvas.width * 0.25, y: canvas.height * 0.75 }, // Bottom-Left
        { x: canvas.width * 0.75, y: canvas.height * 0.75 }  // Bottom-Right
    ];
}

// Initialize on load and on every window resize
window.addEventListener('resize', resize);
resize();

// Nearest neighbor logic remains the same
function getNearestSnapPoint(inputX, inputY) {
    let nearest = snapPoints[0];
    let minDistance = Infinity;
    snapPoints.forEach(point => {
        const dist = Math.sqrt(Math.pow(point.x - inputX, 2) + Math.pow(point.y - inputY, 2));
        if (dist < minDistance) {
            minDistance = dist;
            nearest = point;
        }
    });
    return nearest;
}
/**
 * Maps a temperature value to an RGB color string.
 * 90°C and below = Green
 * 150°C and above = Red
 * In between = Gradient (Yellow/Orange)
 */
function mapTempToColor(temp) {
    // 1. Clamp the temperature between our min and max
    const minTemp = 90;
    const maxTemp = 150;
    let t = Math.max(minTemp, Math.min(maxTemp, temp));

    // 2. Calculate the ratio (0.0 to 1.0)
    const ratio = (t - minTemp) / (maxTemp - minTemp);

    // 3. Interpolate colors
    // From Green (0, 255, 0) to Red (255, 0, 0)
    // At ratio 0.5, this produces Yellow (127, 127, 0)
    const r = Math.floor(255 * ratio);
    const g = Math.floor(255 * (1 - ratio));
    const b = 0;

    return `rgb(${r}, ${g}, ${b})`;
}

async function update() {
    try {
        const res = await fetch('/api/temps');
        if (!res.ok) {
            // Server responded with an error (e.g., 500 or 404)
            throw new Error(`Server Error: ${res.status}`);
        }
        const data = await res.json();

        statusEl.innerText = "● Server Connection: Online";
        statusEl.className = "online";
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Debug the canvas dimensions
        console.log(`Canvas Size: ${canvas.width}x${canvas.height}`);        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const useF = document.getElementById('unitToggle').checked;
        for (let s in data) {
            const zone = data[s];
            // Map 256x192 coords to full screen coords
            const screenX = zone.center[0] * scaleX;
            const screenY = zone.center[1] * scaleY;
            const snapped = getNearestSnapPoint(screenX, screenY);
            const color = mapTempToColor(zone.temp);
            // Scale the radius relative to the overall screen width
            const radius = zone.radius * scaleX;

            // Temp Conversion & Rounding
            let val = zone.temp;
            let label = "°C";
            if (useF) { val = (val * 1.8) + 32; label = "°F"; }
            val = Math.round(val);

            // Draw
            ctx.beginPath();
            ctx.arc(snapped.x, snapped.y, radius, 0, 7);
            ctx.fillStyle = color.replace('rgb', 'rgba').replace(')', ', 0.2)');
            ctx.fill();
            ctx.strokeStyle = color;
            ctx.lineWidth = Math.max(2, 5 * (canvas.width / 1280)); // Responsive line width
            ctx.stroke();

            ctx.fillStyle = "white";
            ctx.font = `bold ${Math.round(canvas.width / 25)}px Arial`; // Responsive font
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(val + label, snapped.x, snapped.y);
        }
    } catch (e) { 
        // --- Connection Failed ---
        console.error("Connection lost:", e);
        
        statusEl.innerText = "○ Server Connection: Offline";
        statusEl.className = "offline";
        
        // Optional: Clear canvas or dim it when offline
        ctx.fillStyle = "rgba(0, 0, 0, 0.05)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
}
setInterval(update, 1500);
</script>
</body>
</html>
"""

@app.route('/')
def index(): return render_template_string(HTML_PAGE)

@app.route('/api/temps')
def get_temps(): 
    return jsonify({s: {"temp": round(d['temp'],1), "center": d['center'], "radius": d['radius']} for s, d in engine.regions.items()})

def camera_worker():
    global last_heartbeat

    cap = cv2.VideoCapture('/dev/video1', cv2.CAP_V4L)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,392)
    
    if args.local_gui:
        handler = MouseDragHandler(lambda s, e: engine.update_region(s, e))
        cv2.namedWindow("Thermal")
        cv2.setMouseCallback("Thermal", handler.handle_mouse)
    else:
        print("Running in HEADLESS mode. No local GUI will open.")

    while cap.isOpened():
        time.sleep(0.25)
        last_heartbeat = time.time()
        # Grab a fram rapidly to ensure the next .read() is fresh
        cap.grab()
        ret, frame = cap.read()
        if not ret: 
            print("Error reading frame")
            break
        
        processed = engine.process_frame(frame, produce_ui_image=args.local_gui)
        if args.local_gui:
            cv2.imshow("Thermal", processed)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if chr(key) in '1234': engine.current_slot = chr(key)
        if key == ord('c'): 
            engine.regions.clear()
            engine.save_regions()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Thermal Server")
    parser.add_argument('--local_gui', action='store_true', dest='local_gui', help='Run with OpenCV GUI window')
    args = parser.parse_args()
    threading.Thread(target=watchdog_thread_function, daemon=True).start()
    threading.Thread(target=camera_worker, daemon=True).start()
    # 3. Start the Flask Server (Main Thread)
    # Using the main thread for Flask allows it to handle shutdown signals
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
