import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Function to dehaze the image using the dark channel prior method
# ... (same as before)

# Function to get the dark channel prior
# ... (same as before)

# Function to get the atmospheric light
# ... (same as before)

# Function to get the transmission
# ... (same as before)

# Function to update the image display

# Function to dehaze the image using the dark channel prior method
def dehaze_image(img, patch_size=15, omega=0.95):
    img = img.astype(np.float32) / 255.0
    dark_channel = get_dark_channel(img, patch_size)
    atmospheric_light = get_atmospheric_light(img, dark_channel)
    transmission = get_transmission(img, atmospheric_light, omega, patch_size)
    
    # Dehaze the image
    dehazed = np.empty(img.shape, img.dtype)
    for i in range(3):
        dehazed[:, :, i] = (img[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]
    
    # Clip and convert to uint8
    dehazed = np.clip(dehazed, 0, 1)
    dehazed = (dehazed * 255).astype(np.uint8)

    return dehazed

# Function to get the dark channel prior
def get_dark_channel(img, patch_size):
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

# Function to get the atmospheric light
def get_atmospheric_light(img, dark_channel, top_percent=0.1):
    img_size = img.shape[0] * img.shape[1]
    num_pixels = int(max(img_size * top_percent, 1))
    flat_img = img.reshape(img_size, 3)
    flat_dark = dark_channel.reshape(img_size)
    indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
    top_pixels = flat_img[indices]
    atmospheric_light = np.max(top_pixels, axis=0)
    return atmospheric_light

# Function to get the transmission
def get_transmission(img, atmospheric_light, omega=0.95, patch_size=15):
    normalized_img = img / atmospheric_light
    dark_channel = get_dark_channel(normalized_img, patch_size)
    transmission = 1 - omega * dark_channel
    return transmission

def update_image_display(original, dehazed):
    global panel_original, panel_dehazed
    # Convert the images to PIL format
    original_pil = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    dehazed_pil = Image.fromarray(cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB))
    # Convert the PIL image to a Tkinter-compatible format
    original_photo = ImageTk.PhotoImage(image=original_pil)
    dehazed_photo = ImageTk.PhotoImage(image=dehazed_pil)
    
    # Update the panels with the new images
    if panel_original is not None:
        panel_original.configure(image=original_photo)
        panel_original.image = original_photo
    else:
        panel_original = tk.Label(image=original_photo)
        panel_original.image = original_photo
        panel_original.pack(side=tk.LEFT, padx=10)
    
    if panel_dehazed is not None:
        panel_dehazed.configure(image=dehazed_photo)
        panel_dehazed.image = dehazed_photo
    else:
        panel_dehazed = tk.Label(image=dehazed_photo)
        panel_dehazed.image = dehazed_photo
        panel_dehazed.pack(side=tk.RIGHT, padx=10)

# Function to capture and dehaze video frames
def video_capture():
    global stop_video
    # Try different camera indices if 0 doesn't work
    for camera_index in range(10):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            break
    else:
        messagebox.showerror("Error", "Could not open video source")
        return

    stop_video = False
    while not stop_video:
        ret, frame = cap.read()
        if ret:
            dehazed_frame = dehaze_image(frame)
            update_image_display(frame, dehazed_frame)
        else:
            messagebox.showerror("Error", "Could not read frame")
            break

    cap.release()

# Function to start video capture
def start_capture():
    global video_thread
    video_thread = threading.Thread(target=video_capture)
    video_thread.start()

# Function to stop video capture
def stop_capture():
    global stop_video
    stop_video = True
    video_thread.join()

# Main application
root = tk.Tk()
root.title("Live Video Dehazing")

# Variables to hold the panels
panel_original = None
panel_dehazed = None

# Video capture control variables
stop_video = False
video_thread = None

# Start and stop buttons
start_button = tk.Button(root, text="Start", command=start_capture)
start_button.pack(side=tk.TOP, pady=10)

stop_button = tk.Button(root, text="Stop", command=stop_capture)
stop_button.pack(side=tk.TOP, pady=10)

# Start the GUI loop
root.mainloop()