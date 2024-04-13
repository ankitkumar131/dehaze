import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading

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

# Function to dehaze a video frame
def dehaze_frame(frame):
    # Convert the frame to RGB and dehaze it
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dehazed_rgb_frame = dehaze_image(rgb_frame)
    
    # Convert the dehazed frame back to BGR
    dehazed_frame = cv2.cvtColor(dehazed_rgb_frame, cv2.COLOR_RGB2BGR)
    return dehazed_frame

# Function to dehaze a video and save it to a file
def dehaze_video(video_path, output_path):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        messagebox.showerror("Error", "Could not open video file")
        return
    
    # Get video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame and write to the output video
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Dehaze the frame
        dehazed_frame = dehaze_frame(frame)
        
        # Write the dehazed frame to the output video
        out.write(dehazed_frame)
    
    # Release everything when done
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

# Function to browse and load the video
# Function to browse and load the video
# Function to browse and load the video
def load_video():
    global original_video, dehazed_video, panel
    path = filedialog.askopenfilename(title="Select Video", filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi")])
    if path:
        # Dehaze the video
        output_path = filedialog.asksaveasfilename(title="Save Dehazed Video", defaultextension=".avi", filetypes=[("AVI files", "*.avi")])
        if output_path:
            # Dehaze the video in a separate thread to keep the GUI responsive
            dehaze_thread = threading.Thread(target=dehaze_video, args=(path, output_path))
            dehaze_thread.start()

# Function to play both videos side by side
def play_videos(original_video_path, dehazed_video_path):
    # Open both videos
    original_video_capture = cv2.VideoCapture(original_video_path)
    dehazed_video_capture = cv2.VideoCapture(dehazed_video_path)

    # Check if videos were opened successfully
    if not original_video_capture.isOpened() or not dehazed_video_capture.isOpened():
        messagebox.showerror("Error", "Could not open one or both video files")
        return

    # Play both videos side by side
    while original_video_capture.isOpened() and dehazed_video_capture.isOpened():
        ret1, original_frame = original_video_capture.read()
        ret2, dehazed_frame = dehazed_video_capture.read()

        if not ret1 or not ret2:
            break

        # Display the frames side by side
        combined_frame = cv2.hconcat([original_frame, dehazed_frame])
        cv2.imshow('Original vs Dehazed Video', combined_frame)

        # Play the sound from both videos (if available)
        # Note: This requires the videos to have audio tracks

        # Break the loop if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video captures and destroy all windows
    original_video_capture.release()
    dehazed_video_capture.release()
    cv2.destroyAllWindows()

# Main GUI Setup
root = tk.Tk()
root.title("Dehaze Video")

# Browse and load video button
load_video_button = tk.Button(root, text="Load Video", command=load_video)
load_video_button.pack(side=tk.TOP, pady=10)

# Main Loop
root.mainloop()