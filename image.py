import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

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

# Function to update the image display
def update_image_display():
    global original_image, dehazed_image, panel
    if original_image is not None and dehazed_image is not None:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Display the original image
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Display the dehazed image
        ax2.imshow(cv2.cvtColor(dehazed_image, cv2.COLOR_BGR2RGB))
        ax2.set_title('Dehazed Image')
        ax2.axis('off')
        
        # Convert the figure to a PhotoImage
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        image_side_by_side = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        side_by_side_photo = ImageTk.PhotoImage(image_side_by_side)
        
        # Update the panel with the side-by-side image
        if panel is not None:
            panel.destroy()
        panel = tk.Label(root, image=side_by_side_photo)
        panel.image = side_by_side_photo
        panel.pack(side=tk.TOP, pady=10)
        
        # Clear the figure for the next update
        plt.close(fig)
def update_image_display():
    global original_image, dehazed_image, panel, canvas, toolbar
    if original_image is not None and dehazed_image is not None:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Display the original image
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Display the dehazed image
        ax2.imshow(cv2.cvtColor(dehazed_image, cv2.COLOR_BGR2RGB))
        ax2.set_title('Dehazed Image')
        ax2.axis('off')
        
        # Add zoom functionality to the figure
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
        toolbar.update()
        
        # Pack the canvas and toolbar
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        # Clear the figure for the next update
        plt.close(fig)

# Function to browse and load the image
def load_image():
    global original_image, dehazed_image, panel, canvas, toolbar
    path = filedialog.askopenfilename()
    if path:
        original_image = cv2.imread(path)
        dehazed_image = dehaze_image(original_image.copy())
        if panel is not None:
            panel.destroy()
        if canvas is not None:
            canvas.get_tk_widget().destroy()
        if toolbar is not None:
            toolbar.destroy()
        update_image_display()

# Function to save the dehazed image
def save_image():
    global dehazed_image
    if dehazed_image is not None:
        path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
        if path:
            cv2.imwrite(path, cv2.cvtColor(dehazed_image, cv2.COLOR_RGB2BGR))
            messagebox.showinfo("Success", "Image saved successfully.")

# Initialize the original and dehazed images
original_image = None
dehazed_image = None
panel = None
canvas = None
toolbar = None

# Create the main window
root = tk.Tk()
root.title("Dehazing App")

# Create a frame for the buttons
button_frame = tk.Frame(root)
button_frame.pack(side=tk.TOP, pady=10)

# Create and place the buttons
load_button = tk.Button(button_frame, text="Load Image", command=load_image)
load_button.pack(side=tk.LEFT, padx=5)

save_button = tk.Button(button_frame, text="Save Dehazed Image", command=save_image)
save_button.pack(side=tk.LEFT, padx=5)

# Start the GUI loop
root.mainloop()