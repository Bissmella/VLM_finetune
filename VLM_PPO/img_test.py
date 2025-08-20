from PIL import Image
import numpy as np
import cv2

img = Image.open("/home/bahaduri/RL4VLM/outputs/2/182.png")
img_np = np.array(img)  # shape: (H, W, 3) in RGB

# Step 3: Convert RGB to BGR
img_bgr = img_np[:, :, ::-1]  # Reverse the channels

# Step 4: Save using OpenCV
cv2.imwrite("/home/bahaduri/RL4VLM/outputs/2/00.png", img_bgr)

img_bgr = cv2.imread("/home/bahaduri/RL4VLM/outputs/2/182.png")

# Step 2: Convert BGR to RGB
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Step 3: Convert to PIL image
img_pil = Image.fromarray(img_rgb)
img_pil.save("/home/bahaduri/RL4VLM/outputs/2/000.png")