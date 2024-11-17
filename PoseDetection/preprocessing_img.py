from PIL import Image
import numpy as np

image_data = []  # Replace with actual image loading logic
# for img_path in image_paths:
img = Image.open("firstFrame.jpeg").resize((128, 128))
image_data.append(np.array(img))
np.save("image_data.npy", np.array(image_data))
