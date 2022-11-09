import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# print(model)
img = 'celeb.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()

# %matplotlib inline
plt.imshow(np.squeeze(results.render()))
plt.title('Traffic image with Yolo V5')
plt.show()
plt.savefig("Celeb_V5.jpg")