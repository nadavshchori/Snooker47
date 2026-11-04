from ultralytics import YOLO
import PIL
from PIL import ImageEnhance

model = YOLO("yoloe-26n-seg.pt")
model.set_classes(["ball on green background"])


img = PIL.Image.open('pool.jpg')
converter = PIL.ImageEnhance.Color(img)
img2 = converter.enhance(2)

results = model(img2)
result = results[0]

# Each detection carries a class, a confidence score, and a box in pixels
for box in result.boxes:
    name = result.names[int(box.cls)]
    corners = box.xyxy[0].int().tolist()
    print(f"{name} {box.conf.item():.2f} {corners}")


boxes = result.boxes  # Boxes object for bounding box outputs
masks = result.masks  # Masks object for segmentation masks outputs
keypoints = result.keypoints  # Keypoints object for pose outputs
probs = result.probs  # Probs object for classification outputs
obb = result.obb  # Oriented boxes object for OBB outputs
result.show()  # display to screen
# save to disk