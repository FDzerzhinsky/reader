from ultralytics import YOLO

import os

def get_file_names(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Пример использования
directory_path = 'test'
file_names = get_file_names(directory_path)
file_names = [directory_path + '/' + file_name for file_name in file_names][:10]


# Load a model
model = YOLO("best_2.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
# results = model(["test/Image_354.jpg", "test/Image_355.jpg"])  # return a list of Results objects
results = model(file_names)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk