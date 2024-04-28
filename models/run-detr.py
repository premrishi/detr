import cv2
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Load the DETR model
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval()

# Define preprocessing transformations
transform = Compose([Resize(800), ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Function to preprocess a single frame
def preprocess_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor_image = transform(image)
    return tensor_image

# Function to visualize the output
def visualize_output(frame, output):
    # Extract predicted bounding boxes and class labels
    boxes = output['pred_boxes']
    labels = output['pred_logits'].argmax(-1)

    # Get image dimensions
    height, width = frame.shape[:2]

    # Iterate through the boxes
    for box, label in zip(boxes, labels):
        # Convert box tensor to list and extract box coordinates
        box = box.tolist()
        if len(box) != 4:
            continue  # Skip this box if it doesn't contain four values

        # Scale bounding box coordinates to match resized frame
        x_min, y_min, x_max, y_max = map(int, box)
        target_width = 10
        target_height = 10
        x_min = max(0, min(width, int(x_min * (target_width / width))))
        y_min = max(0, min(height, int(y_min * (target_height / height))))
        x_max = max(0, min(width, int(x_max * (target_width / width))))
        y_max = max(0, min(height, int(y_max * (target_height / height))))

        # Draw red bounding box around the wire
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red color for the bounding box

    # Display the frame
    cv2.imshow('Object Detection', frame)
    cv2.waitKey(1)



# Path to the video file
video_path = 'D:/New Volume E/E drive backup/dataset/wire_dataset.mp4'
cap = cv2.VideoCapture(video_path)

# Process the video frame by frame
with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        tensor_frame = preprocess_frame(frame)
        
        # Make predictions with the model
        output = model(tensor_frame.unsqueeze(0))
        

        # Visualize the output
        visualize_output(frame, output)
