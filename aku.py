import cv2
import numpy as np

# Load the YOLO v5.onnx model
model = cv2.dnn.readNetFromONNX("C:\Users\Lenovo\yolv5\coba\Deteksi-Objek-dengan-YOLOV5\yolov5s\best.onnx")

# Load the class names
class_names = ["a", "b", "c"]
#with open("coco.names", "r") as f:
    #class_names = f.readlines()

# Define the input size
input_size = 416

# Create a video capture object
cap = cv2.VideoCapture("7.mp4")

known_height = 4.5
known_distance = 5

while True:
    # Capture a frame
    ret, frame = cap.read()

    # Convert the frame to a blob
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_size, input_size), (0, 0, 0), True, crop=False)

    # Set the input of the network
    model.setInput(blob)

    # Run the inference
    out = model.forward()

    # Get the predictions
    predictions = out[0]

    # Get the bounding boxes
    boxes = predictions[:, :, :4]

    # Get the scores
    scores = predictions[:, :, 5:]

    # Get the classes
    classes = predictions[:, :, 6:]

    # Filter out low-confidence predictions
    keep = np.where(scores > 0.5)[0]

    # Draw the bounding boxes and labels on the frame
    for i in keep:
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, class_names[classes[i]], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Calculate the height of the object
        object_height = known_height * h / w * known_distance

        # Display the height of the object on the frame
        cv2.putText(frame, f"Object height: {object_height} meters", (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # If the key is ESC, break from the loop
    if key == 27:
        break

    # Save the output of the program to a file
    cv2.imwrite("output.jpg", frame)

# Close the video capture object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()