import cv2
import numpy as np
import json

def detect_object(frame):
    # Detect objects
    classes, scores, boxes = model.detect(frame, 0.4, 0.3)

    # Plot the image with bounding boxes and labels
    for (classid, score, box) in zip(classes, scores, boxes):
        label = "%s : %f" % (class_names[classid], score)
        cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)

    return frame

def main():
    try:
        classes = json.load(open("dataset.json"))
        # with open("classes.txt", "r") as objects_file:
        # classes = [e_g.strip() for e_g in objects_file.readlines()]
        input_size = 416

        cap = cv2.VideoCapture("7.mp4")
        net = cv2.dnn.readNetFromONNX("yolov5s/yolov5s.onnx")
        known_height = 4.5
        known_distance = 5

        while True:
            ret, img = cap.read()
            if not ret:
                break

            blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255, size=(640, 640), mean=(0, 0, 0), swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward()[0]
            predictions = detections[0]
            boxes = predictions[0.3]
            scores = predictions[:, :, 5:]
            classes = predictions[:, :, 6:]
            keep = np.where(scores > 0.5)[0]

            classes_ids = []
            confidences = []
            boxes = []
            rows = detections.shape[0]

            img_width, img_height = img.shape[1], img.shape[0]
            x_scale = img_width / 640
            y_scale = img_height / 640

            for i in range(rows):
                row = detections[i]
                confidence = row[4]
                if confidence > 0.2:
                    classes_score = row[5:]
                    ind = np.argmax(classes_score)
                    if classes_score[ind] > 0.2:
                        classes_ids.append(ind)
                        confidences.append(confidence)
                        cx, cy, w, h = row[:4]
                        x1 = int((cx - w / 2) * x_scale)
                        y1 = int((cy - h / 2) * y_scale)
                        width = int(w * x_scale)
                        height = int(h * y_scale)
                        box = np.array([x1, y1, width, height])
                        boxes.append(box)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.2)

            for i in indices:
                x1, y1, w, h = boxes[i]
                label = classes[classes_ids[i]]
                conf = confidences[i]
                text = label + "{:.2f}".format(conf)
                cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
                cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                object_height = known_height * h / w * known_distance
                cv2.putText(img, f"Object height: {object_height} meters", (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            detected_frame = detect_object(img)  # Call the detect_object function
            cv2.imshow("Object Detection", detected_frame)  # Show the detected_frame

            exit_key_press = cv2.waitKey(1)
            if exit_key_press == ord('q'):
                break
    except KeyboardInterrupt:
        cap.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("Select the WebCam or Camera index properly, in my case it is 2")

if __name__ == "__main__":
    main()