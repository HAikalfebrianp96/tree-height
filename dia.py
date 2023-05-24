from __future__ import print_function
import cv2
import numpy as np


PIXEL_TO_METER = 0.01  # Assuming 1 pixel is equal to 0.01 meters

def main():
    with open("classes.txt", "r") as objects_file:
        classes = [obj.strip() for obj in objects_file.readlines()]
    
    cap = cv2.VideoCapture("pohon_1.mp4")  # Replace "input_video.mp4" with your input video file

    back_sub = cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=25, detectShadows=True)
    kernel = np.ones((20, 20), np.uint8)
    net = cv2.dnn.readNetFromONNX("yolov5s/best.onnx")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255, size=[640, 640], mean=[0, 0, 0], swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()[0]
        classes_ids = []
        confidences = []
        boxes = []
        rows = detections.shape[0]
        img_width, img_height = frame.shape[1], frame.shape[0]
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
        fg_ukuran = back_sub.apply(frame)
        fg_ukuran = cv2.morphologyEx(fg_ukuran, cv2.MORPH_CLOSE, kernel)
        fg_ukuran = cv2.medianBlur(fg_ukuran, 5)
        _, fg_ukuran = cv2.threshold(fg_ukuran, 127, 255, cv2.THRESH_BINARY)
        fg_ukuran_bb = fg_ukuran
        contours, hierarchy = cv2.findContours(fg_ukuran_bb, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [cv2.contourArea(c) for c in contours]

        if len(areas) < 1:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for i in indices:
            if i >= len(classes_ids):  # Check if index is valid
                continue

            min_area = np.inf
            min_box = None

            for j in range(len(contours)):
                cnt = contours[j]
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)

                # Check if the contour is a square based on aspect ratio
                aspect_ratio = float(w) / h
                if aspect_ratio >= 0.9 and aspect_ratio <= 1.1:
                    if area < min_area:
                        min_area = area
                        min_box = cv2.boundingRect(cnt)

            if min_box is not None:
                x, y, w, h = min_box
                if classes_ids[i] >= len(classes):
                    continue
                label = classes[classes_ids[i]]
                conf = confidences[i]
                text = label + " {:.2f}".format(conf)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)

                # Convert height and width from pixels to meters
                length_pixels = max(w, h)
                length_meters = 4.5  # Length of the object in meters
                distance_meters = 5.0  # Distance from the camera to the object in meters

                conversion_factor = length_meters / length_pixels
                height_m = h * conversion_factor
                height_text = "Height: {:.2f} m".format(height_m)
                text_dimensions, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_x = x + (w - text_dimensions[0]) // 2
                text_y = y + (h + text_dimensions[1]) // 2
                cv2.rectangle(frame, (text_x - 10, text_y - text_dimensions[1] - 5),
                              (text_x + text_dimensions[0] + 10, text_y + 5), (36, 255, 12), -1)
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                if 4 <= height_m < 5:
                    cv2.putText(frame, 'Tinggi:  is (4 - 5) meter', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                elif 6 <= height_m < 7:
                    cv2.putText(frame, 'Tinggi:  is (6 - 7) meter', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                elif 8 <= height_m < 9:
                    cv2.putText(frame, 'Tinggi:  is (8 - 9) meter', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                elif 10 <= height_m < 11:
                    cv2.putText(frame, 'Tinggi:  is (10 - 11) meter', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    cv2.putText(frame, 'Telalu tinggi : ', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print(__doc__)
    main()
