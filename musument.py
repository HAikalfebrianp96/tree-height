from __future__ import print_function
import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)
    back_sub =cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=25, detectShadows= True)
    kernel = np.ones((20,20),np.uint8)
    while(True):
        ret, frame = cap.read()
        fg_ukuran = back_sub.apply(frame)
        fg_ukuran = cv2.morphologyEx(fg_ukuran,cv2.MORPH_CLOSE, kernel)
        fg_ukuran = cv2.medianBlur(fg_ukuran,5)
        _,fg_ukuran = cv2.threshold(fg_ukuran, 127, 255, cv2.THRESH_BINARY)
        fg_ukuran_bb = fg_ukuran 
        contours, hierarchy = cv2.findContours(fg_ukuran_bb, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [cv2.contourArea(c) for c in contours]
        if len(areas)<1:
            cv2.imshow("frame",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        else:
            max_index = np.argmax(areas)
        cnt = contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(36,255,12),2)
        w2 = int(w)
        h2 = int(h)

        text = "lebar(mm) : " + str(w2) + ",  tinggi(mm) " + str(h2) 
        cv2.putText(frame, text. format(w2,h2),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12),2)
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    print(__doc__)
    main()



