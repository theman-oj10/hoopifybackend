import cv2
def getFirstFrame(path):
    cap = cv2.VideoCapture(path)
    success, img = cap.read()
    cap.release()
    return img
def find_rim(path):
    rim = []
    frame = getFirstFrame(path)

    cv2.imshow('Select Rim', frame)
    def Capture_Event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            # This circle doesn't work
            #cv2.circle(img=frame, center=(x,y), radius=50, color=(0,255,0), thickness=5)
            print(x)
            rim.append((x, y))

    cv2.setMouseCallback('Select Rim', Capture_Event)
    #cv2.rectangle(frame, rim[0], rim[3], (0, 255, 0), 3)
    # Press any key to exit
    cv2.waitKey(0)
    #print(rim_coordinates)
    # Destroy all the windows
    # print(rim)
    #cv2.line(frame, rim[0], rim[1], (0, 255, 0), 3)
    cv2.rectangle(frame, rim[0], rim[3], (0, 255, 0), 3)
    cv2.imshow('Show Rim', frame)
    cv2.waitKey(0)

    return rim

# (x,y) of left end of rim, (x,y) of right of rim, (x,y) of left of net, (x,y) of right of net
#find_rim("TestFootage/BALL_DETECTION_1.mov")
