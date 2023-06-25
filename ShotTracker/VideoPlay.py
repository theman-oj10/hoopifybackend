import cv2
'''Input: video File, runs inference and returns analysed video with ball tracking'''
def video_play(videoPath):
    cap = cv2.VideoCapture(videoPath)
    while True:
        success, img = cap.read()

        # Create a new window to display the modified frame
        cv2.imshow("Modified Frame", img)
    cap.release()
    cv2.destroyAllWindows()