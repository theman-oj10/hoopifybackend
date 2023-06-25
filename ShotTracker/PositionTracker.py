import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
from ultralytics import YOLO
# model = YOLO('yolov8l.pt')
#results = model.predict(source="TestFootage/MAIN_TEST.mov", show=True, device='mps', stream='True')

def find_homography_matrix(path_to_video, path_to_courtImg):
# new plot
    plt.figure(figsize=(10,8))

    # stores a set of fixed points on the basketball court in 3D space in the image
    pts_3D = np.array([(0,235), (70, 236), (230, 235), (370, 232), (530, 230), (600, 227), (300, 360), (200, 315), (405, 310)])
    # stores the corresponding points in 2D space
    pts_2D = np.array([(0, 0), (63, 0),(226, 0), (370, 0), (535, 0), (600, 0), (300, 216), (226, 160), (370, 160) ])

    cap = cv2.VideoCapture(path_to_video)
    success, img = cap.read()
    cap.release()
    # image of the video_frame
    frame = img
    frame = cv2.resize(frame, (600,400))

    # 2d court image
    court = cv2.imread(path_to_courtImg)
    court = cv2.resize(court, (600,400))

    # region of interest from the video_frame
    roi = frame[150:400, 0: 600]

    # storing original frame dimensions and dimensions of roi
    r_h, r_w, r_c = roi.shape
    i_h, i_w, i_c = frame.shape

    # line detection works better with grayscale images (less pixel range to process)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Canny edge detection to detect line edges
    # line edges are the lines in the image
    edges = cv2.Canny(roi, 50, 150, apertureSize=3)
    # lines are the mathematical representation of line edges detected in the image
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # calculate (x1,y1) and (x2, y2) coordinates
    for line in lines:
        # rho is the distance of a perpendicular from the origin to the line
        # theta is the angle from the x axis
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        # extend lines out by a factor of a 1000 to ensure they cover the whole image
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # shows each line
        cv2.line(frame, (x1, y1 + (i_h - r_h)), (x2, y2 + (i_h - r_h)), (255, 0 ,100), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    fig.suptitle("PLAYER POSITION EXTRACTION")

    ax1.set_title("3D Image")
    for p in range(0, len(pts_3D)):
        ax1.scatter(pts_3D[p][0], pts_3D[p][1], s=100, c='r', marker='o')

    ax2.set_title("2D Image")
    for p in range(0, len(pts_2D)):
        ax2.scatter(pts_2D[p][0], pts_2D[p][1], s=200, c='r')

    #ax1.imshow(frame)
    #ax2.imshow(court)
    #plt.show()

    # finding homography matrix:
    matrix, status = cv2.findHomography(pts_3D, pts_2D)

    return matrix

# for result in results:
#     boxes = result.boxes
#
#     # boxes contains all the bboxes in a frame
#     # if a frame contains both a person and a ball
#     # check if frame and ball overlap
#     for box in boxes:
#         if int(box.cls) == 0:  # '0' is the class index for 'person'
#             personbBox = box.xywh[0]
#             x_person = personbBox[0]
#             y_person = personbBox[1]
#             # apply homography transformation
#             original_coord = np.array([[x_person, y_person, 1]])
#             transformed_coord = np.dot(matrix, original_coord.T)
#             # Normalize the transformed coordinate
#             normalized_coord = transformed_coord / transformed_coord[2]
#             # Extract the x and y values of the normalized coordinate
#             transformed_x = normalized_coord[0, 0]
#             transformed_y = normalized_coord[1, 0]

def show_hotzones(court_img, positions):
    court = cv2.imRead(court_img)
    for i in positions:
        cv2.circle(img=court, center=(i[0], i[1]), radius=50, color=(0,255,0), thickness=5)
