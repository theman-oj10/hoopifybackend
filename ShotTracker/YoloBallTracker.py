import cv2
from ultralytics import YOLO
import numpy as np
import FindRim
import PositionTracker as pt

def intersect(ballX, ballY, x1, y1, x2, y2, x3, y3, x4, y4):
    # equation of rim line
    m = (y2 - y1)/(x2 - x1)
    c = y2 - (m * x2)

    minX = min(x1, x2, x3, x4)
    maxX = max(x1, x2, x3, x4)
    minY = min(y1, y2, y3, y4)
    maxY = max(y1, y2, y3, y4)
    # check if ball intersects
    #return ballY == m*ballX + c
    return minX <= ballX <= maxX and minY <= ballY <= maxY
def yoloTrack(path_to_video, path_to_court_img):
    model = YOLO('yolov8l.pt')
    # Filtering the classes: !Doesn't work still tracks everything
    model.classes = ['person', 'sports ball']

    # recording the rim-cordinates
    rim_coordinates = FindRim.find_rim(path_to_video)
    # checking shot attempts
    # in possesion => overlap between ball and person bbox

    #cv2.destroyAllWindows()
    print(f"rim coordinates: {rim_coordinates}")
    results = model.predict(source=path_to_video, show=False, stream='True')
    shots_taken = 0
    crossed_rim = False
    score = 0
    ball_coordinates = []
    shot_attempts = []
    made_shots = []
    def check_overlap(ball_bbox, ball_x, ball_y, person_bbox, person_x, person_y):
        ball_w = ball_bbox[2]
        ball_h = ball_bbox[3]
        person_w = person_bbox[2]
        person_h = person_bbox[3]
        ball_rightTop = (ball_x + ball_w/2, ball_y + ball_h/2)
        ball_leftTop = (ball_x - ball_w/2, ball_y + ball_h/2)
        ball_rightBottom = (ball_x + ball_w / 2, ball_y - ball_h / 2)
        ball_leftBottom = (ball_x - ball_w / 2, ball_y - ball_h / 2)
        person_rightTop = (person_x + person_w / 2, person_y + person_h / 2)
        person_leftTop = (person_x - person_w / 2, person_y + person_h / 2)
        person_rightBottom = (person_x + person_w / 2, person_y - person_h / 2)
        person_leftBottom =(person_x - person_w / 2, person_y - person_h / 2)

        return intersect(ball_x, ball_y, person_leftTop[0], person_leftTop[1], person_rightTop[0], person_rightTop[1], person_leftBottom[0],
                         person_leftBottom[1], person_rightBottom[0], person_rightBottom[1])
    person_exists = False
    shot_tobeTaken = False
    # finding homography matrix:
    matrix = pt.find_homography_matrix(path_to_video, path_to_court_img)
    # each result is a frame
    for result in results:
        boxes = result.boxes
        frame = 0

        # boxes contains all the bboxes in a frame
        # if a frame contains both a person and a ball
        # check if frame and ball overlap
        for box in boxes:
            if int(box.cls) == 0: # '0' is the class index for 'person'
                personbBox = box.xywh[0]
                x_person = personbBox[0]
                y_person = personbBox[1]
                # player position tracking
                # apply homography transformation
                original_coord = np.array([[x_person, y_person, 1]])
                transformed_coord = np.dot(matrix, original_coord.T)
                # Normalize the transformed coordinate
                normalized_coord = transformed_coord / transformed_coord[2]
                # Extract the x and y values of the normalized coordinate
                transformed_x = normalized_coord[0, 0]
                transformed_y = normalized_coord[1, 0]
                person_exists = True
            if int(box.cls) == 32:  # '32' is the class index for 'ball'
                currbBox = box.xywh[0]
                x_center = currbBox[0]
                y_center = currbBox[1]
                # checking for # shots taken
                # check if bbox of person and ball overlap => shots to be taken
                if person_exists:
                    if check_overlap(currbBox, x_center, y_center, personbBox, x_person, y_person):
                        print('Shot to be Taken')
                        shot_tobeTaken = True
                        # bbox of person and ball no longer overlap => shot taken
                if shot_tobeTaken and not check_overlap(currbBox, x_center, y_center, personbBox, x_person, y_person):
                    shot_tobeTaken = False
                    print("Shot Taken")
                    shots_taken += 1
                    curr_shot = (transformed_x, transformed_y)
                    shot_attempts.append(curr_shot)
                    print(f"x-player: {transformed_x}")
                    print(f"y-player: {transformed_y}")
                    person_exists = False
                ball_coordinates.append((x_center, y_center))
                #print(ball_coordinates)
                # checking if it crosses the rim
                # check for y value then x
                x1 = rim_coordinates[0][0] - 10
                y1 = rim_coordinates[0][1] + 10
                x2 = rim_coordinates[1][0] + 10
                y2 = rim_coordinates[1][1] + 10
                x3 = rim_coordinates[2][0] - 10
                y3 = rim_coordinates[2][1] - 10
                x4 = rim_coordinates[3][0] + 10
                y4 = rim_coordinates[3][1] - 10
                if min(y1, y2, y3, y4) < y_center < max(y1, y2, y3, y4) and min(x1, x2, x3, x4) < x_center < max(x1, x2, x3, x4):
                        print("crossed rim")
                        crossed_rim = True
                else:
                    if crossed_rim:
                        score +=1
                        made_shots.append(curr_shot)
                        print(f"score: {score}")
                        crossed_rim = False
                        frame = 0
                frame += 1
                    # (x,y) of left end of rim, (x,y) of right of rim, (x,y) of left of net, (x,y) of right of net
    court = cv2.imread(path_to_court_img)
    # hotzones
    for i in shot_attempts:
        if i in made_shots:
            made_shots.remove(i)
            cv2.circle(img=court, center=(int(i[0]), int(i[1])), radius=5, color=(0, 255, 0), thickness=5)
        #print(i)
        else:
            cv2.circle(img=court, center=(int(i[0]), int(i[1])), radius=5, color=(0, 0, 255), thickness=5)
    cv2.imshow("Hotzones", court)
    cv2.waitKey(0)
    # returning the co-ordinates of the center of the ball
    print(f"Score: {score}")
    print(f"Shots: {shots_taken}")
    # print(results)
    return score

# need to have 2 checks
# pass through rim and the middle of the net
# first find the co-ordinates of the middle of the net
# checking if it crosses the rim

yoloTrack("TestFootage/MAIN_TEST.mov", "TestFootage/court_invert.png")

