import cv2
import mediapipe as mp
import time
from math import dist


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=4,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

mpFace = mp.solutions.face_detection
face = mpFace.FaceDetection(model_selection=0, min_detection_confidence=0.5)

mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
p1LeftPunch = False
p1RightPunch = False
p2LeftPunch = False
p2RightPunch = False
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

def averageCoordinates(hand):
    # Returns the average center point from the points on a hand 
    x = 0
    y = 0
    for num in range(21):
        x += hand[num][0]
        y += hand[num][1]
    x //= 21
    y //= 21
    return x, y

def getSize(hand):
    # Returns the estimated radius of the hand
    x = []
    y = []

    for num in range(21):
        x.append(hand[num][0])
        y.append(hand[num][1])
    xRange = max(x) - min(x)
    yRange = max(y) - min(y)

    if xRange > yRange:
        return xRange//2
    else:
        return yRange//2



class Health():
    
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.health = 100
        
    def draw(self):
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), RED, -1)
        greenBar = self.x + self.w * self.health//100
        if greenBar < self.x:
            greenBar = self.x
        cv2.rectangle(img, (self.x, self.y), (greenBar, self.y + self.h), GREEN, -1)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), BLACK, 3)


p1Health = Health(60, 20, 200, 20)
p2Health = Health(380, 20, 200, 20)

while True:
    singleList = {}
    lmList = {}
    facePoints = []
    bothFacePoints = {}
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = hands.process(imgRGB)
    faceResults = face.process(imgRGB)

    p1Health.draw()
    p1Health.health -= 1
    p2Health.draw()
    p2Health.health -= 2


    h, w, c = img.shape
    cv2.line(img, (w // 2, 0), (w // 2, h), BLACK, 3)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                singleList[id] = cx, cy
                # if id ==0:
                cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
            lmList[results.multi_hand_landmarks.index(handLms)] = singleList
            singleList = {}
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    x = 0
    if faceResults.detections:
        for faceLms in faceResults.detections:
            mpDraw.draw_detection(img, faceLms)
            h,w,c = img.shape
            xmin, ymin = int(faceLms.location_data.relative_bounding_box.xmin * w), int(faceLms.location_data.relative_bounding_box.ymin * h)
            xmax, ymax = int(faceLms.location_data.relative_bounding_box.width * w + xmin), int(faceLms.location_data.relative_bounding_box.height * h + ymin)
            facePoints = [xmin, ymin, xmax, ymax]
            bothFacePoints[x] = facePoints
            x += 1

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

    overlay = img.copy()
    if len(bothFacePoints) == 2: #change to 2
        h1 = []
        h2 = []
        for playerHead in bothFacePoints:

            if (bothFacePoints[playerHead][0]+bothFacePoints[playerHead][2])/2 < w/2:
                h1 = bothFacePoints[playerHead]

                h1Target = [(h1[2] + h1[0])//2 + w//2,(h1[3] + h1[1]) // 2]
                h1TargetWidth = (h1[2] - h1[0])//2

                cv2.circle(overlay, h1Target, h1TargetWidth, (255, 0, 0), cv2.FILLED)



            elif (bothFacePoints[playerHead][2] + bothFacePoints[playerHead][0])/2 > w/2:
                h2 = bothFacePoints[playerHead]

                h2Target = [(h2[2] + h2[0])//2 - w//2,(h2[3] + h2[1]) // 2]
                h2TargetWidth = (h2[2] - h2[0])//2

                cv2.circle(overlay,h2Target, h2TargetWidth, (0, 0, 255), cv2.FILLED)

    if len(lmList) == 4: #change to 4
        p1 = []
        p2 = []  
        
        for hand in lmList:
            if lmList[hand][0][0] < w / 2:
                p1.append(lmList[hand])
                center = averageCoordinates(lmList[hand])
                radius = getSize(lmList[hand])
                cv2.circle(overlay, (center[0] + w//2, center[1]), radius, (0, 0, 255), cv2.FILLED)
            
            else:
                p2.append(lmList[hand])
                center = averageCoordinates(lmList[hand])
                radius = getSize(lmList[hand])
                cv2.circle(overlay, (center[0] - w//2, center[1]), radius, (255, 0, 0), cv2.FILLED)
        
        if len(p1) == 2 and len(p2) == 2:

            if p1[0][0][0] > p1[1][0][0]:
                p1.reverse()
            if p2[0][0][0] > p2[1][0][0]:
                p2.reverse()

            # Player 1 Punch detection
            leftKnuckleDif = abs(p1[0][17][1] - p1[0][5][1])
            leftIndexDif = abs(p1[0][5][1] - p1[0][6][1])

                                              # Thumb tip is bellow thumb start
            if leftKnuckleDif < leftIndexDif and p1[0][4][1] > p1[0][1][1]:
                if not p1LeftPunch:
                    print("P1 LEFT PUNCH")
                    p1LeftPunch = True
                    leftHandCenter = averageCoordinates(p1[0])
                    h2Target = [(h2[2] + h2[0])//2 - w//2,(h2[3] + h2[1]) // 2]
                    h2TargetWidth = (h2[2] - h2[0])//2  
                    handRadius = getSize(p1[0])
                                                                        # //2 in order to make it easier
                    if dist(leftHandCenter, h2Target) <= h2TargetWidth - handRadius//2:
                        print("========== TRUE HIT =========")


            else:
                p1LeftPunch = False

            rightKnuckleDif = abs(p1[1][17][1] - p1[1][5][1])
            rightIndexDif = abs(p1[1][5][1] - p1[1][6][1])

                                              # Thumb tip is bellow thumb start
            if rightKnuckleDif < rightIndexDif and p1[1][4][1] > p1[1][1][1]:
                if not p1RightPunch:
                    print("P1 RIGHT PUNCH")
                    p1RightPunch = True
                    rightHandCenter = averageCoordinates(p1[1])
                    h2Target = [(h2[2] + h2[0])//2 - w//2,(h2[3] + h2[1]) // 2]
                    h2TargetWidth = (h2[2] - h2[0])//2  
                    handRadius = getSize(p1[1])
                                                                        # //2 in order to make it easier
                    if dist(rightHandCenter, h2Target) <= h2TargetWidth - handRadius//2:
                        print("========== TRUE HIT =========")



            else:
                p1RightPunch = False

            # Player 1 Defence Mode ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            leftHandCenter = averageCoordinates(p1[0])
            rightHandCenter = averageCoordinates(p1[1])

            if h1[0] <= leftHandCenter[0] <= h1[2] and h1[1] <= leftHandCenter[1] <= h1[3] and         h1[0] <= rightHandCenter[0] <= h1[2] and h1[1] <= rightHandCenter[1] <= h1[3]:
                p1Defence = True
                cv2.circle(img, (leftHandCenter), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (rightHandCenter), 10, (0, 255, 0), cv2.FILLED)
            else:
                p1Defence = False


            # Player 2 Punch detection
            leftKnuckleDif = abs(p2[0][17][1] - p2[0][5][1])
            leftIndexDif = abs(p2[0][5][1] - p2[0][6][1])

                                              # Thumb tip is bellow thumb start
            if leftKnuckleDif < leftIndexDif and p2[0][4][1] > p2[0][1][1]:
                if not p2LeftPunch:
                    print("P2 LEFT PUNCH")
                    p2LeftPunch = True
                    leftHandCenter = averageCoordinates(p2[0])
                    h1Target = [(h1[2] + h1[0])//2 + w//2, (h1[3] + h1[1]) // 2]
                    h1TargetWidth = (h1[2] - h1[0])//2
                    handRadius = getSize(p2[0])
                                                                        # //2 in order to make it easier
                    if dist(leftHandCenter, h1Target) <= h1TargetWidth - handRadius//2:
                        print("========== TRUE HIT =========")



                    
            else:
                p2LeftPunch = False

            rightKnuckleDif = abs(p2[1][17][1] - p2[1][5][1])
            rightIndexDif = abs(p2[1][5][1] - p2[1][6][1])

                                              # Thumb tip is bellow thumb start
            if rightKnuckleDif < rightIndexDif and p2[1][4][1] > p2[1][1][1]:
                if not p2RightPunch:
                    print("P2 RIGHT PUNCH")
                    p2RightPunch = True
                    rightHandCenter = averageCoordinates(p2[1])
                    h1Target = [(h1[2] + h1[0])//2 + w//2, (h1[3] + h1[1]) // 2]
                    h1TargetWidth = (h1[2] - h1[0])//2
                    handRadius = getSize(p2[1])
                                                                        # //2 in order to make it easier
                    if dist(rightHandCenter, h1Target) <= h1TargetWidth - handRadius//2:
                        print("========== TRUE HIT =========")


            else:
                p2RightPunch = False

            # Player 2 Defence Mode ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            leftHandCenter = averageCoordinates(p2[0])
            rightHandCenter = averageCoordinates(p2[1])

            if h2[0] <= leftHandCenter[0] <= h2[2] and h2[1] <= leftHandCenter[1] <= h2[3] and         h2[0] <= rightHandCenter[0] <= h2[2] and h2[1] <= rightHandCenter[1] <= h2[3]:
                p2Defence = True
                cv2.circle(img, (leftHandCenter), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (rightHandCenter), 10, (0, 255, 0), cv2.FILLED)
            else:
                p2Defence = False

        # MEANT FOR 1 PLAYER TESTING
        elif len(lmList) == 2 and len(bothFacePoints) == 1:

            if p1[0][0][0] > p1[1][0][0]:
                p1.reverse()

            leftKnuckleDif = abs(p1[0][17][1] - p1[0][5][1])
            leftIndexDif = abs(p1[0][5][1] - p1[0][6][1])

                                            # Thumb tip is bellow thumb start
            if leftKnuckleDif < leftIndexDif and p1[0][4][1] > p1[0][1][1]:
                if not p1LeftPunch:
                    print("P1 LEFT PUNCH")
                    p1LeftPunch = True
            else:
                p1LeftPunch = False

            rightKnuckleDif = abs(p1[1][17][1] - p1[1][5][1])
            rightIndexDif = abs(p1[1][5][1] - p1[1][6][1])

                                            # Thumb tip is bellow thumb start
            if rightKnuckleDif < rightIndexDif and p1[1][4][1] > p1[1][1][1]:
                if not p1RightPunch:
                    print("P1 RIGHT PUNCH")
                    p1RightPunch = True
            else:
                p1RightPunch = False


            # Defence mode            
            h1 = bothFacePoints[0]
 
            leftHandCenter = averageCoordinates(p1[0])
            rightHandCenter = averageCoordinates(p1[1])

    
            if h1[0] <= leftHandCenter[0] <= h1[2] and h1[1] <= leftHandCenter[1] <= h1[3] and         h1[0] <= rightHandCenter[0] <= h1[2] and h1[1] <= rightHandCenter[1] <= h1[3]:
                defence = True
                cv2.circle(img, (leftHandCenter), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (rightHandCenter), 10, (0, 255, 0), cv2.FILLED)
            else:
                defence = False


        else:
            print("KEEP HANDS TO THEIR SIDES")

    
    new_img = cv2.addWeighted(overlay, 0.5, img, 0.5, 1.0)

    cv2.imshow("Image", new_img)
    cv2.waitKey(1)