import time
import mediapipe as mp
import cv2
import math
from pynput.mouse import Button, Controller
import ctypes
import numpy as np


class AirMouse():
    def __init__(self):
        #Mouse Controls
        self.mouse = Controller()
        self.isClicked = False
        self.isReleased = True

        #Media Pipe Utils
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        #Media Pipe Model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        #Screen Size
        user32 = ctypes.windll.user32
        self.screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    

    def start(self):
        #Set Video Capture to Webcam 0
        vcap = cv2.VideoCapture(0)

        with self.hands as hands:
            #Start video capture
            while vcap.isOpened():
                response, frame = vcap.read()
                #if frame errors breaks
                if not response:
                    break

                #Get Frame Results
                results, frame = self.process(frame, hands)

                #Landmark Parsing
                if results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        #Determine Handedness
                        handedness = self.get_handedness(idx, results)

                        #Extract Coordinates
                        self.right_coords, self.left_coords = self.extract_coordinates(hand_landmarks.landmark, handedness, frame)

                        #Draw Landmarks on frame
                        self.drawHands(frame, hand_landmarks, handedness, 'MIDDLE_FINGER_MCP')
                    
                    if self.right_coords:
                        #Get Middle MCP Coord
                        x, y = self.right_coords['MIDDLE_FINGER_TIP']
                        
                        #Move mouse based on middle finger mcp
                        self.moveMouseRelative(x * self.screensize[0], y * self.screensize[1])

                        #Get index and thumb coords
                        xidx, yidx = self.right_coords['INDEX_FINGER_TIP']
                        xthmb, ythmb = self.right_coords['THUMB_TIP']

                        #Handles Click when indx and thumb are close
                        self.handleLeftClick(xthmb * self.screensize[0], ythmb * self.screensize[1], xidx * self.screensize[0], yidx * self.screensize[1])
                    

                #Display Window
                cv2.imshow('MediaPipe Hands', cv2.resize(frame, (frame.shape[1], frame.shape[0])))
                cv2.moveWindow('MediaPipe Hands',0,0)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

            vcap.release()
    
    def process(self, frame, model):
        #Fix for detecting right hand as left vice versa
        frame = cv2.flip(frame, 1)

        #BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Detect Results
        results = model.process(frame)

        #Set Writeable Flag
        frame.flags.writeable = True

        #RGB 2 BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return results, frame

    def get_handedness(self, index, results):
        #Hand classification results
        hands = results.multi_handedness

        #Gets whether hand is left or right
        label = hands[index].classification[0].label

        #Gets confidence score of hand
        # score = hands[index].classification[0].score

        #Returns hand left or right
        return label

    def extract_coordinates(self, landmarks, handedness, frame):

        #Dictionary containing conversions of mediapipe id to keyword
        id_dict = {
        0:'WRIST', 
        1:'THUMB_CMC', 
        2:'THUMB_MCP', 
        3: 'THUMB_IP', 
        4: 'THUMB_TIP', 
        5: 'INDEX_FINGER_MCP', 
        6: 'INDEX_FINGER_PIP', 
        7: 'INDEX_FINGER_DIP', 
        8: 'INDEX_FINGER_TIP', 
        9: 'MIDDLE_FINGER_MCP', 
        10: 'MIDDLE_FINGER_PIP', 
        11: 'MIDDLE_FINGER_DIP', 
        12: 'MIDDLE_FINGER_TIP', 
        13: 'RING_FINGER_MCP', 
        14: 'RING_FINGER_PIP', 
        15: 'RING_FINGER_DIP', 
        16: 'RING_FINGER_TIP', 
        17: 'PINKY_MCP', 
        18: 'PINKY_PIP', 
        19: 'PINKY_DIP', 
        20: 'PINKY_TIP'}

        #Empty dicts that will be filled with left hand landmarks and right hand landmarks
        right_coord_dict = dict()
        left_coord_dict = dict()


        #Adds coords to the respective dictionary
        for key in id_dict:
            if handedness == 'Right':
                right_coord_dict[id_dict[key]] = landmarks[key].x, landmarks[key].y
            elif handedness == 'Left':
                left_coord_dict[id_dict[key]] = landmarks[key].x, landmarks[key].y
        
        #Returns both dictionary
        return right_coord_dict, left_coord_dict

    def drawHands(self, frame, hand_landmarks, handedness, marker = 'CENTER'):

        #Draw Hand Landmarks
        self.mp_drawing.draw_landmarks(
            frame, hand_landmarks,  self.mp_hands.HAND_CONNECTIONS,  
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=6, circle_radius=1),  
            self.mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=2, circle_radius=1)
        )

        #Extract Coordinates for displaying handedness marker
        if self.right_coords:
            if marker == 'CENTER': markX, markY = self.getHandCenter(self.right_coords)
            else: markX, markY = self.right_coords[marker]

        if self.left_coords:
            if marker == 'CENTER': markX, markY = self.getHandCenter(self.left_coords)
            else: markX, markY = self.left_coords[marker]

        #Display Handedness marker on extracted coordinates
        if handedness:
            cv2.putText(frame, handedness, (int(markX * frame.shape[1]), int(markY * frame.shape[0])) , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (106, 206, 141), 2, cv2.LINE_AA)
    
    def getHandCenter(self, hand_coord_dict):
        #Calculates Geometric Median of markers
        markers = ['WRIST', 'THUMB_CMC', 'INDEX_FINGER_MCP', 'PINKY_MCP']
        x_s = [hand_coord_dict[marker][0] for marker in markers]
        y_s = [hand_coord_dict[marker][1] for marker in markers]

        xlogs = np.log(x_s)
        ylogs = np.log(y_s)

        x_gmedian = np.exp(xlogs.mean())
        y_gmedian = np.exp(ylogs.mean())

        if x_gmedian == np.nan or y_gmedian == np.nan:
            return 0, 0

        #Returns coords of approximate center of the given markers
        return int(x_gmedian), int(y_gmedian)

    def moveMouseRelative(self, newx, newy):
        #Get current Mouse Position
        prevx, prevy = self.mouse.position

        #Calculate change in x and y
        dx =  newx - prevx
        dy =  newy - prevy

        #Calculate distance
        distance = round(math.hypot(dx, dy))

        #Smoothing mouse movement
        if distance > 10:
            self.mouse.move(dx, dy)
    
    def handleLeftClick(self, thumb_x, thumb_y, index_x, index_y):
        #Calculate change in x and y
        dx = thumb_x - index_x
        dy = thumb_y - index_y

        #Calculate distance
        distance = round(math.hypot(dx, dy))


        #Determine click bounds and click if acceptable
        if distance <= 50 and not self.isClicked:
            print(f'Clicked')
            self.mouse.press(Button.left)
            self.isClicked = True
            self.isReleased = False

        if distance >= 70 and not self.isReleased:
            print(f'Released')
            self.mouse.release(Button.left)
            self.isReleased = True
            self.isClicked = False

if __name__ == '__main__':
    airMouse = AirMouse()
    airMouse.start()
