from turtle import left
import mediapipe as mp
import cv2
import math
from pynput.mouse import Button, Controller
import ctypes


class AirMouse():
    def __init__(self, maxHands = 2):
        #Mouse Controls
        self.mouse = Controller()
        self.isClicked = False
        self.isReleased = True

        #Media Pipe Utils
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands


        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        #Screen Size
        user32 = ctypes.windll.user32
        self.screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    
    def handleClick(self, thumb_x, thumb_y, palm_x, palm_y):
        dx = thumb_x - palm_x
        dy = thumb_y - palm_y

        distance = round(math.hypot(dx, dy))


        if distance <= 50 and not self.isClicked:
            print(f'ThumbPalm Distance:{distance}, Clicked')
            self.mouse.press(Button.left)
            self.isClicked = True
            self.isReleased = False

        if distance >= 70 and not self.isReleased:
            print(f'ThumbPalm Distance:{distance}, Released')
            self.mouse.release(Button.left)
            self.isReleased = True
            self.isClicked = False

    def moveMouseRelative(self, newx, newy):
        prevx, prevy = self.mouse.position
        newx = self.screensize[0] - newx
        dx =  newx - prevx
        dy =  newy - prevy

        distance = round(math.hypot(dx, dy))

        # print(f'hand:({newx,newy}) mouse:({prevx, prevy}) |dx: {dx} | dy: {dy} | distance: {distance}')
        if distance > 10 * self.sens:
            print(f"Mouse Move {dx}, {dy} ")
            self.mouse.move(dx,dy)

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

    def draw_hand(self, results, frame):
        if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    xindex = round(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * self.screensize[0])
                    yindex = round(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * self.screensize[1])

                    xthumb = round(hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x * self.screensize[0])
                    ythumb = round(hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y * self.screensize[1])

                    xindex_palm = round(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].x * self.screensize[0])
                    yindex_palm = round(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y * self.screensize[1])

                    self.moveMouseRelative(xindex*self.sens, yindex *self.sens)
                    self.handleClick(xthumb, ythumb, xindex_palm, yindex_palm)
                        
                    self.mp_drawing.draw_landmarks(frame,
                                                    hand_landmarks, 
                                                    self.mp_hands.HAND_CONNECTIONS, 
                                                    self.mp_drawing_styles.get_default_hand_landmarks_style(), 
                                                    self.mp_drawing_styles.get_default_hand_connections_style())
        return frame

    def start(self):
        vcap = cv2.VideoCapture(0)

        with self.hands as hands:
            #Start video capture
            while vcap.isOpened():
                response, frame = vcap.read()
                if not response:
                    break

                #Get Frame Results
                results, frame = self.process(frame, hands)

                #Draw Logic
                if results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        #Determine Handedness
                        handedness = self.get_handedness(idx, results)

                        #Extract Coordinates
                        self.right_coords, self.left_coords = self.extract_coordinates(hand_landmarks.landmark, handedness, frame)

                        #Draw Landmarks
                        self.drawHands(frame, hand_landmarks, handedness, 'WRIST')


                cv2.imshow('MediaPipe Hands', cv2.resize(frame, (int(frame.shape[1]/1.5), int(frame.shape[0]/1.5))))
                cv2.moveWindow('MediaPipe Hands',0,0)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

            vcap.release()

    def drawHands(self, frame, hand_landmarks, handedness, marker = 'WRIST'):

        #Draw Hand Landmarks
        self.mp_drawing.draw_landmarks(
            frame, hand_landmarks,  self.mp_hands.HAND_CONNECTIONS,  
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=6, circle_radius=1),  
            self.mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=2, circle_radius=1)
        )

        #Extract Coordinates for displaying handedness marker
        if self.right_coords: wrist_land = self.right_coords[marker]
        if self.left_coords: wrist_land = self.left_coords[marker]

        #Display Handedness marker on extracted coordinates
        if handedness:
            cv2.putText(frame, handedness, wrist_land, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (106, 206, 141), 2, cv2.LINE_AA)

    def get_handedness(self, index, results):
        hands = results.multi_handedness
        label = hands[index].classification[0].label
        score = hands[index].classification[0].score
        return label

    def extract_coordinates(self, landmarks, handedness, frame):
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

        right_coord_dict = dict()
        left_coord_dict = dict()

        for key in id_dict:
            if handedness == 'Right':
                right_coord_dict[id_dict[key]] = (int(landmarks[key].x * frame.shape[1]), int(landmarks[key].y * frame.shape[0]))
            elif handedness == 'Left':
                left_coord_dict[id_dict[key]] = (int(landmarks[key].x * frame.shape[1]), int(landmarks[key].y * frame.shape[0]))
            

        return right_coord_dict, left_coord_dict



if __name__ == '__main__':
    airMouse = AirMouse()
    airMouse.start()
