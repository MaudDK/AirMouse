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
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks,  self.mp_hands.HAND_CONNECTIONS,  
                        self.mp_drawing.DrawingSpec(color=(69, 66, 237), thickness=4, circle_radius=2),  
                        self.mp_drawing.DrawingSpec(color=(79, 187, 91), thickness=2, circle_radius=2)
                        )


                    # for hand in results.multi_handedness:
                    #     handType=hand.classification

                    # for hand_landmarks in results.multi_hand_landmarks:
                    #     self.mp_drawing.draw_landmarks(frame,
                    #                                 hand_landmarks, 
                    #                                 self.mp_hands.HAND_CONNECTIONS, 
                    #                                 self.mp_drawing_styles.get_default_hand_landmarks_style(), 
                    #                                 self.mp_drawing_styles.get_default_hand_connections_style())


                # self.draw_hand(results, frame)

                cv2.imshow('MediaPipe Hands', frame)
                cv2.moveWindow('MediaPipe Hands',0,0)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
            vcap.release()

if __name__ == '__main__':
    airMouse = AirMouse()
    airMouse.start()
