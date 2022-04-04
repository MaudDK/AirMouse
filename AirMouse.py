import mediapipe as mp
import cv2
from pynput.mouse import Controller
import ctypes


class AirMouse():
    def __init__(self):
        #Mouse Controls
        self.mouse = Controller()

        #Media Pipe Utils
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)

        #Screen Size
        user32 = ctypes.windll.user32
        self.screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

    def moveMouseTo(self, newx, newy):
        currentX, currentY = self.mouse.position
        dispX =  newx-currentX
        dispY = newy - currentY
        self.mouse.move(dispX, dispY)

    def predict(self, frame, model):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return results, frame

    def draw_hand(self, results, frame):
        if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x = round(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * self.screensize[0])
                    y = round(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * self.screensize[1])
                    self.moveMouseTo(x,y)
                        
                    self.mp_drawing.draw_landmarks(frame,
                                                    hand_landmarks, 
                                                    self.mp_hands.HAND_CONNECTIONS, 
                                                    self.mp_drawing_styles.get_default_hand_landmarks_style(), 
                                                    self.mp_drawing_styles.get_default_hand_connections_style())
        return frame

    def handleHand(self):
        vcap = cv2.VideoCapture(0)
        with self.hands as hands:
            while vcap.isOpened():
                response, frame = vcap.read()

                if not response:
                    break

                #Process
                results, frame = self.predict(frame, hands)
                self.draw_hand(results, frame)
                frame = cv2.resize(frame, self.screensize)
                cv2.imshow('MediaPipe Hands', frame)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
            vcap.release()

if __name__ == '__main__':
    cam = AirMouse()
    cam.handleHand()
