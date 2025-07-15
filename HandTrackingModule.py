import cv2 
import mediapipe as mp
import time
class handDetector():
    def __init__(self,mode=False,maxhands=2,model_complx=1,detection_con=0.5
                 ,tracking_con=0.5):
        self.mode=mode
        self.maxhands=maxhands
        self.model_complx=model_complx
        self.detection_con=detection_con
        self.tracking_con=tracking_con
        self.mpHands=mp.solutions.hands 
        self.mpDraw=mp.solutions.drawing_utils           
        self.hands=self.mpHands.Hands(self.mode,self.maxhands,self.model_complx,self.detection_con,self.tracking_con)
    def findHands(self,frame,draw=True):
        imgRgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRgb)
        #print(self.results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame,hand,self.mpHands.HAND_CONNECTIONS)
                
        return frame
    def findPOSI(self,frame,handNO=0,draw=True):
        lmList=[]
        if self.results.multi_hand_landmarks:
            myhand=self.results.multi_hand_landmarks[handNO]
            for id,lm in enumerate(myhand.landmark):
                h,w,c=frame.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                #print("id:-",id,cx,cy)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(frame,(cx,cy),15,(255,0,255),-1)
        return lmList        
 ## 39 MIN

def main():
    prev_time=0
    curr_time=0
    cap=cv2.VideoCapture(0)
    detector= handDetector()

    while True:
        ret,frame=cap.read()
        frame=detector.findHands(frame,True)
        lmList=detector.findPOSI(frame)
        if len(lmList)!=0:
            print(lmList[4])


        curr_time=time.time()
        fps=1/(curr_time-prev_time)
        prev_time=curr_time

        cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow('frame',frame)

   
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ =="__main__":
   main()