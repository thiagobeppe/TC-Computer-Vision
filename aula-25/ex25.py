import cv2
import numpy as np

def draw_rect(event,x,y,flags,params):
    global x_init, y_init, drawing, top_left_pt, bottom_right_pt

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_init, y_init = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            top_left_pt = (min(x_init,x), min(y_init,y))
            bottom_right_pt = (max(x_init,x), max(y_init,y))
            img[y_init:y, x_init:x] = 255 - img[y_init:y,x_init:x]

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        top_left_pt = (min(x_init,x), min(y_init,y))
        bottom_right_pt = (max(x_init,x), max(y_init,y))
        img[y_init:y, x_init:x] = 255 - img[y_init:y,x_init:x]

if __name__== '__main__':
    drawing = False
    top_left_pt,bottom_right_pt = (-1,-1),(-1,-1)

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Cam")
    cv2.setMouseCallback("Cam", draw_rect)


    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    
    while True:
        ret,frame = cap.read()
        img = cv2.resize(frame,None,fx=0.5,fy=0.5, interpolation=cv2.INTER_AREA)
        (x0,y0), (x1,y1) = top_left_pt, bottom_right_pt
        img[y0:y1,x0:x1] = 255 - img[y0:y1,x0:x1]

        
        # ret, old_frame = cap.read()
        # old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        # p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        # mask = np.zeros_like(old_frame)

        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # # calculate optical flow
        # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # # Select good points
        # good_new = p1[st==1]
        # good_old = p0[st==1]

        # # draw the tracks
        # for i,(new,old) in enumerate(zip(good_new,good_old)):
        #     a,b = new.ravel()
        #     c,d = old.ravel()
        #     mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        #     frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        # img = cv2.add(frame,mask)
        cv2.imshow("Cam",img)

        

        c = cv2.waitKey(1)
        if c == 27:
            break
        # old_gray = frame_gray.copy()
        # p0 = good_new.reshape(-1,1,2)

    cap.release()
    cv2.destroyAllWindows()