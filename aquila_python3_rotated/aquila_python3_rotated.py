import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
frame = None; frame_gray = None
goodFeatures_old = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
goodFeatures_new = None

c_x = 320
c_y = 240
v_x = 100
v_y = 100

while(1):
    try:
        ret,frame = cap.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        goodFeatures_new, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, goodFeatures_old, None, **lk_params)
        res = cv.estimateRigidTransform(goodFeatures_old, goodFeatures_new, False)

        print('\n', res)

        v_res = np.array([[res[0][0], res[0][1]],
                         [res[1][0], res[1][1]]]).dot(np.array([[v_x], [v_y]]))

        v_x, v_y = v_res[0][0], v_res[1][0]
        c_res = np.array([[res[0][0], res[0][1]],
                         [res[1][0], res[1][1]]]).dot(np.array([[c_x], [c_y]]))
        c_x, c_y = c_res[0][0] + res[0][2], c_res[1][0] + res[1][2]

        cv.arrowedLine(frame, (int(round(c_x)),int(round(c_y))), (int(round(c_x+v_x)),int(round(c_y+v_y))), (255,255,0), 2)
        cv.imshow('frame',frame)
    except:
        pass
    finally:
        old_gray = frame_gray.copy()
        goodFeatures_old = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        k = cv.waitKey(1) & 0xff
        if k == 27:
            break
        if k == ord('r'):
            c_x = 320
            c_y = 240
            v_x = 100
            v_y = 100
cv.destroyAllWindows()
cap.release()