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
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
frame_gray = 0
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

c_x = 320
c_y = 240
v_x = 100
v_y = 100
while(1):
    try:
        ret,frame = cap.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        res = cv.estimateRigidTransform(p1, p0, False)
        print('\n', res)
        # draw the tracks
        img = cv.add(frame,mask)

        mat_a = np.array([[res[0][0], res[0][1]],
                         [res[1][0], res[1][1]]])
        mat_b = np.array([v_x, v_y])
        mat_res = np.linalg.solve(mat_a, mat_b)
        v_x, v_y = mat_res[0], mat_res[1]

        mat_b = np.array([c_x, c_y])
        mat_res = np.linalg.solve(mat_a, mat_b)

        c_x = mat_res[0] - res[0][2]
        c_y = mat_res[1] - res[1][2]

        cv.arrowedLine(img, (int(round(c_x)),int(round(c_y))), (int(round(c_x+v_x)),int(round(c_y+v_y))), (255,255,0), 2)
        cv.imshow('frame',img)
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
    except:
        old_gray = frame_gray.copy()
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        mask = np.zeros_like(old_frame)
    finally:
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