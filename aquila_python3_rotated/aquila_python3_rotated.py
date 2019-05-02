import numpy as np
import cv2 as cv
import time



class Aquila:
    def __init__(self):
        ret = False
        iter = -1
        while (not ret):
            iter += 1
            iter %= 16
            self.capture = cv.VideoCapture(iter)
            ret, self.frame = self.capture.read()
            time.sleep(0.001)
        self.old_gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        self.frame_gray = self.old_gray.copy()
        self.goodFeatures_old = cv.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)

    def update_frame(self):
        ret, self.frame = self.capture.read()
        self.old_gray = self.frame_gray.copy()
        self.frame_gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        return ret

    def update_flow(self):
        self.goodFeatures_new, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, self.frame_gray, self.goodFeatures_old, None,**self.lk_params)
        transform = cv.estimateRigidTransform(self.goodFeatures_old, self.goodFeatures_new, False)
        self.goodFeatures_old = cv.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
        if (transform is None):
            print('update_flow error')
            return False
        transform, bias = np.array([[transform[0][0], transform[0][1]],
                          [transform[1][0], transform[1][1]]]), np.array([[transform[0][2]], [transform[1][2]]])
        if (self.object_current[0] != None and self.object_current[1] != None):
            object_current_new = transform.dot(np.array(self.object_current).transpose())
            self.object_current[0], self.object_current[1] = object_current_new[0] + bias[0][0], object_current_new[1] + bias[1][0]
        if (self.object_destination[0] != None and self.object_destination[1] != None):
            object_destination_new = transform.dot(np.array(self.object_destination).transpose())
            self.object_destination[0], self.object_destination[1] = object_destination_new[0] + bias[0][0], object_destination_new[1] + bias[1][0]
        return True


    def global_do(self):
        while (1):
            try:
                self.update_frame()
                self.update_flow()
                try:
                    cv.arrowedLine(self.frame, (int(round(self.object_current[0])), int(round(self.object_current[1]))),
                               (int(round(self.object_destination[0])), int(round(self.object_destination[1]))), (255, 255, 0), 2)
                except:
                    pass
                cv.imshow('frame', self.frame)
            except:
                pass
            finally:
                k = cv.waitKey(1) & 0xff
                if k == 27:
                    break
                if k == ord('r'):
                    self.object_current[0] = 320
                    self.object_current[1] = 240
                    self.object_destination[0] = self.object_current[0] + 100
                    self.object_destination[1] = self.object_current[1] + 100

    def __del__(self):
        self.capture.release()
        cv.destroyAllWindows()

    object_current = [None, None]
    object_destination = [None, None]
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                        maxLevel = 2,
                        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


aquila = Aquila()
aquila.global_do()