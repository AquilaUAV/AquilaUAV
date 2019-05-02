import numpy as np
import cv2 as cv
import time

from math import *


class Aquila:
    def __init__(self):
        ret = False
        iter = -1
        while (not ret):
            iter += 1
            iter %= 16
            self.capture = cv.VideoCapture(iter)
            self.capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv.CAP_PROP_FPS, 60)
            ret, self.frame = self.capture.read()
            time.sleep(0.001)
        self.frame_height, self.frame_width, channels = self.frame.shape
        self.old_gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        self.frame_gray = self.old_gray.copy()
        self.goodFeatures_old = cv.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)

    def update_frame(self):
        ret, self.frame = self.capture.read()
        self.old_gray = self.frame_gray.copy()
        self.frame_gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        return ret

    def update_flow(self):
        if (self.goodFeatures_old is None):
            self.__log_info('goodFeatures_old is None')
            self.goodFeatures_old = cv.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
            return False
        self.goodFeatures_new, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, self.frame_gray, self.goodFeatures_old, None,**self.lk_params)
        transform = cv.estimateRigidTransform(self.goodFeatures_old, self.goodFeatures_new, False)
        self.goodFeatures_old = cv.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
        if (transform is None):
            self.__log_info('update_flow error')
            return False
        transform, bias = np.array([[transform[0][0], transform[0][1]],
                          [transform[1][0], transform[1][1]]]), np.array([[transform[0][2]], [transform[1][2]]])
        if (self.object_current != None):
            object_current_new = transform.dot(np.array(self.object_current).transpose())
            self.object_current[0], self.object_current[1] = object_current_new[0] + bias[0][0], object_current_new[1] + bias[1][0]
        if (self.object_destination != None):
            object_destination_new = transform.dot(np.array(self.object_destination).transpose())
            self.object_destination[0], self.object_destination[1] = object_destination_new[0] + bias[0][0], object_destination_new[1] + bias[1][0]
        return True

    def update_object_list(self):
        ret, thresh = cv.threshold(self.frame_gray, self.threshold_value, 255, 0)
        cv.imshow('frame2', thresh)
        frame2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        self.object_list = self.recognize_rects(contours, (1000, 10000), (1.5, 4.5), (0.6, 1.6), 0.05, 2)

    def control_nearest_rect(self):
        nearest_rect = self.find_nearest_rect((self.frame_width // 2, self.frame_height // 2))
        if (nearest_rect == None):
            return None
        return (nearest_rect[0][0] - self.frame_width // 2,
                self.frame_height // 2 - nearest_rect[0][1],
                nearest_rect[2] % 180 - 90)

    def control_object_current(self):
        if (self.object_current == None):
            return None
        nearest_rect = self.find_nearest_rect((self.object_current[0], self.object_current[1]))
        if (nearest_rect == None):
            return None
        return (nearest_rect[0][0] - self.frame_width // 2,
                self.frame_height // 2 - nearest_rect[0][1],
                nearest_rect[2] % 180 - 90)

    def update_objects(self):
        if (not self.object_current is None):
            nearest_rect = self.find_nearest_rect((self.object_current[0],self.object_current[1]))
            if (nearest_rect != None):
                self.object_current = [nearest_rect[0][0], nearest_rect[0][1]]
        if (not self.object_destination is None):
            nearest_rect = self.find_nearest_rect((self.object_destination[0],self.object_destination[1]))
            if (nearest_rect != None):
                self.object_destination = [nearest_rect[0][0], nearest_rect[0][1]]

    def init_object_current(self):
        nearest_rect = self.find_nearest_rect((self.frame_width // 2, self.frame_height // 2))
        if (nearest_rect == None):
            return False
        self.object_current = [nearest_rect[0][0], nearest_rect[0][1]]
        return True

    def init_object_destination(self, distance):
        if (self.object_current == None):
            return False
        nearest_rect = self.find_nearest_rect((self.frame_width // 2, self.frame_height // 2 - distance))
        if (nearest_rect != None):
            self.object_destination = [nearest_rect[0][0], nearest_rect[0][1]]
        return False

    def bad_init_object_destination(self):
        nearest_rect = self.find_nearest_rect((self.frame_width // 2, self.frame_height // 2))
        if (nearest_rect == None):
            return False
        self.object_destination = [nearest_rect[0][0], nearest_rect[0][1]]
        return True

    def __log_info(self, message):
        print('[ {:.6f} ] '.format(time.time()), message, sep='')
    def global_do(self):

        while (1):
            try:
                self.update_frame()
                self.update_flow()
                self.update_object_list()
                self.update_objects()
                self.control_object_current()
                self.__drawAllRect()
                try:
                    cv.circle(self.frame, ((int(round(self.object_current[0])), int(round(self.object_current[1])))),
                               5, (255, 127, 0), 2)
                except:
                    pass
                try:
                    cv.arrowedLine(self.frame, (int(round(self.object_current[0])), int(round(self.object_current[1]))),
                               (int(round(self.object_destination[0])), int(round(self.object_destination[1]))), (255, 255, 0), 2)
                except:
                    pass
                cv.imshow('frame', self.frame)
            except Exception as ex:
                print(ex)
            finally:
                k = cv.waitKey(1) & 0xff
                if k == 27:
                    break
                if k == ord('1'):
                    self.init_object_current()
                if k == ord('3'):
                    self.bad_init_object_destination()
                if k == ord('2'):
                    self.init_object_destination(100)


    def __getCorrectListRect(self, rect):
        new_rect = []
        new_rect.append(list(rect[0]))
        new_rect.append(list(rect[1]))
        new_rect.append(rect[2])
        if (new_rect[1][1] > new_rect[1][0]):
            new_rect[1][1], new_rect[1][0] = new_rect[1][0], new_rect[1][1]
            new_rect[2] = (new_rect[2] + 90) % 360
        return new_rect

    def __getCorrectTupleRect(self, rect):
        new_rect = []
        new_rect.append(tuple(rect[0]))
        new_rect.append(tuple(rect[1]))
        new_rect.append(rect[2])
        return tuple(new_rect)

    def __drawRect(self, rect, color):
        rect = self.__getCorrectTupleRect(rect)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(self.frame, [box], 0, color, 1)

    def __drawAllRect(self):
        if (self.object_list.__len__() == 0):
            return
        nearest_rect = None
        for rect in self.object_list:
            self.__drawRect(rect, (0, 255, 0))
            if ((nearest_rect is None)
                    or (
                            ((self.frame_width // 2 - rect[0][0]) ** 2 + (self.frame_height // 2 - rect[0][1]) ** 2) <
                            ((self.frame_width // 2 - nearest_rect[0][0]) ** 2 + (self.frame_height // 2 - nearest_rect[0][1]) ** 2)
                    )):
                nearest_rect = rect
        self.__drawRect(nearest_rect, (0, 0, 255))


    def find_nearest_rect(self, point):
        if (self.object_list.__len__() == 0):
            return None
        nearest_rect = None
        for rect in self.object_list:
            if ((nearest_rect is None)
                    or (
                            ((point[0] - rect[0][0]) ** 2 + (point[1] - rect[0][1]) ** 2) <
                            ((point[0] - nearest_rect[0][0]) ** 2 + (point[1] - nearest_rect[0][1]) ** 2)
                    )):
                nearest_rect = rect
        return nearest_rect

    def recognize_rects(self, contours, area, aspect_ratio, filling, power, multiplication):
        rectList = []
        area_solve = np.linalg.solve(np.array([[area[1] ** 2, (0.5 * (area[1] + area[0])) ** 2, area[0] ** 2],
                      [area[1], (0.5 * (area[1] + area[0])), area[0]],
                      [1, 1, 1]]).transpose(), np.array([0, 1, 0]))
        aspect_ratio_solve = np.linalg.solve(np.array([[aspect_ratio[1] ** 2, (0.5 * (aspect_ratio[1] + aspect_ratio[0])) ** 2, aspect_ratio[0] ** 2],
                      [aspect_ratio[1], (0.5 * (aspect_ratio[1] + aspect_ratio[0])), aspect_ratio[0]],
                      [1, 1, 1]]).transpose(), np.array([0, 1, 0]))
        filling_solve = np.linalg.solve(np.array([[filling[1] ** 2, (0.5 * (filling[1] + filling[0])) ** 2, filling[0] ** 2],
                      [filling[1], (0.5 * (filling[1] + filling[0])), filling[0]],
                      [1, 1, 1]]).transpose(), np.array([0, 1, 0]))
        for cnt in contours:
            contour_area = cv.contourArea(cnt)
            if (contour_area < area[0]):
                continue
            contribution_rate = 1.0
            rect = cv.minAreaRect(cnt)
            rect_area = rect[1][0] * rect[1][1]
            rect_aspect_ratio = max(rect[1][0], rect[1][1]) / min(rect[1][0], rect[1][1])
            rect_filling = contour_area/rect_area
            area_rate = area_solve[0] * rect_area * rect_area + area_solve[1] * rect_area + area_solve[2]
            if (area_rate < 0.0):
                continue
            aspect_ratio_rate = aspect_ratio_solve[0] * rect_aspect_ratio * rect_aspect_ratio + aspect_ratio_solve[1] * rect_aspect_ratio + aspect_ratio_solve[2]
            if (aspect_ratio_rate < 0.0):
                continue
            filling_rate = filling_solve[0] * rect_filling * rect_filling + filling_solve[1] * rect_filling + filling_solve[2]
            if (filling_rate < 0.0):
                continue
            contribution_rate = pow(contribution_rate*area_rate*aspect_ratio_rate*filling_rate, power)
            rect = self.__getCorrectListRect(rect)
            rect[1][0] *= contribution_rate*multiplication
            rect[1][1] *= contribution_rate*multiplication
            rect = self.__getCorrectTupleRect(rect)
            rectList.append(rect)
        return rectList


    def __del__(self):
        self.capture.release()
        cv.destroyAllWindows()

    frame_height, frame_width = None, None
    object_list = None
    object_current = None
    object_destination = None
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize  = (15,15),
                        maxLevel = 2,
                        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    threshold_value = 90


aquila = Aquila()
aquila.global_do()