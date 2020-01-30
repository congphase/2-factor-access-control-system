import cv2
from datetime import datetime
import rtsp_cam

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
DARK_GREEN = (0, 128, 0)
ORANGE = (0, 140, 255)
YELLOW = (0, 255, 255)
BLACK = (0, 0, 0)

BOTTOM_LEFT_CORNER_OF_TEXT_CONF = (28, 695)
BOTTOM_LEFT_CORNER_OF_TEXT_F = (28, 640)

BOTTOM_LEFT_CORNER_OF_TEXT_CONF_SPREAD_ROI = (19, 250)
BOTTOM_LEFT_CORNER_OF_TEXT_F_SPREAD_ROI = (19, 270)

FONT = cv2.FONT_HERSHEY_PLAIN


def open_window(window_name, width, height, window_title):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.moveWindow(window_name, 0, 0)
    cv2.setWindowTitle(window_name, window_title)


def draw_info(img, bb, conf, frame_id):
    tmp = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), DARK_GREEN, 2)
    tmp = cv2.putText(tmp, str(frame_id), BOTTOM_LEFT_CORNER_OF_TEXT_F, FONT, 4, BLUE, 2, cv2.LINE_AA)
    tmp = cv2.putText(tmp, "{:.3f}".format(conf), BOTTOM_LEFT_CORNER_OF_TEXT_CONF, FONT, 3, YELLOW, 2,
                      cv2.LINE_AA)


def write_img_log(img, log_dir, frame_id):
    cv2.imwrite("{}{}.jpg".format(log_dir, str(frame_id)), img)


def transform_coordinates(spread_roi_coords, spread_roi_bb_coords):
    if max(spread_roi_coords) == 1280:
        return spread_roi_bb_coords

    # (X1, Y1, X2, Y2) =  (left, top, right, bottom) of the full scale coords
    X1 = 0
    Y1 = 0
    X2 = rtsp_cam.WIDTH
    Y2 = rtsp_cam.HEIGHT

    # (x1, y1, x2, y2) = (left, top, right, bottom) of the spread rois
    x1 = spread_roi_coords[0]
    y1 = spread_roi_coords[1]
    x2 = spread_roi_coords[2]
    y2 = spread_roi_coords[3]

    # (m1, n1, m2, n2) = (left, top, right, bottom) of the spread rois' bb
    m1 = spread_roi_bb_coords[0]
    n1 = spread_roi_bb_coords[1]
    m2 = spread_roi_bb_coords[2]
    n2 = spread_roi_bb_coords[3]

    # now the bb in the scale of full-size
    full_scale_coords = (m1 + x1, n1 + y1, m2 + x1, n2 + y1)

    return full_scale_coords

def stick_name(img, bb, name):
    # Draw a label with a name below the face
    cv2.rectangle(img, (bb[0], bb[3]), (bb[2], bb[3] + 20), BLACK, cv2.FILLED)
    cv2.rectangle(img, (bb[0], bb[3]), (bb[2], bb[3] + 20), GREEN, 2)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, name, (bb[0] + 3, bb[3] + 16), font, 0.5, GREEN, 1)
