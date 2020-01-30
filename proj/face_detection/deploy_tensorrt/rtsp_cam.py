import cv2

WIDTH = 1280
HEIGHT = 720
URI = "rtsp://assistant:cuacong267Assist@192.168.100.5"
LATENCY = 10

#SPREAD_ROI = (358, 277, 784, 568)
SPREAD_ROI = (0, 0, 1280, 720)

def open_cam_rtsp(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! nvv4l2decoder ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)