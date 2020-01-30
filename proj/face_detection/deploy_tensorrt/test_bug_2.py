import cv2
import face_recognition
import dlib

import pycuda.autoinit
from pycuda.tools import clear_context_caches

test_img = cv2.imread("/home/gate/lffd-dir/A-Light-and-Fast-Face-Detector-for-Edge-Devices/face_detection/deploy_tensorrt/check_453.jpg")
# cnn_detector = dlib.cnn_face_detection_model_v1("/home/gate/Downloads/mmod_human_face_detector.dat")
test_img_dlib = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
# face_locations = cnn_detector(test_img_dlib, 0)

# face_location = face_locations[0]
# bb_left = face_location.rect.left()
# bb_top = face_location.rect.top()
# bb_right = face_location.rect.right()
# bb_bottom = face_location.rect.bottom()

# css_type_face_location = [(bb_top, bb_right, bb_bottom, bb_left)]

# pycuda.autoinit.context.detach()
pycuda.autoinit.context.pop()
pycuda.tools.clear_context_caches()
# pycuda.autoinit.activate_context_2()
# face_encoding = face_recognition.face_encodings(test_img_dlib, css_type_face_location, 0)[0]
# pycuda.autoinit.deactivate_context_2()
# pycuda.autoinit.context.push()

print(f'Result:\n{face_encoding}')