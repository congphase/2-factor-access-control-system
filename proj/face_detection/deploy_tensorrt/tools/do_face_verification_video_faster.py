#################################################################
######################### RELEASE NOTES #########################
#################################################################
'''
 - Process order:
    RFID scanned --> This script run --> Verify the scanned RFID number against current face
        This order, compared to face-first_RFID-later or face-only, provides a more balanced approach,
        which is a compromise between demanding actual use and the complex generalization of the algorithm
 - How to run:
    python DO_FACE_VERIFICATION.py --video_in <> --video_out <> --candidate <>

 - SPREAD_ROI is used, detection speed is reduced to around 90ms/frame, but sacrificing competency
 -
'''

import argparse
import sys
import dlib
import time
import cv2
import os
import numpy as np

import face_recognition

import api_dirs
import freq_cv
import rtsp_cam
import tnt_info

# from time import gmtime, strftime
from datetime import datetime
from datetime import date

####################################
## WHERE THE DEFINITIONS ARE LAID ##
####################################

# ROI FOR REDUCING COMPUTATION
SPREAD_ROI = rtsp_cam.SPREAD_ROI

# Messages
MSG_NHIN_TRUC_TIEP_VAO_CAM = "MSG_HAY_NHIN_TRUC_TIEP_VAO_CAMERA"
MSG_XAC_THUC_KHUON_MAT_THANH_CONG = "MSG_XAC_THUC_KHUON_MAT_THANH_CONG"
MSG_XAC_THUC_KHUON_MAT_KHONG_THANH_CONG = "MSG_XAC_THUC_KHUON_MAT_KHONG_THANH_CONG"
MSG_KHONG_TIM_THAY_KHUON_MAT_NAO = "MSG_KHONG_TIM_THAY_KHUON_MAT_NAO"
MSG_HAY_QUET_LAI_THE_VA_NHIN_THANG_VAO_CAMERA = "MSG_HAY_QUET_LAI_THE_VA_NHIN_THANG_VAO_CAMERA"

# VARIABLES

# The cnn version of flib face detection
cnn_detector = dlib.cnn_face_detection_model_v1(api_dirs.face_detection_model_cnn)

# Window names
CAP_WINDOW_NAME = 'CameraDemo'
VERIF_WINDOW_NAME = 'Verification Window'


def parse_args():
    # Parse input arguments
    desc = 'Do face verification on video'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--video_in", dest='video_in', required=True,
                        help='Video input')
    parser.add_argument("--video_out", dest='video_out', required=True,
                        help='Video output')
    parser.add_argument("--candidate_id", dest='candidate_id', required=True,
                        type=int,
                        help="Candidate index: [1, 2, 3, ...]")
    args = parser.parse_args()
    return args


import cProfile, pstats, io


def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner

@profile
def run_inference(video_in, video_out, candidate_id, current_time):
    """
    :param video_in: input video needed to be processed, or, rtsp video stream feed
    :param video_out: output video of the process, used for re-checking
    :param current_time: 'd' or 'n', which is day or night -- the time this script is run
    :return: not yet known
    """

    # Initialize some frequently called methods to reduce time
    get_face_encodings = face_recognition.face_encodings
    compare_faces = face_recognition.compare_faces
    green = freq_cv.GREEN
    transform_coordinates = freq_cv.transform_coordinates
    cvtColor = cv2.cvtColor
    imshow = cv2.imshow
    COLOR_BGR2RGB = cv2.COLOR_BGR2RGB
    COLOR_RGB2BGR = cv2.COLOR_RGB2BGR

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    candidate_known_face_encodings = tnt_info.get_smpl_encs(candidate_id, current_time)
    # known_face_encodings =

    '''
    ********************************************************************************************************************
        VIDEO FILE/VIDEO STREAM HANDLING 
    ********************************************************************************************************************
    '''
    # Initialize video stuff
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    input_movie = cv2.VideoCapture(video_in)
    output_movie = cv2.VideoWriter(video_out, fourcc, 29.97, (rtsp_cam.WIDTH, rtsp_cam.HEIGHT))

    # Handling the video
    while input_movie.isOpened():
        ret, frame = input_movie.read()
        # Bail out when the video file ends
        if not ret:
            print("[log  ] Video file ends")
            break

        # Convert the frame to RGB
        frame = cvtColor(frame, COLOR_BGR2RGB)
        frame_spread_roi = frame[SPREAD_ROI[1]:SPREAD_ROI[3], SPREAD_ROI[0]:SPREAD_ROI[2]]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video (USING DLIB)
            start = time.time()
            face_locations = cnn_detector(frame_spread_roi, 0)
            end = time.time()
            print("[debug] detection time: {:.3f}s".format(end - start))

            if len(face_locations) == 0:
                print("[log  ] detected: 0 face")
                frame = cvtColor(frame, COLOR_RGB2BGR)
                imshow('Video', frame)
                output_movie.write(frame)
                continue

            elif len(face_locations) == 1:
                print("[log  ] detected: 1 face")
                # get the (only) face location of this frame
                face_location = face_locations[0]
                bb_conf = face_location.confidence
                bb_left = face_location.rect.left()
                bb_top = face_location.rect.top()
                bb_right = face_location.rect.right()
                bb_bottom = face_location.rect.bottom()

                print("[debug] (conf | left, top, right, bottom) = ({:.3f} | {}, {}, {}, {})".format(
                    bb_conf, bb_left, bb_top, bb_right, bb_bottom))

                if bb_conf < 0.6:
                    frame = cvtColor(frame, COLOR_RGB2BGR)
                    spread_roi_bb_coords = (bb_left, bb_top, bb_right, bb_bottom)
                    # Convert the coordinates and update the bb values
                    bb_left, bb_top, bb_right, bb_bottom = transform_coordinates(SPREAD_ROI,
                                                                                         spread_roi_bb_coords)
                    # Draw a box around the face
                    cv2.rectangle(frame, (bb_left, bb_top), (bb_right, bb_bottom), green, 2)
                    imshow('Video', frame)
                    output_movie.write(frame)
                    print("[log  ] Confidence is low. Skipping ...")
                    continue

                # css_type is needed for face_recognition.face_encodings
                css_type_face_location = [(bb_top, bb_right, bb_bottom, bb_left)]

                # get encoding
                start = time.time()
                face_encoding = get_face_encodings(frame_spread_roi, css_type_face_location, 0)[0]
                end = time.time()
                print("[debug] ec takes: {:.4f}s".format(end - start))

                # See if the face is a match for the known face(s)
                start = time.time()
                matches = compare_faces(candidate_known_face_encodings, face_encoding, 0.5)
                end = time.time()
                print("[debug] e compare takes: {:.4f}s".format(end - start))
                print("[debug] matches: {}".format(matches))
                name = "Unknown"

                # If num of matches is over 50%, then it's it
                print("[debug] True/total = {}/{}".format(matches.count(True), len(matches)))
                if matches.count(True) > (len(matches) / 2):
                    print("[debug] that's it")
                    name = tnt_info.tnt_name_tup[candidate_id].split()[-1]

                # Or instead, use the known face with the smallest distance to the new face
                # face_distances = face_recognition.face_distance(candidate_known_face_encodings, face_encoding)
                # best_match_index = np.argmin(face_distances)
                # if matches[best_match_index]:
                #     name = known_face_names[best_match_index]

                spread_roi_bb_coords = (bb_left, bb_top, bb_right, bb_bottom)
                # Convert the coordinates and update the bb values
                bb_left, bb_top, bb_right, bb_bottom = transform_coordinates(SPREAD_ROI, spread_roi_bb_coords)

                # Convert to BGR
                frame = cvtColor(frame, COLOR_RGB2BGR)
                # Draw a box around the face
                cv2.rectangle(frame, (bb_left, bb_top), (bb_right, bb_bottom), green, 2)
                freq_cv.stick_name(frame, (bb_left, bb_top, bb_right, bb_bottom), name)

                # Display the resulting image
                imshow('Video', frame)
                output_movie.write(frame)

                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break

            else:
                print("[log  ] detected: {} face".format(len(face_locations)))


        process_this_frame = not process_this_frame

    '''
    ********************************************************************************************************************
        DETECTION
    ********************************************************************************************************************
    '''

    '''
    ********************************************************************************************************************
        VERIFICATION
    ********************************************************************************************************************
    '''

def main():
    args = parse_args()
    print('Called with args:')
    print(args)
    print('OpenCV version: {}\n'.format(cv2.__version__))

    # assign the variables
    candidate_id = args.candidate_id

    # get current time when this script is called
    now = datetime.now()
    today_night_time = now.replace(hour=18, minute=0, second=0, microsecond=0)
    print("[log  ] now: {}".format(now))
    if now < today_night_time:
        current_time = 'd'
    else:
        current_time = 'n'

    # freq_cv.open_window(CAP_WINDOW_NAME, rtsp_cam.WIDTH, rtsp_cam.HEIGHT, "Captured")

    run_inference(args.video_in, args.video_out, candidate_id, current_time)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
