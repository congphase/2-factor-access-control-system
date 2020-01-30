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
import logging
import os
import sys
# import dlib
import time
# from time import gmtime, strftime
from datetime import datetime
from typing import Any, Tuple

import cv2
import face_recognition
import numpy
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

import freq_cv
import rtsp_cam
import tnt_info
import api_dirs


####################################
## WHERE THE DEFINITIONS ARE LAID ##
####################################

# Messages
MSG_NHIN_TRUC_TIEP_VAO_CAM = "MSG_HAY_NHIN_TRUC_TIEP_VAO_CAMERA"
MSG_XAC_THUC_KHUON_MAT_THANH_CONG = "MSG_XAC_THUC_KHUON_MAT_THANH_CONG"
MSG_XAC_THUC_KHUON_MAT_KHONG_THANH_CONG = "MSG_XAC_THUC_KHUON_MAT_KHONG_THANH_CONG"
MSG_KHONG_TIM_THAY_KHUON_MAT_NAO = "MSG_KHONG_TIM_THAY_KHUON_MAT_NAO"
MSG_HAY_QUET_LAI_THE_VA_NHIN_THANG_VAO_CAMERA = "MSG_HAY_QUET_LAI_THE_VA_NHIN_THANG_VAO_CAMERA"

# VARIABLES

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


logging.getLogger().setLevel(logging.DEBUG)


def NMS(boxes, overlap_threshold):
    '''

    :param boxes: numpy nx5, n is the number of boxes, 0:4->x1, y1, x2, y2, 4->score
    :param overlap_threshold:
    :return:
    '''
    if boxes.shape[0] == 0:
        return boxes

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype != numpy.float32:
        boxes = boxes.astype(numpy.float32)

    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    sc = boxes[:, 4]
    widths = x2 - x1
    heights = y2 - y1

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = heights * widths
    idxs = numpy.argsort(sc)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # compare secend highest score boxes
        xx1 = numpy.maximum(x1[i], x1[idxs[:last]])
        yy1 = numpy.maximum(y1[i], y1[idxs[:last]])
        xx2 = numpy.minimum(x2[i], x2[idxs[:last]])
        yy2 = numpy.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the box
        w = numpy.maximum(0, xx2 - xx1 + 1)
        h = numpy.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = numpy.delete(idxs, numpy.concatenate(([last], numpy.where(overlap > overlap_threshold)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class Inference_TensorRT:
    def __init__(self, onnx_file_path,
                 receptive_field_list,
                 receptive_field_stride,
                 bbox_small_list,
                 bbox_large_list,
                 receptive_field_center_start,
                 num_output_scales):

        temp_trt_file = os.path.join('trt_file_cache/', os.path.basename(onnx_file_path).replace('.onnx', '.trt'))

        load_trt_flag = False
        if not os.path.exists(temp_trt_file):
            if not os.path.exists(onnx_file_path):
                logging.error('ONNX file does not exist!')
                sys.exit(1)
            logging.info('Init engine from ONNX file.')
        else:
            load_trt_flag = True
            logging.info('Init engine from serialized engine.')

        self.receptive_field_list = receptive_field_list
        self.receptive_field_stride = receptive_field_stride
        self.bbox_small_list = bbox_small_list
        self.bbox_large_list = bbox_large_list
        self.receptive_field_center_start = receptive_field_center_start
        self.num_output_scales = num_output_scales
        self.constant = [i / 2.0 for i in self.receptive_field_list]

        # init log
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        self.engine = None
        if load_trt_flag:
            with open(temp_trt_file, 'rb') as fin, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(fin.read())
        else:
            # declare builder object
            logging.info('Create TensorRT builder.')
            builder = trt.Builder(TRT_LOGGER)

            # get network object via builder
            logging.info('Create TensorRT network.')
            network = builder.create_network()

            # create ONNX parser object
            logging.info('Create TensorRT ONNX parser.')
            parser = trt.OnnxParser(network, TRT_LOGGER)

            with open(onnx_file_path, 'rb') as onnx_fin:
                parser.parse(onnx_fin.read())

            # print possible errors
            num_error = parser.num_errors
            if num_error != 0:
                logging.error('Errors occur while parsing the ONNX file!')
                for i in range(num_error):
                    temp_error = parser.get_error(i)
                    print(temp_error.desc())
                sys.exit(1)

            # create engine via builder
            builder.max_batch_size = 1
            builder.average_find_iterations = 2
            logging.info('Create TensorRT engine...')
            engine = builder.build_cuda_engine(network)

            # serialize engine
            if not os.path.exists('trt_file_cache/'):
                os.makedirs('trt_file_cache/')
            logging.info('Serialize the engine for fast init.')
            with open(os.path.join('trt_file_cache/', os.path.basename(onnx_file_path).replace('.onnx', '.trt')), 'wb') as fout:
                fout.write(engine.serialize())
            self.engine = engine

        self.output_shapes = []
        self.input_shapes = []
        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                self.input_shapes.append(tuple([self.engine.max_batch_size] + list(self.engine.get_binding_shape(binding))))
            else:
                self.output_shapes.append(tuple([self.engine.max_batch_size] + list(self.engine.get_binding_shape(binding))))
        if len(self.input_shapes) != 1:
            logging.error('Only one input data is supported.')
            sys.exit(1)
        self.input_shape = self.input_shapes[0]
        logging.info('The required input size: %d, %d, %d' % (self.input_shape[2], self.input_shape[3], self.input_shape[1]))

        # create executor
        self.executor = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = self.__allocate_buffers(self.engine)

    def __allocate_buffers(self, engine):
        inputs = []
        outputs = []
        bindings = []
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings

    def do_inference(self, image, score_threshold=0.4, top_k=10000, NMS_threshold=0.4, NMS_flag=True, skip_scale_branch_list=[]):

        if image.ndim != 3 or image.shape[2] != 3:
            print('Only RGB images are supported.')
            return None
        input_height = self.input_shape[2]
        input_width = self.input_shape[3]
        if image.shape[0] != input_height or image.shape[1] != input_width:
            logging.info('The size of input image is not %dx%d.\nThe input image will be resized keeping the aspect ratio.' % (input_height, input_width))

        input_batch = numpy.zeros((1, input_height, input_width, self.input_shape[1]), dtype=numpy.float32)
        left_pad = 0
        top_pad = 0
        if image.shape[0] / image.shape[1] > input_height / input_width:
            resize_scale = input_height / image.shape[0]
            input_image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)
            left_pad = int((input_width - input_image.shape[1]) / 2)
            input_batch[0, :, left_pad:left_pad + input_image.shape[1], :] = input_image
        else:
            resize_scale = input_width / image.shape[1]
            input_image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)
            top_pad = int((input_height - input_image.shape[0]) / 2)
            input_batch[0, top_pad:top_pad + input_image.shape[0], :, :] = input_image

        input_batch = input_batch.transpose([0, 3, 1, 2])
        input_batch = numpy.array(input_batch, dtype=numpy.float32, order='C')
        self.inputs[0].host = input_batch

        [cuda.memcpy_htod(inp.device, inp.host) for inp in self.inputs]
        self.executor.execute(batch_size=self.engine.max_batch_size, bindings=self.bindings)
        [cuda.memcpy_dtoh(output.host, output.device) for output in self.outputs]
        outputs = [out.host for out in self.outputs]
        outputs = [numpy.squeeze(output.reshape(shape)) for output, shape in zip(outputs, self.output_shapes)]

        bbox_collection = []
        for i in range(self.num_output_scales):
            if i in skip_scale_branch_list:
                continue

            score_map = numpy.squeeze(outputs[i * 2])

            # show feature maps-------------------------------
            # score_map_show = score_map * 255
            # score_map_show[score_map_show < 0] = 0
            # score_map_show[score_map_show > 255] = 255
            # cv2.imshow('score_map' + str(i), cv2.resize(score_map_show.astype(dtype=numpy.uint8), (0, 0), fx=2, fy=2))
            # cv2.waitKey()

            bbox_map = numpy.squeeze(outputs[i * 2 + 1])

            RF_center_Xs = numpy.array([self.receptive_field_center_start[i] + self.receptive_field_stride[i] * x for x in range(score_map.shape[1])])
            RF_center_Xs_mat = numpy.tile(RF_center_Xs, [score_map.shape[0], 1])
            RF_center_Ys = numpy.array([self.receptive_field_center_start[i] + self.receptive_field_stride[i] * y for y in range(score_map.shape[0])])
            RF_center_Ys_mat = numpy.tile(RF_center_Ys, [score_map.shape[1], 1]).T

            x_lt_mat = RF_center_Xs_mat - bbox_map[0, :, :] * self.constant[i]
            y_lt_mat = RF_center_Ys_mat - bbox_map[1, :, :] * self.constant[i]
            x_rb_mat = RF_center_Xs_mat - bbox_map[2, :, :] * self.constant[i]
            y_rb_mat = RF_center_Ys_mat - bbox_map[3, :, :] * self.constant[i]

            x_lt_mat = x_lt_mat
            x_lt_mat[x_lt_mat < 0] = 0
            y_lt_mat = y_lt_mat
            y_lt_mat[y_lt_mat < 0] = 0
            x_rb_mat = x_rb_mat
            x_rb_mat[x_rb_mat > input_width] = input_width
            y_rb_mat = y_rb_mat
            y_rb_mat[y_rb_mat > input_height] = input_height

            select_index = numpy.where(score_map > score_threshold)
            for idx in range(select_index[0].size):
                bbox_collection.append((x_lt_mat[select_index[0][idx], select_index[1][idx]] - left_pad,
                                        y_lt_mat[select_index[0][idx], select_index[1][idx]] - top_pad,
                                        x_rb_mat[select_index[0][idx], select_index[1][idx]] - left_pad,
                                        y_rb_mat[select_index[0][idx], select_index[1][idx]] - top_pad,
                                        score_map[select_index[0][idx], select_index[1][idx]]))

        # NMS
        bbox_collection = sorted(bbox_collection, key=lambda item: item[-1], reverse=True)
        if len(bbox_collection) > top_k:
            bbox_collection = bbox_collection[0:top_k]
        bbox_collection_numpy = numpy.array(bbox_collection, dtype=numpy.float32)
        bbox_collection_numpy = bbox_collection_numpy / resize_scale

        if NMS_flag:
            final_bboxes = NMS(bbox_collection_numpy, NMS_threshold)
            final_bboxes_ = []
            for i in range(final_bboxes.shape[0]):
                final_bboxes_.append((final_bboxes[i, 0], final_bboxes[i, 1], final_bboxes[i, 2], final_bboxes[i, 3], final_bboxes[i, 4]))

            return final_bboxes_
        else:
            return bbox_collection_numpy


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2

    line = cv2.line
    ellipse = cv2.ellipse

    # Top left
    line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def run_inference(video_in, video_out, anchor_conf, current_time):
    """
    :param video_in: input video needed to be processed, or, rtsp video stream feed
    :param video_out: output video of the process, used for re-checking
    :param anchor_conf: anchor confidence score to decide wether to do recognition
    :param current_time: 'd' or 'n'
    :return: not yet known
    """

    # Initialize some frequently called methods to reduce time
    get_face_encodings = face_recognition.face_encodings
    compare_faces = face_recognition.compare_faces
    get_face_distance = face_recognition.face_distance

    imshow = cv2.imshow
    imwrite = cv2.imwrite
    rectangle = cv2.rectangle
    cvtColor = cv2.cvtColor
    COLOR_BGR2RGB = cv2.COLOR_BGR2RGB
    COLOR_RGB2BGR = cv2.COLOR_RGB2BGR

    get_time = time.time
    GREEN = freq_cv.GREEN
    transform_coordinates = freq_cv.transform_coordinates
    get_smpl_encs = tnt_info.get_smpl_encs

    log_info = logging.info
    log_debug = logging.debug
    log_warning = logging.warning
    log_error = logging.error

    '''
    ********************************************************************************************************************
        VIDEO FILE/VIDEO STREAM HANDLING 
    ********************************************************************************************************************
    '''

    # Initialize video stuff
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    input_movie = cv2.VideoCapture(video_in)
    output_movie = cv2.VideoWriter(video_out, fourcc, 24, (rtsp_cam.WIDTH, rtsp_cam.HEIGHT))

    movie_read = input_movie.read
    movie_isOpened = input_movie.isOpened
    movie_write = output_movie.write


    '''
    ********************************************************************************************************************
        DETECTION MODEL STUFF INITIALIZATION
    ********************************************************************************************************************
    '''
    import sys
    sys.path.append('..')
    from config_farm import configuration_10_320_20L_5scales_v2 as cfg
    #from config_farm import configuration_10_560_25L_8scales_v1 as cfg

    onnx_file_path = './onnx_files/v2.onnx'
    myInference = Inference_TensorRT(
        onnx_file_path=onnx_file_path,
        receptive_field_list=cfg.param_receptive_field_list,
        receptive_field_stride=cfg.param_receptive_field_stride,
        bbox_small_list=cfg.param_bbox_small_list,
        bbox_large_list=cfg.param_bbox_large_list,
        receptive_field_center_start=cfg.param_receptive_field_center_start,
        num_output_scales=cfg.param_num_output_scales)

    do_inference = myInference.do_inference

    process_this_frame = True
    
    # Handling the video
    while movie_isOpened():
        ret, frame = movie_read()
        # Bail out when the video file ends
        if not ret:
            log_info('video file ends')
            break

        # Only process every other frame of video to save time
        if process_this_frame:
            tic = get_time()
            bboxes = do_inference(frame, score_threshold=0.6, top_k=1000, NMS_threshold=0.2, NMS_flag=True)
            toc = get_time()

            log_debug(f'\ndetection takes: {(toc-tic):.3f}s')
            log_debug(f'len(bboxes) = {len(bboxes)}')
            log_debug(f'bboxes looks like: {bboxes}')

            ###############################################
            # NO FACE DETECTED
            ###############################################
            if len(bboxes) == 0:
                log_info('detected: 0 face')

                imshow('Video', frame)
                movie_write(frame)
                continue
            ###############################################
            # ONLY ONE FACE DETECTED
            ###############################################
            elif len(bboxes) == 1:
                log_info('detected: 1 face')

                # get the (only) face location of this frame
                bbox = bboxes[0]
                bb_conf = float(bbox[4])
                bb_left = int(bbox[0])
                bb_top = int(bbox[1])
                bb_right = int(bbox[2])
                bb_bottom = int(bbox[3])

                log_debug(f'(conf | left, top, right, bottom) = ({bb_conf:.2f}, '
                          f'{bb_left}, {bb_top}, {bb_right}, {bb_bottom})')

                # if detected bb has confidence lower than anchor confidence, don't do recognition
                if bb_conf < anchor_conf:
                    draw_border(frame, (bb_left, bb_top), (bb_right, bb_bottom), GREEN, 2, 5, 10)
                    imshow('Video', frame)
                    movie_write(frame)

                    log_info('confidence is low. Skipping ... ')
                    continue
                log_debug(f'confidence is high: {bb_conf}')

                # go get the proposed candidate id
                file = open(r"/home/gate/lffd-dir/Pyro-test/id_from_client_js.txt", 'r+')
                candidate_id = file.readline().strip()
                log_debug(f'candidate id passed: {candidate_id}, type: {type(candidate_id)}')
                if candidate_id == "":
                    continue
                candidate_id = int(candidate_id)
                log_debug(f"he's {tnt_info.tnt_name_tup[candidate_id]}")
                # write to "reset" the file
                file.write("")
                
                draw_border(frame, (bb_left, bb_top), (bb_right, bb_bottom), GREEN, 2, 5, 10)

                # confidence meets the requirement for doing recognition
                # css_type is needed for face_recognition.face_encodings
                css_type_face_location = [(bb_top, bb_right, bb_bottom, bb_left)]

                log_debug(f'css_type_face_location: {css_type_face_location}')

                # conversion to RGB is needed for face_recognition
                frame = cvtColor(frame, COLOR_BGR2RGB)

                # get encoding for the face detected
                tic = get_time()
                face_encoding = get_face_encodings(frame, css_type_face_location, 1)[0]
                toc = get_time()

                log_debug(f'calculating encoding takes: {(toc-tic):.4f}s')

                candidate_known_encs = get_smpl_encs(candidate_id, current_time)

                tic = get_time()
                matches = compare_faces(candidate_known_encs, face_encoding, 0.5)
                toc = get_time()

                log_debug(f'comparing encodings takes: {(toc-tic):.4f}s')
                log_debug(f'matches: {matches}')
                name = "Unknown"

                # If num of matches is over 50%, then it's it
                log_debug(f'True/total: {matches.count(True)}/{len(matches)}')

                if matches.count(True) > (len(matches) / 2):
                    print(MSG_XAC_THUC_KHUON_MAT_THANH_CONG)
                    name = tnt_info.tnt_name_tup[candidate_id].split()[-1]
                
                # Draw a box around the face
                draw_border(frame, (bb_left, bb_top), (bb_right, bb_bottom), GREEN, 2, 5, 10)
                freq_cv.stick_name(frame, (bb_left, bb_top, bb_right, bb_bottom), name)

                # Display the resulting image
                frame = cvtColor(frame, COLOR_RGB2BGR)
                imshow('Video', frame)
                movie_write(frame)

                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break

            ###############################################
            # MULTIPLE FACES DETECTED
            ###############################################
            elif 1 < len(bboxes):
                #################
                # left-most face
                #################
                log_info('detected: > 1 face')

                min_x1_idx, min_x1 = (0, 1280)
                all_other_face_locations = []
                for i, bbox in enumerate(bboxes):
                    log_debug(f'#{i}: {bbox}')
                    tmp_tuple = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                    all_other_face_locations.append(tmp_tuple)
                    if min_x1 > bbox[0]:
                        min_x1 = bbox[0]
                        min_x1_idx = i
                    draw_border(frame, (tmp_tuple[0], tmp_tuple[1]), (tmp_tuple[2], tmp_tuple[3]), GREEN, 2, 5, 10)

                # first, pop the left-most face location out of the list, later used
                all_other_face_locations.pop(min_x1_idx)

                # start to process the left-most: get its info
                bbox = bboxes.pop(min_x1_idx)
                bb_conf = float(bbox[4])
                bb_left = int(bbox[0])
                bb_top = int(bbox[1])
                bb_right = int(bbox[2])
                bb_bottom = int(bbox[3])

                # decide if take this bb to get face encs
                if anchor_conf > bb_conf:
                    imshow('Video', frame)
                    movie_write(frame)
                    log_info('confidence is low. Skipping ... ')
                    continue

                imshow('Video', frame)
                log_debug(f'confidence is high: {bb_conf}')

                # confidence meets the requirement for doing recognition
                css_type_face_location = [(bb_top, bb_right, bb_bottom, bb_left)]

                log_debug(f'css_type_face_location: {css_type_face_location}')

                # conversion to RGB is needed for face_recognition
                frame = cvtColor(frame, COLOR_BGR2RGB)

                # get encoding for the face detected
                tic = get_time()
                face_encoding = get_face_encodings(frame, css_type_face_location, 1)[0]
                toc = get_time()
                log_debug(f'calculating encoding takes: {(toc - tic):.4f}s')

                # go get the proposed candidate id
                file = open(r"/home/gate/lffd-dir/Pyro-test/id_from_client_js.txt", 'r+')
                candidate_id = file.readline().strip()
                print(f'candidate_id passed: {candidate_id}, type: {type(candidate_id)}')
                if candidate_id == "":
                    continue
                candidate_id = int(candidate_id)
                print(f"he's {tnt_info.tnt_name_tup[candidate_id]}")
                # write to "reset" the file
                file.write("")

                candidate_known_encs = get_smpl_encs(candidate_id, current_time)

                tic = get_time()
                matches = compare_faces(candidate_known_encs, face_encoding, 0.5)
                toc = get_time()
                log_debug(f'comparing encodings takes: {(toc - tic):.4f}s')
                log_debug(f'matches: {matches}')

                # If num of matches is over 50%, then it's it
                log_debug(f'True/total: {matches.count(True)}/{len(matches)}')

                if matches.count(True) > (len(matches) / 2):
                    print(MSG_XAC_THUC_KHUON_MAT_THANH_CONG)
                    name = tnt_info.tnt_name_tup[candidate_id].split()[-1]
                    log_info(f'Xac thuc khuon mat thanh cong: (id: {candidate_id}; name: {name})')

                    # for debug purpose
                    frame = cvtColor(frame, COLOR_RGB2BGR)
                    imshow('left-most', frame[bb_top:bb_bottom, bb_left:bb_right])
                    BOTTOM_LEFT_CORNER_OF_TEXT_F = (28, 640)
                    FONT = cv2.FONT_HERSHEY_PLAIN
                    frame = cv2.putText(frame, name, BOTTOM_LEFT_CORNER_OF_TEXT_F, FONT, 4, GREEN, 2, cv2.LINE_AA)
                    movie_write(frame)
                else:
                    print(MSG_XAC_THUC_KHUON_MAT_KHONG_THANH_CONG)
                    print(MSG_HAY_QUET_LAI_THE_VA_NHIN_THANG_VAO_CAMERA)
                    name = "Unknown"
                    log_info(f'Xac thuc khuon mat khong thanh cong: (id: {candidate_id})')

                    # for debug purpose
                    frame = cvtColor(frame, COLOR_RGB2BGR)
                    imshow('left-most', frame[bb_top:bb_bottom, bb_left:bb_right])
                    BOTTOM_LEFT_CORNER_OF_TEXT_F = (28, 640)
                    FONT = cv2.FONT_HERSHEY_PLAIN
                    frame = cv2.putText(frame, name, BOTTOM_LEFT_CORNER_OF_TEXT_F, FONT, 4, GREEN, 2, cv2.LINE_AA)
                    movie_write(frame)
                    #continue

                ##############################
                # all_other_faces
                ##############################
                for i, x in enumerate(all_other_face_locations):
                    imwrite(f'{i}_{x[1]}.jpg', frame[x[1]:x[3], x[0]:x[2]])

                known_face_encodings = numpy.load(f'{api_dirs.tnt_smpl_embs_dir}list_of_kfencs.npy')
                known_face_names = tnt_info.tnt_name_tup

                for i, bbox in enumerate(bboxes):
                    face_encodings = get_face_encodings(frame, all_other_face_locations, 0)

                log_debug(f'face_encodings of all other people:\n{face_encodings}')
                all_other_face_names = []

                for face_encoding in face_encodings:
                    name = "Unknown"
                    face_distances = get_face_distance(known_face_encodings, face_encoding)
                    log_debug(f'face_distances:\n{face_distances}')
                    best_match_index = numpy.argmin(face_distances)
                    if face_distances[best_match_index] < 0.7:
                        name = known_face_names[best_match_index]
                    all_other_face_names.append(name)
                for i, name_ in enumerate(all_other_face_names):
                    print(f'[  info] aofn #{i}: {name_}')

                # log text
                #log_info()
                # log image
                #imwrite(api_dirs.log_imgs_dir, frame)



            else:
                log_error('num of faces detected < 0')
                exit()

        if max(frame.shape[:2]) > 1440:
            scale = 1440 / max(frame.shape[:2])
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        cv2.imshow('Video', frame)
        cv2.waitKey(10)

        process_this_frame = not process_this_frame


    '''
    ********************************************************************************************************************
        VERIFICATION
    ********************************************************************************************************************
    '''



@profile
def main():
    args = parse_args()
    print('Called with args:')
    print(args)

    # get current time when this script is called
    now = datetime.now()
    today_night_time = now.replace(hour=18, minute=0, second=0, microsecond=0)
    print("[log  ] now: {}".format(now))
    if now < today_night_time:
        current_time = 'd'
    else:
        current_time = 'n'

    # freq_cv.open_window(CAP_WINDOW_NAME, rtsp_cam.WIDTH, rtsp_cam.HEIGHT, "Captured")

    run_inference(args.video_in, args.video_out, 1.7, current_time)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
