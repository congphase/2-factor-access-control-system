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
 A-Light-and-Fast-Face-Detector-for-Edge-Devices
 -
'''

import argparse
import logging
import os
import sys
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
import tegra_cam


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


def parse_args():
    # Parse input arguments
    desc = 'Capture and display live camera video on Jetson TX2/TX1'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--rtsp', dest='use_rtsp',
                        help='use IP CAM (remember to also set --uri)',
                        action='store_true')
    parser.add_argument('--uri', dest='rtsp_uri',
                        help='RTSP URI, e.g. rtsp://192.168.1.64:554',
                        default=None, type=str)
    parser.add_argument('--latency', dest='rtsp_latency',
                        help='latency in ms for RTSP [200]',
                        default=200, type=int)
    parser.add_argument('--usb', dest='use_usb',
                        help='use USB webcam (remember to also set --vid)',
                        action='store_true')
    parser.add_argument('--vid', dest='video_dev',
                        help='device # of USB webcam (/dev/video?) [1]',
                        default=1, type=int)
    parser.add_argument('--width', dest='image_width',
                        help='image width [1920]',
                        default=1920, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [1080]',
                        default=1080, type=int)
    args = parser.parse_args()
    return args


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


def run_inference(cap, anchor_conf, current_time):
    """
    :param cap: input video needed to be processed, or, rtsp/usb/onboard camera feed
    :param video_out: output video of the process, used for re-checking
    :param anchor_conf: anchor confidence score to decide wether to do recognition
    :param current_time: 'd' or 'n'
    :return: not yet known
    """

    # Initialize some frequently called methods to reduce time
    get_face_encodings = face_recognition.face_encodings
    compare_faces = face_recognition.compare_faces
    get_face_distance = face_recognition.face_distance

    line = cv2.line
    imshow = cv2.imshow
    imwrite = cv2.imwrite
    WINDOW_NAME = tegra_cam.CAP_WINDOW_NAME
    rectangle = cv2.rectangle
    cvtColor = cv2.cvtColor
    COLOR_BGR2RGB = cv2.COLOR_BGR2RGB
    COLOR_RGB2BGR = cv2.COLOR_RGB2BGR

    get_time = time.time
    GREEN = freq_cv.GREEN
    YELLOW = freq_cv.YELLOW
    transform_coordinates = freq_cv.transform_coordinates
    get_smpl_encs = tnt_info.get_smpl_encs

    pop_context = pycuda.autoinit.context.pop
    push_context = pycuda.autoinit.context.push

    face_root_dir = api_dirs.face_dir

    '''
    ********************************************************************************************************************
        VIDEO FILE/VIDEO STREAM HANDLING 
    ********************************************************************************************************************
    '''


    '''
    ********************************************************************************************************************
        DETECTION MODEL STUFF INITIALIZATION
    ********************************************************************************************************************
    '''
    import sys
    sys.path.append('..')
    from config_farm import configuration_10_320_20L_5scales_v2 as cfg
    #from config_farm import configuration_10_560_25L_8scales_v1 as cfg

    #onnx_file_path = './onnx_files/v2_smallest.onnx'
    onnx_file_path = './onnx_files/v2_small.onnx'
    #onnx_file_path = './onnx_files/v1.onnx'
    myInference = Inference_TensorRT(
        onnx_file_path=onnx_file_path,
        receptive_field_list=cfg.param_receptive_field_list,
        receptive_field_stride=cfg.param_receptive_field_stride,
        bbox_small_list=cfg.param_bbox_small_list,
        bbox_large_list=cfg.param_bbox_large_list,
        receptive_field_center_start=cfg.param_receptive_field_center_start,
        num_output_scales=cfg.param_num_output_scales)

    do_inference = myInference.do_inference

#    process_this_frame = True
    
    # Handling the camera feed
    while True:
        ret, frame = cap.read()
        # Bail out when the camera feed ends
        if not ret:
            print('[   info] camera feed ends')
            break
        
        imshow(WINDOW_NAME, frame)

        # Only process every other frame of video to save time
#    if process_this_frame:
        # first check if id is passed before any face is detected
        file_ = open(r"/home/gate/lffd-dir/id_from_client_js.txt", 'r+')
        candidate_id = file_.readline().strip()
        if candidate_id == "":
            print(f'[  debug] id not passed yet\n')
        else:
            candidate_id = int(candidate_id) - 1
            candidate_name = tnt_info.tnt_name_tup[candidate_id]
            print(f"[  debug] id passed: {candidate_id}, it's {candidate_name}")
            # write to "reset" the file
            file_.seek(0)
            file_.truncate()
            file_.write("")
            file_.close()

        tic = get_time()
        bboxes = do_inference(frame, score_threshold=0.6, top_k=1000, NMS_threshold=0.2, NMS_flag=True)
        toc = get_time()

        print(f'\n[  debug] detection takes: {(toc-tic):.3f}s')
        #print(f'[  debug] len(bboxes) = {len(bboxes)}')
        #print(f'[  debug] bboxes looks like: {bboxes}')

        ###############################################
        # NO FACE DETECTED
        ###############################################
        if len(bboxes) == 0:
            #imshow(WINDOW_NAME, frame)
            print('[   info] detected: 0 face')
            if isinstance(candidate_id, int) and candidate_id >= 0:
                file_ = open(r"/home/gate/lffd-dir/msg_buffer.txt", 'w')
                file_.write(MSG_HAY_QUET_LAI_THE_VA_NHIN_THANG_VAO_CAMERA)
                file_.close()
            
        ###############################################
        # ONLY ONE FACE DETECTED
        ###############################################
        elif len(bboxes) == 1:
            print('[   info] detected: 1 face')

            # get the (only) face location of this frame
            bbox = bboxes[0]
            bb_conf = float(bbox[4])
            bb_left = int(bbox[0])
            bb_top = int(bbox[1])
            bb_right = int(bbox[2])
            bb_bottom = int(bbox[3])

            print(f'[  debug] (c | l, t, r, b) = ({bb_conf:.2f}, '
                        f'{bb_left}, {bb_top}, {bb_right}, {bb_bottom})')

            #draw_border(frame, (bb_left, bb_top), (bb_right, bb_bottom), GREEN, 2, 5, 10)
            rectangle(frame, (bb_left, bb_top), (bb_right, bb_bottom), GREEN, 2)
            #imshow(WINDOW_NAME, frame)

            # if detected bb has confidence lower than anchor confidence, don't do recognition
            if bb_conf < anchor_conf:
                ##movie_write(frame)
                print('[   info] confidence is low. Skipping ... ')
                if type(candidate_id) is int:
                    file_ = open(r"/home/gate/lffd-dir/msg_buffer.txt", 'w')
                    file_.write(MSG_XAC_THUC_KHUON_MAT_KHONG_THANH_CONG)
                    file_.close()
                continue

            print(f'[  debug] confidence is high: {bb_conf:.2f}')
            if type(candidate_id) is int:
                print("[  debug] got id, starting verification")

                # css_type is needed for face_recognition.face_encodings
                css_type_face_location = [(bb_top, bb_right, bb_bottom, bb_left)]

                print(f'[  debug] css_type_face_location: {css_type_face_location}')

                # conversion to RGB is needed for face_recognition
                frame = cvtColor(frame, COLOR_BGR2RGB)

                pop_context()
                # get encoding for the face detected
                tic = get_time()
                face_encoding = get_face_encodings(frame, css_type_face_location, 1)[0]
                toc = get_time()
                print(f'[  debug] calculating encoding takes: {(toc-tic):.4f}s')
                push_context()

                candidate_known_encs = get_smpl_encs(candidate_id, current_time)

                tic = get_time()
                matches = compare_faces(candidate_known_encs, face_encoding, 0.5)
                toc = get_time()

                # Showing verification window
                face_left = frame[bb_top:bb_bottom, bb_left:bb_right]
                person_dir = os.path.join(face_root_dir, str(candidate_id))
                
                for file_ in os.listdir(person_dir):
                    template_face = imread(os.path.join(person_dir, file_), cv2.IMREAD_COLOR)

                print(f'[  debug] comparing encodings takes: {(toc-tic):.4f}s')
                print(f'[  debug] matches: {matches}')

                # If num of matches is over 50%, then it's it
                print(f'[  debug] True/total: {matches.count(True)}/{len(matches)}')

                
                if matches.count(True) > (len(matches) / 2):
                    file_ = open(r"/home/gate/lffd-dir/msg_buffer.txt", 'w')
                    file_.write(MSG_XAC_THUC_KHUON_MAT_THANH_CONG)
                    file_.close()
                else:
                    candidate_name = "Unknown"
                    file_ = open(r"/home/gate/lffd-dir/msg_buffer.txt", 'w')
                    file_.write(MSG_XAC_THUC_KHUON_MAT_KHONG_THANH_CONG)
                    file_.close()

                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break

        ###############################################
        # MULTIPLE FACES DETECTED
        ###############################################
        elif 1 < len(bboxes):
            print('[   info] detected: > 1 face')

            min_x1_idx, min_x1 = (0, 1280)
            css_type_all_other_face_locations = []
            for i, bbox in enumerate(bboxes):
                print(f'[  debug] #{i}: {bbox}')
                tmp_tuple = (int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[0]))
                css_type_all_other_face_locations.append(tmp_tuple)
                if min_x1 > bbox[0]:
                    min_x1 = bbox[0]
                    min_x1_idx = i
                #draw_border(frame, (tmp_tuple[3], tmp_tuple[0]), (tmp_tuple[1], tmp_tuple[2]), GREEN, 2, 5, 10)
                rectangle(frame, (tmp_tuple[3], tmp_tuple[0]), (tmp_tuple[1], tmp_tuple[2]), GREEN, 2)
            #imshow(WINDOW_NAME, frame)

            #################
            # left-most face
            #################
            # first, pop the left-most face location out of the list, later used
            css_type_all_other_face_locations.pop(min_x1_idx)

            # start to process the left-most: get its info
            bbox = bboxes.pop(min_x1_idx)
            bb_conf = float(bbox[4])
            bb_left = int(bbox[0])
            bb_top = int(bbox[1])
            bb_right = int(bbox[2])
            bb_bottom = int(bbox[3])

            # decide if take this bb to get face encs
            if anchor_conf > bb_conf:
                #movie_write(frame)
                print('[   info] confidence is low. Skipping ... ')
                continue

            print(f'[  debug] confidence is high: {bb_conf}')

            if type(candidate_id) is int:
                # confidence meets the requirement for doing recognition
                css_type_face_location = [(bb_top, bb_right, bb_bottom, bb_left)]

                print(f'[  debug] css_type_face_location: {css_type_face_location}')

                # conversion to RGB is needed for face_recognition
                frame = cvtColor(frame, COLOR_BGR2RGB)

                pop_context()
                # get encoding for the face detected
                tic = get_time()
                face_encoding = get_face_encodings(frame, css_type_face_location, 1)[0]
                toc = get_time()
                print(f'[  debug] calculating encoding takes: {(toc - tic):.4f}s')
                push_context()

                candidate_known_encs = get_smpl_encs(candidate_id, current_time)

                tic = get_time()
                matches = compare_faces(candidate_known_encs, face_encoding, 0.5)
                toc = get_time()
                print(f'[  debug] comparing encodings takes: {(toc - tic):.4f}s')
                print(f'[  debug] matches: {matches}')

                print(f'[  debug] True/total: {matches.count(True)}/{len(matches)}')
                # If num of matches is over 50%, then it's it
                if matches.count(True) > (len(matches) / 2):
                    file_ = open(r"/home/gate/lffd-dir/msg_buffer.txt", 'w')
                    file_.write(MSG_XAC_THUC_KHUON_MAT_THANH_CONG)
                    file_.close()
                    print(f'[   info] Xac thuc khuon mat thanh cong: (id: {candidate_id}; name: {candidate_name})')

                    # for debug purpose
                    frame = cvtColor(frame, COLOR_RGB2BGR)
                    imshow('left-most', frame[bb_top:bb_bottom, bb_left:bb_right])
                    #movie_write(frame)
                else:
                    file_ = open(r"/home/gate/lffd-dir/msg_buffer.txt", 'w')
                    file_.write(MSG_XAC_THUC_KHUON_MAT_KHONG_THANH_CONG)
                    file_.close()
                    candidate_id = "Unknown"
                    print(f'[   info] Xac thuc khuon mat khong thanh cong: (id: {candidate_id})')

                    # for debug purpose
                    frame = cvtColor(frame, COLOR_RGB2BGR)
                    imshow('left-most', frame[bb_top:bb_bottom, bb_left:bb_right])
                    #movie_write(frame)
                    #continue

            ##############################
            # all_other_faces
            ##############################
            #for i, x in enumerate(css_type_all_other_face_locations):
            #    imwrite(f'{i}_{x[1]}.jpg', frame[x[1]:x[3], x[0]:x[2]])

            known_face_encodings = numpy.load(f'{api_dirs.tnt_smpl_embs_dir}list_of_kfencs.npy')
            known_face_names = tnt_info.tnt_name_tup

            pop_context()
            # get encodings for all other face locations
            face_encodings = get_face_encodings(frame, css_type_all_other_face_locations, 0)
            push_context()

            print(f'[  debug] face_encodings of all other people:\n{face_encodings}')
            all_other_face_names = []

            for face_encoding in face_encodings:
                name = "Unknown"
                face_distances = get_face_distance(known_face_encodings, face_encoding)
                print(f'[  debug] face_distances:\n{face_distances}')
                best_match_index = numpy.argmin(face_distances)
                if face_distances[best_match_index] < 0.7:
                    name = known_face_names[best_match_index]
                all_other_face_names.append(name)
            for i, name_ in enumerate(all_other_face_names):
                print(f'[  info] aofn #{i}: {name_}')


        else:
            print('[  error] num of faces detected < 0')
            exit()
        
        candidate_id = ""

        if max(frame.shape[:2]) > 1440:
            scale = 1440 / max(frame.shape[:2])
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(10)

#        process_this_frame = not process_this_frame

@profile
def main():
    args = parse_args()
    print('Called with args:')
    print(args)
    
    if args.use_rtsp:
        cap = tegra_cam.open_cam_rtsp(args.rtsp_uri,
                            args.image_width,
                            args.image_height,
                            args.rtsp_latency)
    elif args.use_usb:
        cap = tegra_cam.open_cam_usb(args.video_dev,
                           args.image_width,
                           args.image_height)
    else: # by default, use the Jetson onboard camera
        cap = tegra_cam.open_cam_onboard(args.image_width,
                               args.image_height)

    if not cap.isOpened():
        sys.exit('Failed to open camera!')

    tegra_cam.open_window(args.image_width, args.image_height, tegra_cam.CAP_WINDOW_NAME, 'Camera feed')
    tegra_cam.open_window(640, 360, tegra_cam.VERIF_WINDOW_NAME, 'Verification Window')

    # get current time when this script is called
    now = datetime.now()
    today_night_time = now.replace(hour=18, minute=0, second=0, microsecond=0)
    print("[log  ] now: {}".format(now))
    if now < today_night_time:
        current_time = 'd'
    else:
        current_time = 'n'


    run_inference(cap, 1.7, current_time)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
