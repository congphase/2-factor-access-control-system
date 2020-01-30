# consider changing if(len(face_locations) == 1): in some line when refering to this script
# to create official script of full pipeline, because this script is designed to deal with
# images that contain one face only (for embedding vector extraction)

import numpy
import cv2
import dlib
import face_recognition

import sys
import time
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import api_dirs
import freq_cv
import tnt_info
import logging

#######################
## DECLARE CONSTANTS ##
#######################
BATCH_SIZE = 1

ENCODINGS_DIR = "/home/gate/gate-resources/on-field-resource/tnt_smpl_embs/"
CHECK_BY_BB_DIR = "/home/gate/gate-resources/on-field-resource/tmp_test_dir/"
SPREAD_ROI = (358, 277, 784, 568)
SPREAD_ROI_WIDTH = SPREAD_ROI[2] - SPREAD_ROI[0]
SPREAD_ROI_HEIGHT = SPREAD_ROI[3] - SPREAD_ROI[1]

#######################
## DECLARE VARIABLES ##
#######################

# THE CNN MODEL OF DLIB FACE DETECTION
start = time.time()
cnn_detector = dlib.cnn_face_detection_model_v1(sys.argv[1])
end = time.time()
print("[LOG] Model loading time: {:.3f}".format(end - start))

# THE BASEPATH THAT HOLDS SUBFOLDERS CONTAINING IMAGES OF SPECIFIC PERSONS
basepath = sys.argv[2]

# THE LIST THAT HOLDS IMAGES TO BE PROCESSED IN BATCH
frames = []
frame_count = 0

global count
count_d = 0
global count_2
count_n = 0

tnt_full_name = tnt_info.tnt_name_tup

logging.basicConfig(filename=f'{api_dirs.log_texts_dir}texts.log',
                    format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    filemode='w')
                    

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

###################
## PROCESS STUFF ##
###################
# BROWSE INSIDE THE BASEPATH
for entry in os.listdir(basepath):
    person_dir = os.path.join(basepath, entry)

    # if it is a directory
    if os.path.isdir(person_dir):
        print("\n[LOG] Handling folder {} ({})".format(person_dir, tnt_full_name[int(entry)]))
        smpl_embs_d_list = []
        smpl_embs_n_list = []
        count_d = 0
        count_n = 0
        # specific file
        for file in os.listdir(person_dir):
            print("[LOG] Handling file {}".format(file))
            if file.startswith('d') and file.endswith('.jpg'):
                count_d += 1
                print("[DEBUG] count_d = {}".format(count_d))
            elif file.startswith('n') and file.endswith('.jpg'):
                count_n += 1
                print("[DEBUG] count_n = {}".format(count_n))
            else:
                print("[WARNING] Wrong file name format. Check it now")
                exit()

            frame = cv2.imread(os.path.join(person_dir, file), cv2.IMREAD_COLOR)

            '''
            # convert to RGB which dlib needs
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.shape[1] != SPREAD_ROI_WIDTH or frame.shape[0] != SPREAD_ROI_HEIGHT:
                print("[ERROR] sample image size does not meet the SPREAD ROI size. Check again")
                exit()

            frames.append(frame)
            frame_count += 1

            # start to process detections and recognitions if batch is filled
            if len(frames) == BATCH_SIZE:
                print("\n[LOG] PROCESSING NEW BATCH")
                print("****************************")

                # start timing the detections
                start = time.time()
                batch_of_face_locations = cnn_detector(frames, 0, BATCH_SIZE)
                end = time.time()

                # display timing results of detections
                print("[LOG] dlib batch d_time: {0:0.3f}s".format(end - start))
                print("[LOG] avg time per frame: {0:0.3f}s".format((end - start) / BATCH_SIZE))
                # display how many frames that contain faces
                print("[LOG] Face(s) detected in {}/{} frames".format(len(batch_of_face_locations), BATCH_SIZE))

                # go into the detections in each frame of the batch
                for frame_id_in_batch, face_locations in enumerate(batch_of_face_locations):
                    # every face_locations entry is a mmod_rectangles (with s) object holding one or more
                    # mmod_rectangle (without s) objects, which are the face locations of all faces of that single frame in batch

                    # get numbering stuff
                    number_of_faces_in_frame = len(face_locations)
                    frame_id = frame_count - BATCH_SIZE + frame_id_in_batch

                    # skip if this frame contains more than one faces, otherwise keep on going
                    if number_of_faces_in_frame == 1:
                        # batch detection returns mmod_rectangles object, which does not contain confidence attribute
                        # in order to get confidence score, mmod_rectangle (without s) object should be popped out
                        # from mmod_rectangles so that confidence score can be available

                        # pop (and remove) the detected face in the current frame
                        popped = face_locations.pop()

                        # display what is fuckin mentioned
                        # the line commented below is needed for official full pipeline script, where more than one faces can appear
                        # print("[LOG] f#{}: {} face(s)".format(frame_id, number_of_faces_in_frame))
                        print(
                            "[LOG] d#{}: (conf | left, top, right, bottom) = ({:.3f} | {}, {}, {}, {})".format(frame_id,
                                                                                                               popped.confidence,
                                                                                                               popped.rect.left(),
                                                                                                               popped.rect.top(),
                                                                                                               popped.rect.right(),
                                                                                                               popped.rect.bottom()))

                        ## getting the encodings of this face
                        
                        #  create a list of face location's coordinates
                        css_type_face_location = [(popped.rect.top(), popped.rect.right(),
                                                   popped.rect.bottom(), popped.rect.left())]
                        # get the fuckin encodings and append to list

                        # write to image to check the results visually
                        tmp = cv2.cvtColor(frames[frame_id_in_batch], cv2.COLOR_RGB2BGR)
                        tmp = cv2.rectangle(tmp, (popped.rect.left(), popped.rect.top()),
                                            (popped.rect.right(), popped.rect.bottom()), freq_cv.DARK_GREEN, 2)
                        tmp = cv2.putText(tmp, "{}".format(file), freq_cv.BOTTOM_LEFT_CORNER_OF_TEXT_F_SPREAD_ROI, freq_cv.FONT, 2,
                                          freq_cv.BLUE, 1, cv2.LINE_AA)
                        tmp = cv2.putText(tmp, "{:.3f}".format(popped.confidence), freq_cv.BOTTOM_LEFT_CORNER_OF_TEXT_CONF_SPREAD_ROI,
                                          freq_cv.FONT, 1, freq_cv.YELLOW, 1, cv2.LINE_AA)
                        if file.startswith('d'):
                            print("[LOG] Writing {} ... ".format(
                                CHECK_BY_BB_DIR + "{}_{}_{}.jpg".format(entry, 'd', count_d)))
                            cv2.imwrite(CHECK_BY_BB_DIR + "{}_{}_{}.jpg".format(entry, 'd', count_d), tmp)

                        elif file.startswith('n'):
                            print("[LOG] Writing {} ... ".format(
                                CHECK_BY_BB_DIR + "{}_{}_{}.jpg".format(entry, 'n', count_n)))
                            cv2.imwrite(CHECK_BY_BB_DIR + "{}_{}_{}.jpg".format(entry, 'n', count_n), tmp)

                        start = time.time()
                        smpl_emb = face_recognition.face_encodings(tmp, css_type_face_location, 0)[0]
                        end = time.time()
                        print("[LOG] ec_time: {}s".format(end - start))
                        print("[DEBUG] smpl_embs: \n{}".format(smpl_emb))
                        print("[DEBUG] smpl_embs's type: {}\n".format(type(smpl_emb)))

                        if file.startswith('d'):
                            smpl_embs_d_list.append(smpl_emb)
                        elif file.startswith('n'):
                            smpl_embs_n_list.append(smpl_emb)
                        else:
                            print("[WARNING] Wrong file name format. Take a look at it")

                    else:
                        print("[LOG] Containing more than one face. Skipping ...")

                frames = []
            '''
            
            bboxes = do_inference(frame, score_threshold=0.6, top_k=1000, NMS_threshold=0.2, NMS_flag=True)
            if 1 < len(bboxes):
                print('[  info] containing more than 1 face. Check now!')
                exit()
            elif 1 == len(bboxes):
                bbox = bboxes[0]
                print(f'bbox: {bbox}')
                l_, t_, r_, b_ = bbox[0], bbox[1], bbox[2], bbox[3]
                css_type = [(t_, r_, b_, l_)]
                face_encoding = face_recognition.face_encodings(frame, css_type, 0)[0]
                recheck_img = cv2.rectangle(frame, (l_, t_), (r_, b_), (0, 255, 0), 2)
                if file.startswith('d'):
                    cv2.imwrite(f'{CHECK_BY_BB_DIR}d_{entry}_recheck_{count_d}.jpg', recheck_img)
                    smpl_embs_d_list.append(face_encoding)
                elif file.startswith('n'):
                    cv2.imwrite(f'{CHECK_BY_BB_DIR}n_{entry}_recheck_{count_n}.jpg', recheck_img)
                    smpl_embs_n_list.append(face_encoding)
                print(f'[ debug] enc:{face_encoding}\n')
            else:
                print('[  info] no face found. Replace with another image!')
                exit()
                
        print("[LOG] Finished getting encodings for {}".format(tnt_full_name[int(entry)]))
        numpy.save(ENCODINGS_DIR + "{}_{}_{}.npy".format(entry, 'd', tnt_full_name[int(entry)].split()[-1]),
                smpl_embs_d_list)
        numpy.save(ENCODINGS_DIR + "{}_{}_{}.npy".format(entry, 'n', tnt_full_name[int(entry)].split()[-1]),
                smpl_embs_n_list)

        print("\n\n###### TESTING ######")
        if len(smpl_embs_d_list) != 0:
            dir_to_load = ENCODINGS_DIR + "{}_{}_{}.npy".format(entry, 'd', tnt_full_name[int(entry)].split()[-1])
            print("Loaded numpy day file {}: ".format(dir_to_load))
            loaded = numpy.load(dir_to_load)
            print(loaded)
            print("loaded's type: {}".format(type(loaded)))
            print("Numpy array dimension: {}\n".format(loaded.shape))
            print("Converted day list:")
            converted = list(loaded)
            print(converted)

            print("**********")
        if len(smpl_embs_n_list) != 0:
            dir_to_load = ENCODINGS_DIR + "{}_{}_{}.npy".format(entry, 'n', tnt_full_name[int(entry)].split()[-1])
            print("Loaded numpy night file {}: ".format(dir_to_load))
            loaded = numpy.load(dir_to_load)
            print(loaded)
            print("loaded's type: {}".format(type(loaded)))
            print("Numpy array dimension: {}\n".format(loaded.shape))
            print("Converted night list:")
            converted = list(loaded)
            print(converted)


        print("##################################################################################################\n")
print("\nDone!")

