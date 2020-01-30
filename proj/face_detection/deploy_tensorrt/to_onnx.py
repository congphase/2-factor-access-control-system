import logging
import numpy
import sys
sys.path.append('/home/gate/lffd-dir/incubator-mxnet/python')  # add mxnet python path if need
import mxnet
from mxnet.contrib import onnx as onnx_mxnet
from onnx import checker
import onnx


def generate_onnx_file():
    logging.basicConfig(level=logging.INFO)

    # set the proper symbol path, param path and onnx path
    symbol_path = '../symbol_farm/symbol_10_320_20L_5scales_v2_deploy.json'
    param_path = '../saved_model/configuration_10_320_20L_5scales_v2/train_10_320_20L_5scales_v2_iter_1000000.params'
    onnx_path = './onnx_files/v2_smallest.onnx'

    #symbol_path = '../symbol_farm/symbol_10_560_25L_8scales_v1_modified_deploy.json'
    #param_path = '../saved_model/configuration_10_560_25L_8scales_v1/train_10_560_25L_8scales_v1_iter_1400000.params'
    #onnx_path = './onnx_files/v1.onnx'

    net_symbol = mxnet.symbol.load(symbol_path)
    net_params_raw = mxnet.nd.load(param_path)
    net_params = dict()
    for k, v in net_params_raw.items():
        tp, name = k.split(':', 1)
        net_params.update({name: v})

    #input_shape = (1, 3, 480, 640)  # CAUTION: in TensorRT, the input size cannot be changed dynamically, so you must set it here.
    #input_shape = (1, 3, 240, 320)  # CAUTION: in TensorRT, the input size cannot be changed dynamically, so you must set it here.
    input_shape = (1, 3, 140, 160)  # CAUTION: in TensorRT, the input size cannot be changed dynamically, so you must set it here.

    onnx_mxnet.export_model(net_symbol, net_params, [input_shape], numpy.float32, onnx_path, verbose=True)

    # Load onnx model
    model_proto = onnx.load_model(onnx_path)

    # Check if converted ONNX protobuf is valid
    checker.check_graph(model_proto.graph)


if __name__ == '__main__':
    generate_onnx_file()
