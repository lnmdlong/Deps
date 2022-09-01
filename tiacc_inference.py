from typing import *
# from black import T
import torch
import sys
import GPUtil
import json
import re
import copy
import os
import random
from threading import Lock
from tiacc_inference.tiacc_torchscript_inference import optimize_torchscript_module 
from tiacc_inference.tiacc_nn_module_inference import optimize_nn_module, save_nn_module, load_nn_module
from tiacc_inference.utils import gen_shape_from_data, gen_shape_from_data_v2, gen_report, seperate_shape_v2, convert_shape_to_data_v2, seperate_shapes, shapes_align, wrapper_half
from tiacc_inference.status import StatusCode, Status
from tiacc_inference.tiacc_params_pb2 import *

type_pattern = '(int)|(float)|(fp16)|(int8)|(long)'
range_pattern = '(range)|(seperate)'
def seperate_key(key: str):
    if re.search('long', key):
        return 'long'
    if re.search('int', key):
        return 'int32'
    if re.search('int8', key):
        return 'int8'
    if re.search('fp16', key):
        return 'fp16'
    return 'float'

shape_type = "^[0-9]+(\*[0-9]+)*$"

def convert_to_tnn_name(obj, pre) -> dict:
    '''
    Returns:
        {
            'min_shapes' : dict of prefix and shape,
            'max_shapes' : dict of prefix and shape,
        },
        Status(TIACC_OK/TIACC_ERR, msg)
    '''
    status = Status(StatusCode.TIACC_OK, '')
    min_shapes, max_shapes, types = {}, {}, {}
    if isinstance(obj, dict):
        # print("dict")
        # filter keyword 'range'
        key = list(obj.keys())[0]
        if re.search(type_pattern, key) or re.search(range_pattern, key):
            if isinstance(obj[key], list):
                min_shape, max_shape, rtn = seperate_shapes(obj[key])
                if rtn.code != StatusCode.TIACC_OK:
                    return None, rtn
            else:
                min_shape, max_shape, rtn = seperate_shapes([obj[key]])
                if rtn.code != StatusCode.TIACC_OK:
                    return None, rtn
            min_shapes[pre] = min_shape
            max_shapes[pre] = max_shape
            types[pre] = seperate_key(key)
            return {'min_shapes': min_shapes, 'max_shapes': max_shapes, 'types': types}, status
        # if 'range' in obj:
        #     min_shape, max_shape = seperate_shapes(obj['range'])
        #     min_shapes[pre] = min_shape
        #     max_shapes[pre] = max_shape
        #     return {'min_shapes': min_shapes, 'max_shapes': max_shapes}
        # if 'seperate' in obj:
        #     min_shape, max_shape = seperate_shapes(obj['seperate'])
        #     min_shapes[pre] = min_shape
        #     max_shapes[pre] = max_shape
        #     return {'min_shapes': min_shapes, 'max_shapes': max_shapes}
        for key,value in obj.items():
            shapes, status = convert_to_tnn_name(value, pre + '[' + key + ']')
            min_shapes = {**min_shapes, **shapes['min_shapes']}
            max_shapes = {**max_shapes, **shapes['max_shapes']}
            types      = {**types, **shapes['types']}
        return {'min_shapes': min_shapes, 'max_shapes': max_shapes, 'types': types}, status

    elif isinstance(obj, list):
        # print("list")
        for i in range(len(obj)):
            shapes, status = convert_to_tnn_name(obj[i], pre + '[' + str(i) + ']')
            min_shapes = {**min_shapes, **shapes['min_shapes']}
            max_shapes = {**max_shapes, **shapes['max_shapes']}
            types      = {**types, **shapes['types']}
        return {'min_shapes': min_shapes, 'max_shapes': max_shapes, 'types': types}, status

    elif isinstance(obj, tuple):
        # print("tuple")
        for i in range(len(obj)):
            shapes, status = convert_to_tnn_name(obj[i], pre + '(' + str(i) + ')')
            min_shapes = {**min_shapes, **shapes['min_shapes']}
            max_shapes = {**max_shapes, **shapes['max_shapes']}
            types      = {**types, **shapes['types']}
        return {'min_shapes': min_shapes, 'max_shapes': max_shapes, 'types': types}, status

    elif isinstance(obj, str):
        # print("string")
        if re.match(shape_type, obj) != None:
            min_shape, max_shape, rtn = seperate_shapes([obj])
            if rtn.code != StatusCode.TIACC_OK:
                return None, rtn
            min_shapes[pre] = min_shape
            max_shapes[pre] = max_shape
            types[pre] = 'float'
            return {'min_shapes': min_shapes, 'max_shapes': max_shapes, 'types': types}, status
        else:
            print("Error shape format! Shape format should be positive numbers splited by '*', \n\
                           e.g. 'n*c*h*w'.")

            status = Status(StatusCode.TIACCERR_INVALID_SHAPE, "Error shape format! Shape format should be positive numbers splited by '*', \n\
                            e.g. 'n*c*h*w'.")
            return None, status
    else:
        print('Error type for tnn input name convert!')

        status = Status(StatusCode.TIACCERR_INVALID_SHAPE, 'Error type for tnn input name convert!')
        return None, status

def seperate_shape(input):
    status = Status(StatusCode.TIACC_OK, '')
    min_input, max_input, types, formats = {}, {}, {}, {}
    for ii in range(len(input)):
        name = "input_" + str(ii)
        shapes, rtn = convert_to_tnn_name(input[ii], name)
        if rtn.code != StatusCode.TIACC_OK:
            return None, None, None, rtn

        min_input = {**min_input, **shapes['min_shapes']}
        max_input = {**max_input, **shapes['max_shapes']}
        types     = {**types,     **shapes['types']}
    for name,val in min_input.items():
        formats[name] = 'tensor'
    return min_input, max_input, types, formats, status

def convert_shape_to_data(obj, device_type):
    status = Status(StatusCode.TIACC_OK, '')
    if isinstance(obj, list):
        test_data = []
        for i in range(len(obj)):
            data, rtn = convert_shape_to_data(obj[i], device_type)
            if rtn.code != StatusCode.TIACC_OK:
                return None, rtn
            test_data.append(data)
        return test_data, status

    elif isinstance(obj, dict):
        key = list(obj.keys())[0]
        if re.search(type_pattern, key) or re.search(range_pattern, key):
            if isinstance(obj[key], list):
                min_shape, max_shape, rtn = seperate_shapes(obj[key])
                if rtn.code != StatusCode.TIACC_OK:
                    return None, rtn
            else:
                min_shape, max_shape, rtn = seperate_shapes([obj[key]])
                if rtn.code != StatusCode.TIACC_OK:
                    return None, rtn
            type = seperate_key(key)
            data_tmp = gen_torch_tensor(max_shape, type)
            if device_type == 0:
                data_tmp = data_tmp.cuda()
            return data_tmp, status

        test_data = {}
        for key,value in obj.items():
            data, rtn = convert_shape_to_data(obj[key], device_type)
            if rtn.code != StatusCode.TIACC_OK:
                return None, rtn
            test_data[key] = data
        return test_data, status

    elif isinstance(obj, tuple):
        test_data = []
        for i in range(len(obj)):
            data, rtn = convert_shape_to_data(obj[i], device_type)
            if rtn.code != Status.TIACC_OK:
                return None, rtn
            test_data.append(data)
        return list(test_data), status

    elif isinstance(obj, str):
        min_shape, max_shape, rtn = seperate_shapes([obj])
        if rtn.code != StatusCode.TIACC_OK:
            return None, rtn

        data_tmp = torch.rand(*max_shape)
        if device_type == 0:
            data_tmp = data_tmp.cuda()
        return data_tmp, status
    
    else:
        print('Error input shape format!')

        status = Status(StatusCode.TIACCERR_INVALID_SHAPE, 'Error input shape format!')
        return None, status

def get_model_type(input_model):
    status = Status(StatusCode.TIACC_OK, '')
    model_type = ''
    try:
        import mmdet
    except ModuleNotFoundError as err:
        is_mmdet_found = False
    else:
        is_mmdet_found = True

    if is_mmdet_found:
        from mmdet import models
        if (isinstance(input_model, mmdet.models.detectors.BaseDetector)):
            model_type = 'mmdet'

    try:
        import detectron2
    except ModuleNotFoundError as err:
        is_dt2_found = False
    else:
        is_dt2_found = True

    if is_dt2_found:
        from detectron2 import modeling
        support_model_set = (modeling.meta_arch.rcnn.GeneralizedRCNN)
        if (isinstance(input_model, support_model_set)):
            model_type = 'detectron2'

    return model_type

def get_cpu_name():
    cmd = os.popen('lscpu')
    lines = cmd.readlines()
    for line in lines:
        have_name = re.search('Model name:', line)
        if have_name:
            line = line.split('Model name:')
            name = line[1].lstrip().replace('\n', '')
            return name

def optimize_ori_fp16(input_model):
    opt = copy.deepcopy(input_model)
    opt = opt.half()
    opt = wrapper_half(opt)
    return opt

# True: v2; False: v1
def chooseShapeFunc(input_shapes):
    if (isinstance(input_shapes[0], str)):
        if (input_shapes[0].count(':') == 0):
            return False
        else:
            return True
    else:
        return False

def get_mmdet_input_shapes(input_shapes):
    mmdet_input_name_list = [
        'input_1[0][0][border]',
        'input_1[0][0][flip]',
        'input_1[0][0][img_shape](0)',
        'input_1[0][0][img_shape](1)',
        'input_1[0][0][img_shape](2)',
        'input_1[0][0][ori_shape](0)',
        'input_1[0][0][ori_shape](1)',
        'input_1[0][0][ori_shape](2)',
        'input_1[0][0][pad_shape](0)',
        'input_1[0][0][pad_shape](1)',
        'input_1[0][0][pad_shape](2)',
        'input_1[0][0][scale_factor]',
        'input_2']
    mmdet_input_shape_list = [
        'input_1[0][0][border]:array.float(4)',
        'input_1[0][0][flip]:scalar.int32(0)',
        'input_1[0][0][img_shape](0):scalar.int32()',
        'input_1[0][0][img_shape](1):scalar.int32()',
        'input_1[0][0][img_shape](2):scalar.int32()',
        'input_1[0][0][ori_shape](0):scalar.int32()',
        'input_1[0][0][ori_shape](1):scalar.int32()',
        'input_1[0][0][ori_shape](2):scalar.int32()',
        'input_1[0][0][pad_shape](0):scalar.int32()',
        'input_1[0][0][pad_shape](1):scalar.int32()',
        'input_1[0][0][pad_shape](2):scalar.int32()',
        'input_1[0][0][scale_factor]:array.float(4)',
        'input_2:scalar.int32(0)']
    for idx, mmdet_input_name in enumerate(mmdet_input_name_list):
        name_found = False
        for input_shape in input_shapes:
            name, info = input_shape.split(':')
            if name == mmdet_input_name:
                name_found = True
                break
        if not name_found:
            input_shapes.insert(idx + 1, mmdet_input_shape_list[idx])
    return input_shapes

def get_dt2_input_shapes(input_shapes):
    # right shape
    dt2_input_name_list = [
        'input_0[0][image]:',
        'input_0[0][height]:',
        'input_0[0][width]:'
    ]

    # compatible shape
    dt2_input_name_com_list = [
        'input_0[image]',
        'input_0[height]',
        'input_0[width]'
    ]

    for idx, dt2_input_name in enumerate(dt2_input_name_com_list):
        for shape_idx, input_shape in enumerate(input_shapes):

            name, info = input_shape.split(':')
            
            if name == 'input_0[0][image]:' or name == 'input_0[image]':
                info = shapes_align(info, 32)

            if name == dt2_input_name:
                input_shapes[shape_idx] = dt2_input_name_list[idx] + info
                break
    
    return input_shapes

def optimize(
    input_model: Any,
    optimization_level: int,
    device_type: int,
    input_shapes = {},
    input_nodes_names = [],
    output_nodes_names = [],
    test_data = [],
    save_path = "",
    device_id = 0,
    optimization_config: OptimizeConfig=OptimizeConfig(),
) -> Tuple[Any, str]:
    # print parameters 
    # print("input_model:", input_model)
    # print("optimization_level:", optimization_level)
    # print("device_type:", device_type)
    # print("input_shapes:", input_shapes)
    # print("input_nodes_names:", input_nodes_names)
    # print("output_nodes_names:", output_nodes_names)
    # print("test_data:", test_data)
    # print("save_path:", save_path)
    model_type = ''
    report = gen_report('')
    if isinstance(input_model, str):
        try:
            loaded_input_model = torch.jit.load(input_model)
        except Exception as ex:
            try:
                loaded_input_model = torch.load(input_model)
            except Exception as ex:
                print("Not a torch.nn.Module or a torch.jit.ScriptModule, Please check the model again: {}".format(ex))
                report['status'] = Status(StatusCode.TIACCERR_INVALID_MODEL,
                                        "Not a torch.nn.Module or a torch.jit.ScriptModule, Please check the model again: {}".format(ex)).get_dict()
                report = json.dumps(report, indent=4, separators=(',', ': '))
                return (None, report)

        if isinstance(loaded_input_model, dict) and 'meta' in loaded_input_model and 'config' in loaded_input_model['meta']:
            model_type = 'mmdet'
            # mmdet model init
            try:
                import mmcv
                import mmdet
                from mmdet.models import build_detector, init_detector
                from mmcv.runner import load_checkpoint
            except ModuleNotFoundError as err:
                print('mmcv/mmdet import failed')

                report = gen_report('')
                report['status'] = Status(StatusCode.TIACCERR_ENVIRONMENT_ERR,
                                          'mmcv/mmdet import failed').get_dict()
                report = json.dumps(report, indent=4, separators=(',', ': '))
                return (None, report)
            checkpoint = loaded_input_model
            config = mmcv.Config.fromstring(checkpoint['meta']['config'], '.py')
            #config.model.pretrained = None
            if 'pretrained' in config.model:
                config.model.pretrained = None
            elif 'init_cfg' in config.model.backbone:
                config.model.backbone.init_cfg = None
            config.model.train_cfg = None
            loaded_input_model = build_detector(config.model, test_cfg=config.get('test_cfg'))
            checkpoint = load_checkpoint(loaded_input_model, input_model, map_location='cpu')
            if 'CLASSES' in checkpoint.get('meta', {}):
                classes = checkpoint['meta']['CLASSES']
            else:
                classes = mmdet.core.get_classes('coco')
            loaded_input_model.CLASSES = classes
            loaded_input_model.cfg = config
            if device_type == 0:
                loaded_input_model.to('cuda:0')
            else:
                loaded_input_model.to('cpu')
            loaded_input_model.eval()
        else:
            try:
                if device_type == 0:
                    loaded_input_model = loaded_input_model.cuda()
                else:
                    loaded_input_model = loaded_input_model.cpu()
            except Exception as ex:
                print("Error: input model convert device failed, exception:", ex)
        input_model = loaded_input_model
    
    if isinstance(input_model, torch.nn.Module):
        model_type = get_model_type(input_model)

    # adhoc get input shapes for mmdet
    if model_type == 'mmdet' and len(input_shapes) > 0:
        input_shapes = get_mmdet_input_shapes(input_shapes)
    if model_type == 'detectron2' and len(input_shapes) > 0:
        input_shapes = get_dt2_input_shapes(input_shapes)

    # set min&max input shapes
    types = {}
    if len(input_shapes) > 0:
        if chooseShapeFunc(input_shapes):
            min_input_shapes, max_input_shapes, types, formats, status = seperate_shape_v2(input_shapes)
        else:
            min_input_shapes, max_input_shapes, types, formats, status = seperate_shape(input_shapes)

        if status.code != StatusCode.TIACC_OK:
            report = gen_report('')
            report['status'] = status.get_dict()
            report = json.dumps(report, indent=4, separators=(',', ': '))
            return (None, report)
    elif len(test_data) > 0:
        min_input_shapes, max_input_shapes, types, status = gen_shape_from_data(test_data)
        if status.code != StatusCode.TIACC_OK:
            input_shapes, status = gen_shape_from_data_v2(test_data)
            if status.code != StatusCode.TIACC_OK:
                report = gen_report('')
                report['status'] = status.get_dict()
                report = json.dumps(report, indent=4, separators=(',', ': '))
                return (None, report)

            if chooseShapeFunc(input_shapes):
                min_input_shapes, max_input_shapes, types, formats, status = seperate_shape_v2(input_shapes)
            else:
                min_input_shapes, max_input_shapes, types, formats, status = seperate_shape(input_shapes)

            if status.code != StatusCode.TIACC_OK:
                report = gen_report('')
                report['status'] = status.get_dict()
                report = json.dumps(report, indent=4, separators=(',', ': '))
                return (None, report)
    else:
        report = gen_report('')
        report['status'] = Status(StatusCode.TIACCERR_INVALID_INPUT_DATA,
                                  'Error: At least one between input_shapes and test_data should be provided!').get_dict()
        report = json.dumps(report, indent=4, separators=(',', ': '))
        return (None, report)
    report = {}

    # set test_data
    if len(test_data) == 0:
        #test_data, status = convert_shape_to_data(input_shapes, device_type)
        test_data, status = convert_shape_to_data_v2(max_input_shapes, types, formats, device_type)
        report = gen_report(max_input_shapes)
        report['test_data_info']['test_data_source'] = 'tiacc provided'

        if status.code != StatusCode.TIACC_OK:
            report['status'] = status.get_dict()
            report = json.dumps(report, indent=4, separators=(',', ': '))
            return (None, report)
    else:
        report = gen_report(max_input_shapes)
    
    # print(min_input_shapes)
    # print(max_input_shapes)
    # print(types)
    # print(formats)
    # print(test_data)
    report['test_data_info']['test_data_type'] = str(types)

    if device_type == 0:
        report['hardware_environment']['device_type'] = 'GPU'
    else:
        report['hardware_environment']['device_type'] = 'CPU'
        try:
            cpu_name = get_cpu_name()
        except:
            cpu_name = ''
        report['hardware_environment']['microarchitecture'] = cpu_name

        # print("Error: Unsupported Device Type! ")
        # report = json.dumps(report, indent=4, separators=(',', ': '))
        # return input_model, report


    if isinstance(input_model, torch.jit.ScriptModule): 
        res = optimize_torchscript_module(input_model, optimization_level, device_type, device_id,
            min_input_shapes, max_input_shapes, types, test_data, report, optimization_config)
        status = res[2]
        if status.code != StatusCode.TIACC_OK:
            # try optimize with ori_fp16
            from tiacc_inference.utils import optimize_ori_fp16, save_ori_fp16
            res = optimize_ori_fp16(input_model, test_data, report)
            status = res[2]
            if status.code != StatusCode.TIACC_OK:
                report['status'] = status.get_dict()
                report = json.dumps(report, indent=4, separators=(',', ': '))
                return (None, report)
            model_type = 'ori_fp16'

        if len(save_path) > 0:
            if model_type == 'ori_fp16':
                save_ori_fp16(res[0], report, save_path)
            else:
                torch.jit.save(res[0], save_path)

        return (res[0], res[1])

    elif isinstance(input_model, torch.nn.Module):
        with torch.no_grad():
            res = optimize_nn_module(input_model, optimization_level, device_type, device_id,
                min_input_shapes, max_input_shapes, types, test_data, report, optimization_config)

        status = res[2]
        if status.code != StatusCode.TIACC_OK:
            report['status'] = status.get_dict()
            report = json.dumps(report, indent=4, separators=(',', ': '))
            return (None, report)

        if model_type == 'mmdet':
            if "enable_fast_mmdet_ppl" in optimization_config.parameter_map:
                enable_fast_mmdet_ppl = optimization_config.parameter_map["enable_fast_mmdet_ppl"].b
                if enable_fast_mmdet_ppl:
                    try:
                        import mmdet
                    except ModuleNotFoundError as err:
                        is_mmdet_found = False
                    else:
                        is_mmdet_found = True
                        import tiacc_inference.tiacc_mmdet_ppl as tiacc_mmdet_ppl

                    status = tiacc_mmdet_ppl.optimize_mmdet_ppl(res[0], optimization_level, device_type, device_id, report, optimization_config)
                    if status.code != StatusCode.TIACC_OK:
                        report['status'] = status.get_dict()
                        report = json.dumps(report, indent=4, separators=(',', ': '))
                        return (None, report)

        if len(save_path) > 0:
            # need to fix: support opt model save later
            status = save_nn_module(input_model, res[0], save_path, device_type)
            if status.code != StatusCode.TIACC_OK:
                report['status'] = status.get_dict()
                report = json.dumps(report, indent=4, separators=(',', ': '))
                return (None, report)

        report = json.dumps(res[1], indent=4, separators=(',', ': '))

        return (res[0], report)
    else:
        status = Status(StatusCode.TIACCERR_INVALID_MODEL, "Not a torch.nn.Module or a torch.jit.ScriptModule, Please check the model again.")
        report['status'] = status.get_dict()
        report = json.dumps(report, indent=4, separators=(',', ': '))
        return (None, report)

def load(
    input_model: Any,
    load_path = ""
) -> Tuple[Any, str]:

    if isinstance(input_model, torch.nn.Module):
        # to support mmdet load
        import copy
        opt_model = copy.deepcopy(input_model)
        res = load_nn_module(opt_model, load_path)
        rtn = res[2]
        if rtn.code != StatusCode.TIACC_OK:
            report = gen_report('')
            report['status'] = rtn.get_dict()
            report = json.dumps(report, indent=4, separators=(',', ': '))
            return (None, report)
        report = json.dumps(res[1], indent=4, separators=(',', ': '))
        return (res[0], report)
    if isinstance(input_model, str):
        try:
            opt_model = torch.jit.load(input_model)
        except Exception as ex:
            # nn.module
            try:
                loaded_input_model = torch.load(input_model)

                device_type = 'cuda:0'
                model_type  = ''
                # load fp16
                if 'model_type' in loaded_input_model:
                    buffer = loaded_input_model['model_type']
                    buffer.seek(0)
                    model_type = buffer.read()
                
                if model_type == 'ori_fp16':
                    from tiacc_inference.utils import load_ori_fp16
                    res = load_ori_fp16(loaded_input_model)
                    report = json.dumps(res[1], indent=4, separators=(',', ': '))
                    return (res[0], report)

                # end load fp16
                if 'tiacc_device_type' in loaded_input_model:
                    buffer = loaded_input_model['tiacc_device_type']
                    buffer.seek(0)
                    device_type = buffer.read()

                if isinstance(loaded_input_model, dict):
                    if 'tiacc_mmdet_meta' in loaded_input_model:
                        # mmdet model init
                        try:
                            import mmcv
                            import mmdet
                            from mmdet.models import build_detector
                            import tiacc_inference.tiacc_mmdet_ppl
                        except ModuleNotFoundError as err:
                            print('mmcv/mmdet import failed')

                            report = gen_report('')
                            report['status'] = Status(StatusCode.TIACCERR_ENVIRONMENT_ERR,
                                                    'mmcv/mmdet import failed').get_dict()
                            report = json.dumps(report, indent=4, separators=(',', ': '))
                            return (None, report)
                        config = mmcv.Config.fromstring(loaded_input_model['tiacc_mmdet_meta']['config'], '.py')
                        #config.model.pretrained = None
                        if 'pretrained' in config.model:
                            config.model.pretrained = None
                        elif 'init_cfg' in config.model.backbone:
                            config.model.backbone.init_cfg = None
                        config.model.train_cfg = None
                        model = build_detector(config.model, test_cfg=config.get('test_cfg'))
                        if 'CLASSES' in loaded_input_model.get('meta', {}):
                            classes = loaded_input_model['meta']['CLASSES']
                        else:
                            classes = mmdet.core.get_classes('coco')
                        model.CLASSES = classes
                        model.cfg = config
                    elif 'tiacc_origin_model' in loaded_input_model:
                        model_buffer = loaded_input_model['tiacc_origin_model']
                        model_buffer.seek(0)
                        model = torch.load(model_buffer)
                del loaded_input_model
                torch.cuda.empty_cache()
                model.to(device_type)
                model.eval()

                res = load_nn_module(model, load_path=input_model)
                rtn = res[2]
                if rtn.code != StatusCode.TIACC_OK:
                    report = gen_report('')
                    report['status'] = rtn.get_dict()
                    report = json.dumps(report, indent=4, separators=(',', ': '))
                    return (None, report)
                report = json.dumps(res[1], indent=4, separators=(',', ': '))
                return (res[0], report)
            except Exception as ex:
                print('tiacc load model caught exception:', ex)

                report = gen_report('')
                report['status'] = Status(StatusCode.TIACCERR_OPTIMIZE_ERR,
                                        'tiacc load model caught exception: {}' .format(ex)).get_dict()
                report = json.dumps(report, indent=4, separators=(',', ': '))
                return (None, report)
        else:
            return (opt_model, '')
    else:
        print('tiacc load only support torchscript/torch.nn.Module for now')

        report = gen_report('')
        report['status'] = Status(StatusCode.TIACCERR_UNSUPPORT_MODEL,
                                  'tiacc load only support torchscript/torch.nn.Module for now').get_dict()
        report = json.dumps(report, indent=4, separators=(',', ': '))
        return (None, report)
