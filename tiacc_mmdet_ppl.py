import numpy as np
import torch
import torchvision

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets import replace_ImageToTensor
from mmcv.parallel import DataContainer as DC
from tiacc_inference.status import StatusCode, Status
from mmcv import ConfigDict

@PIPELINES.register_module()
class TIACCPIPELINE:
    def __init__(self,
                 mean,
                 std,
                 keys=['img'],
                 img_to_float=True,
                 to_rgb=True,
                 size=None,
                 size_divisor=None,
                 pad_to_square=False,
                 pad_val=dict(img=0, masks=0, seg=255),
                 to_dc=False,
                 ):
        # Normalize
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

        # Pad
        self.size = size
        self.size_divisor = size_divisor
        if isinstance(pad_val, float) or isinstance(pad_val, int):
            warnings.warn(
                'pad_val of float type is deprecated now, '
                f'please use pad_val=dict(img={pad_val}, '
                f'masks={pad_val}, seg=255) instead.', DeprecationWarning)
            pad_val = dict(img=pad_val, masks=pad_val, seg=255)
        assert isinstance(pad_val, dict)
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        assert not self.pad_to_square, \
            'TI-ACC not support pad_to_square transform'

        assert img_to_float, \
            'TI-ACC only support float img'

        # ImageToTensor
        self.keys = keys

        # DefaultFormatBundle
        self.to_dc = to_dc

    def __call__(self, results):
        """Call function to normalize + pad + to_tensor.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        for key in results.get('img_fields', ['img']):
            assert key in self.keys, \
                'img key {} not in keys'.format(key)
            img = torch.from_numpy(results[key])
            img = img.cuda().float()
            # HWC -> CHW
            img = torch.transpose(img, 0, 2)
            results[key] = torch.transpose(img, 1, 2)

            # bgr to rgb
            if self.to_rgb:
                permute = [2, 1, 0]
                results[key] = results[key][permute, :, :]

            results[key] = torchvision.transforms.Normalize(self.mean, self.std)(results[key])

            img_w = results[key].shape[2]
            img_h = results[key].shape[1]

            padding = None
            if self.size is not None:
                padding = (0, 0, self.size[1] - img_w, self.size[0] - img_h)
            elif self.size_divisor is not None:
                pad_h = int(np.ceil(img_h / self.size_divisor)) * self.size_divisor
                pad_w = int(np.ceil(img_w / self.size_divisor)) * self.size_divisor
                padding = (0, 0, pad_w - img_w, pad_h - img_h)
            if padding is not None:
                results[key] = torchvision.transforms.Pad(padding, fill=self.pad_val['img'])(results[key])

            if self.to_dc:
                results[key] = DC(results[key], padding_value=self.pad_val['img'], stack=True)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

def convert_tiacc_pipeline(base_pipeline):
    tiacc_pipeline = {'type': 'TIACCPIPELINE'}
    for transform in base_pipeline:
        for key, value in transform.items():
            if key != 'type':
                tiacc_pipeline[key] = value
            elif value == 'DefaultFormatBundle':
                tiacc_pipeline['to_dc'] = True
    return tiacc_pipeline

def convert_transform(root_node, support_mmdet_ppl_patterns):
    if isinstance(root_node, list):
        transforms = []
        end_idx = -1
        for idx, transform in enumerate(root_node):
            if idx < end_idx:
                continue
            else:
                end_idx = -1
            match = False
            for support_mmdet_ppl_pattern in support_mmdet_ppl_patterns:
                if isinstance(transform, dict) and 'type' in transform and transform['type'] == support_mmdet_ppl_pattern[0]:
                    if idx + len(support_mmdet_ppl_pattern) <= len(root_node):
                        # check pattern match
                        match = True
                        for pattern_idx in range(len(support_mmdet_ppl_pattern)):
                            cur_transform = root_node[idx + pattern_idx]
                            if 'type' not in cur_transform or cur_transform['type'] != support_mmdet_ppl_pattern[pattern_idx]:
                                match = False
                        if match:
                            end_idx = idx + len(support_mmdet_ppl_pattern)
                            break
            if match:
                transforms.append(convert_tiacc_pipeline(root_node[idx:end_idx]))
            else:
                transforms.append(convert_transform(transform, support_mmdet_ppl_patterns))
        return transforms
    elif isinstance(root_node, dict):
        transforms = ConfigDict()
        for key, value in root_node.items():
            transforms[key] = convert_transform(value, support_mmdet_ppl_patterns)
        return transforms
    else:
        return root_node

def convert_pipeline(base_pipeline):
    # supported preprocess ppl pattern list
    support_mmdet_ppl_patterns = (['Normalize', 'Pad', 'ImageToTensor'],
                                  ['Normalize', 'Pad', 'DefaultFormatBundle'],
                                  ['Normalize', 'ImageToTensor'],
                                  ['Normalize', 'DefaultFormatBundle'])
    opt_pipeline = convert_transform(base_pipeline, support_mmdet_ppl_patterns)

    return opt_pipeline

def optimize_mmdet_ppl(input_model, optimization_level, device_type, device_id, report, optimization_config):
    status = Status(StatusCode.TIACC_OK, '')
    cfg = input_model.cfg
    test_pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    try:
        input_model.cfg.data.test.pipeline = convert_pipeline(test_pipeline)
    except Exception as ex:
        print('Optimize mmdet ppl failed, exception:{}'.format(ex))

        status = Status(StatusCode.TIACCERR_OPTIMIZE_ERR, 'Optimize mmdet ppl failed, exception:{}'.format(ex))
        return status

    return status
