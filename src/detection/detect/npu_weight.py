from threading import Lock
import time
import torch
import numpy as np
import cv2
import yaml



# set path
import sys
from pathlib import Path
import os


CAROM_BASE_DIR=Path(__file__).resolve().parent.parent.parent
FILE = Path(__file__).resolve()
ROOT = FILE.parent


tmp = ROOT / 'npu_yolov5'
if os.path.isabs(tmp):
    NPU_YOLO_DIR = tmp  # add yolov5 ROOT to PATH

tmp = ROOT / 'gpu_yolov5'
if os.path.isabs(tmp):
    GPU_YOLO_DIR = tmp  # add yolov5 ROOT to PATH

# Set weight directory
WEIGHT_DIR = None
tmp = ROOT / 'weights'
if str(tmp) not in sys.path and os.path.isabs(tmp):
    WEIGHT_DIR= (tmp)  # add Weights ROOT to PATH

from IWeight import IWeight
from Singleton import NPU_YOLO_Singleton
from detect_utills import (
    select_device, Path, is_test
)

def is_test_npu_weights()->bool:
    return True and is_test()

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test_npu_weights():
        print("npu weights test : ", s, s1, s2, s3, s4, s5, end=end)


class InferenceEngine:
    def __init__(self) -> None:
        pass

    def get_input_shapes(self):
        raise NotImplementedError

    def get_output_shapes(self):
        raise NotImplementedError

    def infer(self, *x):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def __call__(self, *x):
        return self.infer(*x)
    
class InferenceEngineOnnx(InferenceEngine):
    def __init__(self, predictor, onnx_file, input_names=None, output_names=None) -> None:
        super().__init__()
        import onnxruntime

        self.ort_session = onnxruntime.InferenceSession(str(onnx_file))

        if input_names is None:
            input_names = [i.name for i in self.ort_session.get_inputs()]

        if output_names is None:
            output_names = [i.name for i in self.ort_session.get_outputs()]

        self.input_names = input_names
        self.output_names = output_names

    def get_input_shapes(self):
        inputs = self.ort_session.get_inputs()
        shapes = [i.shape for i in inputs]
        return shapes

    def get_output_shapes(self):
        outputs = self.ort_session.get_outputs()
        shapes = [i.shape for i in outputs]
        return shapes

    def infer(self, *x):
        if len(x) == 1 and isinstance(x, dict):
            input_dict = x[0]
        else:
            input_dict = {k: v for k, v in zip(self.input_names, x)}
        out = self.ort_session.run(self.output_names, input_dict)

        return out

    def close(self):
        pass
    
class InferenceEngineFuriosa(InferenceEngine):
    def __init__(self,enf_file, device=None) -> None:
        super().__init__()
        test_print("InferenceEngineFuriosa, 109:",enf_file)
        from furiosa.runtime import session
        
        input_format = "hwc"
        input_prec = "i8"
        
        self.input_format = input_format
        self.input_prec = input_prec
        
        assert input_prec in ("f32", "i8")
        assert input_format in ("chw", "hwc")

        assert not (input_prec == "f32" and input_format != "chw")
        assert input_prec == "i8", "Nothing to do"
        
        compile_config = {
                "without_quantize": {
                    "parameters": [
                        {
                            "input_min": 0.0, "input_max": 1.0, 
                            "permute": [0, 2, 3, 1]
                        }
                    ]
                },
            }
        
        self.sess = session.create(str(enf_file), compiler_config = compile_config)

    def get_input_shapes(self):
        inputs = self.sess.inputs()
        shapes = [i.shape for i in inputs]
        return shapes

    def get_output_shapes(self):
        outputs = self.sess.outputs()
        shapes = [i.shape for i in outputs]
        return shapes

    def infer(self, *x):
        x = list(x)
        test_print("start infer")
        # print(x, type(x))
        outputs = self.sess.run(x)
        test_print("end infer")
        outputs = [outputs[i].numpy() for i in range(len(outputs))]

        return outputs

    def close(self):
        self.sess.close()


class Predictor:
    def __init__(self) -> None:
        pass

    def get_calib_dataset(self):
        raise NotADirectoryError

class Yolov5Detector(Predictor):
    def __init__(self, model_file, enf_file, cfg_file, framework, conf_thres=0.25, iou_thres=0.45, 
        input_color_format="bgr", box_decoder="c") -> None:
        # ADD gpu_yolov5 to env list
        if str(GPU_YOLO_DIR) in sys.path:
            sys.path.remove(str(GPU_YOLO_DIR))
        if str(NPU_YOLO_DIR) not in sys.path:
            sys.path.append(str(NPU_YOLO_DIR))  # add yolov5 ROOT to PATH
        from npu_yolov5.utils.box_decode.box_decoder import BoxDecoderPytorch, BoxDecoderC
        
        
        self.input_color_format = input_color_format


        self.input_format = input_format= "hwc"
        self.input_prec = input_prec = "i8"

        # load input name and shape in advance from onnx file
        input_name, input_shape = Yolov5Detector._get_input_name_shape(model_file)
        b, c, h, w = input_shape
        assert b == 1, "Code only supports batch size 1"

        self.input_name = input_name
        self.input_size = w, h

        if framework == "furiosa":
            infer = InferenceEngineFuriosa( enf_file )
        elif framework == "onnx":
            infer = InferenceEngineOnnx(self, model_file)

        assert input_format in ( "hwc")
        assert input_prec in ("i8")
        assert input_color_format in ("rgb", "bgr")
        assert box_decoder in ("pytorch", "c")

        self.infer = infer

        with open(cfg_file, "r") as f:
            cfg = yaml.safe_load(f)
            self.anchors = np.float32(cfg["anchors"])
            self.class_names = cfg["class_names"]
        
        self.stride = self._compute_stride()

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        if box_decoder == "pytorch":
            box_decoder = BoxDecoderPytorch(nc=len(self.class_names), anchors=self.anchors, stride=self.stride, conf_thres=self.conf_thres)
        elif box_decoder == "c":
            box_decoder = BoxDecoderC(nc=len(self.class_names), anchors=self.anchors, stride=self.stride, conf_thres=self.conf_thres)

        self.box_decoder = box_decoder

    @staticmethod
    def _get_input_name_shape(onnx_file):
        temp_sess = InferenceEngineOnnx(None, onnx_file)
        input_shape = temp_sess.get_input_shapes()[0]
        input_name = temp_sess.input_names[0]

        return input_name, input_shape

    def get_class_count(self):
        return len(self.class_names)

    def get_output_feat_count(self):
        return self.anchors.shape[0]

    def get_anchor_per_layer_count(self):
        return self.anchors.shape[1]

    def _compute_stride(self):
        img_h = self.input_size[1]
        feat_h = np.float32([shape[2] for shape in self.infer.get_output_shapes()])
        strides = img_h / feat_h
        return strides

    def _resize(self, img):
        from npu_yolov5.utils.transforms import letterbox
        
        w, h = self.input_size
        return letterbox(img, (h, w), auto=False)

    def _cvt_color(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _transpose(self, img):
        return img.transpose(2, 0, 1)

    def _normalize(self, img):
        img = img.astype(np.float32) / 255
        return img

    def _reshape_output(self, feat):
        return np.ascontiguousarray(feat.reshape(
            feat.shape[0], self.get_anchor_per_layer_count(), self.get_class_count() + 5, feat.shape[2], feat.shape[3]
        ).transpose(0, 1, 3, 4, 2))

    def preproc(self, img, input_format=None, input_prec=None):
        if input_format is None:
            input_format = self.input_format

        if input_prec is None:
            input_prec = self.input_prec
        
        img, (sx, sy), (padw, padh) = self._resize(img)

        if self.input_color_format == "bgr":
            img = self._cvt_color(img)

        if input_format == "chw":
            img = self._transpose(img)

        if input_prec == "f32":
            img = self._normalize(img)
        assert sx == sy
        scale = sx

        return img, (scale, (padw, padh))

    def postproc(self, feats_batched, preproc_params):
        from npu_yolov5.utils.nms import nms    
        boxes_batched = []

        for i, (scale, (padw, padh)) in enumerate(preproc_params):
            feats = [f[i:i+1] for f in feats_batched]
            feats = [self._reshape_output(f) for f in feats]
            boxes = self.box_decoder(feats)
            boxes = nms(boxes, self.iou_thres)[0]

            # rescale boxes
            boxes[:, [0, 2]] = (1 / scale) * (boxes[:, [0, 2]] - padw)
            boxes[:, [1, 3]] = (1 / scale) * (boxes[:, [1, 3]] - padh)

            boxes_batched.append(boxes)

        return boxes_batched

    def __call__(self, imgs):
        test_print("================start inference==========")
        single_input = not isinstance(imgs, (tuple, list))
        if single_input:
            imgs = [imgs]

        inputs, preproc_params = zip(*[self.preproc(img) for img in imgs])
        inputs = np.stack(inputs)
        feats = self.infer(inputs)
        test_print("================infer==========")
        res = self.postproc(feats, preproc_params)
        test_print("================postproc==========")
        if single_input:
            res = res[0]
        
        test_print("================end inference==========")

        return res

    def close(self):
        self.infer.close()

class NPUDetectObjectWeight(metaclass=NPU_YOLO_Singleton):
    def __init__(
        self,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=7,
        cls=[0, 1],
        imgsz=(640,640),
        device= 'furiosa'
        ) -> None:
        t1 = time.time()
        # 고정값
        WEIGHTS = "npu_yolo_ball"
        weights_dir = WEIGHT_DIR / WEIGHTS
        self.device = select_device(model_name="FURIOSA YOLOv5", device=device)
        
        # 변하는 값(입력 값)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        # classes = None  # filter by class: --class 0, or --class 0 2 3
        self.cls = cls
        self.imgsz = imgsz  # inference size (height, width)
        self.lock = Lock()
        
        framework = device
        onnx = weights_dir / "weights_i8.onnx"
        enf = weights_dir / "weights.enf"
        cfg_file = weights_dir / "cfg.yaml"
        
        ### load model ### Yolov5Detector가 들어가야함
        test_print("======weights=====", onnx, enf)
        model = Yolov5Detector(
            model_file= onnx, 
            enf_file = enf, 
            cfg_file = cfg_file, 
            framework = framework,
            conf_thres = conf_thres,
            iou_thres= iou_thres,
            input_color_format = "bgr",
            box_decoder="c" )
        self.model = model
        ############
        
        t2 = time.time()
        print( f'[NPU YOLOv5 init {(t2-t1):.1f}s]')
        
        
    def inference(self, im, origin_size=(640,640)):
        result = [self.model(im)]
        return result
    
    def preprocess(self, im):
        return im