def is_test()->bool:
    return True

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test():
        print("pipe cls exe : ", s, s1, s2, s3, s4, s5, end=end)

from random import random, randrange
import glob
import logging
from typing import Optional
import inspect
import cv2
import numpy as np


import sys
from pathlib import Path
import os



FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory

# ADD gpu_yolov5 to env list
tmp = FILE.parent / 'gpu_yolov5'
if str(tmp) not in sys.path and os.path.isabs(tmp):
    sys.path.append(str(tmp))  # add yolov5 ROOT to PATH

from gpu_yolov5.utils.torch_utils import select_device, time_sync
from gpu_yolov5.utils.general import (check_img_size, non_max_suppression, scale_boxes)
from gpu_yolov5.models.common import DetectMultiBackend
from gpu_yolov5.utils.plots import Annotator, colors, save_one_box


IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

class LoadImages:
    def __init__(self, path, vid_stride=1):   
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)
        
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.vid_stride = vid_stride  # video frame-rate stride
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'
                            
    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        return im0, path, s
    
    def _new_video(self, path):
        # Create a new video capture object
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
        # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable https://github.com/ultralytics/yolov5/issues/8493

    def _cv2_rotate(self, im):
        # Rotate a cv2 video manually
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        return self.nf  # number of files


class PipeResource:
    def __init__(self, im=None, metadata=dict(), images=dict(), s=None) -> None:
        self.im = im            # target image
        self.dets = list()
        self.unset_det_num = dict()
        
        # get dateset  (path, im, im0s, vid_cap, s)
        self.s=s
        self.metadata=dict()
        self.metadata.update(metadata)
        # images
        self.images=dict()
        self.images.update(images)
    
    def __str__(self) -> str:
        return self.s
    
    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self) -> dict:
        if self.count == self.__len__():
            raise StopIteration
        
        cnt = self.count 
        
        contents = self.dets[cnt]
        
        self.count += 1
        
        return contents
    
    def __len__(self):
        #test_print("len ==>",self.dets.__len__())
        return self.dets.__len__()
    
    def print(self, on=True):
        if on:
            print("==============================================")
            print(f'{self.s}',"dets")
            for i, det in enumerate(self.dets):
                print(f'det {i} :', det)
            print("==============================================")
        
    def auto_set_dets(self):
        for i in range(randrange(1, 7)):
            det = {"xmin": randrange(10,1970), "ymin": randrange(10,1070), "xmax":randrange(18,31), "ymax":randrange(18,31), "cls":randrange(0,2), "conf":random()}
            self.dets.append(det)
    def set_det(self, idx=0, xyxy=None, cls=0, conf=0.0):
        #test_print(self.dets.__len__(), "det.len()")
        if self.dets.__len__() < idx:
            raise IndexError
        if xyxy is not None:
            det = self.dets[idx]
            det["xmin"] = xyxy[0]
            det["ymin"] = xyxy[1]
            det["xmax"] = xyxy[2]
            det["ymax"] = xyxy[3]
            det["cls"] = cls
            det["conf"] = conf

    def append_det(self, xywh, id=-1, cls=0, conf=0.0):
        det = dict()
        det["xmin"] = xywh[0]
        det["ymin"] = xywh[1]
        det["xmax"] = xywh[2]
        det["ymax"] = xywh[3]
        det["id"] = id
        det["cls"] = cls
        det["conf"] = conf
        self.dets.append(det)
            
    def imshow(self, name="no name", idx_names=["1","2","3","4","5","6"],cls_names=["Overall", "Bottom", "Top", "Outer","Shose"],line_thickness=2, hide_labels=True, hide_conf = True):
        im0= self.images["origin"]
        annotator = Annotator(
            im0, line_width=line_thickness, example=str(cls_names))
        # Write results
        for i, det in enumerate(self.dets):
            c = int(det["cls"])  # integer class
            id = ""
            try:
               id = f"{idx_names[int(det['id'])]} "
            except KeyboardInterrupt:sys.exit()
            except:
                pass
            label = None if hide_labels else (f"{id}{cls_names[c]}" if hide_conf else f'{cls_names[c]} {det["conf"]:.2f}')
            xyxy = [det["xmin"], det["ymin"], det["xmax"], det["ymax"]]
            annotator.box_label(xyxy, label, color=colors(c, True))
        im0 = annotator.result()
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, im0.shape[1], im0.shape[0])
        cv2.imshow(name, im0)
        
    def is_detkey(self, key = "id") ->bool:
        try:
            for det in self.dets:
                a = det[key]
                test_print(f"is_detkey({key} : {a})")
        except KeyboardInterrupt:sys.exit()
        except:
            return False
        return True
    
    def len_unset_detkey(self, key = "id") -> int:
        try:
            return self.unset_det_num[key]
        except KeyboardInterrupt:sys.exit()
        except:
            cnt = 0
            for det in self.dets:
                try:
                    a = det[key]
                    test_print(f"is_detkey({key} : {a})")
                except:
                    cnt += 1
            self.unset_det_num[key] = cnt
            return self.unset_det_num[key]
    def len_detkey_match(self, key = "cls", value="1") -> int:
        cnt = 0
        if len(self.dets) <= 0:
            return 0
        
        for det in self.dets:
            if int(det[key]) == int(value):
                cnt += 1
        return cnt
    def update_id(self, key, value, xywh, conf, cls=1):
        for det in self.dets:
            if float(det["conf"]) == float(conf) and int(det["cls"]) == int(cls):
                if same_box([],[]):
                    det[key] = value
                        
def same_box(box1, box2, iou_th=0.9) -> bool:
    return True

def xywh2xyxy(x):
    y=list(x)
    y[0] = float(x[0]) - float(x[2]) / 2  # top left x
    y[1] = float(x[1]) - float(x[3]) / 2  # top left y
    y[2] = float(x[0]) + float(x[2]) / 2  # bottom right x
    y[3] = float(x[1]) + float(x[3]) / 2  # bottom right y
    return y

def xyxy2xywh(x):
    y=list(x)
    y[0] = (float(x[0]) + float(x[2])) / 2  # x center
    y[1] = (float(x[1]) + float(x[3])) / 2  # y center
    y[2] = float(x[2]) - float(x[0])  # width
    y[3] = float(x[3]) - float(x[1]) # height
    return y

def copy_piperesource(src:PipeResource)->PipeResource:
    dst = PipeResource()
    
    
    dst.dets = src.dets.copy()
    dst.im = src.im.copy()      # image
    
    # get dateset  (path, im0s, vid_cap, s)
    dst.metadata.update(src.metadata)
    for key, value in src.images.items():
        dst.images[key] = value.copy()
        
    return dst



def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def make_padding_image(im0, img_size=640, stride=32, auto=True, transforms=None):
    if transforms:
        dst = transforms(im0)
    else:
        dst = letterbox(im0, img_size, stride=stride, auto=auto)[0]  # padded resize
        dst = dst.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        dst = np.ascontiguousarray(dst)  # contiguous
    return dst



import os
import platform
def is_colab():
    # Is environment a Google Colab instance?
    return 'COLAB_GPU' in os.environ
def is_kaggle():
    # Is environment a Kaggle Notebook?
    return os.environ.get('PWD') == '/kaggle/working' and os.environ.get('KAGGLE_URL_BASE') == 'https://www.kaggle.com'

RANK = int(os.getenv('RANK', -1)) # rank in world for Multi-GPU trainings
VERBOSE = str(os.getenv('Fashion_VERBOSE', True)).lower() == 'true'  # global verbose mode
def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str
def set_logging(name=None, verbose=VERBOSE):
    # Sets level and returns logger
    if is_kaggle() or is_colab():
        for h in logging.root.handlers:
            logging.root.removeHandler(h)  # remove all handlers associated with the root logger object
    rank = RANK
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    log = logging.getLogger(name)
    log.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)
    log.addHandler(handler)
    
set_logging()  # run before defining LOGGER
LOGGER = logging.getLogger("fashion_detector")  # define globally (used in train.py, val.py, detect.py, etc.)
if platform.system() == 'Windows':
    for fn in LOGGER.info, LOGGER.warning:
        setattr(LOGGER, fn.__name__, lambda x: fn(emojis(x)))  # emoji safe logging
        
import warnings
import torch
warnings.filterwarnings('ignore', message='User provided device_type of \'cuda\', but CUDA is not available. Disabling')

def select_device(model_name="Resnet34" ,device='', batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = f'{model_name} 🚀 Python-{platform.python_version()} torch-{torch.__version__} '
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = 'cuda:0'
    elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():  # prefer MPS if available
        s += 'MPS\n'
        arg = 'mps'
    else:  # revert to CPU
        s += 'CPU\n'
        arg = 'cpu'

    if not newline:
        s = s.rstrip()
    LOGGER.info(s)
    return torch.device(arg)


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    LOGGER.info(colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))