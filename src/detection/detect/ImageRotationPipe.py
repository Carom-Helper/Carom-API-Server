import cv2
import numpy as np
from pipe_cls import One2OnePipe, ResourceOne
from detect_utills import PipeResource, line_show, is_test

def is_test_image_rotate()->bool:
    return True and is_test()

# set path
import os
from pathlib import Path
import sys
CAROM_BASE_DIR=Path(__file__).resolve().parent.parent.parent

def check_aline_corners(width, top_left, top_right, bottom_left, bottom_right)-> bool:
    # top_left, top_right, bottom_left, bottom_right 들의 배치가 잘 맞는지 확인해 주는 함수
    origin_list = [top_left, top_right, bottom_left, bottom_right]
    sorted_list = [top_left, top_right, bottom_left, bottom_right]
    sorted_list.sort(key=lambda x:x[0] + x[1]*width)
    
    for origin, sorted in zip(origin_list,sorted_list):
        if origin != sorted:
            return False
    return True
    
def aline_corner_in_dict(metadata:dict)->list:
    # 들어온 값을 정렬하여서 [TL, TR, BL, BR] 순서로 반환한다.
    pts = list()
    width = metadata['WIDTH']
    for key, value in metadata.items():
        if key == 'TL' or key == 'BL' or key == 'TR' or key == 'BR':
            pts.append(value)
    pts.sort(key=lambda x:x[0] + x[1]*width)
    pts = [pts[0],pts[1],pts[3],pts[2]]
    return pts

def rotate_origin_90(xy, shift_len, right_direct=False):
    import math 
    
    x_shift = 1 if right_direct else 0
    y_shift =  0 if right_direct else 1
    
    degree = 90 
    theta = math.radians(degree)
    
    cos = 0 #math.cos(theta) * (1 if right_direct else -1)
    sin = 1 if right_direct else -1 #math.sin(theta)
    
    """Only rotate a point around the origin (0, 0)."""
    x, y = xy
    xx = x * cos - y * sin + shift_len * x_shift
    yy = x * sin + y * cos + shift_len * y_shift

    return int(xx), int(yy)

class ResizeingPipe(One2OnePipe):
    def __init__(self, target_size=(1080,1920)) -> None:
        super().__init__()
        self.size = target_size
        
    def exe(self, input: PipeResource) -> PipeResource:
        output = input
        size = self.size
        
        
        if is_test_image_rotate():
            conners = [output.metadata["TL"],
                        output.metadata["TR"],
                        output.metadata["BL"],
                        output.metadata["BR"]]
            line_show(output.im, conners, f"ResizeingPipe{size}")
            cv2.waitKey(10000)
        
        # 홀수 회전시 size를 바꿔준다.
        if "rotation_num" in output.metadata.keys():
            rotation_num = int(output.metadata["rotation_num"])
            if rotation_num % 2 == 1:
                size = (size[1], size[0])
        
        output.im = cv2.resize(output.im, size)
        output.images["origin"] = output.im
        [ output.metadata["WIDTH"], output.metadata["HEIGHT"] ]= [size[0],size[1]]
        
        if is_test_image_rotate():
                conners = [output.metadata["TL"],
                           output.metadata["TR"],
                           output.metadata["BL"],
                           output.metadata["BR"]]
                line_show(output.im, conners, f"ResizeingPipe{size}")
                cv2.waitKey(10000)
        return output
    
    def get_regist_type(self, idx=0) -> str:
        return f"Resizeing{self.size}"
class ImageRotationPipe(One2OnePipe):
    def __init__(self, table_size = None) -> None:
        super().__init__()
        self.table_size = table_size
    
    def exe(self, input: PipeResource) -> PipeResource:
        max = 4
        output = input
        
        #set conners
        try:
            conner_list = [ output.metadata["TL"], output.metadata["TR"], output.metadata["BL"], output.metadata["BR"] ]
        except:
            print("not a key ['TL','TR','BL','BR']ImageRotationPipe.+ input.metadata") 
            conner_list[0,0,0,0]
        wh = [output.im.shape[1], output.im.shape[0]] if self.table_size is None else[self.table_size[0], self.table_size[1]]
        direct = False
        img_direct = cv2.ROTATE_90_CLOCKWISE if direct else cv2.ROTATE_90_COUNTERCLOCKWISE
        now_conners = conner_list.copy()
        
        
        #set img
        rotated_img = output.im.copy()
        
        count = 0
        for cnt in range(max):
            if is_test_image_rotate():
                line_show(rotated_img, now_conners, "ImageRotationPipe")
                cv2.waitKey(10000)
            
            # 각도가 맞는다면, 중간에 종료한다.
            if check_aline_corners(wh[0] ,*now_conners):
                break
            
            # rotation start!!!!
            count=cnt+1
            # 회전 방향 설정
            if direct: idx = (cnt+1)%2
            else: idx = (cnt)%2
            shift_len = wh[idx]
            
            #corner rotation
            for idx in range(4):
                x, y = rotate_origin_90(now_conners[idx], shift_len, direct)
                now_conners[idx] = [x,y]
                
            # if direct:
            #     now_conners = [now_conners[2],now_conners[0],now_conners[3],now_conners[1]]
            # else :
            #     now_conners = [now_conners[1],now_conners[3],now_conners[0],now_conners[2]]
            
            wh = [wh[1], wh[0]] # swap
            
            # image rotation
            rotated_img = cv2.rotate(rotated_img, img_direct)

        if not max > count:
            print(count)
            raise IndexError(f"ImageRotationPipe. + not collect position : {conner_list}")
        
        # reset corner
        [ output.metadata["TL"], output.metadata["TR"], output.metadata["BL"], output.metadata["BR"] ] = now_conners
        [ output.metadata["WIDTH"], output.metadata["HEIGHT"] ]= wh
        output.metadata["rotation_num"] = count
        
        # reset image
        output.images["origin"] = rotated_img
        output.im = rotated_img
        
        return output
    
    def get_regist_type(self, idx=0) -> str:
        return "image_rotater"


def test_check_aline(direct = True):
    max = 10
    
    sample_path = np.fromfile(CAROM_BASE_DIR / "media" / "test2" / "sample.jpg", np.uint8)
    sample = cv2.imdecode(sample_path, cv2.IMREAD_COLOR)
    
    ca3655_path = np.fromfile(CAROM_BASE_DIR / "media" / "carom" / "CAP3825091495947943655.jpg ", np.uint8)
    ca3655 = cv2.imdecode(ca3655_path, cv2.IMREAD_COLOR)
    ca3655 = cv2.resize(ca3655, (720,1280))
    
    #sample image test
    sample_points = [[549, 109], [942, 111], [180, 565], [1270, 580]] # sample
    sample_wh=[1440, 681]
    
    # CAP3825091495947943655 image test
    ca3655_points = [[662, 452], [662, 752], [57, 147], [57, 1057]] # CAP3825091495947943655.jpg
    
    ca3655_wh=[720,1280]
    
    now_sample_points = sample_points.copy()
    now_ca3655_points = ca3655_points.copy()
    
    print(now_sample_points)
    print(f"seample.jpg : {check_aline_corners(sample_wh[0], *now_sample_points)}")
    
    print(now_ca3655_points)
    print(f"CAP3825091495947943655.jpg : {check_aline_corners(ca3655_wh[0], *now_ca3655_points)}")
    
    line_show(sample, now_sample_points, "sample.jpg")
    line_show(ca3655, now_ca3655_points, "CAP3825091495947943655.jpg")
    cv2.waitKey()
    
    for cnt in range(1, max+1):
        if direct:
            idx = (cnt)%2
        else:
            idx = (cnt+1)%2
        
        sample_shift_len = sample_wh[idx]
        ca3655_shift_len = ca3655_wh[idx]
        # rotation
        target_list = [now_sample_points]#, now_ca3655_points]
        target_shift_len_list = [sample_shift_len]#, ca3655_shift_len]
        
        # conner point rotation
        for target_num in range(len(target_list)):
            target = target_list[target_num]
            shift_len = target_shift_len_list[target_num]
            print('Rotation : ',shift_len, target)
            for idx in range(4):
                
                x, y = rotate_origin_90(target[idx], shift_len, direct)
                target[idx] = [x,y]
                
            if direct:
                target = [target[2],target[0],target[3],target[1]]
            else :
                target = [target[1],target[3],target[0],target[2]]

        # image rotation
        img_direct = cv2.ROTATE_90_CLOCKWISE if direct else cv2.ROTATE_90_COUNTERCLOCKWISE
        
        sample = cv2.rotate(sample, img_direct)
        ca3655 = cv2.rotate(ca3655, img_direct)
        
        # view result
        print(f"=================== Rotation {cnt}s ====================")
        print(now_sample_points)
        print(f"seample.jpg : {check_aline_corners(sample_shift_len, *now_sample_points)}")
        print(now_ca3655_points)
        print(f"CAP3825091495947943655.jpg : {check_aline_corners(ca3655_shift_len, *now_ca3655_points)}")
        print("======================================================")
        line_show(sample, now_sample_points, "sample.jpg")
        line_show(ca3655, now_ca3655_points, "CAP3825091495947943655.jpg")
        cv2.waitKey()
        
        
def test_line_show():
    sample_path = np.fromfile(CAROM_BASE_DIR / "media" / "test2" / "sample.jpg", np.uint8)
    sample = cv2.imdecode(sample_path, cv2.IMREAD_COLOR)
    
    ca3655_path = np.fromfile(CAROM_BASE_DIR / "media" / "carom" / "CAP3825091495947943655.jpg ", np.uint8)
    ca3655 = cv2.imdecode(ca3655_path, cv2.IMREAD_COLOR)
    ca3655 = cv2.resize(ca3655, (720,1280))
    
    sample_points = [[549, 109], [942, 111], [180, 565], [1270, 580]] # sample
    ca3655_points = [[57, 147], [662, 452], [57, 1057], [662, 752]] # sample
    
    line_show(sample, sample_points, "sample.jpg")
    line_show(ca3655, ca3655_points, "CAP3825091495947943655.jpg")
    cv2.waitKey()

def get_int(coord:str)->list:
    x,y=map(int, coord.split('_'))
    return [x,y]

def test_pipe(src, top_left, top_right, bottom_left, bottom_right, display=True):
    pipe = ImageRotationPipe()
    path = np.fromfile(src, np.uint8)
    im0 = cv2.imdecode(path, cv2.IMREAD_COLOR)
    
    topLeft = get_int( top_left )
    topRight = get_int( top_right )
    bottomLeft = get_int( bottom_left )
    bottomRight = get_int( bottom_right )
    
    s=src
    metadata = {"TL":topLeft, "BR":bottomRight, "TR":topRight, "BL":bottomLeft, "WIDTH":im0.shape[1], "HEIGHT":im0.shape[0]}
    images = {"origin":im0}
    input = PipeResource(im=im0, metadata=metadata, images=images, s=s)
    input.print()
    output = pipe.exe(input)
    output.print()
    output.imshow(name=s, guide_line=True)
    cv2.waitKey(10000)

def test_connect_pipe(src, top_left, top_right, bottom_left, bottom_right, display=True):
    rotation_pipe = ImageRotationPipe()
    size_pipe = ResizeingPipe()
    bag = ResourceOne()
    
    pipe = size_pipe
    next_pipe = rotation_pipe
    
    pipe.connect_pipe(next_pipe)
    next_pipe.connect_pipe(bag)
    
    path = np.fromfile(src, np.uint8)
    im0 = cv2.imdecode(path, cv2.IMREAD_COLOR)
    
    topLeft = get_int( top_left )
    topRight = get_int( top_right )
    bottomLeft = get_int( bottom_left )
    bottomRight = get_int( bottom_right )
    
    s=src
    metadata = {"TL":topLeft, "BR":bottomRight, "TR":topRight, "BL":bottomLeft, "WIDTH":im0.shape[1], "HEIGHT":im0.shape[0]}
    images = {"origin":im0}
    input = PipeResource(im=im0, metadata=metadata, images=images, s=s)
    input.print()
    pipe.push_src(input)
    
    output = bag.get_src()
    
    output.print()
    output.imshow(name="output", guide_line=True)
    cv2.waitKey(10000)

def runner(args):
    # test_line_show()
    # test_check_aline()
    # test_pipe(args.src,
    #           args.top_left,
    #           args.top_right,
    #           args.bottom_left,
    #           args.bottom_right,
    #           not args.no_display)
    test_connect_pipe(args.src,
                      args.top_left,
                      args.top_right,
                      args.bottom_left,
                      args.bottom_right,
                      not args.no_display)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--src', default=CAROM_BASE_DIR / "media" / "carom" / "CAP3825091495947943655.jpg")
    parser.add_argument('--top_left', default='662_452', help='input x,y coord [format : x_y]')
    parser.add_argument('--top_right', default='662_752', help='input x,y coord [format : x_y]')
    parser.add_argument('--bottom_left', default='57_147', help='input x,y coord [format : x_y]')
    parser.add_argument('--bottom_right', default='57_1057', help='input x,y coord [format : x_y]')
    parser.add_argument('--no_display', default=False, action="store_true")
    args = parser.parse_args()
    runner(args)