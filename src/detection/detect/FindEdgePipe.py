from pipe_cls import *
import math
import numpy as np


class FindEdgePipe(IPipeObserver):
    def __init__(self):
        super().__init__()
        self.src_list = list()

    def push_src(self, input: PipeResource) -> None:
        self.src_list.append(input)

    def get_edge(self) -> dict:
        top_left = list()
        top_right = list()
        bottom_left = list()
        bottom_right = list()

        edge_not_found = list()
        edge_found = list()

        result = dict()

        # 초기화
        for i, src in enumerate(self.src_list):
            threshold_x = int(src.im0s.shape[1]/2)
            threshold_y = int(src.im0s.shape[0]/2)

            for i in range(len(src.dets)):
                if src.dets[i]['x'] > threshold_x and src.dets[i]['y'] > threshold_y:
                    bottom_right.append(src.dets[i])
                elif src.dets[i]['x'] <= threshold_x and src.dets[i]['y'] > threshold_y:
                    bottom_left.append(src.dets[i])
                elif src.dets[i]['x'] > threshold_x and src.dets[i]['y'] <= threshold_y:
                    top_right.append(src.dets[i])
                else:
                    top_left.append(src.dets[i])

            result["TL"] = reset_coord(return_max(top_left), "TL")
            result["TR"] = reset_coord(return_max(top_right), "TR")
            result["BL"] = reset_coord(return_max(bottom_left), "BL")
            result["BR"] = reset_coord(return_max(bottom_right), "BR")

            # Edge 개수 파악
            for key in result:
                if result[key] == -1:
                    edge_not_found.append(key)
                else:
                    edge_found.append(key)

            # Edge 경우의 수에 따른 좌표 재설정
            if len(edge_not_found) == 2:
                if "TL" in edge_not_found and "TR" in edge_not_found:
                    distance = get_distance(result["BL"], result["BR"])
                    result["TL"] = (result["BL"][0], int(
                        result["BL"][1]-(distance/2)))
                    result["TR"] = (result["BR"][0], int(
                        result["BR"][1]-(distance/2)))
                elif "BL" in edge_not_found and "BR" in edge_not_found:
                    distance = get_distance(result["TL"], result["TR"])
                    result["BL"] = (result["TL"][0], int(
                        result["TL"][1]+(distance/2)))
                    result["BR"] = (result["TR"][0], int(
                        result["TR"][1]+(distance/2)))
                elif "TL" in edge_not_found and "BL" in edge_not_found:
                    distance = get_distance(result["TR"], result["BR"])
                    result["TL"] = (
                        result["TR"][0] - (distance*2), result["TR"][1])
                    result["BL"] = (
                        result["BR"][0] - (distance*2), result["BR"][1])
                elif "TR" in edge_not_found and "BR" in edge_not_found:
                    distance = get_distance(result["TL"], result["BL"])
                    result["TR"] = (
                        result["TL"][0] + (distance*2), result["TL"][1])
                    result["BR"] = (
                        result["BL"][0] + (distance*2), result["BL"][1])
                elif "TL" in edge_not_found and "BR" in edge_not_found:
                    result["TL"] = (result["BL"][0],
                                    result["TR"][1])
                    result["BR"] = (result["TR"][0],
                                    result["BL"][1])
                elif "TR" in edge_not_found and "BL" in edge_not_found:
                    result["TR"] = (result["BR"][0],
                                    result["TL"][1])
                    result["BL"] = (result["TL"][0],
                                    result["BR"][1])
            elif len(edge_not_found) == 1:
                if "TL" in edge_not_found:
                    dist_x = result["BR"][0] - result["BL"][0]
                    dist_y = result["BR"][1] - result["BL"][1]
                    result["TL"] = (
                        result["TR"][0] - dist_x, result["TR"][1] - dist_y)
                elif "TR" in edge_not_found:
                    dist_x = result["BR"][0] - result["BL"][0]
                    dist_y = result["BR"][1] - result["BL"][1]
                    result["TR"] = (
                        result["TL"][0] + dist_x, result["TL"][1] + dist_y)
                elif "BL" in edge_not_found:
                    dist_x = result["TR"][0] - result["TL"][0]
                    dist_y = result["TR"][1] - result["TL"][1]
                    result["BL"] = (
                        result["BR"][0] - dist_x, result["BR"][1] - dist_y)
                elif "BR" in edge_not_found:
                    dist_x = result["TR"][0] - result["TL"][0]
                    dist_y = result["TR"][1] - result["TL"][1]
                    result["BR"] = (
                        result["BL"][0] + dist_x, result["BL"][1] + dist_y)

        return result


def return_max(coord):
    if len(coord) == 0:
        return -1
    else:
        x1 = max(list(map(lambda x: x['x'], coord)), key=list(
            map(lambda x: x['x'], coord)).count)
        y1 = max(list(map(lambda x: x['y'], coord)), key=list(
            map(lambda x: x['y'], coord)).count)
        x2 = max(list(map(lambda x: x['w'], coord)), key=list(
            map(lambda x: x['w'], coord)).count)
        y2 = max(list(map(lambda x: x['h'], coord)), key=list(
            map(lambda x: x['h'], coord)).count)

    return [x1, y1, x2, y2]


def reset_coord(coord, type):
    if coord == -1:
        return -1

    if type == "TL":
        return (coord[2], coord[3])
    elif type == "TR":
        return (coord[0], coord[3])
    elif type == "BR":
        return (coord[0], coord[1])
    else:
        return (coord[2], coord[1])


def get_distance(coord1, coord2):
    a = coord2[0] - coord1[0]
    b = coord2[1] - coord1[1]
    return math.sqrt(a**2+b**2)


if __name__ == '__main__':
    input = PipeResource(
        vid=-1, f_num=1, im0s=np.zeros((1080, 1920, 3), np.uint8))
    input.append_det([50, 300, 60, 310])  # TL
    # input.append_det([0, 700, 10, 710])  # BL
    input.append_det([1050, 250, 1060, 260])  # TR
    input.append_det([1000, 650, 1010, 660])  # BR

    pipe = FindEdgePipe()
