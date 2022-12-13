# django base
from django.db import DatabaseError
from django.http import Http404
from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
# add base
import urllib.parse as uparse
from time import sleep

# add our project
from .serializers import *

# calc route
from .route.simulate import simulation

def is_debug():
    return False

# Create your views here.
class PositionViewSet(viewsets.ModelViewSet):
    lookup_field = 'id'
    queryset = position.objects.all()
    serializer_class = PositionSerializer
    def create(self, request, *args, **kwargs):
        # =======================detect coord와 매칭시켜준다.
        try:
            coord = eval(request.data["coord"])
        except:
            coord = request.data["coord"]
            # print(type(coord))
        # 존재하면 기존 것을 반환해 준다.
        pos = position.objects.filter(
            coord__cue=coord["cue"], 
            coord__obj1=coord["obj1"],
            coord__obj2=coord["obj2"]
            ).first()
        if pos is None:
            return super().create( request, args, kwargs)
        else:
            serializer = PositionSerializer(pos)
            return Response(serializer.data, status=status.HTTP_200_OK)
        
            
        
            
class RouteViewSet(viewsets.ModelViewSet):
    lookup_field = 'id'
    queryset = soultion_route.objects.all()
    serializer_class = RouteSerializer

def synchronized(wrapped):
    import threading
    lock = threading.RLock()
    def _wrapper(*args, **kwargs):
        with lock:
            return wrapped(*args, **kwargs)
    return _wrapper

@synchronized
def get_ball_state(issue_id):
    # issue id 상태확인 - 연산 완료 상태 확인
    try:
        pos = position.objects.get(id=issue_id)
    except position.DoesNotExist: # 초기 배치가 아직 저장 안된 경우
        return  Response({"message":"Position doesn't exist"},status=status.HTTP_404_NOT_FOUND)
    state = pos.state
    # 상태가 N 이라면 A로 진행시킨다.
    if state == 'N':
        pos.state = 'A'
        pos.save()
    # pos의 state확인
    return state



class Make_Coordinate_Singleton(type):
    from threading import Lock
    _instance =None
    _lock = Lock()
    
    def __call__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(Make_Coordinate_Singleton, cls).__call__(*args, **kwargs)
        return cls._instance

class Simulate_route(metaclass=Make_Coordinate_Singleton):
    def __init__(self) -> None:
        import threading
        self.lock = threading.Lock()
    
    def RUN_THREAD(issue_id, display=False, save=False):
        simlate = Simulate_route()
        simlate.run(issue_id=issue_id, display=display, save=save)
        

    def run(self, issue_id, display=False, save=False):
        with self.lock:
            # 연산 중인지 확인
            state = get_ball_state(issue_id)
            if state == 'P' or state == 'D':
                return
            
            # 연산 중이 아니라면 이제 연산 해준다.
            pos = position.objects.get(id=issue_id)
            print(f'Make_Coord : ', end=' ')
            
            pos.state="P"
            if not display:
                pos.save()
        Simulate_route.make_cord(issue_id, display, save=save)
    
    def make_cord(issue_id, display=False, save=False):
        postion = position.objects.get(id=issue_id)
        
        #좌표 받아오기 cue, 목적구, 목적구2
        cue = postion.coord["cue"]
        obj1 =  postion.coord["obj1"]
        obj2 =  postion.coord["obj2"]
        cue = (cue[0], cue[1])
        obj1 = (obj1[0],obj1[1])
        obj2 = (obj2[0],obj2[1])
        soultion_list = simulation(cue, obj1, obj2, display=display, save=save)
        for soultion in soultion_list:
            if display:
                print("=============== add route =======================")
                for key, value in soultion.items():
                    print(f"==={key}===\n{value}")
            route = soultion_route(
                issue_id=issue_id, 
                route=soultion, 
                algorithm_ver=ROUTE_ALGORITHM_VERSION
                )
            if not display:
                route.save()
                print(f'add route',end=', ')
        # for i in range(3 - len(soultion_list)):
        #     route = soultion_route(
        #         issue_id=issue_id, 
        #         route={
        #                 "power": 0,
        #                 "clock": 0,
        #                 "tip": 1,
        #                 "thick": 0,
        #                 "cue": [[cue[0], cue[1]]],
        #                 "tar1": [obj1[0],obj1[1]],
        #                 "tar2": [obj2[0],obj2[1]]
        #             }, 
        #         algorithm_ver=ROUTE_ALGORITHM_VERSION
        #         )
        postion.state="D"
        if not display:
            postion.save()

class RouteRequestAPIView(APIView):
    def make_route(self, issue_id, usr):
        #detect PIPE
        try:
            import threading
            thrd = threading.Thread(target=Simulate_route.RUN_THREAD, args=(issue_id, False, False))
            thrd.start()
        except Exception as ex:
            print("make_route ex : "+ str(ex))
        
    def get(self, request, issue_id, usr, format=None):
        state = get_ball_state(issue_id=issue_id)
        
        #   None - A로 변경후 경로찾기
        if state == "N":
            self.make_route(issue_id, usr)
            # return Response({"state":"Accepted"}, status=status.HTTP_202_ACCEPTED) #기다리라고 한다.
            return Response({"state":"Accept"}, status=status.HTTP_202_ACCEPTED) #기다리라고 한다.
        
        #   Done - solution_route issue_id로 검색하여 결과반환 && 요청로그 남기기
        if state == "D":
            try:
                soultion = soultion_route.objects.filter(issue_id = issue_id)
                http_status = status.HTTP_200_OK
                serializer = RouteSerializer(soultion, many=True)
            except:
                return Response({"message":"Route doesn't exist"}, status=status.HTTP_404_NOT_FOUND)
            # requester 로그 쌓기
            try:
                # request를 찾고
                requester = route_request.objects.filter(issue_id=issue_id, requester=usr)
            except route_request.DoesNotExist:
                #존재 하지 않으면 새로 생성
                rr = route_request(issue_id=issue_id, requester=usr).save()
            return Response(serializer.data, status=http_status)
        #   Progress - 기다리라고 하기 
        return Response({"state":"Progress"}, status=status.HTTP_202_ACCEPTED) #기다리라고 한다.


def lastpos_route(user):
    pos = position.objects.last()
    print(f"=== {pos.id} : start ===")
    route = RouteRequestAPIView()
    response = route.get(request=None ,issue_id=pos.id, usr=user, format=None)
    print(f"=== {user}, {response} ===")
    for data in response.data:
        print(data)
    return response