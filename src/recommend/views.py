# django base
from django.db import DatabaseError
from django.http import Http404
from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response

# add base
import urllib.parse as uparse
from time import sleep

# add our project
from .serializers import *


# Create your views here.
class PositionViewSet(viewsets.ModelViewSet):
    lookup_field = 'id'
    queryset = position.objects.all()
    serializer_class = PositionSerializer
    
    def create(self, request, *args, **kwargs):
        # =======================detect coord와 매칭시켜준다.
        
        coord = eval(request.data["coord"])
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
    
def test_make_route(issue_id, usr, t=1):
    print("======== test ============")
    pos = position.objects.get(id=issue_id)
    pos.state="P"
    pos.save()
    print("======== start Process ============")
    sleep(t*3)
    print("======== detect done ============")
    route = soultion_route(issue_id=issue_id, route={
			"power": 2.4,
			"cue": [(200, 200), (590, 200), (680, 10), (780, 150), (480, 390), (210, 150), (200, 120)],
			"obj1": [(600, 200), (700, 120)],
			"obj2": [(200, 150), (150, 130)]
		}, algorithm_ver=ROUTE_ALGORITHM_VERSHION)
    print(route)
    route.save()
    pos.state="D"
    pos.save()
    print("======== save ball_coord ============")

class RouteRequestAPIView(APIView):
    def make_route(self, issue_id, usr):
        import threading
        #detect PIPE
        try:
            runner = threading.Thread(target=test_make_route, args=(issue_id, usr))
            runner.start()
        except Exception as ex:
            print("make_route ex : "+ str(ex))
    
    
    def get(self, request, issue_id, usr, format=None):
        # issue id 상태확인 - 연산 완료 상태 확인
        try:
            pos = position.objects.get(id=issue_id)
        except position.DoesNotExist: # 초기 배치가 아직 저장 안된 경우
            return  Response({"message":"Position doesn't exist"},status=status.HTTP_404_NOT_FOUND)
        
        # pos의 state확인
        state = pos.state
        #   None - A로 변경후 경로찾기
        #   Accepted - 경로찾기 
        if state == "A" or state == "N":
            self.make_route(issue_id, usr)
            return Response({"state":"Accepted"}, status=status.HTTP_202_ACCEPTED) #기다리라고 한다.
        
        
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
                requester = route_request.objects.get(issue_id=issue_id, requester=usr)
            except route_request.DoesNotExist:
                #존재 하지 않으면 새로 생성
                rr = route_request(issue_id=issue_id, requester=usr).save()
            return Response(serializer.data, status=http_status)
        
        #   Progress - 기다리라고 하기 
        return Response({"state":"Progress"}, status=status.HTTP_202_ACCEPTED) #기다리라고 한다.
