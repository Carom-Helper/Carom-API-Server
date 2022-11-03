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
class CoordViewSet(viewsets.ModelViewSet):
    lookup_field = 'id'
    queryset = balls_coord.objects.all()
    serializer_class = CoordSerializer
    
class ProjectionViewSet(viewsets.ModelViewSet):
    lookup_field = 'id'
    queryset = projection.objects.all()
    serializer_class = ProjectionSerializer
    

def test_make_coord(carom_id, usr, t=1):
    print("======== test ============")
    carom_img = carom.objects.get(id=carom_id)
    carom_img.detect_state="P"
    carom_img.save()
    print("======== start Process ============")
    sleep(t*3)
    print("======== detect done ============")
    coord = balls_coord(carom_id=carom_id, coord={"cue" : [200, 200], "obj1" : [600, 200], "obj2" : [200, 150]})
    coord.save()
    carom_img.detect_state="D"
    carom_img.save()
    print("======== save ball_coord ============")

class DetectRequestAPIView(APIView):
    def get_coord(self, carom_id):
        try:
            return balls_coord.objects.get(carom_id=carom_id)
        except balls_coord.DoesNotExist:
            raise Http404
    def make_coord(self, carom_id, usr="ANOYMOUS"):
        import threading
        #detect PIPE
        try:
            runner = threading.Thread(target=test_make_coord, args=(carom_id, usr))
            runner.start()
        except Exception as ex:
            print("make_coord ex : "+ str(ex))
            
    def get(self, request, carom_id, usr, format=None):
        # carom에서 carom_id 있는지 확인
        try:
            carom_obj = carom.objects.get(id=carom_id)
        except carom.DoesNotExist: # 이미지가 아직 저장 안된 경우
            return Response({"message":"Image doesn't exist"},status=status.HTTP_404_NOT_FOUND)
        
        # carom의 detect_state 확인
        state = carom_obj.detect_state
        
        if state == "A" or state == "N":
            self.make_coord(carom_id, usr)
            return Response({"state":"Accepted"}, status=status.HTTP_202_ACCEPTED) #기다리라고 한다.
            
        if state == "D":
            try:
                coord_obj = balls_coord.objects.get(carom_id=carom_id)
                http_status = status.HTTP_200_OK
                serializer = CoordSerializer(coord_obj)
            except:
                return Response({"message":"Ball Corrdination doesn't exist"}, status=status.HTTP_404_NOT_FOUND)
            # requester 로그 쌓기
            try:
                # request를 찾고
                requester = detect_request.objects.get(carom_id=carom_id, requester=usr)
            except detect_request.DoesNotExist:
                #존재 하지 않으면 새로 생성
                detect_request(carom_id=carom_id, requester=usr).save()
            return Response(serializer.data, status=http_status)
        
        #진행중이라면
        return Response({"state":"Progress"}, status=status.HTTP_202_ACCEPTED) #기다리라고 한다.