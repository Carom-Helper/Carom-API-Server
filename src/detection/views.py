# django base
from django.db import DatabaseError
from django.http import Http404
from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.settings import api_settings

# add base
import urllib.parse as uparse

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
    from time import sleep
    print("======== test ============")
    carom_img = carom.objects.get(id=carom_id)
    carom_img.detect_state="P"
    carom_img.save()
    sleep(t*3)
    print("======== detect done ============")
    coord = balls_coord(carom_id=carom_id, coord={"cue" : [200, 200], "obj1" : [600, 200], "obj2" : [200, 150]})
    coord.save()
    carom_img.detect_state="D"
    carom_img.save()
    print("======== save ball_coord ============")

class RequestAPIView(APIView):
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
            print("======== start Process ============")
            runner.start()
        except Exception as ex:
            print("make_coord ex : "+ str(ex))
            
    def get(self, request, carom_id, usr, format=None):
        obj_request = None
        try:#존재할 경우
            obj_request = detect_request.objects.select_related('carom').values(
                "requester", "carom_id", "carom__detect_state"
                ).get(carom_id=carom_id)
            state = obj_request["carom__detect_state"]
        except detect_request.DoesNotExist:
            #존재하지 않으면 새로 만든다.
            print("========make coord============")
            detect_request(carom_id=carom_id, requester=usr).save()
            carom_img = carom.objects.get(id=carom_id)
            carom_img.detect_state = 'A'
            carom_img.save()
            self.make_coord(carom_id)
            return Response({"state":"Create"}, status.HTTP_202_ACCEPTED)
        except detect_request.MultipleObjectsReturned:
            # 같은 요청이 존재한다면, 
            # 해당 작업이 이미 끝났으면, 요청 로그를 남기고, 결과 반환하고
            # 아직 안 끝났으면, 기다리라고 한다.
            # < 로직상 두개가 만들어 졌다면, 이미 Done인 상태일 것이다. >
            state = "D"
            
        #아직 연산중 이라면
        if state == "A":
            #detect PIPE
            print("========== I'm state Create ================")
            self.make_coord(carom_id)
            return Response({"state":"Accepted"}, status=status.HTTP_202_ACCEPTED) #기다리라고 한다.
        elif state == "P":
            return Response({"state":"Progress"}, status=status.HTTP_202_ACCEPTED) #기다리라고 한다.
            
        # 다 완료됐다면
        elif state == "D":
            coord = self.get_coord(carom_id=carom_id)
            http_status = status.HTTP_200_OK
            serializer = CoordSerializer(coord)
            
            # 요청자가 같은지 찾아보고, 다르다면 새로 만든다.
            if obj_request is None:
                # 객체가 여러개 라면, 이미 이전에 답을 찾았을 것이라고 가정한다.
                try:
                    detect_request(carom_id=carom_id, requester=usr).save()
                    http_status = status.HTTP_201_CREATED
                except DatabaseError:
                    pass
            return Response(serializer.data, status=http_status)
            
    def get_success_headers(self, data):
        try:
            return {'Location': str(data[self.settings.URL_FIELD_NAME])}
        except (TypeError, KeyError):
            return {}