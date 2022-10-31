# django base
from django.http import Http404
from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response

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
    


def test_make_coord(carom_id, usr, t=5):
    from time import sleep
    print("======== test ============")
    sleep(t)
    obj_request = detect_request.objects.get(carom_id=carom_id)
    obj_request.state="P"
    obj_request.save()
    sleep(t*3)
    print("======== detect done ============")
    balls_coord(carom_id=carom_id, coord= MEDIA_ROOT / "test" / "test.json").save()
    obj_request.state="D"
    obj_request.save()
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
            runner = threading(target=test_make_coord, args=(carom_id, usr))
            print("======== start Process ============")
            runner.start()
        except Exception as ex:
            print("ex : "+ str(ex))
        
    
    def get(self, request, carom_id, usr, format=None):
        obj_request = None
        try:#존재할 경우
            obj_request = detect_request.objects.get(carom_id=carom_id)
        except detect_request.DoesNotExist:
            #존재하지 않으면 새로 만든다.
            print("========make coord============")
            detect_request(carom_id=carom_id, requester=usr, state="C").save()
            self.make_coord(carom_id)
            return Response({"state":"Create"}, status.HTTP_202_ACCEPTED)
        
        #아직 연산중 이라면
        if obj_request.state == "C":
            #detect PIPE
            self.make_coord(carom_id)
            return Response({"state":obj_request.get_state_display()}, status=status.HTTP_202_ACCEPTED) #기다리라고 한다.
        elif obj_request.state == "P":
            return Response({"state":obj_request.get_state_display()}, status=status.HTTP_202_ACCEPTED) #기다리라고 한다.
            
        # 다 완료됐다면
        elif obj_request.state == "D":
            coord = self.get_coord(carom_id=carom_id)
            serializer = CoordSerializer(coord)
            return Response(serializer.data, status=status.HTTP_200_OK)