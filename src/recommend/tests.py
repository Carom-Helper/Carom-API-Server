from django.test import TestCase, LiveServerTestCase
from .views import *
from threading import Thread

# Create your tests here.
class RouteTestClass(LiveServerTestCase):
    def setUp(self):
        coord = {
            "cue":[300,400], 
            "obj1":[100,750],
            "obj2":[300,300],
        }
        position(coord=coord).save()
        
        pos = position.objects.last()
        print("+++++++++++++++++ position ++++++++++++++++++++")
        print(pos.coord, pos.state)
        self.assertDictEqual(pos.coord, coord)
        
    def test_make_route(self):
        pos = position.objects.last()
        test_make_route(pos.id, display=True)
        
    def test_get_route(self):
        response = lastpos_route("tglee")
        self.assertNotEqual(response , Response({"state":"Progress"}, status=status.HTTP_202_ACCEPTED))
        self.assertNotEqual(response , Response({"message":"Route doesn't exist"}, status=status.HTTP_404_NOT_FOUND))