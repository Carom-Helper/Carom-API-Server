from django.test import TestCase, LiveServerTestCase
from .views import *
from random import randint

def ball_creater():
    x = randint(7, 393)
    y = randint(7, 793)
    return [x,y]

# Create your tests here.
class RouteTestClass(LiveServerTestCase):
    def setUp(self):
        coord = {
            "cue":ball_creater(), 
            "obj1":ball_creater(),
            "obj2":ball_creater(),
        }
        position(coord=coord).save()
        
        pos = position.objects.last()
        print("+++++++++++++++++ position ++++++++++++++++++++")
        print(pos.coord, pos.state)
        self.assertDictEqual(pos.coord, coord)
        
    def test_make_route(self):
        pos = position.objects.last()
        try:
            test_make_route(pos.id, display=True)
        except:
            pass
        
    def test_get_route(self):
        response = lastpos_route("tglee")
        self.assertNotEqual(response , Response({"state":"Progress"}, status=status.HTTP_202_ACCEPTED))
        self.assertNotEqual(response , Response({"message":"Route doesn't exist"}, status=status.HTTP_404_NOT_FOUND))