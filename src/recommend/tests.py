from django.test import TestCase, LiveServerTestCase
from .views import *
from random import randint
from carom_api.settings import FRAME_WORK

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
        import threading
        # 연산 돌리기 전에 경로의 수 확인
        before_route_num = len(soultion_route.objects.all())
        print("before) route len = ", before_route_num)
        # 연산!
        pos = position.objects.last()
        id = pos.id
        
        thrd = threading.Thread(target=Simulate_route.RUN_THREAD, args=(id, False))
        #                                       make_coord.run(id, display= False)#False if FRAME_WORK=='furiosa' else True)
        thrd.start()
        
        delay_time = 10
        repeat_num = 20
        max_calc_time = delay_time * repeat_num
        
        import time
        for i in range(1, repeat_num):
            time.sleep(delay_time)    
            pos = position.objects.get(id=id)
            state = pos.state
            
            if state == 'D':
                break
            else:
                print(f"({i}:{state})", end="")
            
        print()
        after_route = soultion_route.objects.all().values_list("issue", "route", "algorithm_ver")
        print('\n',len(after_route))
        for route in after_route:
            print(route)
        print("after) route len = ", len(after_route))
        self.assertEqual(before_route_num + 3, len(after_route))
        for route in after_route:
            print(route)
        
    # def test_get_route(self):
    #     response = lastpos_route("tglee")
    #     self.assertNotEqual(response , Response({"state":"Progress"}, status=status.HTTP_202_ACCEPTED))
    #     self.assertNotEqual(response , Response({"message":"Route doesn't exist"}, status=status.HTTP_404_NOT_FOUND))
