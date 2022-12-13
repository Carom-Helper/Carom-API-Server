from django.test import TestCase, LiveServerTestCase
from u_img.models import carom_img, carom_data
from .views import test_make_coord, balls_coord
from carom_api.settings import FRAME_WORK
from .detect.pipe_factory import PipeFactory

import cv2

# Create your tests here.
class DetectTestClass(LiveServerTestCase):
    image_size=(720, 1280)
    
    def setUp(self):
        #DB에 이미지 삽입
        view = "B"
        guide =  {
            "TL": [662, 452],
            "TR": [662, 752],
            "BL": [57, 147],
            "BR": [57, 1057]
        }
        #C:/Users/qjrm6/inte/Carom-API-Server/src/media/carom/2022/11/15/
        img = carom_img(img='carom/CAP3825091495947943655.jpg')
        img.save()
        data = carom_data(img=img, guide=guide, view=view)
        data.save()
        self.img_id = img.id

        data = carom_data.objects.get(img_id=self.img_id)
        self.assertEquals(data.img_id, self.img_id)
        
    def test_singleton(self):
        print("Call PipeFactory ",end="|")
        pipe = PipeFactory(device=FRAME_WORK, display=True, image_size=self.image_size, inDB=True).pipe
        print("Call PipeFactory ",end="|")
        pipe = PipeFactory(device=FRAME_WORK, display=True, image_size=self.image_size, inDB=True).pipe
        print("Call PipeFactory ",end="|")
        pipe = PipeFactory(device=FRAME_WORK, display=True, image_size=self.image_size, inDB=True).pipe
        print("Call PipeFactory ",end="|")
        pipe = PipeFactory(device=FRAME_WORK, display=True, image_size=self.image_size, inDB=True).pipe
        print("Call PipeFactory ",end="|")
        pipe = PipeFactory(device=FRAME_WORK, display=True, image_size=self.image_size,inDB=True).pipe
        print("Call PipeFactory ",end="|")
        pipe = PipeFactory(device=FRAME_WORK, display=True, image_size=self.image_size, inDB=True).pipe
        print()
    
    def test_detect(self):
        self.assertEqual(len(balls_coord.objects.all()), 0)
        img = carom_img.objects.last()
        test_make_coord(img.id, image_size=self.image_size, display=False if FRAME_WORK=='furiosa' else True)
        self.assertEqual(len(balls_coord.objects.all()), 1)
        ball = balls_coord.objects.last()
        print("=========== detect ball ==================")
        print(ball.coord)
        x,y = ball.coord["1"]
        self.assertEqual(x*y > 0, True)
        x,y = ball.coord["2"]
        self.assertEqual(x*y > 0, True)
        x,y = ball.coord["3"]
        self.assertEqual(x*y > 0, True)
        

    def test_framework(self):
        self.assertEqual(FRAME_WORK, 'furiosa')
