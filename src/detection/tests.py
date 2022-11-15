from django.test import TestCase, LiveServerTestCase
from u_img.models import carom
from .views import test_make_coord, balls_coord
import cv2

# Create your tests here.
class testclass(LiveServerTestCase):
    def setUp(self):
        #DB에 이미지 삽입
        guide =  {
            "TL": [
                549,
                109
            ],
            "BR": [
                1270,
                580
            ],
            "TR": [
                942,
                111
            ],
            "BL": [
                180,
                565
            ]
        }
        #C:/Users/qjrm6/inte/Carom-API-Server/src/media/carom/2022/11/15/
        img = carom(img='carom/sample.jpg', guide=guide)
        img.save()
        self.img_id = img.id

        img = carom.objects.get(id=self.img_id)
        self.assertEquals(img.id, self.img_id)

    def test_test(self):
        self.assertEqual(len(balls_coord.objects.all()), 0)
        test_make_coord(1, display=False)
        self.assertEqual(len(balls_coord.objects.all()), 1)


