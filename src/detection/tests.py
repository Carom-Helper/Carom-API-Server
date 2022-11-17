from django.test import TestCase, LiveServerTestCase
from u_img.models import carom_img, carom_data
from .views import test_make_coord, balls_coord
from carom_api.settings import FRAME_WORK
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
                112
            ],
            "BL": [
                180,
                565
            ]
        }
        #C:/Users/qjrm6/inte/Carom-API-Server/src/media/carom/2022/11/15/
        img = carom_img(img='carom/sample.jpg')
        img.save()
        data = carom_data(img=img, guide=guide)
        data.save()
        self.img_id = img.id

        data = carom_data.objects.get(img_id=self.img_id)
        self.assertEquals(data.img_id, self.img_id)

    def test_test(self):
        self.assertEqual(len(balls_coord.objects.all()), 0)
        img = carom_img.objects.last()
        test_make_coord(img.id, display=True)
        self.assertEqual(len(balls_coord.objects.all()), 1)

    def test_framework(self):
        self.assertEqual(FRAME_WORK, 'furiosa')
