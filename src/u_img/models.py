from django.db import models
from django.db.models import CASCADE, Model, F
from django_db_views.db_view import DBView

from carom_api.settings import BASE_DIR, MEDIA_ROOT

# Create your models here.

class carom_img(Model):
    img = models.ImageField(upload_to="carom/", verbose_name="Image")

class carom_data(Model):
    WORK_STATE = [
        ('N', "None"),
        ('A', "Accepted"),
        ('P', "Progress"),
        ('D', "Done"),
    ]
    img = models.ForeignKey(to="carom_img", on_delete=CASCADE, verbose_name="Image ID")
    guide = models.JSONField(default=dict, verbose_name="Guide JSON")
    detect_state = models.CharField(max_length=1, choices=WORK_STATE, verbose_name="Work State", default="N")
    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["img"],
                name = "unique img id"
            )
        ]
    
class CaromTable(DBView):
    WORK_STATE = [
        ('N', "None"),
        ('A', "Accepted"),
        ('P', "Progress"),
        ('D', "Done"),
    ]
    
    guide = models.JSONField(verbose_name="Guide Point", default=dict, null=True)
    img = models.ImageField(verbose_name="Image", default="/", null=True)
    detect_state = models.CharField(max_length=1, choices=WORK_STATE, verbose_name="Work State", default="N", null=True)
    
    
    view_definition = lambda: str(
        carom_data.objects.select_related(
            "img"
            ).values(
                "img_id", 
                "img__img", 
                "guide", 
                "detect_state"
            ).annotate(
                id = F('img_id'),
                img = F('img__img')
            ).all().query
    )
    class Meta:
        managed = False 
        db_table = "CaromTable"