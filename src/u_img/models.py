from django.db import models
from django.db.models import CASCADE, Model

from carom_api.settings import BASE_DIR, MEDIA_ROOT

# Create your models here.


# class projection_method(Model):
#     name = models.CharField(max_length=100, verbose_name="NAME")
#     value = models.PositiveSmallIntegerField(verbose_name="VALUE", null=True)
    
#     def __str__(self) -> str:
#         return str(self.name)

class carom(Model):
    WORK_STATE = [
        ('N', "None"),
        ('A', "Accepted"),
        ('P', "Progress"),
        ('D', "Done"),
    ]
    
    img = models.ImageField(upload_to="carom/%Y/%m/%d/", verbose_name="Image")
    guide = models.JSONField(default=dict, verbose_name="Guide JSON")
    detect_state = models.CharField(max_length=1, choices=WORK_STATE, verbose_name="Work State", default="N")
    # method = models.ForeignKey(to="projection_method", on_delete=CASCADE, verbose_name="Type")
    # table_size = models.DecimalField(verbose_name="Persent", max_digits=4, decimal_places=1)
    
    
