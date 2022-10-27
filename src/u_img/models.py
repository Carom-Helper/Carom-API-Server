from django.db import models
from django.db import models
from django.db.models import CASCADE, Model
# from django_db_views.db_view import DBView

# Create your models here.

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

from detection.models import projection_method

class carom(Model):
    img = models.ImageField(upload_to="carom/%Y/%m/%d/", verbose_name="Image")
    method = models.ForeignKey(to="detection.projection_method", on_delete=CASCADE, verbose_name="Type")
    table_size = models.DecimalField(verbose_name="Persent", max_digits=3, decimal_places=1)