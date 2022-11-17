# python manage.py shell_plus --print-sql
from u_img.models import carom_data,carom_img

print("==== Start ====")

#Create projection_method
print("===== Create sample Image ====")
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
                94,
                111
            ],
            "BL": [
                180,
                565
            ]
        }

img = carom_img(img='carom/sample.jpg')
img.save()
data = carom_data(img=img, guide=guide)
data.save()
# try:
#     projection_method(name="top-view", value=10).save()
#     print("Create Projection : "+ str(projection_method.objects.all()))
# except Exception as ex:
#     print("Pass Projection : " + str(ex))