# python manage.py shell_plus --print-sql
from u_img.models import carom_data,carom_img
from recommend.models import position

print("==== Start ====")

#Create projection_method
print("===== Create sample Image ====")
guide =  {
            "BR": [
                1270,
                580
            ],
            "TL": [
                549,
                109
            ],
            
            "TR": [
                180,
                565
            ],
            "BL": [
                942,
                112
            ],
        }

img = carom_img(img='carom/sample.jpg')
img.save()
data = carom_data(img=img, guide=guide)
data.save()

print("==== Create init position ===")
coord = {
    "cue":[300,400], 
    "obj1":[100,750],
    "obj2":[300,300]
}
pos = position(coord=coord)
pos.save()
# try:
#     projection_method(name="top-view", value=10).save()
#     print("Create Projection : "+ str(projection_method.objects.all()))
# except Exception as ex:
#     print("Pass Projection : " + str(ex))