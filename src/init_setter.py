from detection.models import projection_method

print("==== Start ====")

#Create projection_method
print("===== Create Projection Method ====")
try:
    projection_method(name="top-view", value=10).save()
    print("Create person : "+ str(projection_method.objects.all()))
except Exception as ex:
    print("Pass person : " + str(ex))