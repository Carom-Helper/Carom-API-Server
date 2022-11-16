from action_cls import CaromBall

temp = CaromBall()
temp.start_param(clock = 12, tip = 1)
temp.print_param()

temp.add_xy(300,400,0)
temp.get_vector_to_tar(350,450,0)
print(temp.move_by_time(1))