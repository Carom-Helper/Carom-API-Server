from action_cls import CaromBall, set_vec

cue = CaromBall()
cue.start_param(clock = 12, tip = 1)
cue.print_param()

tar = CaromBall()
tar.set_xy(350,450)

cue.set_xy(300,400)
set_vec(cue, tar, 0)
cue.set_mover(cue.move_by_time)
print(cue.move(1))