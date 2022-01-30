from main import WavePacket
from utils import tunneling_probability

barrier_height = [x *0.1 for x in range(1, 6)]
barrier_width =  [x *0.1 for x in range(1, 10)]
# barrier_width = [x for x in range(5, 50, 5)]

for height in barrier_height:
  curr_wp = WavePacket(barrier_height=height)
  print(f"Energy: {curr_wp.E}, Height: {height}")
  with open('barrier_height.csv', 'a+') as f:
    print(str(height) + "," + str(tunneling_probability(curr_wp.V0, curr_wp.E, curr_wp.m,
                                                      curr_wp.hbar, curr_wp.THCK).real ), file=f)
  
for width in barrier_width:
  curr_wp = WavePacket(barrier_width=width, barrier_height=0.1)
  print(f"Energy: {curr_wp.E}, Width: {width}")
  with open('barrier_width.csv', 'a+') as f:
    print(str(width) + "," + str(tunneling_probability(curr_wp.V0, curr_wp.E, curr_wp.m,
                                                      curr_wp.hbar, curr_wp.THCK).real ), file=f)
 
print("Cvs files created")