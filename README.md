# COM3528Team3
Cognitive and Biomimetic Robotics

How to run:

1. Download the repository.
2. Move the Sim folder to the MDK sim folder. Merge all conflicts.
3. In terminal 1:
4. Ensure simulator mode is enabled.
5. $ roscore
6. In terminal 2:
7. Navigate to the sim folder.
8. $ ./launch_sim_fix_cog.sh
9. In terminal 3:
10. Go to catkin_ws
11. $ catkin build
12. $ source devel/setup.bash
13. Go to src folder of the project
14. $ python3 main.py
