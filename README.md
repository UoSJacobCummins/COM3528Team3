# COM3528Team3
Cognitive and Biomimetic Robotics

Motivation in animals can be modelled through an oscillating framework - needs oscillate back and forth in strength, such as the need to find food or to drink water. This framework has been developed as a dynamical system for MiRo to simulate motivational drives; this project adds a cost function to that system, modulating the probability of taking an action based on its relative cost\footnote{Model provided by Alejandro Jiminez-Rodriguez}. The cost function takes distance to stimuli into account when considering what action to take.  We find that although the cost function is sound in theory and has non-negligible impact on the dynamical system, further experiments with larger roaming areas for the system, and greater variance in stimuli, are required to yield more decisive data.

Included in the repo is a video showing expected results when running with the default parameters.

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
