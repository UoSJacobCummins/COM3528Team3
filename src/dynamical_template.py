# This is a simplified version of the code that can be readily used inside one
# ROS node. 
# The class structure is as follows:
# 

from ast import Global
import numpy as np
import matplotlib.pyplot as plt
import cost
from enum import Enum


from datetime import datetime

# MiroActions
import random
import os
import cv2

import miro2 as miro  # MiRo Developer Kit library
try:  # For convenience, import this util separately
    from miro2.lib import wheel_speed2cmd_vel  # Python 3
except ImportError:
    from miro2.utils import wheel_speed2cmd_vel  # Python 2
    
import rospy  # ROS Python interface
from std_msgs.msg import (
    Float32MultiArray,
    UInt32MultiArray,
    UInt16,
)  # Used in callbacks
from geometry_msgs.msg import TwistStamped  # ROS cmd_vel (velocity control)

from sensor_msgs.msg import CompressedImage  # ROS CompressedImage message
from cv_bridge import CvBridge, CvBridgeError  # ROS -> OpenCV converter

DISTANCE = [100,100,100]
DEPTH = -1

class State(Enum):
# Enumeration of the possible states of the actionsystem
    SEARCH = 1
    FOLLOW = 2
    CONSUME = 3

class Percept:
# Represents a percept with useful information to locate it in egocentric space
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.depth = 2
        self.intensity = 0.0

class Controller:
# The controller simulates the dynamical system
    def __init__(self) -> None:
        self.start_time = datetime.now()
        # Parameters, check the main document
        self.a1 = 2.0
        self.a2 = 2.0
        self.b1 = 50.0
        self.b2 = 50.0
        self.sigma = 10.0
        self.epsilon = 0.05 
        
        # ρ symbol is rho
        self.rho1 = -1.0
        self.rho2 = 1.0
        self.h = 0.05 # small time step
        # Component classes
        self.perceptualSystem = PerceptualSystem()
        self.hunger = HungerMotivationalSystem( self.perceptualSystem )
        self.thirst = ThirstMotivationalSystem( self.perceptualSystem )
        self.X0 = [0.0, 1.0, 1.0] # Initial state
        self.X = self.X0 # Current state, if you want to keep the time series, this should be an array
        self.time = 0.0
        # η symbol is eta. This is motivation to do something
        self.eta1 = 0.0
        self.eta2 = 0.0
        
        self.data = {'E1':[], 'E2': [], 'rho': [], 'xi_r': [], 'xi_g': [], 
					 'reward_g': [], 'reward_r': [], 'vel_l': [], 'vel_r':[],
					 'a':[], 'b':[], 'total_r1' : [], 'total_r2' : []}
        
        self.miroActions = MiroActions()
        

    def integrate( self, f ):
        # Perform a Runge-Kutta step        
        k1 = f(self.time, self.X)
        k2 = f(self.time + self.h/2.0, self.X + self.h*k1/2.0)
        k3 = f(self.time + self.h/2.0, self.X + self.h*k2/2.0)
        k4 = f(self.time + self.h, self.X + self.h*k3)
        self.X = self.X + self.h*(k1 + 2*k2 + 2*k3 + k4)/6.0
        self.time = self.time + self.h
        
        
    def map(self, t, x, input_u = 0.0, input_v = 0.0 ): 
    # The evolution map for the dynamical system
    # subtract 
        # g1 = lambda q1, q2, rho: -self.a1*q1 - cost.cost_constant_stimulus(DISTANCE[1], input_u) + self.b1*(1.0 - q1)*np.exp(-self.sigma*(rho - self.rho1)**2)*(1.0 + input_u)
        # g2 = lambda q1, q2, rho: -self.a2*q2 - cost.cost_constant_stimulus(DISTANCE[2], input_v) + self.b2*(1.0 - q2)*np.exp(-self.sigma*(rho - self.rho2)**2)*(1.0 + input_v)

        g1 = lambda q1, q2, rho: -self.a1*q1 + self.b1*(1.0 - q1)*np.exp(-self.sigma*(rho - self.rho1)**2)*(1.0 + input_u)
        g2 = lambda q1, q2, rho: -self.a2*q2 + self.b2*(1.0 - q2)*np.exp(-self.sigma*(rho - self.rho2)**2)*(1.0 + input_v)
        f = lambda q1, q2, rho: -4.0*((rho - self.rho1)*(rho - self.rho2)*(rho - (self.rho1+self.rho2)/2.0) + 
                                    (1-q1)*(rho - self.rho1)/2.0 + (1-q2)*(rho - self.rho2)/2.0)

        calc_g1 = g1(x[0], x[1], x[2])
        calc_g2 = g2(x[0], x[1], x[2])
        #print("g1 = " + str(calc_g1))
        #print("g2 = " + str(calc_g2))

        return np.array([self.epsilon*g1(x[0], x[1], x[2]),
                    self.epsilon*g2(x[0], x[1], x[2]),
                    f(x[0], x[1], x[2])])
    
    def plots(self):
        duration = (datetime.now() - self.start_time).seconds
        #print(self.data['b'])
        data_a = self.data['a']
        data_b = self.data['b']
        if len(data_a) > len(data_b):
            data_a = data_a[:-1]
        no_of_points = len(self.data['a'])
        #print(duration)
        interval = duration / no_of_points
        time_points = np.arange(0, duration, interval).tolist()
        #print(time_points)
        plt.plot(time_points,self.data['a'], label='Eta 1')
        plt.plot(time_points,self.data['b'], label='Eta 2')
        plt.plot(time_points,self.data['total_r1'], label='Total eats')
        plt.plot(time_points,self.data['total_r2'], label='Total drinks')
        plt.xlabel("Time")
        plt.ylabel("eta")
        plt.legend()
        plt.show()


    def step( self ):
    # Performs one integration step updating the inputs and outputs of
    # the model
        # Observe objects in the environment and their distance
        global DISTANCE 
        DISTANCE = self.miroActions.distance_of_objects()
            
        #print("eta1 = " + str(self.eta1))
        r1 = self.hunger.activate( self.eta1 )
        # #print("eta2 = " + str(self.eta2))
        r2 = self.thirst.activate( self.eta2 )

        f = lambda t, x: self.map( t, x, r2*0.0, r1*0.0 )
        self.integrate( f )

        print("X[2]: " + str(self.X[2]) + ", eta1: " + str(self.eta1) + ", eta2: " + str(self.eta2) )
        #print("X1" + str(self.X[1]))

        b_sigma = 5.0
        self.eta1 = np.exp(-b_sigma*(self.X[2] - self.rho1)**2)
        self.eta2 = np.exp(-b_sigma*(self.X[2] - self.rho2)**2)
        
        #print("r1",r1)
        #print("r2",r2)
        self.data['a'].append(self.eta1)
        self.data['b'].append(self.eta2)
        self.data['total_r1'].append(r1)
        self.data['total_r2'].append(r2)

class ActionSystem:
# The action system manages the seach/follow/consume pattern and logic
    def __init__( self, motivationalSystem ):
        self.state = State.SEARCH
        self.motivationalSystem = motivationalSystem
        self.miroActions = MiroActions()
        
    def search( self, eta ):
        # Modulate the search action by eta, you can choose to either set a hard 
        # threshold or a soft activation in which the ghost of each motivational system
        # remains but feeble

        if(eta < 0.2):
            return None

        random_action = random.randint(0,10)
        if random_action < 5:
            # Move forward.
            self.miroActions.forward(rospy.Time.now())
        else:
            # Turn
            self.miroActions.rotate(rospy.Time.now())
        percept = self.motivationalSystem.perceive()
        self.motivationalSystem.express( eta )
        #print("search",percept)
        if percept is None:
            print('Doing search')
        else:
            return percept

    def follow( self, eta ):
        if(eta < 0.2):
            return None

        self.motivationalSystem.express( eta )
        #print('Pursuing the percept')
        percept = self.motivationalSystem.perceive()
        self.motivationalSystem.approach()
        # Need to toggle colours based on food or water.
        #self.miroActions.face_ball(rospy.Time.now(),[255.0,0.0,0.0])
        #self.miroActions.forward(rospy.Time.now())
        return percept

    def consume( self, eta ):
        # Consuming activates action patterns when the target has been achieved
        # In this case the function returns the consumed reward used as forcing term in
        # the dynamics
        if(eta < 0.2):
            return None

        self.motivationalSystem.express( eta )
        r = self.motivationalSystem.consume( eta )
        print('Consuming')
        return r


    def changeState( self ):
        # This function implements the finite state machine for the actions
        # The challenge is to define when something is close
        #print("Hello",self.motivationalSystem.currentPercept)
        if self.motivationalSystem.currentPercept == None:
            #print("IN THIS ONE")
            self.state = State.SEARCH
        else:
            #print("DEPTH",DEPTH)
            if DEPTH < 1 and DEPTH > -1: # Adjust the threshold
                self.state = State.CONSUME
            else:
                self.state = State.FOLLOW

class PerceptualSystem:
# This class is in charge of managing the raw inputs from the robot
# using visual system as an example
    def __init__(self):
        self.currentImage = None

class MotivationalSystem:
# The MotivationalSystem generates the motivational system's specific action patterns.
# This is an abstract class
    def __init__( self ):
        #self.perceptualSystem = perceptualSystem
        self.actionSystem = ActionSystem(self)
        self.currentPercept = None
        self.miroActions = MiroActions()

    def activate( self, eta ):
        #print("eta",eta)
        if self.actionSystem.state == State.SEARCH:
            self.currentPercept = self.actionSystem.search(eta)
        elif self.actionSystem.state == State.FOLLOW:
            currentPercept = self.actionSystem.follow(eta)
        elif self.actionSystem.state == State.CONSUME:
            if eta > 0.5:
                r = self.actionSystem.consume(eta)
            else: 
                currentPercept = None

        self.actionSystem.changeState()
        # Added check. If reward is given by consume then r exists. Otherwise
        # the reward is 0.
        if 'r' in locals():
            return r
        else:
            return 1

    def perceive( self ):
        # Override
        None
    
    def approach( self ):
        # Override
        None

    def express( self, eta ):
        # Override
        None

    def consume( self, eta ):
        print("None")
        # Override
        r = 0 
        return r

# Child classes for the specific motivational systems
class HungerMotivationalSystem(MotivationalSystem):
    def __init__( self, perceptualSystem ):
        MotivationalSystem.__init__(self)

    def perceive( self ):
        # Do the logic to detect the stimulus associated to this motivational system
        #p = None        
        #currentImage = self.perceptualSystem.currentImage
        print("Hunger Percept")
        print(DISTANCE[1])
        if DISTANCE[1] == -1:
            p = None
        else:
            p = DISTANCE[1]
        global DEPTH
        DEPTH = DISTANCE[1]
        #print(self.distance[2])
        return p # or None
    
    def approach( self ):
        # Do the logic to approach the stimulus associated to this motivational system
        self.miroActions.face_ball(rospy.Time.now(),[0.0,255.0,0.0])
        self.miroActions.forward(rospy.Time.now())

    def express( self, eta ):
        # Action patterns related to the motivational system, not directly related to movement
        None

    def consume( self, eta ):
        print ("Eating")
        # define what is the reward based on the interaction with the target
        global CONSUMING_0
        CONSUMING_0 = True
        r = 0
        return r 

class ThirstMotivationalSystem(MotivationalSystem):
    def __init__( self, perceptualSystem ):
        MotivationalSystem.__init__(self )

    def perceive( self ):
        # Do the logic to detect the stimulus associated to this motivational system
        #p = Percept()        
        #currentImage = self.perceptualSystem.currentImage
        print("Thirst Percept")
        print(DISTANCE[2])
        if DISTANCE[2] == -1:
            p = None
        else:
            p = DISTANCE[2]
        global DEPTH
        DEPTH = DISTANCE[2]
        #print(self.distance[2])
        return p # or None
    
    def approach( self ):
        # Do the logic to approach the stimulus associated to this motivational system
        self.miroActions.face_ball(rospy.Time.now(),[255.0,0.0,0.0])
        self.miroActions.forward(rospy.Time.now())

    def express( self, eta ):
        # Action patterns related to the motivational system, not directly related to movement
        None

    def consume( self, eta ):
        print ("Drinking")
        # define what is the reward based on the interaction with the target
        global CONSUMING_1
        CONSUMING_1 = True
        r = 0
        return r

class MiroActions:
    ACTION_DURATION = rospy.Duration(1.0)  # seconds
    
    def __init__( self ):
        topic_root = "/" + os.getenv("MIRO_ROBOT_NAME")
        
        # Define ROS publishers
        self.pub_cmd_vel = rospy.Publisher(
            topic_root + "/control/cmd_vel", TwistStamped, queue_size=0
        )
        
        self.vel_pub = rospy.Publisher(
            topic_root + "/control/cmd_vel", TwistStamped, queue_size=0
        )
        
        self.hostile_colour = [0.0,0.0,255.0]
            
        self.food_colour = [0.0,255.0,0.0]
            
        self.water_colour = [255.0,0.0,0.0]
            
        self.colours = [self.hostile_colour, self.food_colour, self.water_colour]
        
        self.image_converter = CvBridge()
        
        self.sub_caml = rospy.Subscriber(
            topic_root + "/sensors/caml/compressed",
            CompressedImage,
            self.callback_caml,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.sub_camr = rospy.Subscriber(
            topic_root + "/sensors/camr/compressed",
            CompressedImage,
            self.callback_camr,
            queue_size=1,
            tcp_nodelay=True,
        )
        
        # Create handle to store images
        self.input_camera = [None, None]
        # New frame notification
        self.new_frame = [False, False]
        # Create variable to store a list of ball's x, y, and r values for each camera
        self.ball = [None, None]
        # Set the default frame width (gets updated on reciecing an image)
        self.frame_width = 640
        
        rospy.sleep(1.0)
        
    def callback_caml(self, ros_image):  # Left camera
        self.callback_cam(ros_image, 0)

    def callback_camr(self, ros_image):  # Right camera
        self.callback_cam(ros_image, 1)

    def callback_cam(self, ros_image, index):
        """
        Callback function executed upon image arrival
        """
        # Silently(-ish) handle corrupted JPEG frames
        try:
            # Convert compressed ROS image to raw CV image
            image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "rgb8")
            # Convert from OpenCV's default BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Store image as class attribute for further use
            self.input_camera[index] = image
            # Get image dimensions
            self.frame_height, self.frame_width, channels = image.shape
            self.x_centre = self.frame_width / 2.0
            self.y_centre = self.frame_height / 2.0
            # Raise the flag: A new frame is available for processing
            self.new_frame[index] = True
        except CvBridgeError as e:
            # Ignore corrupted frames
            pass
    
    def distance_of_objects(self):
        KNOWN_WIDTH = 0.2
        FOCAL_LENGTH = 261.33966
        
        # Use to calibrate camera focal length to calculate distnaces based
        # on pixel widths.
        #KNOWN_DISTANCE = 1.451887
        #pixel_width = self.find_marker(self.input_camera[0])
        #focalLength = (pixel_width * KNOWN_DISTANCE) / KNOWN_WIDTH
        #print("Flength:", focalLength)
        
        colour_distances = [-1,-1,-1]
        for i in range(len(self.colours)):
            
            # Pixel widths of left and right camera.
            pixel_width_zero = self.find_marker(self.input_camera[0], self.colours[i])
            pixel_width_one = self.find_marker(self.input_camera[1], self.colours[i])
            
            # If left and right camera then average them both.
            if pixel_width_zero and pixel_width_zero:
                pixel_width = (pixel_width_zero + pixel_width_one) / 2
            # If only camera zero then use only pixel width zero.
            elif pixel_width_zero:
                pixel_width = pixel_width_zero
            # Otherwise either only camera one or it can't be seen.
            else:
                pixel_width = pixel_width_one
            if pixel_width:
                colour_distances[i] = self.distance_to_camera(KNOWN_WIDTH,FOCAL_LENGTH,pixel_width)
        #print(self.colours)
        #print(colour_distances)
        return colour_distances
    
    def distance_to_camera(self, knownWidth, focalLength, perWidth):
        # compute and return the distance from the maker to the camera
        return (knownWidth * focalLength) / perWidth
    
    def detect_ball(self, frame, index, colour):
        """
        Image processing operations, fine-tuned to detect a small,
        toy blue ball in a given frame.
        """
        if frame is None:  # Sanity check
            return

        # Flag this frame as processed, so that it's not reused in case of lag
        self.new_frame[index] = False
        # Get image in HSV (hue, saturation, value) colour format
        im_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Specify target ball colour
        bgr_colour = np.uint8([[colour]])  # e.g. Blue (Note: BGR)
        # Convert this colour to HSV colour model
        hsv_colour = cv2.cvtColor(bgr_colour, cv2.COLOR_RGB2HSV)

        # Extract colour boundaries for masking image
        # Get the hue value from the numpy array containing target colour
        target_hue = hsv_colour[0, 0][0]
        hsv_lo_end = np.array([target_hue - 20, 70, 70])
        hsv_hi_end = np.array([target_hue + 20, 255, 255])

        # Generate the mask based on the desired hue range
        mask = cv2.inRange(im_hsv, hsv_lo_end, hsv_hi_end)
        mask_on_image = cv2.bitwise_and(im_hsv, im_hsv, mask=mask)
        
        #cv2.imshow('image', mask_on_image)
        #cv2.waitKey()

        # Clean up the image
        seg = mask
        seg = cv2.GaussianBlur(seg, (5, 5), 0)
        seg = cv2.erode(seg, None, iterations=2)
        seg = cv2.dilate(seg, None, iterations=2)

        # Fine-tune parameters
        ball_detect_min_dist_between_cens = 40  # Empirical
        canny_high_thresh = 10  # Empirical
        ball_detect_sensitivity = 1  # Lower detects more circles, so it's a trade-off
        ball_detect_min_radius = 5  # Arbitrary, empirical
        ball_detect_max_radius = 50  # Arbitrary, empirical

        # Find circles using OpenCV routine
        # This function returns a list of circles, with their x, y and r values
        circles = cv2.HoughCircles(
            seg,
            cv2.HOUGH_GRADIENT,
            1,
            ball_detect_min_dist_between_cens,
            param1=canny_high_thresh,
            param2=ball_detect_sensitivity,
            minRadius=ball_detect_min_radius,
            maxRadius=ball_detect_max_radius,
        )

        if circles is None:
            # If no circles were found, just display the original image
            return

        # Get the largest circle
        max_circle = None
        self.max_rad = 0
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            if c[2] > self.max_rad:
                self.max_rad = c[2]
                max_circle = c
        # This shouldn't happen, but you never know...
        if max_circle is None:
            return

        # Append detected circle and its centre to the frame
        cv2.circle(frame, (max_circle[0], max_circle[1]), max_circle[2], (0, 255, 0), 2)
        cv2.circle(frame, (max_circle[0], max_circle[1]), 2, (0, 0, 255), 3)

        # Normalise values to: x,y = [-0.5, 0.5], r = [0, 1]
        max_circle = np.array(max_circle).astype("float32")
        max_circle[0] -= self.x_centre
        max_circle[0] /= self.frame_width
        max_circle[1] -= self.y_centre
        max_circle[1] /= self.frame_width
        max_circle[1] *= -1.0
        max_circle[2] /= self.frame_width

        # Return a list values [x, y, r] for the largest circle
        return [max_circle[0], max_circle[1], max_circle[2]]
    
    def find_marker(self, image, colour):
        
        # Get image in HSV (hue, saturation, value) colour format
        im_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Specify target ball colour
        bgr_colour = np.uint8([[colour]])  # e.g. Blue (Note: BGR)
        # Convert this colour to HSV colour model
        hsv_colour = cv2.cvtColor(bgr_colour, cv2.COLOR_RGB2HSV)

        # Extract colour boundaries for masking image
        # Get the hue value from the numpy array containing target colour
        target_hue = hsv_colour[0, 0][0]
        hsv_lo_end = np.array([target_hue - 20, 70, 70])
        hsv_hi_end = np.array([target_hue + 20, 255, 255])

        # Generate the mask based on the desired hue range
        mask = cv2.inRange(im_hsv, hsv_lo_end, hsv_hi_end)
        mask_on_image = cv2.bitwise_and(im_hsv, im_hsv, mask=mask)
        
        width = sum(mask.any(axis=0))
        #print(' width:', width)
        #cv2.imshow('image', mask)
        #cv2.waitKey()
        return width
    
    def drive(self, speed_l=0.1, speed_r=0.1):  # (m/sec, m/sec)
        """
        Wrapper to simplify driving MiRo by converting wheel speeds to cmd_vel
        """
        # Prepare an empty velocity command message
        msg_cmd_vel = TwistStamped()

        # Desired wheel speed (m/sec)
        wheel_speed = [speed_l, speed_r]

        # Convert wheel speed to command velocity (m/sec, Rad/sec)
        (dr, dtheta) = wheel_speed2cmd_vel(wheel_speed)

        # Update the message with the desired speed
        msg_cmd_vel.twist.linear.x = dr
        msg_cmd_vel.twist.angular.z = dtheta

        # Publish message to control/cmd_vel topic
        self.vel_pub.publish(msg_cmd_vel)
    
    def face_ball(self, t0, colour):
        #print("MiRo finding ball")
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            ball = False
        
        while not ball:
            # Find ball
            for index in range(2):  # For each camera (0 = left, 1 = right)
                # Skip if there's no new image, in case the network is choking
                if not self.new_frame[index]:
                    continue
                image = self.input_camera[index]
                # Run the detect ball procedure
                self.ball[index] = self.detect_ball(image, index, colour)
            # Once a ball has been detected
            if not self.ball[0] and not self.ball[1]:
                self.drive(0.1 , -0.1)
            
            # Face ball
            for index in range(2):  # For each camera (0 = left, 1 = right)
                # Skip if there's no new image, in case the network is choking
                if not self.new_frame[index]:
                    continue
                image = self.input_camera[index]
                # Run the detect ball procedure
                self.ball[index] = self.detect_ball(image, index, colour)
            # If only the right camera sees the ball, rotate clockwise
            if not self.ball[0] and self.ball[1]:
                self.drive(0.1, -0.1)
            # Conversely, rotate counterclockwise
            elif self.ball[0] and not self.ball[1]:
                self.drive(-0.1, 0.1)
            # Make the MiRo face the ball if it's visible with both cameras
            elif self.ball[0] and self.ball[1]:
                error = 0.05  # 5% of image width
                # Use the normalised values
                left_x = self.ball[0][0]  # should be in range [0.0, 0.5]
                right_x = self.ball[1][0]  # should be in range [-0.5, 0.0]
                rotation_speed = 0.03  # Turn even slower now
                if abs(left_x) - abs(right_x) > error:
                    self.drive(rotation_speed, -rotation_speed)  # turn clockwise
                elif abs(left_x) - abs(right_x) < -error:
                    self.drive(-rotation_speed, rotation_speed)  # turn counterclockwise
                else:
                    # Successfully turned to face the ball
                    ball = True
            #print("SEARCHING BALL")
            rospy.sleep(0.000001)
        self.drive(0,0)
        #self.velocity.twist.linear.x = 0
        #self.velocity.twist.angular.z = 0
        #self.pub_cmd_vel.publish(self.velocity)
        
    def forward(self, t0):
        #print("MiRo moving forward")
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            self.drive(0.2,0.2)
        self.drive(0,0)
        
    def rotate(self, t0):
        #print("MiRo rotating")
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            self.drive(-0.1,0.1)
        self.drive(0,0)