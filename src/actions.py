#!/usr/bin/env python
"""
Simple action selection mechanism inspired by the K-bandit problem

Initially, MiRo performs one of the following actions on random, namely: 
wiggle ears, wag tail, rotate, turn on LEDs and simulate a Braitenberg Vehicle.

While an action is being executed, stroking MiRo's head will reinforce it, while  
 stroking MiRo's body will inhibit it, by increasing or reducing the probability 
 of this action being picked in the future.

NOTE: The code was tested for Python 2 and 3
For Python 3 the shebang line is
#!/usr/bin/env python3
"""
# Imports
##########################
import os
import numpy as np

import rospy  # ROS Python interface
from std_msgs.msg import (
    Float32MultiArray,
    UInt32MultiArray,
    UInt16,
)  # Used in callbacks
from geometry_msgs.msg import TwistStamped  # ROS cmd_vel (velocity control)

from sensor_msgs.msg import CompressedImage  # ROS CompressedImage message
from cv_bridge import CvBridge, CvBridgeError  # ROS -> OpenCV converter
import cv2

import miro2 as miro  # MiRo Developer Kit library
try:  # For convenience, import this util separately
    from miro2.lib import wheel_speed2cmd_vel  # Python 3
except ImportError:
    from miro2.utils import wheel_speed2cmd_vel  # Python 2
##########################


class MiRoClient:

    # Scripts settings below
    ACTION_DURATION = rospy.Duration(3.0)  # seconds
    TICK = 0.3  # Step interval of action execution (in secs)
    VERBOSE = True  # Print out the Q and N values after each iteration
    ##NOTE The following option is relevant in MiRoCODE
    NODE_EXISTS = False  # Disables (True) / Enables (False) rospy.init_node


    def __init__(self):
        """
        Class initialisation
        """
        print("Initialising the controller...")

        # Get robot name
        topic_root = "/" + os.getenv("MIRO_ROBOT_NAME")

        # Initialise a new ROS node to communicate with MiRo
        if not self.NODE_EXISTS:
            rospy.init_node("kbandit", anonymous=True)

        # Define ROS publishers
        self.pub_cmd_vel = rospy.Publisher(
            topic_root + "/control/cmd_vel", TwistStamped, queue_size=0
        )
        self.pub_cos = rospy.Publisher(
            topic_root + "/control/cosmetic_joints", Float32MultiArray, queue_size=0
        )
        self.pub_illum = rospy.Publisher(
            topic_root + "/control/illum", UInt32MultiArray, queue_size=0
        )

        # Define ROS subscribers
        rospy.Subscriber(
            topic_root + "/sensors/touch_body",
            UInt16,
            self.touchBodyListener,
        )
        rospy.Subscriber(
            topic_root + "/sensors/light",
            Float32MultiArray,
            self.lightCallback,
        )
        
        self.vel_pub = rospy.Publisher(
            topic_root + "/control/cmd_vel", TwistStamped, queue_size=0
        )
        
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
        
        self.goal_ball_colour = [255.0,0.0,0.0]
        
        # Create handle to store images
        self.input_camera = [None, None]
        # New frame notification
        self.new_frame = [False, False]
        # Create variable to store a list of ball's x, y, and r values for each camera
        self.ball = [None, None]
        # Set the default frame width (gets updated on reciecing an image)
        self.frame_width = 640

        # List of action functions
        ##NOTE Yes, you can do such things
        ##TODO Try writing your own action functions and add them here
        self.actions = [
            self.earWiggle,
            self.tailWag,
            self.rotate,
            self.shine,
            self.braitenberg2a,
            self.forward,
            self.reverse,
            self.face_ball,
            self.spot_colour,
            self.circle,
        ]

        # Initialise objects for data storage and publishing
        self.light_array = None
        self.velocity = TwistStamped()
        self.cos_joints = Float32MultiArray()
        self.cos_joints.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.illum = UInt32MultiArray()
        self.illum.data = [
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
        ]

        # Utility enums
        self.tilt, self.lift, self.yaw, self.pitch = range(4)
        (
            self.droop,
            self.wag,
            self.left_eye,
            self.right_eye,
            self.left_ear,
            self.right_ear,
        ) = range(6)

        # Variables for Q-learning algorithm
        self.reward = 0
        self.action_to_do = 0  # Current action index
        self.Q = [0] * len(self.actions)  # Highest Q value gets to run
        self.N = [0] * len(self.actions)  # Number of times an action was done
        self.alpha = 0.7  # learning rate
        self.discount = 25  # discount factor (antidamping)

        # Give it a sec to make sure everything is initialised
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
        
    def detect_ball(self, frame, index):
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
        bgr_colour = np.uint8([[self.goal_ball_colour]])  # e.g. Blue (Note: BGR)
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

        # Clean up the image
        seg = mask
        seg = cv2.GaussianBlur(seg, (5, 5), 0)
        seg = cv2.erode(seg, None, iterations=2)
        seg = cv2.dilate(seg, None, iterations=2)

        # Fine-tune parameters
        ball_detect_min_dist_between_cens = 40  # Empirical
        canny_high_thresh = 10  # Empirical
        ball_detect_sensitivity = 10  # Lower detects more circles, so it's a trade-off
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
        
    def first_colour(self):
        colour_thresh = {"Red": ([0,190,100], [10,255,255]), "Green": ([20, 150, 100], [65,255,255]), "Turquoise": ([75,150,100], [100,255,255]),
                         "Blue": ([120, 225, 100], [130,255,255]),"Ball Blue": ([145,180,100], [160,255,255]), "Yellow": ([30, 180, 100], [30, 255, 255])}

        print("Searching for colour..")
        for colour_detected, (low,up) in colour_thresh.items():
            #cv2.imshow('image', self.input_camera[1])
            #cv2.waitKey()
            up_bound=np.array(up)
            low_bound=np.array(low)
            # Just right eye for now (easier).
            mask=cv2.inRange(self.input_camera[1], low_bound, up_bound)
            if mask.any():
                #TypeError: lowerb is not a numpy array, neither a scalar: SOLUTION

                self.colour_detected= colour_detected
                self.low_bound = low_bound
                self.up_bound = up_bound

                print("SEARCH INITIATED: The target beacon colour is {}".format(self.colour_detected))
                break


    def earWiggle(self, t0):
        print("MiRo wiggling ears")
        A = 1.0
        w = 2 * np.pi * 0.2
        f = lambda t: A * np.cos(w * t)
        i = 0
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            self.cos_joints.data[self.left_ear] = f(i)
            self.cos_joints.data[self.right_ear] = f(i)
            self.pub_cos.publish(self.cos_joints)
            i += self.TICK
        self.cos_joints.data[self.left_ear] = 0.0
        self.cos_joints.data[self.right_ear] = 0.0
        self.pub_cos.publish(self.cos_joints)


    def tailWag(self, t0):
        print("MiRo wagging tail")
        A = 1.0
        w = 2 * np.pi * 0.2
        f = lambda t: A * np.cos(w * t)
        i = 0
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            self.cos_joints.data[self.wag] = f(i)
            self.pub_cos.publish(self.cos_joints)
            i += self.TICK
        self.cos_joints.data[self.wag] = 0.0
        self.pub_cos.publish(self.cos_joints)


    def rotate(self, t0):
        print("MiRo rotating")
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            self.velocity.twist.linear.x = 0
            self.velocity.twist.angular.z = 0.2
            self.pub_cmd_vel.publish(self.velocity)
        self.velocity.twist.linear.x = 0
        self.velocity.twist.angular.z = 0
        self.pub_cmd_vel.publish(self.velocity)


    def shine(self, t0):
        print("MiRo turning on LEDs")
        color = 0xFF00FF00
        i = 0
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            ic = int(np.mod(i, 6))
            ip = int(np.mod(i + 1, 6))
            self.illum.data[ic] = color
            self.illum.data[ip] = 0x00000000
            self.pub_illum.publish(self.illum)
            i += self.TICK
        self.illum.data[ic] = 0x00000000
        self.pub_illum.publish(self.illum)


    def braitenberg2a(self, t0):
        print("MiRo simulates a Braitenberg Vehicle")
        if self.light_array is None:
            wheel_speed = [0, 0]
        else:
            wheel_speed = [self.light_array[1], self.light_array[0]]
        (dr, dtheta) = wheel_speed2cmd_vel(wheel_speed)
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            self.velocity.twist.linear.x = dr
            self.velocity.twist.angular.z = dtheta
            self.pub_cmd_vel.publish(self.velocity)
        self.velocity.twist.linear.x = 0
        self.velocity.twist.angular.z = 0
        self.pub_cmd_vel.publish(self.velocity)
        
    def reverse(self, t0):
        print("MiRo reversing")
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            self.velocity.twist.linear.x = -0.2
            self.velocity.twist.angular.z = -0.2
            self.pub_cmd_vel.publish(self.velocity)
        self.velocity.twist.linear.x = 0
        self.velocity.twist.angular.z = 0
        self.pub_cmd_vel.publish(self.velocity)
        
    def forward(self, t0):
        print("MiRo moving forward")
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            self.velocity.twist.linear.x = 0.2
            self.velocity.twist.angular.z = 0.2
            self.pub_cmd_vel.publish(self.velocity)
        self.velocity.twist.linear.x = 0
        self.velocity.twist.angular.z = 0
        self.pub_cmd_vel.publish(self.velocity)
        
    def circle(self, t0):
        print("MiRo going around object")
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            self.drive(speed_l=0.0, speed_r=0.0)
            rospy.sleep(1)
            self.drive(speed_l=-0.1, speed_r=1)
            now = rospy.get_rostime()
            while (rospy.get_rostime() < (now + rospy.Duration(0.5))):
                self.drive(speed_l=-0.05, speed_r=0.05)
            self.drive(speed_l=0.0, speed_r=0.0)
            rospy.sleep(1)
            now = rospy.get_rostime()
            while (rospy.get_rostime() < (now + rospy.Duration(10))):
                self.drive(speed_l=0.2, speed_r=0.15)
            self.drive(speed_l=0.0, speed_r=0.0)
            rospy.sleep(1)
            now = rospy.get_rostime()
            rospy.sleep(1)
            while (rospy.get_rostime() < (now + rospy.Duration(3))):
                self.drive(speed_l=0.1, speed_r=-0.1)
            self.drive(speed_l=0.0, speed_r=0.0)
            rospy.sleep(3)
        self.velocity.twist.linear.x = 0
        self.velocity.twist.angular.z = 0
        self.pub_cmd_vel.publish(self.velocity)
        
    def face_ball(self, t0):
        print("MiRo finding ball")
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
                self.ball[index] = self.detect_ball(image, index)
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
                self.ball[index] = self.detect_ball(image, index)
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
            print("BALL FOUND")
        self.velocity.twist.linear.x = 0
        self.velocity.twist.angular.z = 0
        self.pub_cmd_vel.publish(self.velocity)
        
    def spot_colour(self, t0):
        print("Spot colour")
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            self.first_colour()
        self.velocity.twist.linear.x = 0
        self.velocity.twist.angular.z = 0
        self.pub_cmd_vel.publish(self.velocity)


    def touchBodyListener(self, data):
        """
        Touch the body cycles actions
        """
        if data.data > 0:
            self.action_to_do += 1
            # If action to do is greater than list of actions then loop round to 0.
            if self.action_to_do >= len(self.actions):
                self.action_to_do = 0


    def lightCallback(self, data):
        """
        Get the frontal illumination
        """
        if data.data:
            self.light_array = data.data


    def loop(self):
        """
        Main loop
        """
        print("Starting the loop")

        while not rospy.core.is_shutdown():
            print("Doing action", self.action_to_do)

            # Run the selected action
            start_time = rospy.Time.now()
            #self.actions[self.action_to_do](start_time)
            #self.spot_colour(start_time)
            print("Action finished")


# This is run when the script is called directly
if __name__ == "__main__":
    main = MiRoClient()  # Instantiate class
    main.loop()  # Run the main control loop
