import rospy


class FaceVerificationNode:

    def __init__(self):

        self.serialNode()

    def serialNode(self):
        rospy.init_node('face_verification_node')

        rate = rospy.Rate(100)  # 100hz
        while not rospy.is_shutdown():
            rate.sleep()


FaceVerificationNode()
