import numpy as np

"""Specify the rotational and translational kinematics of the vehicle"""



# The direction-cosne matrix / attitude matrix is a coordinate transform that maps
# vectors fro the reference frame ot the body frame.
def getAttitudeMatrix(q):
	
	A11 = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
	A12 = 2 * (q[0] * q[1] + q[2] * q[3])
	A13 = 2 * (q[0] * q[2] - q[1] * q[3])
	A21 = 2 * (q[0] * q[1] - q[2] * q[3])
	A22 = -q[0]**2 + q[1]**2 - q[2]**2 + q[3]**2
	A23 = 2 * (q[1] * q[2] + q[0] * q[3])
	A31 = 2 * (q[0] * q[2] + q[1] * q[3])
	A32 = 2 * (q[1] * q[2] - q[0] * q[3])
	A33 = -q[0]**2 - q[1]**2 + q[2]**2 + q[3]**2

	return np.array([A11, A12, A13], [A21, A22, A23], [A31, A32, A33])


# Calculate the	Matrix Function of angular velocity, W(omega):
def getW(angVel):
	W = np.array(\
		[0, angVel[2], -angVel[1], angVel(0)], \
		[-angVel[2], 0, angVel[0], angVel[1]], \
		[angVel[1], -angVel[0], 0 , angVel[2]] \
		[-angVel[0], -angVel[1], -angVel[2], 0])

	return W


# Calculate the change in q:
def deltaQ(W, angVel, qInit, dt):

	dQ = 0 # Figure out np.mat or whatever