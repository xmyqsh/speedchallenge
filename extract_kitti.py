import numpy as np
import math

def mat2euler(M, cy_thresh=None, seq='zyx'):
	''' 
	Taken Forom: http://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/eulerangles.py
	Discover Euler angle vector from 3x3 matrix
	Uses the conventions above.
	Parameters
	----------
	M : array-like, shape (3,3)
	cy_thresh : None or scalar, optional
		 threshold below which to give up on straightforward arctan for
		 estimating x rotation.  If None (default), estimate from
		 precision of input.
	Returns
	-------
	z : scalar
	y : scalar
	x : scalar
		 Rotations in radians around z, y, x axes, respectively
	Notes
	-----
	If there was no numerical error, the routine could be derived using
	Sympy expression for z then y then x rotation matrix, which is::
		[                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
		[cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
		[sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
	with the obvious derivations for z, y, and x
		 z = atan2(-r12, r11)
		 y = asin(r13)
		 x = atan2(-r23, r33)
	for x,y,z order
		y = asin(-r31)
		x = atan2(r32, r33)
    z = atan2(r21, r11)
	Problems arise when cos(y) is close to zero, because both of::
		 z = atan2(cos(y)*sin(z), cos(y)*cos(z))
		 x = atan2(cos(y)*sin(x), cos(x)*cos(y))
	will be close to atan2(0, 0), and highly unstable.
	The ``cy`` fix for numerical instability below is from: *Graphics
	Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
	0123361559.  Specifically it comes from EulerAngles.c by Ken
	Shoemake, and deals with the case where cos(y) is close to zero:
	See: http://www.graphicsgems.org/
	The code appears to be licensed (from the website) as "can be used
	without restrictions".
	'''
	M = np.asarray(M)
	if cy_thresh is None:
			try:
					cy_thresh = np.finfo(M.dtype).eps * 4
			except ValueError:
					cy_thresh = _FLOAT_EPS_4
	r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
	# cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
	cy = math.sqrt(r33*r33 + r23*r23)
	if seq=='zyx':
		if cy > cy_thresh: # cos(y) not close to zero, standard form
				z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
				y = math.atan2(r13,  cy) # atan2(sin(y), cy)
				x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
		else: # cos(y) (close to) zero, so x -> 0.0 (see above)
				# so r21 -> sin(z), r22 -> cos(z) and
				z = math.atan2(r21,  r22)
				y = math.atan2(r13,  cy) # atan2(sin(y), cy)
				x = 0.0
	elif seq=='xyz':
		if cy > cy_thresh:
			y = math.atan2(-r31, cy)
			x = math.atan2(r32, r33)
			z = math.atan2(r21, r11)
		else:
			z = 0.0
			if r31 < 0:
				y = np.pi/2
				x = atan2(r12, r13)	
			else:
				y = -np.pi/2
				#x = 
	else:
		raise Exception('Sequence not recognized')
	return z, y, x

def pose2mat(pose):
    mat = np.reshape(pose, [3, 4])
    mat = np.vstack([mat, np.array([0, 0, 0, 1])])

    return mat

poses = np.loadtxt("/mnt/Bulk/kitti/odometry/data_odometry_poses/dataset/poses/00.txt")

pose_deltas = []
prev_mat = pose2mat(poses[0])
for pose in poses[1:]:
    curr_mat = pose2mat(pose)
    delta_inv = np.linalg.inv(prev_mat) @ curr_mat
    delta = np.linalg.inv(delta_inv)
    delta = delta[0:3,:]

    pose_deltas.append(delta)
    prev_mat = curr_mat


for pose in pose_deltas:
    translation = pose[:,-1]
    rotmat = pose[:,:3]
    euler = mat2euler(rotmat, seq='xyz')
    print(euler, translation)