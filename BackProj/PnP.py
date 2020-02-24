import numpy as np
import cv2


class PnP_Solver(object):

	def __init__(self, intrin):
		super(PnP_Solver, self).__init__()
		self.intrinsics = intrin

	def estimate_pose(self, p3ds, p2ds, flag=cv2.SOLVEPNP_EPNP):
		found, rvec, tvec = cv2.solvePnP(p3ds, p2ds, self.intrinsics, distCoeffs=None, flags=flag)
		rotM = cv2.Rodrigues(rvec)[0]
		Rt = np.mat(np.vstack((np.hstack((rotM, tvec)), np.array([0, 0, 0, 1])))).I.A
		return Rt
	
	def evaluate_pose_error(self, pa, pb):
		# trans
		o = np.array([0., 0., 0., 1.])
		o_pa = np.dot(pa, o)
		o_pb = np.dot(pb, o)
		terr = np.linalg.norm(o_pa[0:3]-o_pb[0:3])
		# rot
		pa33_inv = np.matrix(pa[0:3,0:3]).I
		pb33 = np.matrix(pb[0:3,0:3])
		err33 = pa33_inv*pb33
		errv = cv2.Rodrigues(err33)[0]
		rad = np.linalg.norm(errv)
		ang = rad/3.14*180
		return terr, ang	


if __name__ == '__main__':

	intrinsics_path = 'data/syn_calib.txt'
	coord_path = 'data/syn_coord.npy'
	pose_path = 'data/syn_pose.txt'

	intrinsics = np.loadtxt(intrinsics_path)
	img_coord = np.load(coord_path)
	pose = np.loadtxt(pose_path)

	img_w, img_h = img_coord.shape[1], img_coord.shape[0]

	sample_3d = []
	sample_2d = []
	while len(sample_3d) < 6:
		r = np.random.randint(0, img_h)
		c = np.random.randint(0, img_w)
		p3d = img_coord[r,c,:]
		p2d = np.array([c, r])
		if (p3d == np.array([0., 0., 0.])).all():
			continue
		sample_3d.append(p3d)
		sample_2d.append(p2d)
	sample_3d = np.array(sample_3d, dtype=np.float32)
	sample_2d = np.array(sample_2d, dtype=np.float32)
	
	ps = PnP_Solver(intrinsics)
	Rt = ps.estimate_pose(sample_3d, sample_2d)

	terr, rerr = ps.evaluate_pose_error(Rt, pose)
	print('terr %.2f cm, rerr %.2f d'%(terr*100, rerr))
