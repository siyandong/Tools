import numpy as np
import cv2
import CoordGenerator
import PnP


if __name__ == '__main__':

	name = 'rio'
	fid = 0
	intrinsics_path = 'data/%s/%s_calib.txt'%(name, name)
	depth_path = 'data/%s/%d_depth.png'%(name, fid)
	pose_path = 'data/%s/%d_pose.txt'%(name, fid)	
	coord_path = 'data/%s/%d_coord.npy'%(name, fid)
	coord_vis_path = 'data/%s/%d_coord_vis.png'%(name, fid)
	pc_path = 'data/%s/%d_pc.txt'%(name, fid)

	intrinsics = np.loadtxt(intrinsics_path)
	depth = cv2.imread(depth_path, -1)
	if len(depth.shape)==3:
		depth = depth[:,:,0]
	depth = depth/1000
	pose = np.loadtxt(pose_path)
	image_width, image_height = depth.shape[1], depth.shape[0]

	# back project to scene coordinates
	cg = CoordGenerator.CoordGenerator(intrinsics, image_width, image_height)
	coord, coord_vis = cg.depth_pose_2coord(depth, pose)
	points_coord = coord.reshape(-1, 3)

	np.save(coord_path, coord)
	cv2.imwrite(coord_vis_path, coord_vis)
	np.savetxt(pc_path, points_coord)

	# solve camera pose
	img_coord = np.load(coord_path)
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
	
	ps = PnP.PnP_Solver(intrinsics)
	Rt = ps.estimate_pose(sample_3d, sample_2d)

	terr, rerr = ps.evaluate_pose_error(Rt, pose)
	print('terr %.2f cm, rerr %.2f d'%(terr*100, rerr))