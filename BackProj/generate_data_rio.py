from PIL import Image
import numpy as np
import cv2
import os
import CoordGenerator
#import PnP # code used to check coord by pnp


dir_input = 'D:/data/RIO_part'
dir_output = 'D:/data/RIO_part/siyan'
depth_width = 224
depth_height = 172
generate_pc = False


def read_pgm(path):	
	return np.array(Image.open(path)).astype(np.uint16)

def raw_data_convert(seq_id, g_trans, f_beg, f_end, f_step=1):
	print('raw_data_convert\n')
	for idx in range(f_beg, f_end, f_step):
		# color
		img = cv2.imread("%s/%s/sequence/frame-%06d.color.jpg"%(dir_input, seq_id, idx), -1)
		img = cv2.resize(img, (depth_width, depth_height), interpolation=cv2.INTER_AREA)  
		cv2.imwrite("%s/%s/%06d_color.png"%(dir_output, seq_id, idx), img)
		# depth
		img = read_pgm("%s/%s/sequence/frame-%06d.depth.pgm"%(dir_input, seq_id, idx))
		cv2.imwrite("%s/%s/%06d_depth.png"%(dir_output, seq_id, idx), img)
		# pose
		pose = np.loadtxt("%s/%s/sequence/frame-%06d.pose.txt"%(dir_input, seq_id, idx))
		pose = g_trans @ pose
		np.savetxt("%s/%s/%06d_pose.txt"%(dir_output, seq_id, idx), pose)
		#print('%03.2f %%'%((idx-f_beg)/(f_end-f_beg)*100))
	print('done.\n')
	return

def generate_coord(image_intrinsics, seq_id, f_beg, f_end, f_step=1):
	print('generate_coord\n')
	for idx in range(f_beg, f_end, f_step):
		# depth
		depth = cv2.imread("%s/%s/%06d_depth.png"%(dir_output, seq_id, idx), -1).astype(np.float32)
		depth /= 1000
		# pose
		pose = np.loadtxt("%s/%s/%06d_pose.txt"%(dir_output, seq_id, idx))
		# generate coord
		image_width, image_height = depth.shape[1], depth.shape[0]
		cg = CoordGenerator.CoordGenerator(image_intrinsics, image_width, image_height)
		coord, coord_vis = cg.depth_pose_2coord(depth, pose)
		np.save("%s/%s/%06d_coord.npy"%(dir_output, seq_id, idx), coord)
		#cv2.imwrite("%s/%s/%06d_coord_vis.png"%(dir_output, seq_id, idx), coord_vis)
		if generate_pc:
			points_coord = coord.reshape(-1, 3)
			np.savetxt("%s/%s/%06d_pc.txt"%(dir_output, seq_id, idx), points_coord)
		#print('%03.2f %%'%((idx-f_beg)/(f_end-f_beg)*100))
	print('done.\n')
	return

''' # code used to check coord by pnp
def solve_pose(image_intrinsics, seq_id, f_beg, f_end, f_step=1):
	for idx in range(f_beg, f_end, f_step):
		# coord and pose_gt
		img_coord = np.load("%s/%s/%06d_coord.npy"%(dir_output, seq_id, idx))
		pose = np.loadtxt("%s/%s/%06d_pose.txt"%(dir_output, seq_id, idx))
		# solve pose
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
		ps = PnP.PnP_Solver(image_intrinsics)
		Rt = ps.estimate_pose(sample_3d, sample_2d)
		terr, rerr = ps.evaluate_pose_error(Rt, pose)
		print('terr %.2f cm, rerr %.2f d\n'%(terr*100, rerr))
	return
#'''

def generate_seq(image_intrinsics, seq_id, g_trans, f_beg, f_end):
	print('seq_id %s\n'%(seq_id))
	raw_data_convert(seq_id, g_trans, f_beg, f_end)
	generate_coord(image_intrinsics, seq_id, f_beg, f_end)
	#solve_pose(image_intrinsics, seq_id, f_beg, f_end) # code used to check coord by pnp
	return 


if __name__ == '__main__':

	''' # office - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	img_intrin = np.array([[204.75, 0, 111.588], [0, 279.5, 85.5793], [0, 0, 1]], dtype=np.float32)
	# seq 1
	seq_id = '09582250-e2c2-2de1-9541-3efcbdc2dca4'
	g_trans = [0.86197835206985474, 0.50653976202011108, 0.020271344110369682, 0, 
	-0.50639784336090088, 0.86221671104431152, -0.011990923434495926, 0, 
	-0.023552171885967255, 0.000070551417593378574, 0.99972271919250488, 0,
	-0.064294837415218353, -0.23404276371002197, -0.044750727713108063, 1]
	f_beg = 0
	f_end = 233
	g_trans = np.array(g_trans).reshape(4, 4).T
	output_path = '%s/%s'%(dir_output, seq_id)
	folder = os.path.exists(output_path)
	if not folder:
		os.makedirs(output_path)
	generate_seq(img_intrin, seq_id, g_trans, f_beg, f_end+1)
	np.savetxt('%s/%s/calib.txt'%(dir_output, seq_id), img_intrin)
	# seq 2
	seq_id = '09582252-e2c2-2de1-96bf-d41dd0362f39'
	g_trans = [0.62916392087936401, 0.77727258205413818, 0.00032233912497758865, 0,
	-0.77719974517822266, 0.62911045551300049, -0.013442425988614559, 0,
	0.01065121591091156, 0.0082069672644138336, 0.99990963935852051, 0,
	-0.2435276061296463, -0.03114430233836174, -0.048019681125879288, 1]
	f_beg = 0
	f_end = 260
	g_trans = np.array(g_trans).reshape(4, 4).T
	output_path = '%s/%s'%(dir_output, seq_id)
	folder = os.path.exists(output_path)
	if not folder:
		os.makedirs(output_path)
	generate_seq(img_intrin, seq_id, g_trans, f_beg, f_end+1)
	np.savetxt('%s/%s/calib.txt'%(dir_output, seq_id), img_intrin)
	# seq 3
	seq_id = '09582254-e2c2-2de1-9434-162187eb819e'
	g_trans = [0.88824546337127686, 0.45928594470024109, -0.0087462365627288818, 0, 
	-0.4593014121055603, 0.88828045129776001, 0.00026336940936744213, 0, 
	0.0078900726512074471, 0.0037832222878932953, 0.99996167421340942, 0,
	0.038428124040365219, -0.18496222794055939, -0.076817028224468231, 1]
	f_beg = 0
	f_end = 316
	g_trans = np.array(g_trans).reshape(4, 4).T
	output_path = '%s/%s'%(dir_output, seq_id)
	folder = os.path.exists(output_path)
	if not folder:
		os.makedirs(output_path)
	generate_seq(img_intrin, seq_id, g_trans, f_beg, f_end+1)
	np.savetxt('%s/%s/calib.txt'%(dir_output, seq_id), img_intrin)
	# seq 4
	seq_id = '09582256-e2c2-2de1-9662-c4bc7ca7c497'
	g_trans = [0.72929763793945312, -0.6839107871055603, -0.019773958250880241, 0,
	0.68396991491317749, 0.72800743579864502, 0.046803798526525497, 0,
	0.017614036798477173, -0.047658689320087433, 0.99870842695236206, 0,
	-1.2009553909301758, -0.14804369211196899, -0.06389271467924118, 1]
	f_beg = 0
	f_end = 274
	g_trans = np.array(g_trans).reshape(4, 4).T
	output_path = '%s/%s'%(dir_output, seq_id)
	folder = os.path.exists(output_path)
	if not folder:
		os.makedirs(output_path)
	generate_seq(img_intrin, seq_id, g_trans, f_beg, f_end+1)
	np.savetxt('%s/%s/calib.txt'%(dir_output, seq_id), img_intrin)
	# office - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # '''


	''' # kitchen - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	img_intrin = np.array([[177.345, 0, 112.94], [0, 241.689, 84.4557], [0, 0, 1]], dtype=np.float32)
	# seq 1
	seq_id = '9766cbe5-6321-2e2f-8040-4e5b7a5d8ba1'
	g_trans = [0.34942460060119629, -0.93696218729019165, -0.0020918117370456457, 0,
	0.93694949150085449, 0.34943076968193054, -0.0048793721944093704, 0,
	0.0053027304820716381, -0.00025494955480098724, 0.99998587369918823, 0,
	0.17242573201656342, -0.28566098213195801, -0.074938319623470306, 1]
	f_beg = 0
	f_end = 847
	g_trans = np.array(g_trans).reshape(4, 4).T
	output_path = '%s/%s'%(dir_output, seq_id)
	folder = os.path.exists(output_path)
	if not folder:
		os.makedirs(output_path)
	generate_seq(img_intrin, seq_id, g_trans, f_beg, f_end+1)
	np.savetxt('%s/%s/calib.txt'%(dir_output, seq_id), img_intrin)
	# seq 2
	seq_id = '9766cbf5-6321-2e2f-8131-78c4e204635d'
	g_trans = [0.19051408767700195, -0.98165875673294067, -0.0071061272174119949, 0, 
	0.98168021440505981, 0.1905294805765152, -0.0015493285609409213, 0, 
	0.0028748386539518833, -0.0066807763651013374, 0.9999735951423645, 0, 
	0.62913650274276733, -0.24034136533737183, 0.044620595872402191, 1]
	f_beg = 0
	f_end = 1261
	g_trans = np.array(g_trans).reshape(4, 4).T
	output_path = '%s/%s'%(dir_output, seq_id)
	folder = os.path.exists(output_path)
	if not folder:
		os.makedirs(output_path)
	generate_seq(img_intrin, seq_id, g_trans, f_beg, f_end+1)
	np.savetxt('%s/%s/calib.txt'%(dir_output, seq_id), img_intrin)
	# kitchen - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # '''


	''' # restroom - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	img_intrin = np.array([[176.594, 0, 114.613], [0, 240.808, 85.7915], [0, 0, 1]], dtype=np.float32)
	# seq 1
	seq_id = '0ad2d386-79e2-2212-9b40-43d081db442a'
	g_trans = [1., 0., 0., 0.,
	0., 1., 0., 0.,
	0., 0., 1., 0.,
	0., 0., 0., 1.]
	f_beg = 0
	f_end = 106
	g_trans = np.array(g_trans).reshape(4, 4).T
	output_path = '%s/%s'%(dir_output, seq_id)
	folder = os.path.exists(output_path)
	if not folder:
		os.makedirs(output_path)
	generate_seq(img_intrin, seq_id, g_trans, f_beg, f_end+1)
	np.savetxt('%s/%s/calib.txt'%(dir_output, seq_id), img_intrin)
	# seq 2
	seq_id = '0ad2d389-79e2-2212-9b0a-5f0e6f4982b5'
	g_trans = [0.46398162841796875, -0.88564765453338623, -0.018682058900594711, 0,
	0.88583731651306152, 0.46378573775291443, 0.013984784483909607, 0,
	-0.0037211207672953606, -0.023037942126393318, 0.99972784519195557, 0,
	0.55887115001678467, 0.39800447225570679, 0.10449286550283432, 1]
	f_beg = 0
	f_end = 200
	g_trans = np.array(g_trans).reshape(4, 4).T
	output_path = '%s/%s'%(dir_output, seq_id)
	folder = os.path.exists(output_path)
	if not folder:
		os.makedirs(output_path)
	generate_seq(img_intrin, seq_id, g_trans, f_beg, f_end+1)
	np.savetxt('%s/%s/calib.txt'%(dir_output, seq_id), img_intrin)
	# restroom - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # '''
	

	# home - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	img_intrin = np.array([[176.594, 0, 114.613], [0, 240.808, 85.7915], [0, 0, 1]], dtype=np.float32)
	# seq 1
	seq_id = 'dcb6a32b-5526-23f1-9cbc-3d257a8096fa'
	g_trans = [-0.98312866687774658, 0.18266454339027405, 0.0095770386978983879, 0,
	-0.1827181875705719, -0.98315232992172241, -0.005055790301412344, 0, 
	0.0084921745583415031, -0.0067203915677964687, 0.99994140863418579, 0,
	2.86199951171875, -1.4074712991714478, -0.12345875799655914, 1]
	f_beg = 0
	f_end = 106
	g_trans = np.array(g_trans).reshape(4, 4).T
	output_path = '%s/%s'%(dir_output, seq_id)
	folder = os.path.exists(output_path)
	if not folder:
		os.makedirs(output_path)
	generate_seq(img_intrin, seq_id, g_trans, f_beg, f_end+1)
	np.savetxt('%s/%s/calib.txt'%(dir_output, seq_id), img_intrin)
	# seq 2
	seq_id = 'dcb6a329-5526-23f1-9d81-7718f682269c'
	g_trans = [-0.33231633901596069, -0.94307488203048706, 0.013266079127788544, 0, 
	0.94316446781158447, -0.33224475383758545, 0.0073329056613147259, 0, 
	-0.0025078938342630863, 0.014948934316635132, 0.99988508224487305, 0, 
	-0.45759952068328857, -0.5762677788734436, -0.080425910651683807, 1]
	f_beg = 0
	f_end = 251
	g_trans = np.array(g_trans).reshape(4, 4).T
	output_path = '%s/%s'%(dir_output, seq_id)
	folder = os.path.exists(output_path)
	if not folder:
		os.makedirs(output_path)
	generate_seq(img_intrin, seq_id, g_trans, f_beg, f_end+1)
	np.savetxt('%s/%s/calib.txt'%(dir_output, seq_id), img_intrin)
	# home - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # '''


	print('done.')
