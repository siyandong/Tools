import numpy as np

class Kabsch:

	# A[i] and B[i] should be correspondences.
	A = [] # list of ndarray
	B = [] # list of ndarray
	dim = 3

	# init.
	def __init__(self, a_vertexes, b_vertexes):
		self.A = a_vertexes
		self.B = b_vertexes
		assert len(self.A) == len(self.B) # the number must be same
		self.dim = len(self.A[0])
		return

	# P[i] = np.dot(R, Q[i]).
	def solve_R(self, P, Q):
		# compute H
		H = np.zeros((self.dim, self.dim))
		for i in range(len(P)):
			q = Q[i].reshape(3, 1)
			pT = P[i].reshape(1, 3)
			H = H + np.dot(q, pT)
		# compute R
		U, Sigma, VT = np.linalg.svd(H)
		if np.linalg.det(U) * np.linalg.det(VT) < 0:
			U[:,2] = -U[:,2]
		R = np.dot(VT.T, U.T)
		return R

	# P[i] = np.dot(R, Q[i]) + t.
	def solve_R_t(self):
		# centroids 
		cen_A = np.zeros((self.dim))
		cen_B = np.zeros((self.dim))
		for i in range(len(self.A)):
			cen_A = cen_A + self.A[i]
			cen_B = cen_B + self.B[i]
		cen_A = cen_A / len(self.A)
		cen_B = cen_B / len(self.B)
		# subtraction
		P = []
		Q = []
		for i in range(len(self.A)):
			P.append(self.A[i] - cen_A)
			Q.append(self.B[i] - cen_B)
		# compute R
		R = self.solve_R(P, Q)
		# compute t
		t = cen_A - np.dot(R, cen_B)
		return R, t

	# P[i] = np.dot(Rt, Q[i]), in homogenous coordinate system.
	def solve_Rt(self):
		R, t = self.solve_R_t()
		Rt = np.zeros((self.dim+1, self.dim+1))
		for i in range(self.dim):
			for j in range(self.dim):
				Rt[i,j] = R[i,j]
			Rt[i, self.dim] = t[i]
		Rt[self.dim, self.dim] = 1.
		#print('R\n', R)
		#print('t\n', t)
		#print('Rt\n', Rt)
		return Rt

'''# test	
A = []
B = []
dim = 3
v_num = 100
angle = np.pi/3
gt_R = np.array([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
gt_t = np.random.rand(dim)
for i in range(v_num):
	A.append(np.random.rand(dim))
for i in range(v_num):
	B.append(np.dot(gt_R, A[i])+gt_t)
kab = Kabsch(A, B)
R, t = kab.solve_R_t()
for i in range(v_num):
	print('A[i]\n', A[i])
	print('np.dot(R, B[i])\n', np.dot(R, B[i])+t)
#'''