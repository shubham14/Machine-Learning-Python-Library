'''Principal Component Analysis in python'''

import numpy as np

#for reproducibility
np.random.seed(234151353354)

mu_vec1=np.array([0,0,0])
cov_mat1=np.array([[1,0,0],[0,1,0],[0,0,1]])

class1_sample=np.random.multivariate_normal(mu_vec1,cov_mat1,20).T

assert class1_sample.shape==(3,20)

mu_vec2 = np.array([1,1,1])
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
assert class2_sample.shape == (3,20), "The matrix has not the dimensions 3x20"

all_samples=np.concatenate((class1_sample,class2_sample),axis=1)
assert all_samples.shape==(3,40)

mean_x=np.mean(all_samples[0,:])
mean_y=np.mean(all_samples[1,:])
mean_z=np.mean(all_samples[2,:])

mean_vector=np.array([[mean_x],[mean_y],[mean_z]])

scatter_matrix=np.zeros((3,3))
for i in range(all_samples.shape[1]):
	scatter_matrix+=(all_samples[:,i].reshape(3,1)-mean_vector).dot((all_samples[:,i].reshape(3,1)-mean_vector).T)

#Covariance Matrix
cov_mat = np.cov([all_samples[0,:],all_samples[1,:],all_samples[2,:]])

# eigenvectors and eigenvalues for the from the scatter matrix
eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

# eigenvectors and eigenvalues for the from the covariance matrix
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

for i in range(len(eig_val_sc)):
    eigvec_sc = eig_vec_sc[:,i].reshape(1,3).T
    eigvec_cov = eig_vec_cov[:,i].reshape(1,3).T
    assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvectors are not identical'

for ev in eig_vec_sc:
    numpy.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

#sorting the eigenvalues,eigenvectors pairs
eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)

#Choosing the k highest eigenvectors 
matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))
