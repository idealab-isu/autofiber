import sys
import numpy as np


def calcunitvector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def calc2d(obj, points):
    coord_sys = obj.implpart.surfaces[0].inplanemats
    coord_sys = np.transpose(coord_sys, axes=(0, 2, 1))
    points_2d = np.matmul(points, coord_sys)
    return points_2d


def minor(arr, i, j):
    # https://stackoverflow.com/questions/3858213/numpy-routine-for-computing-matrix-minors
    # ith row, jth column removed
    return ((-1) ** (i + j)) * arr[:, np.array(range(i)+range(i+1, arr.shape[1]))[:, np.newaxis],
                               np.array(range(j)+range(j+1, arr.shape[2]))]


def build_checkerboard(w, h):
    # https://stackoverflow.com/questions/2169478/how-to-make-a-checkerboard-in-numpy
    re = np.r_[w * [1, -1]]  # even-numbered rows
    ro = np.r_[w * [-1, 1]]  # odd-numbered rows
    return np.row_stack(h * (re, ro))[:w, :h]


def computeglobalstrain(normalized_2d, fiberpoints, vertexids, stiffness_tensor):
    element_vertices_uv = fiberpoints.reshape(fiberpoints.shape[0]/2, 2)[vertexids]

    centroid_2d = np.sum(normalized_2d, axis=1) / 3
    centroid_uv = np.sum(element_vertices_uv, axis=1) / 3

    rel_uv = np.subtract(element_vertices_uv, centroid_uv[:, np.newaxis])
    rel_2d = np.subtract(normalized_2d, centroid_2d[:, np.newaxis])

    rel_uvw = np.pad(rel_uv, [(0, 0), (0, 0), (0, 1)], "constant", constant_values=1).transpose(0, 2, 1)
    rel_3d = np.pad(rel_2d, [(0, 0), (0, 0), (0, 1)], "constant", constant_values=1).transpose(0, 2, 1)

    areas = 0.5 * np.linalg.det(rel_uvw)

    F = np.matmul(rel_3d, np.linalg.inv(rel_uvw))[:, :2, :2]

    strain = 0.5 * (np.matmul(F.transpose(0, 2, 1), F) - np.identity(F.shape[1]))

    m = np.array([1.0, 1.0, 0.5])[np.newaxis].T
    strain_vector = np.divide(np.array([[strain[:, 0, 0]], [strain[:, 1, 1]], [strain[:, 0, 1]]]).transpose((2, 0, 1)), m).squeeze()

    # http://homepages.engineering.auckland.ac.nz/~pkel015/SolidMechanicsBooks/Part_I/BookSM_Part_I/08_Energy/08_Energy_02_Elastic_Strain_Energy.pdf
    strain_energy_density = 0.5*np.multiply(np.einsum("ei,ei->e", strain_vector, np.matmul(strain_vector, stiffness_tensor)), areas)

    total_strain_energy = np.sum(strain_energy_density)
    print(total_strain_energy)
    import pdb
    pdb.set_trace()
    return total_strain_energy


def computeglobalstrain_grad(normalized_2d, fiberpoints, vertexids, stiffness_tensor):
    element_vertices_uv = fiberpoints.reshape(fiberpoints.shape[0]/2, 2)[vertexids]

    centroid_2d = np.sum(normalized_2d, axis=1) / 3
    centroid_uv = np.sum(element_vertices_uv, axis=1) / 3

    rel_uv = np.subtract(element_vertices_uv, centroid_uv[:, np.newaxis])
    rel_2d = np.subtract(normalized_2d, centroid_2d[:, np.newaxis])

    rel_uvw = np.pad(rel_uv, [(0, 0), (0, 0), (0, 1)], "constant", constant_values=1).transpose(0, 2, 1)
    rel_3d = np.pad(rel_2d, [(0, 0), (0, 0), (0, 1)], "constant", constant_values=1).transpose(0, 2, 1)

    areas = np.abs(0.5 * np.linalg.det(rel_uvw))

    minor_mat = np.zeros(rel_uvw.shape)
    for i in range(0, 3):
        for j in range(0, 3):
            minor_mat[:, i, j] = np.linalg.det(minor(rel_uvw, i, j))

    adj_mat = np.multiply(minor_mat, build_checkerboard(minor_mat.shape[1], minor_mat.shape[2])).transpose(0, 2, 1)

    dareas_duv = np.zeros((rel_uvw.shape[0], 6))
    for j in range(0, 3):
        for i in range(0, 2):
            duvw_duij = np.zeros((rel_uvw.shape[1], rel_uvw.shape[2]))
            duvw_duij[i, j] = 1
            dareas_duv[:, j*2+i] = -0.5 * np.trace(np.matmul(adj_mat, duvw_duij), axis1=1, axis2=2)

    F = np.matmul(rel_3d, np.linalg.inv(rel_uvw))[:, :2, :2]

    strain = 0.5 * (np.matmul(F.transpose(0, 2, 1), F) - np.identity(F.shape[1]))

    m = np.array([1.0, 1.0, 0.5])[np.newaxis].T
    strain_vector = np.divide(np.array([[strain[:, 0, 0]], [strain[:, 1, 1]], [strain[:, 0, 1]]]).transpose((2, 0, 1)), m).squeeze()

    dF_duv = np.zeros((F.shape[0], 6, F.shape[1], F.shape[2]))
    for j in range(0, 3):
        for i in range(0, 2):
            duvw_duij = np.zeros((rel_uvw.shape[1], rel_uvw.shape[2]))
            duvw_duij[i, j] = 1.0
            dF_duv[:, j*2+i, :, :] = np.matmul(rel_3d, np.matmul(np.matmul(np.linalg.inv(rel_uvw), duvw_duij), np.linalg.inv(rel_uvw)))[:, :2, :2]

    dstrainvector_duv = np.zeros((strain_vector.shape[0], strain_vector.shape[1], 6))
    for i in range(0, 6):
        dstrain_du = 0.5 * (np.matmul(dF_duv[:, i, :, :].transpose(0, 2, 1), F) + np.matmul(F.transpose(0, 2, 1), dF_duv[:, i, :, :]))
        dstrainvector_duv[:, :, i] = np.divide(np.array([[dstrain_du[:, 0, 0]], [dstrain_du[:, 1, 1]], [dstrain_du[:, 0, 1]]]).transpose((2, 0, 1)), m).squeeze()

    dE_du = (np.einsum("ei,e->ei", np.einsum("ei,eij->ej", np.matmul(strain_vector, stiffness_tensor), dstrainvector_duv), areas) + 0.5*np.einsum("e,ej->ej", np.einsum("ei,ei->e", np.matmul(strain_vector, stiffness_tensor), strain_vector), dareas_duv)).reshape(dstrainvector_duv.shape[0], 3, 2)

    point_strain_grad = np.zeros((fiberpoints.shape[0]/2, 2))
    for i in range(0, vertexids.shape[0]):
        ele_vertices = vertexids[i]
        ele_strain_grad = dE_du[i]

        point_strain_grad[ele_vertices] = point_strain_grad[ele_vertices] + ele_strain_grad

    import pdb
    # print(point_strain_grad)
    pdb.set_trace()
    return -1*point_strain_grad.flatten()


def optimize(f, grad, x_0, eps=1e-7):
    import pdb
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d

    x = x_0.flatten()
    b = f(x)
    print("Starting Energy: %s" % b)

    its = 10
    strains = np.zeros(its)
    for i in range(0, its):
        cur_grad = grad(x)
        x = x - eps * cur_grad
        b = f(x)

        strains[i] = b
        print("Current Strain Energy: %s" % b)

    fig = plt.figure()
    plt.scatter(x_0[:, 0], x_0[:, 1])
    plt.scatter(x.reshape(x_0.shape)[:, 0], x.reshape(x_0.shape)[:, 1])

    fig = plt.figure()
    plt.plot(range(0, its), strains)

    return x.reshape(x_0.shape)
