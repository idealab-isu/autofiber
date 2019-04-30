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

    # We can exclude the rotation of F by multiplying by it's transpose
    # https://en.wikipedia.org/wiki/Finite_strain_theory
    strain = 0.5 * (np.matmul(F.transpose(0, 2, 1), F) - np.identity(F.shape[1]))

    m = np.array([1.0, 1.0, 0.5])[np.newaxis].T
    strain_vector = np.divide(np.array([[strain[:, 0, 0]], [strain[:, 1, 1]], [strain[:, 0, 1]]]).transpose((2, 0, 1)), m).squeeze()

    # http://homepages.engineering.auckland.ac.nz/~pkel015/SolidMechanicsBooks/Part_I/BookSM_Part_I/08_Energy/08_Energy_02_Elastic_Strain_Energy.pdf
    # J == Pa * m^3 -> J/m = Pa * m^2           ex. GPa * mm^3 = 1 J so to get J/mm -> GPa * mm^2
    strain_energy_density = 0.5*np.multiply(np.einsum("ei,ei->e", strain_vector, np.matmul(strain_vector, stiffness_tensor)), areas)

    total_strain_energy = np.sum(strain_energy_density)
    # print(total_strain_energy)
    return total_strain_energy


duvw_duij_t = np.zeros((6, 3, 3))
for j in range(0, 3):
    for i in range(0, 2):
        duvw_duij_t[j*2+i, i, j] = 1


def computeglobalstrain_grad(normalized_2d, fiberpoints, vertexids, stiffness_tensor, oc):
    element_vertices_uv = fiberpoints.reshape(fiberpoints.shape[0]/2, 2)[vertexids]

    centroid_2d = np.sum(normalized_2d, axis=1) / 3
    centroid_uv = np.sum(element_vertices_uv, axis=1) / 3

    rel_uv = np.subtract(element_vertices_uv, centroid_uv[:, np.newaxis])
    rel_2d = np.subtract(normalized_2d, centroid_2d[:, np.newaxis])

    rel_uvw = np.pad(rel_uv, [(0, 0), (0, 0), (0, 1)], "constant", constant_values=1).transpose(0, 2, 1)
    rel_3d = np.pad(rel_2d, [(0, 0), (0, 0), (0, 1)], "constant", constant_values=1).transpose(0, 2, 1)

    areas = 0.5 * np.linalg.det(rel_uvw)

    minor_mat = np.zeros(rel_uvw.shape)
    for i in range(0, 3):
        for j in range(0, 3):
            minor_mat[:, i, j] = np.linalg.det(minor(rel_uvw, i, j))

    adj_mat = np.multiply(minor_mat, build_checkerboard(minor_mat.shape[1], minor_mat.shape[2])).transpose(0, 2, 1)

    dareas_duv = -0.5*np.trace(np.matmul(adj_mat[:, np.newaxis, :, :], duvw_duij_t), axis1=2, axis2=3)

    F = np.matmul(rel_3d, np.linalg.inv(rel_uvw))[:, :2, :2]

    # We can exclude the rotation of F by multiplying by it's transpose
    strain = 0.5 * (np.matmul(F.transpose(0, 2, 1), F) - np.identity(F.shape[1]))

    m = np.array([1.0, 1.0, 0.5])[np.newaxis].T
    strain_vector = np.divide(np.array([[strain[:, 0, 0]], [strain[:, 1, 1]], [strain[:, 0, 1]]]).transpose((2, 0, 1)), m).squeeze()

    dF_duv = np.matmul(rel_3d[:, np.newaxis, :, :], np.matmul(np.matmul(np.linalg.inv(rel_uvw)[:, np.newaxis, :, :], duvw_duij_t), np.linalg.inv(rel_uvw)[:, np.newaxis, :, :]))[:, :, :2, :2]

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

    point_strain_grad[oc][0] = 0.0
    point_strain_grad[oc][1] = 0.0

    return -1*point_strain_grad.flatten()


# https://medium.com/100-days-of-algorithms/day-69-rmsprop-7a88d475003b
def rmsprop_momentum(F, dF, x_0, precision=None, maxsteps=None, lr=None, decay=None, eps=None, mu=None):
    x = x_0.flatten()
    loss = []
    dx_mean_sqr = np.zeros(x.shape, dtype=float)
    momentum = np.zeros(x.shape, dtype=float)

    if F(x) < eps:
        return x.reshape(-1, 2), loss

    for _ in range(maxsteps):
        b0 = F(x)
        dx = dF(x)
        dx_mean_sqr = decay * dx_mean_sqr + (1 - decay) * dx ** 2
        momentum = mu * momentum + lr * dx / (np.sqrt(dx_mean_sqr) + eps)
        x -= momentum

        loss.append(F(x))

        sys.stdout.write("Residual: %s      \r" % abs(F(x) - b0))
        sys.stdout.flush()

        if abs(F(x) - b0) < precision:
            break
        if F(x) < 0:
            raise ValueError("Negative strain energy detected. Bad parameterization?")
    print("")
    return x.reshape(-1, 2), loss
