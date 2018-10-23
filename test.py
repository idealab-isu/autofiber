import sys, pdb
from scipy import optimize
import numpy as np
from autofiber.generator import AutoFiber as AF

angle = 30
# test = AF(r'C:\Files\Research\Fiber-Generator\demos\plate.x3d', np.array([1.0, -1.0, 0.0]), np.array([-np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0]), np.array([0, 0, 1]), plotting=True, offset=0.1, fiberint=0.01)

# test = AF(r'C:\Files\Research\Fiber-Generator\demos\cylinder.x3d', np.array([1.0, -1.0, 0.0]), np.array([0, np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]), np.array([1, 0, 0]), plotting=True)

# test = AF('C:\\Users\\Nate\\Documents\\UbuntuSharedFiles\\fibergeneration\\demos\\CurvedSurface.x3d', np.array([25.0, 0.0, 25.0]), np.array([-np.sin(np.deg2rad(angle)), 0, -np.cos(np.deg2rad(angle))]), np.array([0, -1, 0]), plotting=True, offset=0.1)

# test = AF('C:\\Users\\Nate\\Documents\\UbuntuSharedFiles\\fibergeneration\\demos\\smallsaddle.x3d', np.array([1.0, 1.0, 1.0]), np.array([-np.cos(np.deg2rad(angle)), 0, -np.sin(np.deg2rad(angle))]), np.array([0, 1, 0]), plotting=True, offset=0.01, fiberint=0.01)

E = 200.0
nu = 0.3
G = E / (2 * (1 + nu))
compliance_tensor = np.array([[1 / E, -nu / E, 0],
                              [-nu / E, 1 / E, 0],
                              [0, 0, 1 / G]])

stiffness_tensor2d = np.linalg.inv(compliance_tensor)

vertexids2d = np.array([[0, 1, 2]])

vertices2d = np.array([[[0, 0],
                        [1, 1],
                        [2, 0]]], dtype=np.float)

fiberpoints2d = np.array([[0, 0],
                        [1, 2],
                        [2, 0]], dtype=np.float).flatten()


def strainenergy(normalized_2d, fiberpoints, vertexids, stiffness_tensor):
    element_vertices_uv = fiberpoints.reshape(fiberpoints.shape[0]/2, 2)[vertexids]

    centroid_2d = np.sum(normalized_2d, axis=1) / 3
    centroid_uv = np.sum(element_vertices_uv, axis=1) / 3

    rel_uv = np.subtract(element_vertices_uv, centroid_uv[:, np.newaxis])
    rel_uv = np.pad(rel_uv, [(0, 0), (0, 0), (0, 1)], "constant", constant_values=1)
    rel_2d = np.subtract(normalized_2d, centroid_2d[:, np.newaxis])

    invrel_uv = np.linalg.inv(rel_uv)
    deform_mat = np.matmul(invrel_uv, rel_2d)[:2, :2]

    # https://en.wikipedia.org/wiki/Infinitesimal_strain_theory
    # deform_mat = deform_mat + np.identity(deform_mat.shape[1])
    # strain = 0.5 * (np.transpose(deform_mat, (0, 2, 1)) + deform_mat) - np.identity(deform_mat.shape[1])
    # Finite strain theory
    # https://www.klancek.si/sites/default/files/datoteke/files/derivativeofprincipalstretches.pdf
    strain = 0.5 * (np.matmul(np.transpose(deform_mat, (0, 2, 1)), deform_mat) - np.identity(deform_mat.shape[1]))

    m = np.array([1, 1, 2])[np.newaxis].T
    strain_vector = np.divide(np.array([[strain[:, 0, 0]], [strain[:, 1, 1]], [strain[:, 0, 1]]]).transpose((2, 0, 1)), m).squeeze()

    # http://homepages.engineering.auckland.ac.nz/~pkel015/SolidMechanicsBooks/Part_I/BookSM_Part_I/08_Energy/08_Energy_02_Elastic_Strain_Energy.pdf
    stress = np.einsum('ij,ej->ei', stiffness_tensor, strain_vector[np.newaxis])
    strain_energy_density = 0.5 * (np.einsum('ei,ei->e', stress, strain_vector[np.newaxis]))

    # sys.stdout.write('Strain Energy Density: %f       \r' % (np.sum(strain_energy_density),))
    # sys.stdout.flush()
    # pdb.set_trace()
    return np.sum(strain_energy_density)


def f(x, *args):
    return strainenergy(vertices2d, x, vertexids2d, stiffness_tensor2d)


res = optimize.minimize(f, fiberpoints2d, method="CG", options={'gtol': 1})

sys.modules["__main__"].__dict__.update(globals())
sys.modules["__main__"].__dict__.update(locals())
raise ValueError()
