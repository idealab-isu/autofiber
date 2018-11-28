import sys
from scipy import optimize
import numpy as np
from autofiber import optimization as OP
from autofiber.generator import AutoFiber as AF

angle = 30
# test = AF(r'C:\Users\Nate\Documents\UbuntuSharedFiles\fibergeneration\demos\plate.x3d', np.array([1.0, -1.0, 0.0]), np.array([-np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0]), np.array([0, 0, 1]), plotting=True, offset=0.1, fiberint=0.01)

# Anisotropic
# test = AF(r'C:\Users\Nate\Documents\UbuntuSharedFiles\fibergeneration\demos\plate.x3d',
#           np.array([1.0, -1.0, 0.0]),
#           np.array([-np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0]),
#           np.array([0, 0, 1]),
#           E=[1.415e11/1.e9, 8.5e9/1.e9, 8.5e9/1.e9],
#           nu=[0.33, 0.33, 0.33],
#           G=[5.02e9/1.e9, 5.02e9/1.e9, 2.35e9/1.e9],
#           plotting=True,
#           offset=0.1,
#           fiberint=0.01)


# test = AF(r'C:\Users\Nate\Documents\UbuntuSharedFiles\fibergeneration\demos\cylinder.x3d', np.array([1.0, -1.0, 0.0]), np.array([0, np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]), np.array([1, 0, 0]), plotting=True)

# Anisotropic
# test = AF(r'C:\Users\Nate\Documents\UbuntuSharedFiles\fibergeneration\demos\cylinder.x3d',
#           np.array([1.0, -1.0, 0.0]),
#           np.array([0, np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]),
#           np.array([1, 0, 0]),
#           E=[1.415e11/1.e9, 8.5e9/1.e9, 8.5e9/1.e9],
#           nu=[0.33, 0.33, 0.33],
#           G=[5.02e9/1.e9, 5.02e9/1.e9, 2.35e9/1.e9],
#           plotting=True)

# test = AF(r'C:\Users\Nate\Documents\UbuntuSharedFiles\fibergeneration\demos\CurvedSurface.x3d', np.array([25.0, 0.0, 25.0]), np.array([-np.sin(np.deg2rad(angle)), 0, -np.cos(np.deg2rad(angle))]), np.array([0, -1, 0]), plotting=True, offset=0.1)

# test = AF(r'C:\Files\Research\Fiber-Generator\demos\smallsaddle.x3d', np.array([1.0, 1.0, 1.0]), np.array([-np.cos(np.deg2rad(angle)), 0, -np.sin(np.deg2rad(angle))]), np.array([0, 1, 0]), plotting=True, offset=0.01, fiberint=0.01)

# Anisotropic
# test = AF(r'C:\Users\Nate\Documents\UbuntuSharedFiles\fibergeneration\demos\smallsaddle.x3d',
#           np.array([1.0, 1.0, 1.0]),
#           np.array([-np.cos(np.deg2rad(angle)), 0, -np.sin(np.deg2rad(angle))]),
#           np.array([0, 1, 0]),
#           E=[1.415e11/1.e9, 8.5e9/1.e9, 8.5e9/1.e9],
#           nu=[0.33, 0.33, 0.33],
#           G=[5.02e9/1.e9, 5.02e9/1.e9, 2.35e9/1.e9],
#           plotting=True,
#           offset=0.01,
#           fiberint=0.01)

E = 200.0
nu = 0.3
G = E / (2 * (1 + nu))
compliance_tensor = np.array([[1 / E, -nu / E, 0],
                              [-nu / E, 1 / E, 0],
                              [0, 0, 1 / G]])

stiffness_tensor2d = np.linalg.inv(compliance_tensor)

compliance_tensor = np.array([[1 / E, -nu / E, -nu / E, 0, 0, 0],
                              [-nu / E, 1 / E, -nu / E, 0, 0, 0],
                              [-nu / E, -nu / E, 1 / E, 0, 0, 0],
                              [0, 0, 0, 1 / G, 0, 0],
                              [0, 0, 0, 0, 1 / G, 0],
                              [0, 0, 0, 0, 0, 1 / G]])

stiffness_tensor3d = np.linalg.inv(compliance_tensor)

vertexids2d = np.array([[0, 1, 2]])

vertices2d_un = np.array([[0, 0],
                          [1, 1],
                          [2, 0]], dtype=np.float)

vertices2d = vertices2d_un.flatten()

fiberpoints2d_un = np.array([[0, 0],
                             [1.5, 2],
                             [2.5, 0]], dtype=np.float)

fiberpoints2d = fiberpoints2d_un.flatten()


def f(x, *args):
    return OP.computeglobalstrain(vertices2d_un[np.newaxis], x, vertexids2d, stiffness_tensor2d)


def gradf(x, *args):
    return OP.computeglobalstrain_grad(vertices2d_un[np.newaxis], x, vertexids2d, stiffness_tensor2d)

# [-0.016,  0.052,  0.04 , -0.13 , -0.024,  0.078]
# [ 2.55999993e-01, -3.20000026e-02,  0.00000000e+00, -1.11022302e-08, -2.56000005e-01,  3.20000026e-02]
# [-0.016,  0.052,  0.04 , -0.13 , -0.024,  0.078]
# [ 0.096,  0.088, -0.32 ,  0.04 ,  0.224, -0.128]
# [-257.14285101, -528.57142094,  -30.76922894]
# [-109.18900841,  -46.39670834,  -11.29670579,  128.56318534, 120.48571705,  -82.16648979]
# testgrad = optimize.approx_fprime(fiberpoints2d, f, 1e-8)

grad_check = optimize.check_grad(f, gradf, fiberpoints2d)

print(grad_check)

# res = optimize.minimize(f, fiberpoints2d, method="CG", options={'gtol': 1e-8})
#
# print("Total strain energy: %s" % res.fun)
#
# fres = res.x.reshape(fiberpoints2d_un.shape)
#
# import matplotlib.pyplot as plt
#
# fig = plt.figure()
#
# plt.scatter(vertices2d_un[:, 0], vertices2d_un[:, 1], color='b')
# t1 = plt.Polygon(vertices2d_un, color='b', fill=False)
# plt.gca().add_patch(t1)
#
# plt.scatter(fiberpoints2d_un[:, 0], fiberpoints2d_un[:, 1], color='r')
# t2 = plt.Polygon(fiberpoints2d_un, color='r', fill=False)
# plt.gca().add_patch(t2)
#
# plt.scatter(fres[:, 0], fres[:, 1], color='g')
# t3 = plt.Polygon(fres, color='g', fill=False)
# plt.gca().add_patch(t3)

sys.modules["__main__"].__dict__.update(globals())
sys.modules["__main__"].__dict__.update(locals())
raise ValueError()
