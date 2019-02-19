import numpy as np
from autofiber.generator import AutoFiber as AF

angle = 30
# test = AF(r'C:\Files\Research\Fiber-Generator\demos\plate.x3d', np.array([1.0, -1.0, 0.0]), np.array([-np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0]), np.array([0, 0, 1]), plotting=True, offset=0.1, fiberint=0.01)

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


test = AF(r'C:\Users\Nate\Documents\UbuntuSharedFiles\fibergeneration\demos\cylinder.x3d', np.array([0.0, 0.0, 0.99]), np.array([np.sin(np.deg2rad(angle)), -1*np.cos(np.deg2rad(angle)), 0]), np.array([0, 0, 1]), plotting=True)

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

# test = AF(r'C:\Users\Nate\Documents\UbuntuSharedFiles\fibergeneration\demos\smallsaddle.x3d', np.array([0.0, 1.0, 0.0]), np.array([-np.cos(np.deg2rad(angle)), 0, -np.sin(np.deg2rad(angle))]), np.array([0, 1, 0]), plotting=True, offset=0.01, fiberint=0.005)

# Anisotropic
# test = AF(r'C:\Files\Research\Fiber-Generator\demos\64smallsaddle.x3d',
#           np.array([1.0, 1.0, 1.0]),
#           np.array([-np.cos(np.deg2rad(angle)), 0, -np.sin(np.deg2rad(angle))]),
#           np.array([0, 1, 0]),
#           E=[1.415e11/1.e9, 8.5e9/1.e9, 8.5e9/1.e9],
#           nu=[0.33, 0.33, 0.33],
#           G=[5.02e9/1.e9, 5.02e9/1.e9, 2.35e9/1.e9],
#           plotting=True,
#           offset=0.01,
#           fiberint=0.005)
