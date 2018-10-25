import numpy as np
from autofiber.generator import AutoFiber as AF

angle = 30
# test = AF(r'C:\Files\Research\Fiber-Generator\demos\plate.x3d', np.array([1.0, -1.0, 0.0]), np.array([-np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0]), np.array([0, 0, 1]), plotting=True, offset=0.1, fiberint=0.01)

# test = AF(r'C:\Files\Research\Fiber-Generator\demos\cylinder.x3d', np.array([1.0, -1.0, 0.0]), np.array([0, np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]), np.array([1, 0, 0]), plotting=True)

test = AF(r'C:\Files\Research\Fiber-Generator\demos\CurvedSurface.x3d', np.array([25.0, 0.0, 25.0]), np.array([-np.sin(np.deg2rad(angle)), 0, -np.cos(np.deg2rad(angle))]), np.array([0, -1, 0]), plotting=True, offset=0.1)

# test = AF(r'C:\Files\Research\Fiber-Generator\demos\smallsaddle.x3d', np.array([1.0, 1.0, 1.0]), np.array([-np.cos(np.deg2rad(angle)), 0, -np.sin(np.deg2rad(angle))]), np.array([0, 1, 0]), plotting=True, offset=0.01, fiberint=0.01)
