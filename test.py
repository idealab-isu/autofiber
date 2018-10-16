import numpy as np
from autofiber.generator import AutoFiber as AF

angle = 30
test = AF('C:\Files\Research\Fiber-Generator\demos\cylinder.x3d', np.array([1.0, -1.0, 0.0]), np.array([0, np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]), np.array([1, 0, 0]), plotting=True)
