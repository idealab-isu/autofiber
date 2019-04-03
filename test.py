import numpy as np
from autofiber.generator import AutoFiber as AF

angles = [0.0]

for angle in angles:
    print("Angle: %s" % angle)
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

    # test = AF(r'C:\Files\Research\Fiber-Generator\demos\cylinder.x3d', np.array([1.0, -1.0, 0.0]), np.array([0, np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]), np.array([1, 0, 0]), fiberint=0.5, plotting=True)
    # test = AF(r'C:\Files\Research\Fiber-Generator\demos\cylinder.x3d', np.array([0.0, 0.0, 1.0]), np.array([np.sin(-1*np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0.0]), np.array([0, 0, 1.0]), fiberint=0.1, plotting=True)

    # Anisotropic
    test = AF(r'C:\Files\Research\Fiber-Generator\demos\cylinder.x3d',
              np.array([0.0, 0.0, 1.0]),
              np.array([np.sin(-1 * np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0.0]),
              np.array([0, 0, 1.0]),
              E=[1.415e11/1.e9, 8.5e9/1.e9, 8.5e9/1.e9],
              nu=[0.33, 0.33, 0.33],
              G=[5.02e9/1.e9, 5.02e9/1.e9, 2.35e9/1.e9],
              fiberint=0.05,
              plotting=True)

    # test = AF(r'C:\Users\Nate\Documents\UbuntuSharedFiles\fibergeneration\demos\CurvedSurface.x3d', np.array([25.0, 0.0, 25.0]), np.array([-np.sin(np.deg2rad(angle)), 0, -np.cos(np.deg2rad(angle))]), np.array([0, -1, 0]), plotting=True, offset=0.1)

    # test = AF(r'C:\Files\Research\Fiber-Generator\demos\smallsaddle.x3d', np.array([0.0, 1.0, 0.0]), np.array([-np.cos(np.deg2rad(angle)), 0, -np.sin(np.deg2rad(angle))]), np.array([0, 1, 0]), plotting=True, offset=0.01, fiberint=0.005)

    # Anisotropic
    # test = AF(r'C:\Files\Research\Fiber-Generator\demos\smallsaddle.x3d',
    #           np.array([0.0, 1.0, 0.0]),
    #           np.array([-np.cos(np.deg2rad(angle)), 0, -np.sin(np.deg2rad(angle))]),
    #           np.array([0, 1, 0]),
    #           E=[1.415e11/1.e9, 8.5e9/1.e9, 8.5e9/1.e9],
    #           nu=[0.33, 0.33, 0.33],
    #           G=[5.02e9/1.e9, 5.02e9/1.e9, 2.35e9/1.e9],
    #           plotting=True,
    #           offset=0.01,
    #           fiberint=0.005)

    # test = AF(r'C:\Files\Research\de-la-mo-v2\data\01_Planar_Single_Delam_ORIGINAL - Open CASCADE STEP translator 6.9 5-1.STL',
    #           np.array([55.0, 55.0, 4.5]),
    #           np.array([-np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0.0]),
    #           np.array([0, 0, 1]),
    #           E=[1.415e11/1.e9, 8.5e9/1.e9, 8.5e9/1.e9],
    #           nu=[0.33, 0.33, 0.33],
    #           G=[5.02e9/1.e9, 5.02e9/1.e9, 2.35e9/1.e9],
    #           plotting=True,
    #           offset=0.01,
    #           fiberint=0.005)
