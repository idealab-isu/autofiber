import pickle
import numpy as np
from autofiber.generator import AutoFiber as AF

# angles = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
angles = [0.0]

for angle in angles:
    print("Angle: %s" % angle)
    # test = AF(r'C:\Files\Research\Fiber-Generator\demos\plate.x3d', np.array([1.0, -1.0, 0.0]), np.array([-np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0]), np.array([0, 0, 1]), fiberint=0.01)

    # Anisotropic
    # test = AF(r'C:\Users\Nate\Documents\UbuntuSharedFiles\fibergeneration\demos\plate.x3d',
    #           np.array([1.0, -1.0, 0.0]),
    #           np.array([-np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0]),
    #           np.array([0, 0, 1]),
    #           E=[1.415e11/1.e9, 8.5e9/1.e9, 8.5e9/1.e9],
    #           nu=[0.33, 0.33, 0.33],
    #           G=[5.02e9/1.e9, 5.02e9/1.e9, 2.35e9/1.e9],
    #           fiberint=0.01)

    # test = AF(r'C:\Files\Research\Fiber-Generator\demos\cylinder.x3d', np.array([1.0, -1.0, 0.0]), np.array([0, np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]), np.array([1, 0, 0]), fiberint=0.5)
    # test = AF(r'C:\Files\Research\Fiber-Generator\demos\cylinder.x3d', np.array([0.0, 0.0, 1.0]), np.array([np.sin(-1*np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0.0]), np.array([0, 0, 1.0]), fiberint=0.1)

    # Anisotropic
    # test = AF(r'C:\Files\Research\Fiber-Generator\demos\cylinder.x3d',
    #           np.array([0.0, 0.0, 1.0]),
    #           np.array([np.sin(-1 * np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0.0]),
    #           np.array([0, 0, 1.0]),
    #           E=[1.415e11/1.e9, 8.5e9/1.e9, 8.5e9/1.e9],
    #           nu=[0.33, 0.33, 0.33],
    #           G=[5.02e9/1.e9, 5.02e9/1.e9, 2.35e9/1.e9],
    #           fiberint=0.05)

    # test = AF(r'C:\Users\Nate\Documents\UbuntuSharedFiles\fibergeneration\demos\CurvedSurface.x3d', np.array([25.0, 0.0, 25.0]), np.array([-np.sin(np.deg2rad(angle)), 0, -np.cos(np.deg2rad(angle))]), np.array([0, -1, 0]))

    # test = AF(r'C:\Files\Research\Fiber-Generator\demos\smallsaddle.x3d', np.array([0.0, 1.0, 0.0]), np.array([-np.cos(np.deg2rad(angle)), 0, -np.sin(np.deg2rad(angle))]), np.array([0, 1, 0]), fiberint=0.005)

    # Anisotropic
    # test = AF(r'C:\Files\Research\Fiber-Generator\demos\smallsaddle.x3d',
    #           np.array([0.0, 1.0, 0.0]),
    #           np.array([-np.cos(np.deg2rad(angle)), 0, -np.sin(np.deg2rad(angle))]),
    #           np.array([0, 1, 0]),
    #           E=[1.415e11/1.e9, 8.5e9/1.e9, 8.5e9/1.e9],
    #           nu=[0.33, 0.33, 0.33],
    #           G=[5.02e9/1.e9, 5.02e9/1.e9, 2.35e9/1.e9],
    #           fiberint=0.01)

    # test = AF(r'C:\Files\Research\de-la-mo-v2\examples\01_Planar_Single_Delam_output\Mold.stl',     # mm
    #           np.array([0.0, 0.0, 3.0]),
    #           np.array([np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0.0]),
    #           np.array([0, 0, 1]),
    #           E=[1.415e11/1.e9, 8.5e9/1.e9, 8.5e9/1.e9],        # GPa
    #           nu=[0.33, 0.33, 0.33],
    #           G=[5.02e9/1.e9, 5.02e9/1.e9, 2.35e9/1.e9],        # GPa
    #           fiberint=1.0)

    # test = AF(pickle.load(open(r'C:\Files\Research\Fiber-Generator\demos\test.pkl', 'rb')),     # mm
    #           np.array([0.0, 0.0, 3.0]),
    #           np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0.0]),
    #           np.array([0, 0, 1]),
    #           E=[1.415e11/1.e9, 8.5e9/1.e9, 8.5e9/1.e9],        # GPa
    #           nu=[0.33, 0.33, 0.33],
    #           G=[5.02e9/1.e9, 5.02e9/1.e9, 2.35e9/1.e9],        # GPa
    #           fiberint=0.25)

    test = AF(r'C:\Files\Research\Fiber-Generator\demos\flatplate.stl',     # mm
              np.array([0.0, 0.0, 3.0]),
              np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0.0]),
              np.array([0, 0, 1]),
              E=[1.415e11/1.e9, 8.5e9/1.e9, 8.5e9/1.e9],        # GPa
              nu=[0.33, 0.33, 0.33],
              G=[5.02e9/1.e9, 5.02e9/1.e9, 2.35e9/1.e9],        # GPa
              fiberint=0.25)

    # meshcoords = np.load("curved_abaqus_mesh_coords.npy")
    meshcoords = np.load("test_Layer_1_LB1.npy")

    texcoords2inplane = test.layup(0.0, orienation_locations=meshcoords, plotting=True)
