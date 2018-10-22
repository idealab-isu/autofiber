import os, time
import numpy as np
from scipy import optimize
from spatialnde import geometry
from spatialnde.coordframes import coordframe
from spatialnde.ndeobj import ndepart

from autofiber import geodesic as GEO
from autofiber import analyze_uv as AUV
from autofiber import optimization as OP


class AutoFiber:

    def __init__(self, cadfile, initpoint, initdirection, initnormal, pois=None, offset=0.01, **kwargs):
        """
        Calculate composite fiber orientations on a surface
        :param cadfile: Path to CAD file (currently supports x3d and stl)
        :param initpoint: 3D point closest to the corner of the surface we would like to work on
        :param initdirection: 3D vector representing the fiber direction at initpoint
        :param initnormal: 3D vector inicating the surface normal at initpoint
        (used to determine which surface to operate on)
        :param pois: Points on the surface where fiber orientation is desired, if None will calculate at all element centers
        :param offset: an offset from the initpoint to start geodesics away from the edge
        :param kwargs: options: E (Young's Modulus) = 228, nu (Poisson's ratio) = 0.2, fiberint = 0.1
                                plotting = False, optimize = True, accel = False
        """
        # Get CAD file of part that we would like to lay composite fiber over
        self.cadfile = cadfile

        # Gather options, set to default if option doesn't exist
        # plotting: activate visual plots
        self.plotting = kwargs.get("plotting", False)
        # optimize: activate strain energy density optimization
        self.optimize = kwargs.get("optimize", True)
        # accel: activate opencl optimization features
        self.accel = kwargs.get("accel", False)

        # Init spatialnde objects
        self.obj = None
        self.objframe = None

        # load model into spatialnde obj
        self.loadobj()

        # Init model variables
        self.vertices = None
        self.vertexids_indices = None
        self.vertexids = None
        self.facetnormals = None
        self.refpoints = None
        self.inplanemat = None
        self.edges = None
        self.adjacencyidx = None
        self.surfaces = None
        self.surface_vertexids = None
        self.texcoords2inplane = None

        # load relevant model variables
        self.loadvars()

        # Calculate the centroid of each element
        self.centroids = self.vertices[self.vertexids].sum(axis=1)/3

        # If no points of interest identified then use centroids
        self.pois = pois

        # Init fiber material properties
        # http://www.performance-composites.com/carbonfibre/mechanicalproperties_2.asp
        self.fiberint = kwargs.get("fiberint", 0.1)
        self.E = kwargs.get("E", np.array([12.3, 12.3, 0.]))
        self.nu = kwargs.get("nu", np.array([[0., 0.53, 0.],
                                             [0.53, 0., 0.],
                                             [0., 0., 0.]]))
        # [G_12, G_13, G_23]
        self.G = kwargs.get("G", np.array([11., 0., 0.]))

        # Init geodesic variables
        self.startpoints = np.empty((0, 3))
        self.startuv = np.empty((0, 2))
        self.startelements = np.empty(0, dtype=int)
        self.sfiberdirections = np.empty((0, 3))
        self.fiberdirections = np.empty((self.vertexids.shape[0], 3)) * np.nan

        # Init geodesic start parameters
        # initpoint: a point close to a corner of the surface we would like to work on
        self.initpoint = initpoint
        # initdirection: a vector in 3 space which corresponds to our
        # desired composite fiber direction at the corner point
        self.initdirection = initdirection
        # initnormal: a vector in 3 space which corresponds to the normal of the surface we are working on
        self.surfacenormal = initnormal
        # offset: an offset from the initpoint to start geodesics away from the edge
        # i.e. attempts to prevent early geodesic start point generation from terminating early
        self.offset = offset
        # Init geodesic path record
        self.georecord = {}
        self.geoints = []

        # Init uv parameterization parameters
        self.geoparameterization = np.empty((self.vertices.shape[0], 2)) * np.nan
        # Init optimization parameterization
        self.optimizedparameterization = None
        # Calculate compliance tensor
        # Orthotropic
        self.compliance_tensor = np.array([[1/self.E[0], -self.nu[1, 0]/self.E[1], -self.nu[2, 0]/self.E[2], 0, 0, 0],
                                           [-self.nu[0, 1]/self.E[0], 1/self.E[1], -self.nu[2, 1]/self.E[2], 0, 0, 0],
                                           [-self.nu[0, 2]/self.E[0], -self.nu[1, 2]/self.E[1], 1/self.E[2], 0, 0, 0],
                                           [0, 0, 0, 1/self.G[2], 0, 0],
                                           [0, 0, 0, 0, 1/self.G[1], 0],
                                           [0, 0, 0, 0, 0, 1/self.G[0]]])
        self.compliance_tensor[np.isnan(self.compliance_tensor)] = 0
        self.compliance_tensor[np.isinf(self.compliance_tensor)] = 0
        import pdb
        pdb.set_trace()
        # Isotropic
        # G = self.E / (2 * (1 + self.nu))
        # self.compliance_tensor = np.array([[1/self.E, -self.nu/self.E, -self.nu/self.E, 0, 0, 0],
        #                                    [-self.nu/self.E, 1/self.E, -self.nu/self.E, 0, 0, 0],
        #                                    [-self.nu/self.E, -self.nu/self.E, 1/self.E, 0, 0, 0],
        #                                    [0, 0, 0, 1/G, 0, 0],
        #                                    [0, 0, 0, 0, 1/G, 0],
        #                                    [0, 0, 0, 0, 0, 1/G]])

        # Calculate stiffness tensor
        self.stiffness_tensor = np.linalg.inv(self.compliance_tensor)
        # Calculate 2D normalized points for each element
        self.normalized_2d = OP.calc2d(self.obj, self.vertices[self.vertexids])
        # Init final orientation arrays
        self.orientations = None
        if self.pois is not None:
            self.orientations = np.zeros(self.pois.shape)
        else:
            self.orientations = np.zeros(self.centroids.shape)

        # Find start points for all geodesics
        self.find_startpoints()

        # Determine which surface we will be laying the geodesics upon
        self.determine_surface()

        # Calculate each geodesic across the surface
        self.calc_geodesics()

        # Interpolate any vertices missed by the geodesics
        self.cleanup()

        # If optimize is True then we will attempt to optimize the geodesic parametrization based on strain energy
        # density
        if self.optimize:
            self.fiberoptimize()

        # With results we will calculate the fiber directions based on the available parametrizations
        self.calcorientations()

        print("Done. Plotting...")

        if self.plotting:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import axes3d

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2])
            ax.scatter(self.startpoints[:, 0], self.startpoints[:, 1], self.startpoints[:, 2])
            for i in self.geoints:
                ax.plot(i[:, 0], i[:, 1], i[:, 2])

            fig = plt.figure()
            plt.scatter(self.geoparameterization[:, 0], self.geoparameterization[:, 1])
            if self.optimize:
                plt.scatter(self.optimizedparameterization[:, 0], self.optimizedparameterization[:, 1])

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2])
            ax.quiver(self.centroids[:, 0], self.centroids[:, 1], self.centroids[:, 2], self.orientations[:, 0],
                      self.orientations[:, 1], self.orientations[:, 2], length=0.1)

            plt.show()

    def loadobj(self):
        if os.path.splitext(self.cadfile)[1] == ".x3d":
            self.objframe = coordframe()
            self.obj = ndepart.fromx3d(self.objframe, None, self.cadfile, tol=1e-6)
        elif os.path.splitext(self.cadfile)[1] == ".stl":
            self.objframe = coordframe()
            self.obj = ndepart.fromstl(self.objframe, None, self.cadfile, tol=1e-6)
        else:
            raise Exception("Unsupported file type.")

    def loadvars(self):
        """ Load spatialnde data """
        self.vertices = self.obj.implpart.surfaces[0].vertices
        self.vertexids_indices = self.obj.implpart.surfaces[0].vertexidx_indices
        self.vertexids = self.obj.implpart.surfaces[0].vertexidx.reshape(self.vertexids_indices.shape[0], 4)[:, 0:3]
        self.facetnormals = self.obj.implpart.surfaces[0].facetnormals
        self.refpoints = self.obj.implpart.surfaces[0].refpoints
        self.inplanemat = self.obj.implpart.surfaces[0].inplanemats
        self.edges = AUV.BuildEdgeDict(self.obj.implpart.surfaces[0])
        self.adjacencyidx = AUV.DetermineAdjacency(self.obj.implpart.surfaces[0], self.edges)
        self.surfaces = AUV.FindTexPatches(self.obj.implpart.surfaces[0], self.adjacencyidx)
        self.texcoords2inplane = self.obj.implpart.surfaces[0].intrinsicparameterization.texcoords2inplane

    def find_startpoints(self):
        """
        Finds the starting points for each geodesic
        Shoots off geodesics in principle directions x, y, z, -x, -y, -z
        For any valid directions we drop start points in interval self.fiberint as the geodesic is traced
        """
        pointdirections = np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1],
                                    [-1, 0, 0],
                                    [0, -1, 0],
                                    [0, 0, -1]])

        for i in range(0, pointdirections.shape[0]):
            pointdirection = pointdirections[i]
            # Find the closest vertex in the mesh that corresponds to the defined start point
            closestvertex_ind = np.where(np.linalg.norm(self.vertices - self.initpoint, axis=1) == np.min(np.linalg.norm(self.vertices - self.initpoint, axis=1)))
            closestvertex = self.vertices[closestvertex_ind[0], :][0]

            # TODO: Offset direction
            point = closestvertex + self.initdirection * self.offset

            # Determine if start point is on a vertex or not then perform
            # the necessary calculation to find the next element
            if 0 in np.linalg.norm(self.vertices - point, axis=1):
                # Current point is a vertex:
                element, newvector = GEO.find_element_vertex(point, pointdirection, self.surfacenormal,
                                                             self.vertices, self.vertexids, self.facetnormals)
            else:
                # Current point is not on a vertex but within a polygon:
                element, newvector = GEO.find_element_within(point, pointdirection, self.surfacenormal, self.vertices,
                                                             self.vertexids, self.facetnormals, self.inplanemat)

            if element is None or newvector is None:
                continue

            if not GEO.check_proj_inplane_pnt(point, self.vertices[self.vertexids[element]]):
                continue

            vdirection = GEO.calcunitvector(np.cross(self.surfacenormal, self.initdirection))
            angle = GEO.angle_between_vectors(vdirection, pointdirection)

            # Rotate the given fiber vector to be in plane with the start element
            try:
                newfibervector = GEO.rot_vector(self.surfacenormal, self.facetnormals[element], self.initdirection)
            except GEO.EdgeError:
                continue

            self.startpoints = np.vstack((self.startpoints, point))
            self.startuv = np.vstack((self.startuv, np.array([0.0, 0.0])))
            self.startelements = np.append(self.startelements, element)
            self.sfiberdirections = np.vstack((self.sfiberdirections, newfibervector))

            pointdirection = GEO.rot_vector(self.surfacenormal, self.facetnormals[element], pointdirection)

            p = 1
            while True:
                dleft = np.abs(self.fiberint / np.cos(angle))
                try:
                    while True:
                        int_pnt_3d, nextunitvector, nextelement, _ = GEO.traverse_element(self, element, point, pointdirection, None, None, None, parameterization=False)

                        d2int = np.linalg.norm(int_pnt_3d - point)

                        dleft = dleft - d2int

                        if dleft <= 0:
                            point = int_pnt_3d + pointdirection * dleft

                            selementvec = GEO.rot_vector(self.facetnormals[self.startelements[-1]],
                                                         self.facetnormals[element], self.sfiberdirections[-1],
                                                         force=True)

                            self.startpoints = np.vstack((self.startpoints, point))
                            self.startelements = np.append(self.startelements, element)
                            self.sfiberdirections = np.vstack((self.sfiberdirections, selementvec))
                            break
                        else:
                            if not nextelement:
                                raise GEO.EdgeError
                            point = int_pnt_3d
                            element = nextelement
                            pointdirection = nextunitvector
                except GEO.EdgeError:
                    break
                self.startuv = np.vstack((self.startuv, np.array([np.abs(p * self.fiberint * np.tan(angle)), np.sign(self.fiberint / np.cos(angle)) * self.fiberint * p])))
                p += 1

    def determine_surface(self):
        # Determine if start point is on a vertex or not then perform the necessary calculation to find the next element
        if 0 in np.linalg.norm(self.vertices - self.startpoints[0], axis=1):
            # Current point is a vertex:
            element, newvector = GEO.find_element_vertex(self.startpoints[0], self.initdirection, self.surfacenormal,
                                                         self.vertices, self.vertexids, self.facetnormals)
        else:
            # Current point is not on a vertex but within a polygon:
            element, newvector = GEO.find_element_within(self.startpoints[0], self.initdirection, self.surfacenormal,
                                                         self.vertices, self.vertexids, self.facetnormals, self.inplanemat)

        surface = [(i, surface.index(element)) for i, surface in enumerate(self.surfaces) if element in surface][0][0]

        surface_polygons = self.surfaces[surface]
        self.surface_vertexids = self.vertexids[surface_polygons]

    def calc_geodesics(self):
        """
        Determines the approximate path of a geodesic along a fiber direction
        :param fiberdirection: Fiber orientation in degrees
        :param iters: Number of elements to iterate through
        :param obj: x3d model object
        :return: An array of fiber points relative to an approximate geodesic of the surface
        """
        # Setup uv parametrization arrays:
        fiberpoints_local = np.empty((self.vertices.shape[0], 2)) * np.nan
        print("Number of geodesics: %i" % self.startpoints.shape[0])
        start_time = time.time()
        for i in range(0, self.startpoints.shape[0]):
            unitfiberdirection = self.sfiberdirections[i]
            point = self.startpoints[i]
            element = self.startelements[i]
            uv_start_i = self.startuv[i]
            # Create an empty array of intersection points to visualize geodesics
            int_pnts = np.array([self.startpoints[i]])

            length = 0
            p = 0
            while True:
                try:
                    int_pnt_3d, nextunitvector, nextelement, fiberpoints_local = GEO.traverse_element(self, element, point, unitfiberdirection, fiberpoints_local, length, uv_start_i)
                except GEO.EdgeError:
                    break

                # Update and store the calculated fiber points and the intersection points
                int_pnts = np.vstack((int_pnts, int_pnt_3d))

                # Calculate the new length of the fiber
                length = np.linalg.norm(int_pnt_3d - point) + length

                if not nextelement:
                    break

                # Update previous iteration values with the next iteration values
                point = int_pnt_3d
                unitfiberdirection = nextunitvector
                element = nextelement
                p += 1
            self.geoints.append(int_pnts)

        stop_time = time.time()
        elapsed = stop_time - start_time
        print("Time to calculate geodesics: %f seconds" % elapsed)

    def cleanup(self):
        mask = np.ones((self.geoparameterization.shape[0]), dtype=bool)
        mask[np.unique(self.surface_vertexids)] = False
        leftover_idxs = np.where((np.isnan(self.geoparameterization).all(axis=1) & np.array(~mask)))[0]
        print("Cleaning up %s points" % leftover_idxs.size)
        while leftover_idxs.shape[0] > 0:
            for i in range(0, leftover_idxs.shape[0]):
                unassigned_facets = np.unique(np.where((self.vertexids == leftover_idxs[i]))[0])
                array_sum = np.array([0, 0])
                count = 0
                for j in unassigned_facets:
                    elementvertices = self.vertices[self.vertexids[j]]
                    assigned_vertsids = np.where((~np.isnan(self.geoparameterization[self.vertexids[j]]).all(axis=1)))[0]
                    if assigned_vertsids.shape[0] > 0:
                        if not np.isnan(self.fiberdirections[j]).any():
                            for k in assigned_vertsids:
                                fdistance, closest_point = GEO.calcclosestpoint(self.fiberdirections[j], elementvertices[k],
                                                                                np.array([self.vertices[leftover_idxs[i]]]),
                                                                                self.facetnormals[j])
                                unassigned_fpoint = self.geoparameterization[self.vertexids[j]][k] + fdistance
                                array_sum = array_sum + unassigned_fpoint
                                count += 1
                        else:
                            neighbors = GEO.find_neighbors(j, self.vertexids_indices, self.adjacencyidx)
                            for neighbor in neighbors:
                                if not np.isnan(self.fiberdirections[neighbor]).any():
                                    est_fiberdir = GEO.rot_vector(self.facetnormals[j], self.facetnormals[neighbor],
                                                                  self.fiberdirections[neighbor], force=True)
                                    self.fiberdirections[j] = est_fiberdir
                if count > 0:
                    fpoint = array_sum / count
                    self.geoparameterization[leftover_idxs[i]] = fpoint
            leftover_idxs = np.where((np.isnan(self.geoparameterization).all(axis=1) & np.array(~mask)))[0]
        assert np.where((np.isnan(self.geoparameterization).all(axis=1) & np.array(~mask)))[0].size == 0

    def fiberoptimize(self):
        def f(x, *args):
            return OP.computeglobalstrain(self.normalized_2d, x, self.vertexids, self.stiffness_tensor)

        def gradf(x, *args):
            return OP.computeglobalstrain_grad(self.normalized_2d, x, self.vertexids, self.stiffness_tensor)

        start_time = time.time()
        # res = optimize.minimize(f, self.geoparameterization, jac=gradf, method="CG", options={'gtol': 1})
        # print("Final Strain Energy Density Value: %f" % res.fun)
        # self.optimizedparameterization = res.x.reshape(self.geoparameterization.shape)
        self.optimizedparameterization = OP.optimize(f, gradf, self.geoparameterization)
        stop_time = time.time()
        elapsed = stop_time - start_time
        print("Time to optimize: %f seconds" % elapsed)

    def calcorientations(self):
        if self.optimize:
            self.calctransform(self.optimizedparameterization)
        else:
            self.calctransform(self.geoparameterization)

        if self.pois is not None:
            for i in range(0, self.pois.shape[0]):
                vert = self.pois[i]
                element = None
                for j in range(0, self.vertexids.shape[0]):
                    if geometry.point_in_polygon_3d(self.vertices[self.vertexids][j], vert, self.inplanemat[j]):
                        element = j
                        continue
                if element:
                    red_texcoords2inplane = self.texcoords2inplane[element][:2, :2]
                    texutexbasis = np.array([1.0, 0.0])
                    texu2dbasis = np.dot(red_texcoords2inplane, texutexbasis)
                    u3D = np.dot(self.inplanemat[element].T, texu2dbasis)
                    self.orientations[i] = GEO.calcunitvector(u3D)
                else:
                    print("Failed to find point on surface: %s" % vert)
        else:
            red_texcoords2inplane = self.texcoords2inplane[:, :2, :2]
            texutexbasis = np.repeat(np.array([1.0, 0.0])[np.newaxis, :], red_texcoords2inplane.shape[0], axis=0)
            texu2dbasis = np.einsum('ijk,ik->ij', red_texcoords2inplane, texutexbasis)
            self.orientations = GEO.calcunitvector(np.einsum('ijk,ik->ij', self.inplanemat.transpose(0, 2, 1), texu2dbasis))

    def calctransform(self, parameterization):
        self.obj.implpart.surfaces[0].intrinsicparameterization.invalidateprojinfo()
        self.obj.implpart.surfaces[0].intrinsicparameterization.invalidateboxes()
        self.obj.implpart.surfaces[0].intrinsicparameterization.texcoord = parameterization
        self.obj.implpart.surfaces[0].intrinsicparameterization.texcoordidx = self.obj.implpart.surfaces[0].vertexidx
        self.obj.implpart.surfaces[0].intrinsicparameterization.buildprojinfo(self.obj.implpart.surfaces[0])

        self.vertices = self.obj.implpart.surfaces[0].vertices
        self.vertexids_indices = self.obj.implpart.surfaces[0].vertexidx_indices
        self.vertexids = self.obj.implpart.surfaces[0].vertexidx.reshape(self.vertexids_indices.shape[0], 4)[:, 0:3]
        self.inplanemat = self.obj.implpart.surfaces[0].inplanemats
        self.texcoords2inplane = self.obj.implpart.surfaces[0].intrinsicparameterization.texcoords2inplane
