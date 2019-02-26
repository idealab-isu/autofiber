import os, time
import numpy as np
from scipy import optimize
from spatialnde.coordframes import coordframe
from spatialnde.ndeobj import ndepart

from autofiber import geodesic as GEO
from autofiber import analyze_uv as AUV
from autofiber import optimization as OP


class AutoFiber:

    def __init__(self, cadfile, initpoint, initdirection, initnormal, pois=None, offset=0.05, **kwargs):
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
        # accel: activate opencl optimization features
        self.accel = kwargs.get("accel", False)
        # save: write orientations to file
        self.save = kwargs.get("save", False)

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
        self.fiberint = kwargs.get("fiberint", 0.1)
        self.E = kwargs.get("E", 228.0)
        self.nu = kwargs.get("nu", 0.2)

        # Init geodesic variables
        self.startpoints = np.empty((0, 3))
        self.startuv = np.empty((0, 2))
        self.startelements = np.empty(0, dtype=int)
        self.sfiberdirections = np.empty((0, 3))
        self.fiberdirections = np.empty((self.vertexids.shape[0], 3)) * np.nan
        self.fiberpoints_local = np.empty((self.vertices.shape[0], 2)) * np.nan

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
        if isinstance(self.E, list):
            # Orthotropic
            G = kwargs.get("G", None)
            if G is None:
                raise ValueError("G property is not defined.")
            self.compliance_tensor = np.array([[1/self.E[0], -self.nu[0]/self.E[1], 0],
                                               [-self.nu[0]/self.E[0], 1/self.E[1], 0],
                                               [0, 0, 1/(2*G[0])]])
        else:
            # Isotropic
            G = self.E / (2 * (1 + self.nu))
            self.compliance_tensor = np.array([[1/self.E, -self.nu/self.E, 0],
                                               [-self.nu/self.E, 1/self.E, 0],
                                               [0, 0, 1/G]])
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

        # Optimize the geodesic parametrization based on strain energy density
        self.fiberoptimize()

        # With results we will calculate the fiber directions based on the available parametrizations
        self.calctransform(self.optimizedparameterization)
        self.orientations = calcorientations_abaqus(self.centroids, self.vertices, self.vertexids, self.inplanemat,
                                                    self.texcoords2inplane, self.obj.implpart.surfaces[0].boxes,
                                                    self.obj.implpart.surfaces[0].boxpolys,
                                                    self.obj.implpart.surfaces[0].boxcoords)

        if self.save:
            np.save("orientation.npy", self.orientations)

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
            plt.scatter(self.startuv[:, 0], self.startuv[:, 1])
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
                    int_pnt_3d, nextunitvector, nextelement, fiberpoints_local = GEO.traverse_element(self, element, point, unitfiberdirection, self.fiberpoints_local, length, uv_start_i)
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

            unitfiberdirection = -1*self.sfiberdirections[i]
            point = self.startpoints[i]
            element = self.startelements[i]
            uv_start_i = self.startuv[i]
            # Create an empty array of intersection points to visualize geodesics
            int_pnts = np.array([self.startpoints[i]])

            length = 0
            p = 0
            while True:
                try:
                    int_pnt_3d, nextunitvector, nextelement, fiberpoints_local = GEO.traverse_element(self, element, point, unitfiberdirection, self.fiberpoints_local, length, uv_start_i, direction=-1)
                except GEO.EdgeError:
                    break

                # Update and store the calculated fiber points and the intersection points
                int_pnts = np.vstack((int_pnts, int_pnt_3d))

                # Calculate the new length of the fiber
                length = -1*np.linalg.norm(int_pnt_3d - point) + length

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
        import pdb
        mask = np.ones((self.geoparameterization.shape[0]), dtype=bool)
        mask[np.unique(self.surface_vertexids)] = False
        leftover_idxs = np.where((np.isnan(self.geoparameterization).all(axis=1) & np.array(~mask)))[0]
        print("Cleaning up %s points" % leftover_idxs.size)
        print(leftover_idxs)

        # leftover_idxs = self.sup_geodesics(leftover_idxs, mask)
        # print("After step one: %s" % leftover_idxs.shape[0])

        leftover_idxs = self.interpolate(leftover_idxs, mask)
        print("After step two: %s" % leftover_idxs.shape[0])

        leftover_idxs = self.average_fpoint(leftover_idxs, mask)
        print("After step three: %s" % leftover_idxs.shape[0])

        self.manual()

        rel_uvw = np.pad(self.geoparameterization[self.vertexids], [(0, 0), (0, 0), (0, 1)], "constant", constant_values=1).transpose(0, 2, 1)
        vdir = 0.5 * np.linalg.det(rel_uvw)

        assert (vdir > 0).any()
        # assert np.where((np.isnan(self.geoparameterization).all(axis=1) & np.array(~mask)))[0].size == 0

    def find_close_geodesic(self, element, point):
        if element in self.georecord.keys():
            # [pointuv (bary), int_pnt (bary), point (3D), unitfiberdirection (3D), closest_point_idx (idx), uv_start, length]
            geodesics = self.georecord[element][0]
            dlist = []
            dlist_arrays = []
            for g in range(0, len(geodesics)):
                if ~np.isnan(self.geoparameterization[geodesics[g][4]]).all():
                    d2left = GEO.calcdistance(geodesics[g][3], geodesics[g][2], point)
                    dperp = np.linalg.norm(d2left[0])
                    dlist_arrays.append(d2left)
                    dlist.append(dperp)
            if len(dlist) > 0:
                mind = np.argmin(dlist)
                return geodesics[mind], dlist_arrays[mind], dlist[mind]
            else:
                raise IndexError("Cannot find a close geodesic")
        else:
            raise IndexError("Cannot find a close geodesic")

    def interpolate(self, leftover_idxs, mask):
        timeout = 0
        while leftover_idxs.shape[0] > 0 and timeout < 50:
            for i in range(0, leftover_idxs.shape[0]):
                unassigned_facets = np.unique(np.where((self.vertexids == leftover_idxs[i]))[0])
                done = False
                for j in unassigned_facets:
                    # try:
                    #     min_geodesic, minparl, minperp = self.find_close_geodesic(j, self.vertices[leftover_idxs[i]])
                    #
                    #
                    #
                    #     import pdb
                    #     pdb.set_trace()
                    # except IndexError:
                    #     pass

                    elementvertices = self.vertices[self.vertexids[j]]
                    assigned_vertsids = np.where((~np.isnan(self.geoparameterization[self.vertexids[j]]).all(axis=1)))[0]
                    if assigned_vertsids.shape[0] > 0:
                        try:
                            min_geodesic, minparl, minperp = self.find_close_geodesic(j, self.vertices[leftover_idxs[i]])
                            fiberdirection = min_geodesic[3]
                        except IndexError:
                            if not np.isnan(self.fiberdirections[j]).any():
                                fiberdirection = self.fiberdirections[j]
                            else:
                                neighbors = GEO.find_neighbors(j, self.vertexids_indices, self.adjacencyidx)
                                for neighbor in neighbors:
                                    if not np.isnan(self.fiberdirections[neighbor]).any():
                                        est_fiberdir = GEO.rot_vector(self.facetnormals[j], self.facetnormals[neighbor],
                                                                      self.fiberdirections[neighbor], force=True)
                                        self.fiberdirections[j] = est_fiberdir
                                        fiberdirection = est_fiberdir
                                        break

                        for k in assigned_vertsids:
                            fdistance, closest_point = GEO.calcclosestpoint(fiberdirection, elementvertices[k],
                                                                            np.array([self.vertices[leftover_idxs[i]]]),
                                                                            self.facetnormals[j])
                            unassigned_fpoint = self.geoparameterization[self.vertexids[j]][k] + np.sign(min_geodesic[6])*fdistance

                            fiberrec = np.copy(self.geoparameterization)
                            fiberrec[leftover_idxs[i]] = unassigned_fpoint
                            rel_uvw = np.pad(fiberrec[self.vertexids], [(0, 0), (0, 0), (0, 1)], "constant", constant_values=1).transpose(0, 2, 1)
                            vdir = 0.5 * np.linalg.det(rel_uvw)
                            if (vdir < 0).any():
                                pass
                            else:
                                self.geoparameterization[leftover_idxs[i]] = unassigned_fpoint
                                done = True
                                break
                    if done:
                        break
            leftover_idxs = np.where((np.isnan(self.geoparameterization).all(axis=1) & np.array(~mask)))[0]
            timeout += 1
        return leftover_idxs

    def sup_geodesics(self, leftover_idxs, mask):
        import pdb
        startpoints = np.empty((0, 3))
        startuv = np.empty((0, 2))
        startelements = np.empty(0, dtype=int)
        sfiberdirections = np.empty((0, 3))
        for i in range(0, leftover_idxs.shape[0]):
            leftover_neighbors = np.unique(np.where((self.vertexids == leftover_idxs[i]))[0])
            for facet in leftover_neighbors:
                try:
                    min_geodesic, minparl, minperp = self.find_close_geodesic(facet, self.vertices[leftover_idxs[i]])
                    spoint = self.vertices[leftover_idxs[i]]
                    v2p = np.cross(min_geodesic[3], self.facetnormals[facet])

                    gpoint = spoint - calcunitvector(v2p) * 0.0001
                    if point_in_polygon_3d(self.vertices[self.vertexids[facet]], gpoint, self.inplanemat[facet]):
                        element = facet
                        point = np.copy(gpoint)
                        pointdirection = -1*np.copy(calcunitvector(v2p))

                        startpoints = np.vstack((startpoints, point))
                        startuv = np.vstack((startuv, np.array([min_geodesic[5][0] + min_geodesic[6] + minparl[1], min_geodesic[5][1] - minperp])))
                        startelements = np.append(startelements, facet)
                        sfiberdirections = np.vstack((sfiberdirections, min_geodesic[3]))

                        p = 1
                        while True:
                            dleft = self.fiberint
                            try:
                                while True:
                                    int_pnt_3d, nextunitvector, nextelement, _ = GEO.traverse_element(self, element,
                                                                                                      point,
                                                                                                      pointdirection,
                                                                                                      None, None, None,
                                                                                                      parameterization=False)
                                    d2int = np.linalg.norm(int_pnt_3d - point)
                                    dleft = dleft - d2int

                                    if dleft <= 0:
                                        point = int_pnt_3d + pointdirection * dleft

                                        try:
                                            neargeo, nearparl, nearperp = self.find_close_geodesic(element, point)
                                            selementvec = neargeo[3]
                                        except IndexError:
                                            selementvec = GEO.rot_vector(self.facetnormals[startelements[-1]],
                                                                         self.facetnormals[element],
                                                                         sfiberdirections[-1],
                                                                         force=True)

                                        startpoints = np.vstack((startpoints, point))
                                        startelements = np.append(startelements, element)
                                        sfiberdirections = np.vstack((sfiberdirections, selementvec))
                                        break
                                    else:
                                        if not nextelement:
                                            raise GEO.EdgeError
                                        point = int_pnt_3d
                                        element = nextelement
                                        pointdirection = nextunitvector
                            except GEO.EdgeError:
                                break
                            # if neargeo is not None:
                            startuv_i = np.array([startuv[-1][0], startuv[-1][1] + self.fiberint])
                            startuv = np.vstack((startuv, startuv_i))
                            # pdb.set_trace()
                            p += 1
                except IndexError:
                    pass

        ints = []
        for s in range(0, startuv.shape[0]):
            unitfiberdirection = np.copy(sfiberdirections[s])
            point = np.copy(startpoints[s])
            element = startelements[s]
            uv_start_i = np.copy(startuv[s])
            # Create an empty array of intersection points to visualize geodesics
            int_pnts = np.array([point])

            length = 0
            p = 0
            while True:
                try:
                    int_pnt_3d, nextunitvector, nextelement, fiberpoints_local = GEO.traverse_element(
                        self, element, point, unitfiberdirection, self.fiberpoints_local, length,
                        uv_start_i)
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
            ints.append(int_pnts)

        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import axes3d
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], alpha=0.1)
        # ax.scatter(startpoints[:, 0], startpoints[:, 1], startpoints[:, 2])
        # ax.quiver(startpoints[:, 0], startpoints[:, 1], startpoints[:, 2], sfiberdirections[:, 0], sfiberdirections[:, 1], sfiberdirections[:, 2])
        # for o in ints:
        #     ax.plot(o[:, 0], o[:, 1], o[:, 2])
        #
        # pdb.set_trace()
        leftover_idxs = np.where((np.isnan(self.geoparameterization).all(axis=1) & np.array(~mask)))[0]
        return leftover_idxs

    def average_fpoint(self, leftover_idxs, mask):
        for i in range(0, leftover_idxs.shape[0]):
            neighbors = np.unique(np.where((self.vertexids == leftover_idxs[i]))[0])
            neighbor_fpoint = self.geoparameterization[self.vertexids[neighbors]]
            neighbor_fpoint = neighbor_fpoint.reshape(neighbor_fpoint.shape[0]*neighbor_fpoint.shape[1], 2)

            count = ~np.isnan(neighbor_fpoint).all(axis=1)

            average_fpoint = np.sum(neighbor_fpoint[count], axis=0)/neighbor_fpoint[count].shape[0]

            fiberrec = np.copy(self.geoparameterization)
            fiberrec[leftover_idxs[i]] = average_fpoint
            rel_uvw = np.pad(fiberrec[self.vertexids], [(0, 0), (0, 0), (0, 1)], "constant", constant_values=1)
            vdir = 0.5 * np.linalg.det(rel_uvw)

            if (np.sign(vdir) < 0).any():
                pass
            else:
                self.geoparameterization[leftover_idxs[i]] = average_fpoint

            del fiberrec
        leftover_idxs = np.where((np.isnan(self.geoparameterization).all(axis=1) & np.array(~mask)))[0]
        return leftover_idxs

    def manual(self):
        import pdb
        import matplotlib.pyplot as plt

        fig = plt.figure()
        for i in range(0, self.vertexids.shape[0]):
            plt.plot(self.geoparameterization[self.vertexids[i]][:, 0], self.geoparameterization[self.vertexids[i]][:, 1])

        pdb.set_trace()

    def fiberoptimize(self):
        def f(x, *args):
            return OP.computeglobalstrain(self.normalized_2d, x, self.vertexids, self.stiffness_tensor)

        def gradf(x, *args):
            return OP.computeglobalstrain_grad(self.normalized_2d, x, self.vertexids, self.stiffness_tensor)

        start_time = time.time()
        print("Optimizing...")
        res = optimize.minimize(f, self.geoparameterization, jac=gradf, method="CG")
        print("Final Strain Energy Value: %f J/m" % res.fun)
        self.optimizedparameterization = res.x.reshape(self.geoparameterization.shape)
        stop_time = time.time()
        elapsed = stop_time - start_time
        print("Time to optimize: %f seconds" % elapsed)

    def calcorientations(self):
        self.calctransform(self.optimizedparameterization)

        if self.pois is not None:
            for i in range(0, self.pois.shape[0]):
                vert = self.pois[i]
                element = None
                for j in range(0, self.vertexids.shape[0]):
                    if point_in_polygon_3d(self.vertices[self.vertexids][j], vert, self.inplanemat[j]):
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


def calcorientations_abaqus(modellocs, vertices, vertexids, inplanemat, texcoords2inplane, boxes, boxpolys, boxcoords):
    orientations = np.zeros((modellocs.shape[0], 3))
    for i in range(0, modellocs.shape[0]):
        vert = modellocs[i]
        element = None

        rangex = np.array([boxcoords[:, 0], boxcoords[:, 3]]).T
        rangey = np.array([boxcoords[:, 1], boxcoords[:, 4]]).T
        rangez = np.array([boxcoords[:, 2], boxcoords[:, 5]]).T

        containers = np.where(np.logical_and(np.logical_and(
            np.logical_and(vert[0] > rangex[:, 0], vert[0] < rangex[:, 1]),
            np.logical_and(vert[1] > rangey[:, 0], vert[1] < rangey[:, 1])),
            np.logical_and(vert[2] > rangez[:, 0], vert[2] < rangez[:, 1])))[0]

        box_idx = boxes[containers][:, -1][boxes[containers][:, -1] != -1][0]

        polys = np.array([boxpolys[box_idx]])
        count = 1
        while True:
            if boxpolys[box_idx+count] == -1:
                break
            polys = np.append(polys, boxpolys[box_idx+count])
            count += 1

        for j in polys:
            if point_in_polygon_3d(vertices[vertexids][j], vert, inplanemat[j]):
                element = j
                continue
        if element is not None:
            red_texcoords2inplane = texcoords2inplane[element][:2, :2]
            texutexbasis = np.array([1.0, 0.0])
            texu2dbasis = np.dot(red_texcoords2inplane, texutexbasis)
            u3D = np.dot(inplanemat[element].T, texu2dbasis)
            orientations[i] = calcunitvector(u3D)
        else:
            print("Failed to find point on surface: %s" % vert)
    return orientations


def calcunitvector(vector):
    """ Returns the unit vector of the vector.  """
    if len(vector.shape) >= 2:
        return vector / np.linalg.norm(vector, axis=1)[:, np.newaxis]
    else:
        return vector / np.linalg.norm(vector)


def point_in_polygon_2d(vertices_rel_point):
    import sys
    # Apply winding number algorithm.
    # This algorithm is selected -- in its most simple form --
    # because it is so  simple and robust in the case of the
    # intersect point being on or near the edge. It may well
    # be much slower than optimal. It tries to return True
    # in the edge case.

    # Should probably implement a faster algorithm then drop
    # down to this for the special cases.

    # See Hormann and Agathos, The point in polygon problem
    # for arbitrary polygons, Computational Geometry 20(3) 131-144 (2001)
    # http://dx.doi.org/10.1016/S0925-7721(01)00012-8
    # https://pdfs.semanticscholar.org/e90b/d8865ddb7c7af2b159d413115050d8e5d297.pdf

    # Winding number is sum over segments of
    # acos((point_to_vertex1 dot point_to_vertex2)/(magn(point_to_vertex1)*magn(point_to_vertex_2))) * sign(det([ point_to_vertex1  point_to_vertex2 ]))
    # where sign(det) is really: What is the sign of the z
    # component of (point_to_vertex1 cross point_to_vertex2)

    # Special cases: magn(point_to_vertex1)==0 or
    #  magn_point_to_vertex2   -> point is on edge
    # det([ point_to_vertex1  point_to_vertex2 ]) = 0 -> point may be on edge

    windingnum = 0.0
    numvertices = vertices_rel_point.shape[0]

    for VertexCnt in range(numvertices):
        NextVertex = VertexCnt + 1
        if NextVertex == numvertices:
            # final vertex... loop back to the start
            NextVertex = 0
            pass

        # calculate (thisvertex - ourpoint) -> vec1
        vec1 = vertices_rel_point[VertexCnt, :]
        magn1 = np.linalg.norm(vec1)

        # calculate (nextvertex - ourpoint) -> vec2
        vec2 = vertices_rel_point[NextVertex, :]
        magn2 = np.linalg.norm(vec2)

        if magn1 == 0.0 or magn2 == 0.0:
            # Got it!!!
            return True

        vec1 = vec1 / magn1
        vec2 = vec2 / magn2

        det = vec1[0] * vec2[1] - vec2[0] * vec1[1]  # matrix determinant

        cosparam = (vec1[0] * vec2[0] + vec1[1] * vec2[1])  # /(magn1*magn2);

        if cosparam < -1.0:
            # Shouldn't be possible...just in case of weird roundoff
            cosparam = -1.0

        if cosparam > 1.0:
            # Shouldn't be possible...just in case of weird roundoff
            cosparam = 1.0

        if det > 0:
            windingnum += np.arccos(cosparam)
        elif det < 0:
            windingnum -= np.arccos(cosparam)
        else:
            # det==0.0
            # Vectors parallel or anti-parallel

            if cosparam > 0.9:
                # Vectors parallel. We are OUTSIDE. Do Nothing
                pass
            elif cosparam < -0.9:
                # Vectors anti-parallel. We are ON EDGE */
                return True
            else:
                assert 0  # Should only be able to get cosparam = +/- 1.0 if abs(det) > 0.0 */
                pass
            pass
        pass

    windingnum = abs(windingnum) * (
                1.0 / (2.0 * np.pi))  # divide out radians to number of winds; don't care about clockwise vs. ccw
    if windingnum > .999 and windingnum < 1.001:
        # Almost exactly one loop... got it!
        return True
    elif windingnum >= .001:
        #
        sys.stderr.write(
            "spatialnde.geometry.point_in_polygon_2d() Got weird winding number of %e; assuming inaccurate calculation on polygon edge\n" % (
                windingnum))
        # Could also be self intersecting polygon
        # got it !!!
        return True

    # If we got this far, the search failed
    return False


def point_in_polygon_3d(vertices, point, inplanemat):
    """ assumes vertices are coplanar, with given orthonormal 2D basis inplanemat.  """
    vert3d_rel_point = vertices-point[np.newaxis, :]
    vert2d_rel_point = np.inner(vert3d_rel_point, inplanemat)

    return point_in_polygon_2d(vert2d_rel_point)
