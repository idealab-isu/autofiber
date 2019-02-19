import numpy as np
from spatialnde import geometry


class EdgeError(Exception):
    pass


def referenceaxis(facetnormal):
    """ Determine a reference axis based on the element normal
        Returns: Global axis pointing to "right" of the element normal
    """
    normal = np.array([0, 0, 0])
    if facetnormal[np.where((np.abs(facetnormal) == max(np.abs(facetnormal))))] > 0:
        normal[np.where((np.abs(facetnormal) == max(np.abs(facetnormal))))] = 1
    else:
        normal[np.where((np.abs(facetnormal) == max(np.abs(facetnormal))))] = -1
    coordsys = np.where((normal != 0))[0][0]
    if normal[coordsys] > 0:
        if coordsys == 0:
            axis = np.array([0, 1, 0])
        elif coordsys == 1:
            axis = np.array([0, 0, 1])
        else:
            axis = np.array([0, 1, 0])
    else:
        if coordsys == 0:
            axis = np.array([0, -1, 0])
        elif coordsys == 1:
            axis = np.array([0, 0, -1])
        else:
            axis = np.array([0, -1, 0])
    return axis


def calcunitvector(vector):
    """ Returns the unit vector of the vector.  """
    if len(vector.shape) >= 2:
        return vector / np.linalg.norm(vector, axis=1)[:, np.newaxis]
    else:
        return vector / np.linalg.norm(vector)


def calcnormal(points):
    """ Returns the normal for the given 2d points"""
    v1 = points[2] - points[0]
    v2 = points[1] - points[0]
    return np.cross(v1, v2)


def angle_between_vectors(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'
        https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    """
    v1_u = calcunitvector(v1)
    v2_u = calcunitvector(v2)
    detarray = np.vstack((v1_u, v2_u))
    detarray = np.vstack((detarray, np.ones(detarray.shape[1]))).T
    det = np.linalg.det(detarray)
    if det < 0:
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    else:
        return 2*np.pi - np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def vector_inbetween(v1, v2, v3):
    """
    Determines if a vector (v1) is between v2 and v3
    https://stackoverflow.com/questions/13640931/how-to-determine-if-a-vector-is-between-two-other-vectors
    :param v1: Test vector
    :param v2: Given vector
    :param v3: Given vector
    :return: True if vector is between v2 and v3, false if not between
    """
    if np.dot(np.cross(v2, v1), np.cross(v2, v3)) >= 0 and np.dot(np.cross(v3, v1), np.cross(v3, v2)) >= 0:
        return True
    else:
        return False


def rot_vector(oldnormal, newnormal, vector, force=False):
    """ Rotate a vector given an axis and an angle of rotation
        Returns: Vector reoriented from an old element face to a new element
        https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula
    """
    if np.linalg.norm(oldnormal - newnormal) < 1e-10:
        return vector
    else:
        vector_a = np.cross(oldnormal, newnormal)
        sinphi = np.linalg.norm(vector_a)
        cosphi = np.dot(oldnormal, newnormal)
        a_hat = vector_a/sinphi
        if np.arccos(cosphi) >= np.deg2rad(85) and not force:
            # print("### Crossed a 90 degree or greater edge ###")
            raise EdgeError
        else:
            return calcunitvector(vector * cosphi - np.cross(vector, a_hat) * sinphi + a_hat * np.dot(vector, a_hat) * (1 - cosphi))


def check_proj_inplane_pnt(point, element_vertices):
    """
    https://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle
    :param point: Test point to check
    :param element_vertices: Vertices of current element
    :return: True or False, depending on if the projected point is inside or outside
    """
    u = element_vertices[1] - element_vertices[0]
    v = element_vertices[2] - element_vertices[0]
    normal = np.cross(u, v)
    w = point - element_vertices[0]

    gamma = np.dot(np.cross(u, w), normal) / np.dot(normal, normal)
    beta = np.dot(np.cross(w, v), normal) / np.dot(normal, normal)
    alpha = 1 - gamma - beta

    projpoint = alpha * element_vertices[0] + beta * element_vertices[1] + gamma * element_vertices[2]
    if 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1:
        return True, projpoint
    else:
        return False, projpoint


def proj_vector(vector, newnormal):
    return calcunitvector(vector - np.dot(vector, newnormal) * newnormal)


def check_intersection(p1, q1, p2, q2):
    """
    https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    """

    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0])*(r[1] - q[1])
        if val is 0:
            return 0
        if val > 0:
            return 1
        else:
            return 2

    def onSegment(p, q, r):
        if q[0] <= np.max([p[0], r[0]]) and q[0] >= np.min([p[0], r[0]]) and q[1] <= np.max([p[1], r[1]]) and q[1] >= np.min([p[1], r[1]]):
            return True
        return False

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    # Special Cases
    # p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if o1 == 0 and onSegment(p1, p2, q1):
        return True
    # p1, q1 and q2 are colinear and q2 lies on segment p1q1
    if o2 == 0 and onSegment(p1, q2, q1):
        return True
    # p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if o3 == 0 and onSegment(p2, p1, q2):
        return True
    # p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if o4 == 0 and onSegment(p2, q1, q2):
        return True

    # Doesn't fall in any of the above cases
    return False


def find_intpnt(P1, P2, P3, P4):
    """ Line-Line intersection method
        Returns: A point in 2d that intersects line P1P2 and P3P4
        https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    """
    return np.array([((P1[0] * P2[1] - P1[1] * P2[0]) * (P3[0] - P4[0]) - (P1[0] - P2[0]) * (P3[0] * P4[1] - P3[1] * P4[0])) /
                     ((P1[0] - P2[0]) * (P3[1] - P4[1]) - (P1[1] - P2[1]) * (P3[0] - P4[0])),
                     ((P1[0] * P2[1] - P1[1] * P2[0]) * (P3[1] - P4[1]) - (P1[1] - P2[1]) * (P3[0] * P4[1] - P3[1] * P4[0])) /
                     ((P1[0] - P2[0]) * (P3[1] - P4[1]) - (P1[1] - P2[1]) * (P3[0] - P4[0]))])


def find_edge(point, direction, error):
    """
    Determines which edge number is intersected first (0, 1, 2) -> (d12, d23, d31)
    https://math.stackexchange.com/questions/2292895/walking-on-the-surface-of-a-triangular-mesh
    :param point: Start point
    :param direction: Current fiber direction
    :param error: Numerical tolerance
    :return: Edge number (0, 1, 2) or -1 if on an edge
    """
    if direction[1] != 0:
        d0 = -point[1] / direction[1]
    else:
        d0 = -1

    if direction[0] + direction[1] != 0:
        d1 = (1 - point[0] - point[1]) / (direction[0] + direction[1])
    else:
        d1 = -1

    if direction[0] != 0:
        d2 = -point[0] / direction[0]
    else:
        d2 = -1

    if d0 >= error and (d0 <= d1 or d1 <= error) and (d0 <= d2 or d2 <= error):
        return 0
    elif d1 >= error and (d1 <= d0 or d0 <= error) and (d1 <= d2 or d2 <= error):
        return 1
    elif d2 > error and (d2 <= d0 or d0 <= error) and (d2 <= d1 or d1 <= error):
        return 2
    elif d0 == -1 or d1 == -1 or d2 == -1:
        raise EdgeError("### Trying to follow an edge, bad direction? ###")
    elif d0 < 0 and d1 < 0 and d2 < 0:
        return find_edge(point, -direction, error)
    else:
        # pdb.set_trace()
        raise EdgeError('### Something weird has happened ###')


def find_neighbors(element, vertexids_indices, adjacencyidx):
    """
    Finds neighboring elements
    :param element: Current element
    :param vertexids_indices: Indices of the mesh indices
    :param adjacencyidx: Built from spatialnde, index of element adjacency
    :return: An array of element numbers that neighbor the current element
    """
    firstidx = vertexids_indices[element]
    neighbors = np.array([element, adjacencyidx[firstidx]])
    indx = 1
    counter = 1
    while indx > 0:
        indx = adjacencyidx[firstidx + counter]
        if np.isscalar(indx):
            if indx != -1:
                neighbors = np.append(neighbors, indx)
            counter += 1
        else:
            print("### Edge encountered ###")
            break
    return neighbors


def calcdistance(unitvector, oldvertex, meshpoints):
    """
    Calculate perpendicular distance between a ray and a point
    :param unitvector: Reference vector to calculate distance from
    :param oldvertex: Start point for unitvector
    :param meshpoints: Test points
    :return: Perpendicular and parallel distance to each mesh point
    """
    perpvectors = -1*((oldvertex - meshpoints) - np.multiply(np.dot((oldvertex - meshpoints), unitvector[:, np.newaxis]), unitvector[np.newaxis, :]))
    paraldistance = np.dot(meshpoints - oldvertex, unitvector)
    return perpvectors, paraldistance


def calcclosestpoint(unitvector, oldpoint, meshpoints, normal):
    """
    Find closest mesh vertex defined by the distances calculated in calcdistance
    :param unitvector: Reference direction vector
    :param oldpoint: Start point for unitvector
    :param meshpoints: All test points
    :return: Closest point relative to unitvector
    """
    trimedmeshpoints = np.delete(meshpoints, np.where((meshpoints == oldpoint).all(axis=1)), axis=0)
    perpdistances, paraldistances = calcdistance(unitvector, oldpoint, trimedmeshpoints)
    point_idx = np.argmin(np.linalg.norm(perpdistances, axis=1))

    fpointu = paraldistances[point_idx]
    point_3d = trimedmeshpoints[point_idx]
    vector2pnt = perpdistances[point_idx]

    # perppoint = oldpoint + unitvector*fpointu

    # rel_uvw = np.vstack((oldpoint, perppoint, point_3d)).T
    # testval = np.sign(0.5*np.linalg.det(rel_uvw))

    testval = np.dot(calcunitvector(np.cross(unitvector, vector2pnt)), normal)

    # if element == 204:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.quiver(oldpoint[0], oldpoint[1], oldpoint[2], unitvector[0], unitvector[1], unitvector[2], color="y", length=0.1)
    #     ax.quiver(oldpoint[0], oldpoint[1], oldpoint[2], normal[0], normal[1], normal[2], color="r",
    #               length=0.1)
    #     ax.quiver(perppoint[0], perppoint[1], perppoint[2], vector2pnt[0], vector2pnt[1], vector2pnt[2], color="orange", length=0.1)
    #     ax.scatter(oldpoint[0], oldpoint[1], oldpoint[2], color="black")
    #     ax.scatter(point_3d[0], point_3d[1], point_3d[2], color="g")
    #     ax.scatter(perppoint[0], perppoint[1], perppoint[2], color="b")
    #     ax.scatter(meshpoints[:, 0], meshpoints[:, 1], meshpoints[:, 2], color="cyan")
    #     import sys
    #     sys.modules["__main__"].__dict__.update(globals())
    #     sys.modules["__main__"].__dict__.update(locals())
    #     pdb.set_trace()

    fpointv = testval * np.linalg.norm(vector2pnt)
    if np.isnan(fpointv):
        fpointv = 0.0
    fpoint = np.array([fpointu, fpointv])

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.quiver(oldpoint[0], oldpoint[1], oldpoint[2], unitvector[0], unitvector[1], unitvector[2], color="r", length=0.1)
    # ax.quiver(perppoint[0], perppoint[1], perppoint[2], vector2pnt[0], vector2pnt[1], vector2pnt[2], color="r", length=0.1)
    # ax.scatter(oldpoint[0], oldpoint[1], oldpoint[2], color="black")
    # ax.scatter(point_3d[0], point_3d[1], point_3d[2], color="g")
    # ax.scatter(perppoint[0], perppoint[1], perppoint[2], color="b")
    #
    # print(unitvector)
    # print(fpoint)
    # pdb.set_trace()
    return fpoint, point_3d


def calcbarycentric(point, element_vertices):
    """
    Convert 3d point to barycenteric coordinates
    https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    https://math.stackexchange.com/questions/2292895/walking-on-the-surface-of-a-triangular-mesh
    :param point: 3d point
    :param element_vertices: Vertices of current element
    :return: point in barycenteric coordinates
    """
    d1 = (element_vertices[0, 0] * (element_vertices[1, 1] - element_vertices[2, 1]) + element_vertices[1, 0] * (element_vertices[2, 1] - element_vertices[0, 1]) + element_vertices[2, 0] * (element_vertices[0, 1] - element_vertices[1, 1]))
    d2 = (element_vertices[0, 0] * (element_vertices[1, 2] - element_vertices[2, 2]) + element_vertices[1, 0] * (element_vertices[2, 2] - element_vertices[0, 2]) + element_vertices[2, 0] * (element_vertices[0, 2] - element_vertices[1, 2]))
    d3 = (element_vertices[0, 1] * (element_vertices[1, 2] - element_vertices[2, 2]) + element_vertices[1, 1] * (element_vertices[2, 2] - element_vertices[0, 2]) + element_vertices[2, 1] * (element_vertices[0, 2] - element_vertices[1, 2]))

    a1 = np.abs(d1)
    a2 = np.abs(d2)
    a3 = np.abs(d3)

    if a1 >= a2 and a1 >= a3:
        u = (point[0] * (element_vertices[2, 1] - element_vertices[0, 1]) + element_vertices[0, 0] * (
            point[1] - element_vertices[2, 1]) + element_vertices[2, 0] * (element_vertices[0, 1] - point[1])) / d1
        v = (point[0] * (element_vertices[0, 1] - element_vertices[1, 1]) + element_vertices[0, 0] * (
            element_vertices[1, 1] - point[1]) + element_vertices[1, 0] * (point[1] - element_vertices[0, 1])) / d1
    elif a2 >= a1 and a2 >= a3:
        u = (point[0] * (element_vertices[2, 2] - element_vertices[0, 2]) + element_vertices[0, 0] * (
            point[2] - element_vertices[2, 2]) + element_vertices[2, 0] * (element_vertices[0, 2] - point[2])) / d2
        v = (point[0] * (element_vertices[0, 2] - element_vertices[1, 2]) + element_vertices[0, 0] * (
            element_vertices[1, 2] - point[2]) + element_vertices[1, 0] * (point[2] - element_vertices[0, 2])) / d2
    else:
        u = (point[1] * (element_vertices[2, 2] - element_vertices[0, 2]) + element_vertices[0, 1] * (
            point[2] - element_vertices[2, 2]) + element_vertices[2, 1] * (element_vertices[0, 2] - point[2])) / d3
        v = (point[1] * (element_vertices[0, 2] - element_vertices[1, 2]) + element_vertices[0, 1] * (
            element_vertices[1, 2] - point[2]) + element_vertices[1, 1] * (point[2] - element_vertices[0, 2])) / d3
    return u, v


def invcalcbarycentric(pointuv, element_vertices):
    """
    Convert barycenteric coordinates into 3d
    https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    https://math.stackexchange.com/questions/2292895/walking-on-the-surface-of-a-triangular-mesh
    :param pointuv: Point in barycenteric coordinates (u, v)
    :param element_vertices: Vertices of current element
    :return: pointuv in 3d coordinates (x, y, z)
    """
    return element_vertices[0] + pointuv[0] * (element_vertices[1] - element_vertices[0]) + pointuv[1] * (element_vertices[2] - element_vertices[0])


def calcbarycentricdirection(vector, element_vertices):
    """
    Convert a direction vector from 3d to barycenteric coordinates
    https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    https://math.stackexchange.com/questions/2292895/walking-on-the-surface-of-a-triangular-mesh
    :param vector: Direction vector in 3d
    :param element_vertices: Vertices of current element
    :return: Vector in barycenteric coordinates (du, dv)
    """
    d1 = (element_vertices[0, 0] * (element_vertices[1, 1] - element_vertices[2, 1]) + element_vertices[1, 0] * (element_vertices[2, 1] - element_vertices[0, 1]) + element_vertices[2, 0] * (element_vertices[0, 1] - element_vertices[1, 1]))
    d2 = (element_vertices[0, 0] * (element_vertices[1, 2] - element_vertices[2, 2]) + element_vertices[1, 0] * (element_vertices[2, 2] - element_vertices[0, 2]) + element_vertices[2, 0] * (element_vertices[0, 2] - element_vertices[1, 2]))
    d3 = (element_vertices[0, 1] * (element_vertices[1, 2] - element_vertices[2, 2]) + element_vertices[1, 1] * (element_vertices[2, 2] - element_vertices[0, 2]) + element_vertices[2, 1] * (element_vertices[0, 2] - element_vertices[1, 2]))

    a1 = np.abs(d1)
    a2 = np.abs(d2)
    a3 = np.abs(d3)

    if a1 >= a2 and a1 >= a3:
        du = (vector[0] * (element_vertices[2, 1] - element_vertices[0, 1]) + vector[1] * (
        element_vertices[0, 0] - element_vertices[2, 0])) / d1
        dv = (vector[0] * (element_vertices[0, 1] - element_vertices[1, 1]) + vector[1] * (
        element_vertices[1, 0] - element_vertices[0, 0])) / d1
    elif a2 >= a1 and a2 >= a3:
        du = (vector[0] * (element_vertices[2, 2] - element_vertices[0, 2]) + vector[2] * (
        element_vertices[0, 0] - element_vertices[2, 0])) / d2
        dv = (vector[0] * (element_vertices[0, 2] - element_vertices[1, 2]) + vector[2] * (
        element_vertices[1, 0] - element_vertices[0, 0])) / d2
    else:
        du = (vector[1] * (element_vertices[2, 2] - element_vertices[0, 2]) + vector[2] * (
        element_vertices[0, 1] - element_vertices[2, 1])) / d3
        dv = (vector[1] * (element_vertices[0, 2] - element_vertices[1, 2]) + vector[2] * (
        element_vertices[1, 1] - element_vertices[0, 1])) / d3
    return du, dv


def invcalcbarycentricdirection(vectoruv, element_vertices):
    """
    Convert vector in barycenteric coordinates into a 3d vector
    https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    https://math.stackexchange.com/questions/2292895/walking-on-the-surface-of-a-triangular-mesh
    :param vectoruv: Vector in barycenteric coordinate (du, dv)
    :param element_vertices: Vertices of current element
    :return: Vectoruv in 3d space (dx, dy, dz)
    """
    # element_vertices = vertices[vertexids[element]]
    return vectoruv[0] * (element_vertices[1] - element_vertices[0]) + vectoruv[1] * (element_vertices[2] - element_vertices[0])


def check_inplane_pnt(point, element_vertices):
    """
    Determines if a point is within the plane of the current element face
    :param point: A point within or on the edge of the current element
    :param element_vertices: Vertices of current element
    :return: True if the point is within in the plane, or False if otherwise
    """
    detarray = np.vstack((element_vertices, point)).T
    detarray = np.vstack((detarray, np.ones(detarray.shape[1])))
    det = np.linalg.det(detarray)
    err = 1e-9
    if det == 0 or -err < det < err:
        return True
    else:
        return False


def check_inplane_vector(vector, normal):
    """
    Determines if a vector is in plane with the current element
    :param vector: Test vector
    :param normal: Normal of element
    :return:
    """
    checkdot = np.dot(normal, vector)
    err = 1e-9
    if -err < checkdot < err:
        return True
    else:
        return False


def find_element_vertex(vertex, unitvector, curnormal, vertices, vertexids, facetnormals):
    """
    Determines which element is next given a vertex and an angle
    :param af: AutoFiber class variable
    :param vertex: Vertex in the mesh
    :param unitvector: Fiber direction vector
    :param curnormal: Current element normal direction vector
    :param vertices: Mesh vertices
    :param vertexids: Id's of mesh element vertices
    :param facetnormals: Normals of each element in mesh
    :return: The element in which the fiber direction vector resides
    """
    newvector = None
    element = np.empty((0, 3), dtype=int)
    # Find neighboring elements:
    vertexid = np.where(
        np.linalg.norm(vertices - vertex, axis=1) == np.min(np.linalg.norm(vertices - vertex, axis=1)))
    neighbors = np.where(vertexids == vertexid)[0]

    # For each neighbor and the given angle determine which will include the next point
    nvectors = vertices[vertexids[neighbors]] - vertex
    for i in neighbors:
        # Calculate fiber direction at current point
        try:
            newvector = rot_vector(curnormal, facetnormals[i], unitvector)
        except EdgeError:
            continue
        if check_inplane_vector(newvector, facetnormals[i]):
            where_ind = np.where((nvectors[np.where((neighbors == i))][0] == 0))[0]
            unique, counts = np.unique(where_ind, return_counts=True)
            evectors = np.delete(nvectors[np.where((neighbors == i))], np.where((counts == 3)), axis=1)[0]
            if vector_inbetween(newvector, evectors[0], evectors[1]):
                element = np.append(element, i)
            else:
                element = np.append(element, i)
    if element.shape[0] > 1:
        print("### On an edge, multiple potential elements ###")
        normal_angles = np.array([])
        for i in range(0, element.shape[0]):
            angle = angle_between_vectors(curnormal, facetnormals[element[i]])
            if angle > np.pi:
                angle = 2 * np.pi - angle
            normal_angles = np.append(normal_angles, angle)
        element = int(element[np.argmin(normal_angles)])
    elif element.shape[0] == 0:
        print("### No elements found ###")
        # pdb.set_trace()
        element = None
    else:
        element = int(element[0])
    if element is not None:
        newvector = rot_vector(curnormal, facetnormals[element], unitvector)
    return element, newvector


def find_element_within(point, unitvector, normal, vertices, vertexids, facetnormals, inplanemat):
    """
    Determines which element a point is within
    :param af: AutoFiber class variable
    :param point: Vertex in the mesh
    :param unitvector: Fiber direction vector
    :param normal: Current element normal direction vector
    :return: The element that the point is within
    """
    vertexid = np.where(np.linalg.norm(vertices - point, axis=1) == np.min(np.linalg.norm(vertices - point, axis=1)))
    neighbors = np.where(vertexids == vertexid)[0]

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vertsn = vertices[vertexids[neighbors]]
    ax.scatter(vertsn.reshape(vertsn.shape[0]*vertsn.shape[1], vertsn.shape[2])[:, 0], vertsn.reshape(vertsn.shape[0]*vertsn.shape[1], vertsn.shape[2])[:, 1], vertsn.reshape(vertsn.shape[0]*vertsn.shape[1], vertsn.shape[2])[:, 2], alpha=0.1)
    ax.scatter(point[0], point[1], point[2])

    for i in neighbors:
        if check_inplane_pnt(point, vertices[vertexids[i, :]]):
            print("point %s in plane with element %s" % (point, i))
            if geometry.point_in_polygon_3d(vertices[vertexids[i]], point, inplanemat[i]):
                print("point in element")
                if check_inplane_vector(unitvector, facetnormals[i]):
                    print("vector in plane")
                    return i, unitvector
                else:
                    print("vector not in plane...adjusting")
                    try:
                        newvector = rot_vector(normal, facetnormals[i], unitvector)
                    except EdgeError:
                        continue
                    if check_inplane_vector(newvector, facetnormals[i]):
                        print("vector adjusted")
                        return i, newvector
        else:
            print("point not in plane...attempting to adjust")
            test, projpnt = check_proj_inplane_pnt(point, vertices[vertexids[i]])
            if test:
                print("point in plane with element %s" % i)
                ax.scatter(vertices[vertexids[i]][:, 0], vertices[vertexids[i]][:, 1], vertices[vertexids[i]][:, 2])
                ax.scatter(projpnt[0], projpnt[1], projpnt[2])
                if geometry.point_in_polygon_3d(vertices[vertexids[i]], projpnt, inplanemat[i]):
                    print("point in element")
                    if check_inplane_vector(unitvector, facetnormals[i]):
                        print("vector in plane")
                        return i, unitvector
                    else:
                        print("vector not in plane...adjusting")
                        try:
                            newvector = rot_vector(normal, facetnormals[i], unitvector)
                        except EdgeError:
                            continue
                        if check_inplane_vector(newvector, facetnormals[i]):
                            print("vector adjusted")
                            return i, newvector
                        else:
                            ax.quiver(projpnt[0], projpnt[1], projpnt[2], newvector[0], newvector[1], newvector[2])
                            ax.quiver(projpnt[0], projpnt[1], projpnt[2], unitvector[0], unitvector[1], unitvector[2])
                            import pdb
                            pdb.set_trace()
    return None, None


def traverse_element(af, element, point, unitfiberdirection, fiberpoints_local, length, uv_start, direction=1, parameterization=True):
    # Determine the elements surrounding the current element
    neighbors = find_neighbors(element, af.vertexids_indices, af.adjacencyidx)

    # Retrieve current element vertices
    element_vertices = af.vertices[af.vertexids[element]]

    # Calculate the barycentric coordinates for each vertex of the current element
    element_vertices_bary = np.zeros((element_vertices.shape[0], 2))
    for i in range(0, element_vertices.shape[0]):
        element_vertices_bary[i, :] = calcbarycentric(element_vertices[i], element_vertices)

    # Calculate the barycentric coordinates for the current 3d point
    pointuv = np.array(calcbarycentric(point, element_vertices))

    # Calculate the barycentric direction for the current 3d fiber direction vector
    duv = np.array(calcbarycentricdirection(unitfiberdirection, element_vertices))

    # Calculate another point in the direction of the fiber in order to calculate intersection point later
    lnpoint = pointuv + duv * 1

    # Signed distance to each edge ([d12, d23, d31])
    edges_dict = np.array([[0, 1], [1, 2], [2, 0]])
    # Determine which edge will be intersected first
    edge_num = find_edge(pointuv, duv, 0.0000000005)
    # Retrieve the corresponding vertex indices to the intersected edge
    edge = edges_dict[edge_num]
    nextedge = element_vertices_bary[edge]

    # Find the point of intersection between the edge and a line in the fiber direction
    int_pnt = find_intpnt(pointuv, lnpoint, nextedge[0], nextedge[1])

    if parameterization:
        test_vertices = np.copy(element_vertices)
        for p in range(0, element_vertices.shape[0]):
            fpoint, closest_point = calcclosestpoint(unitfiberdirection, point, test_vertices, af.facetnormals[element])
            closest_point_idx = np.where((af.vertices == closest_point).all(axis=1))[0][0]

            prev_lines = af.georecord.get(element, [[], None])[0]
            for line in prev_lines:
                if check_intersection(pointuv, int_pnt, line[0], line[1]):
                    raise EdgeError

            if np.isnan(fiberpoints_local[closest_point_idx]).all() or np.abs(fpoint[1]) < np.abs(fiberpoints_local[closest_point_idx][1]):
                fpoint_t = np.array([direction*(length + fpoint[0] + uv_start[0]), fpoint[1] + uv_start[1]])

                if ~np.isnan(fpoint_t).all():
                    fiberrec = np.copy(af.geoparameterization)
                    fiberrec[closest_point_idx] = fpoint_t

                    rel_uvw = np.pad(fiberrec[af.vertexids], [(0, 0), (0, 0), (0, 1)], "constant", constant_values=1)
                    vdir = 0.5 * np.linalg.det(rel_uvw)
                    if (np.sign(vdir) < 0).any():
                        test_vertices = np.delete(test_vertices, np.where((test_vertices == closest_point).all(axis=1))[0][0], axis=0)
                        pass
                    else:
                        if parameterization and element not in list(af.georecord.keys()):
                            af.georecord[element] = [[], None]

                        af.georecord[element][0].append(
                            (pointuv, int_pnt, point, unitfiberdirection, closest_point_idx, uv_start, length))
                        fiberpoints_local[closest_point_idx] = fpoint
                        # For every iteration that isn't the first add the last fiberpoint.u and the u value of the very first point
                        af.geoparameterization[closest_point_idx] = fpoint_t
                        af.fiberdirections[element] = unitfiberdirection
                        del fiberrec
                        break
                    del fiberrec

    # Retrieve the 3d coordinates of the edge vertices
    nextedgec = af.vertices[af.vertexids[element, edge]]
    # Remove the current element from the neighbors array
    neighbors = np.delete(neighbors, np.where((neighbors == element)))
    # Redefine the all_vertices variable to not include the current element
    all_vertices = af.vertices[af.vertexids[neighbors]]
    # Find which element is across the intersected edge
    nextelement = None
    for i in range(0, all_vertices.shape[0]):
        # Check for the first vertex of the edge
        test = np.linalg.norm(all_vertices[i] - nextedgec[0], axis=1)
        if 0 in test:
            # If the first vertex is in this element check for the second vertex of the edge
            test2 = np.linalg.norm(all_vertices[i] - nextedgec[1], axis=1)
            if 0 in test2:
                # If both vertices are in this element set nextelement to the current element
                nextelement = neighbors[i]
                break
    # Convert the intersection point into 3d coordinates
    int_pnt_3d = invcalcbarycentric(int_pnt, element_vertices)

    if nextelement is None:
        return int_pnt_3d, None, nextelement, fiberpoints_local

    # Rotate current 3d fiber vector to match the plane of the next element
    nextunitvector = rot_vector(af.facetnormals[element], af.facetnormals[nextelement], unitfiberdirection)

    if 0 in np.linalg.norm(af.vertices - int_pnt_3d, axis=1):
        print("### Encountered a vertex ###")
    return int_pnt_3d, nextunitvector, nextelement, fiberpoints_local
