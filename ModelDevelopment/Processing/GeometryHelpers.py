import numpy as np

'''Some larger functions used to calculate intersections of pixels with geometric shapes.'''

def calculate_circle_points(image_dim, center, radius):
    '''Identify which pixels intersect with the defined circle.
    center = (down, across).'''

    lhs = max(0, center[1] - radius)
    rhs = min(image_dim[1] - 1, center[1] + radius)
    x_range = (lhs, rhs)
    top = max(0, center[0] - radius)
    bottom = min(image_dim[0] - 1, center[0] + radius)
    y_range = (top, bottom)

    #For each x-value, find and plot corresponding y-values:
    x_to_consider = np.linspace(x_range[0], x_range[1], (x_range[1] - x_range[0] + 1) * 10)

    y_sq = np.ones_like(x_to_consider) * radius**2 - np.square(x_to_consider- np.ones_like(x_to_consider) * center[1])
    y_rt = np.sqrt(y_sq)

    y1_vals = y_rt + np.ones_like(x_to_consider) * center[0]
    y1_vals = np.round(y1_vals, 0).astype(int)

    y2_vals = -y_rt + np.ones_like(x_to_consider) * center[0]
    y2_vals = np.round(y2_vals, 0).astype(int)

    array = np.zeros(shape=image_dim)
    x_to_plot = np.round(x_to_consider, 0).astype(int)

    for x_index in range(0, len(x_to_consider)):
        array[y1_vals[x_index], x_to_plot[x_index]] = 1
        array[y2_vals[x_index], x_to_plot[x_index]] = 1

    #Then for each y-value, find and plot corresponding x-vals
    y_to_consider = np.linspace(y_range[0], y_range[1], (y_range[1] - y_range[0] + 1) * 10)

    x_sq = np.ones_like(y_to_consider) * radius ** 2 - np.square(y_to_consider - np.ones_like(y_to_consider) * center[0])
    x_rt = np.sqrt(x_sq)

    x1_vals = x_rt + np.ones_like(y_to_consider) * center[1]
    x1_vals = np.round(x1_vals, 0).astype(int)

    x2_vals = -x_rt + np.ones_like(y_to_consider) * center[1]
    x2_vals = np.round(x2_vals, 0).astype(int)

    y_to_plot = np.round(y_to_consider, 0).astype(int)

    for y_index in range(0, len(y_to_consider)):
        array[y_to_plot[y_index], x1_vals[y_index]] = 1
        array[y_to_plot[y_index], x2_vals[y_index]] = 1

    return array

def dist(p1, p2):
    sum = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    return np.sqrt(sum)

def find_POI_vertical_line(x, c, r, point_approx):
    '''
    Find the POI between line x=x and circle with given center (down, across) and radius.
    Return the value closest to the given y-value.
    :return:
    '''

    y_sqr = r**2 - (x-c[1])**2
    if y_sqr >= 0:
        #If a solution exists:
        y_rt = np.sqrt(y_sqr)
        y1 = y_rt + c[0]
        y2 = -y_rt + c[0]
        #Take the one closest to the y-val of given.
        if (point_approx[1] - 0.5) <= y1 < (point_approx[1] + 0.5):
            return y1
        elif (point_approx[1]-0.5) <= y2 < (point_approx[1]+0.5):
            return y2
        else:
            return "N"
    else:
        return "N"

def find_POI_horizontal_line(y, c, r, point_approx):
    '''
        Find the POI between line y=y and circle with given center (down, across) and radius.
        Return the value closest to the given x-value.
        :return:
        '''

    x_sqr = r ** 2 - (y - c[0]) ** 2
    if x_sqr >= 0:
        # If a solution exists:
        x_rt = np.sqrt(x_sqr)
        x1 = x_rt + c[1]
        x2 = - x_rt + c[1]
        # Take the one closest to the y-val of given.
        if (point_approx[0] - 0.5) <= x1 < (point_approx[0] + 0.5):
            return x1
        elif (point_approx[0] - 0.5) <= x2 < (point_approx[0] + 0.5):
            return x2
        else:
            return "N"
    else:
        return "N"



def calculate_intersection_points(all_points, c, r):
    '''
    Intakes a set of points and returns the length of the "chunk" of the circle
    which intersects the pixel centered at each point, approximated by a straight line.
    :param all_points: Array containing pixels for which to calculate the intersection.
    :param circle_center: (y, x) as in (vertical, horizontal)
    :param circle_radius: radius of circle to consider.
    :return:
    '''
    n_poi = np.zeros_like(all_points)
    arc_lengths = np.zeros_like(all_points).astype(np.float32)
    intersection_coords = np.zeros(shape=(all_points.shape[0], all_points.shape[1], 2, 2)) #Shape [pixel_y_coord, pixel_x_coord, POI 1 or 2, POI coord]
    for x in range(0, all_points.shape[1]):
        for y in range(0, all_points.shape[0]):
            POIs = []

            # Find POI with each edge of that pixel
            x_l = x - 0.5
            x_r = x + 0.5
            y_t = y - 0.5
            y_b = y + 0.5

            #Find POIs between vertical boundary and the circle:
            x_l_POI_y_val = find_POI_vertical_line(x_l, c, r, (x, y))
            if x_l_POI_y_val != "N":
                POIs.append((x_l, x_l_POI_y_val))
            x_r_POI_y_val = find_POI_vertical_line(x_r, c, r, (x, y))
            if x_r_POI_y_val != "N":
                POIs.append((x_r, x_r_POI_y_val))
            #Find POIs between horizontal boundary and the circle:
            y_t_POI_x_val = find_POI_horizontal_line(y_t, c, r, (x, y))
            if y_t_POI_x_val != "N":
                POIs.append((y_t_POI_x_val, y_t))
            y_b_POI_x_val = find_POI_horizontal_line(y_b, c, r, (x, y))
            if y_b_POI_x_val != "N":
                POIs.append((y_b_POI_x_val, y_b))


            #print(len(POIs))
            n_poi[y, x] = len(POIs)

            # For each pixel with intersections
            # Calculate the distance of the arc approximated by a line
            if n_poi[y, x] == 2:
                arc_len = dist(POIs[0], POIs[1])
                arc_lengths[y, x] = arc_len

                intersection_coords[y, x, 0, :] = POIs[0]
                intersection_coords[y, x, 1, :] = POIs[1]


    #Check that each pixel has two POIs with the circle:
    #plt.imshow(boundary_points * 2 - n_poi)
    #plt.colorbar()
    #plt.show()
    #Note there is a slight mismatch where some pixels flagged as having intersection
    #aren't recorded as boundary points. This is because the method used to calculate
    #intersections accounts for even miniscule intersections, whereas the step size used
    #to find points on the circle is not infinitesimally small. So take the intersection
    #calculation as the correct standard.

    #TODO sense check that the total distance is roughly the circle circumference
    #plt.imshow(arc_lengths)
    #plt.colorbar()
    #plt.show()

    circ = np.pi * 2 * r
    print("Error in arc length approximation (in pixels):")
    print(str(np.round(np.sum(arc_lengths) - circ, 4)))

    return intersection_coords