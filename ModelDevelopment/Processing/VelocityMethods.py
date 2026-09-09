import matplotlib.pyplot as plt
import numpy as np
def plot_dense_flow(flow, image, n, color_array=None, world_coords=False, mask=None):
    # world_coords param: This function assumes flow is in pixel coordinates (if flow is in
    # real world coordinates (where a smaller y value indicates smaller height)
    # then the y-comonents need to be multipled by -1 for correct visualisation.
    if isinstance(mask, np.ndarray):
        flow_mask = np.empty_like(flow)
        flow_mask[:,:,0] = mask
        flow_mask[:,:,1] = mask
        flow = np.ma.masked_where(flow_mask, flow)

    # Downsample every nth flow vector for plotting
    flow_dx = flow[0::n, 0::n, 0]
    flow_dy = flow[0::n, 0::n, 1]
    if isinstance(mask, np.ndarray):
        flow_dx = np.ma.masked_where(flow.mask[0::n, 0::n, 0], flow_dx)
        flow_dy = np.ma.masked_where(flow.mask[0::n, 0::n, 1], flow_dy)

    if world_coords == True:
        flow_dy = -1 * flow_dy

    # Need array of x-coords, y-coords, then dx and dy
    # Downsample to plot every n-th coord point
    x_pixels = np.arange(0, image.shape[1], n)
    y_pixels = np.arange(0, image.shape[0], n)
    X, Y = np.meshgrid(x_pixels, y_pixels)

    if isinstance(mask, np.ndarray):
        downsampled_mask = flow_dx.mask
        X = np.ma.masked_where(flow_dx.mask, X)
        Y = np.ma.masked_where(flow_dx.mask, Y)
        X = X.compressed()
        Y = Y.compressed()
        flow_dx = flow_dx.compressed()
        flow_dy = flow_dy.compressed()


    if isinstance(color_array, np.ndarray):
        color_array = color_array[0::n, 0::n]
        if isinstance(mask, np.ndarray):
            color_array = np.ma.masked_where(downsampled_mask, color_array)
            color_array = color_array.compressed()
        plt.quiver(X, Y, flow_dx, flow_dy, [color_array], scale_units='xy', scale=1, angles='xy')
        plt.colorbar()
    else:
        plt.quiver(X, Y, flow_dx, flow_dy, color='g', scale_units='xy', scale=1, angles='xy')
    plt.gca().invert_yaxis()
    plt.imshow(image, cmap="gray")
    plt.colorbar()
    # if save_loc != None:
    #    plt.savefig(save_loc)
    plt.show()
    # plt.close()
def ones(frame1, frame2):
    return np.ones(shape=(frame1.shape[0], frame1.shape[1], 2))