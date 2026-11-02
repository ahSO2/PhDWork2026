import matplotlib.pyplot as plt
import numpy as np
def show(image, colormap="gray", title="None"):
    fig, axs = plt.subplots()
    plot = axs.imshow(image, cmap=colormap)
    if "None" in title:
        pass
    else:
        fig.suptitle(title)
    fig.colorbar(plot, ax=axs)
    axs.set_axis_off()
    plt.show()

def show_target_mask(image, mask_to_draw, mask_to_exclude):
    image_copy = np.ma.masked_where(mask_to_exclude > 0, image)
    masked_area = np.ones_like(image_copy)
    masked_area = np.ma.masked_where(mask_to_draw == 0, masked_area)
    fig, axs = plt.subplots()
    axs.imshow(image_copy, cmap="gray")
    axs.imshow(masked_area, cmap="Wistia_r")
    plt.show()

