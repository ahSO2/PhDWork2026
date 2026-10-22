import matplotlib.pyplot as plt
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

