import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.restoration import denoise_bilateral


folder_path = "X:/volcano_cameras/Shared/Lascar/2022/2022-07-23"
video_save_path = "C:/Users/ggp24ash/Documents/Scratch Data/Optical Flow Outputs/21 - Videos of Timestep Difference/Lascar_2022-07-23.mp4"
fltr = "A"
mod=1
max_frames = 100
difference_threshold = 5
fps=5

def show(image):
    plt.imshow(image, cmap="gray")
    plt.colorbar()
    plt.show()

original_sequence = []
filtered_sequence = []
#Read a sequence of bandA images

index = 1
for image_name in os.listdir(folder_path):
    if ".png" in image_name:
        if "fltr" + fltr in image_name:
            if index % mod == 0:
                if index <= max_frames:
                    print(image_name)
                    image = cv2.imread(folder_path + "/" + image_name, -1)
                    print("Reading image: " + str(index))
                    # Apply smoothing
                    original_sequence.append(image)
                    image = denoise_bilateral(image.astype("float32"), sigma_color=5, sigma_spatial=10, win_size=20)
                    filtered_sequence.append(image)

            index +=1



#Take the difference between timesteps
filtered_sequence = np.array(filtered_sequence)
print(filtered_sequence.shape)

diff = np.abs(filtered_sequence[1:,:,:].astype("float32") - filtered_sequence[:-1, :, :].astype("float32"))
for vis_index in range(1, diff.shape[0], 2):
    show(diff[vis_index])

diff = np.where(diff > difference_threshold, 1, 0)

def normalise(image):
    image = image.astype("float32") - np.ones_like(image) * np.min(image)
    scale_factor = np.max(image) / 255
    image = np.divide(image, scale_factor)
    return image.astype("uint8")
def create_video(left, right, video_save_path):
    index = 0
    output = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1296,486))

    for index in range(0, right.shape[0]):
        print("Writing index " + str(index))

        whole_frame = np.concatenate([normalise(left[index]), normalise(right[index])], axis=1)

        rgb_frame = cv2.cvtColor(whole_frame, cv2.COLOR_GRAY2BGR)
        #print(rgb_both_bands.shape)
        #max_val = int(rgb_both_bands.max())

        #both_bands_rgb = cv2.rectangle(rgb_both_bands, (0,10), (560,40), (0,0,0), -1)
        #both_bands_rgb = cv2.putText(both_bands_rgb, image_name, (20,30), cv2.FONT_HERSHEY_SIMPLEX,
        #fontScale=0.5, color=(255,255,255), thickness=1)

        ##to_write.append(both_bands)
        output.write(rgb_frame)
        #show(rgb_frame)
        index += 1
    output.release()

create_video(original_sequence[1:], diff, video_save_path)