
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

img = cv2.imread("Data/Reventador_2024-10-03T162045_fltrA_1ag_699997ss_Plume.png")
mask = np.load("Data/PlumeAndExpPixels_Reventador_2024-10-03T162045_fltrA_1ag_699997ss_Plume.npy")
sensor_marks_mask = cv2.imread("Data/Reventador_2024A.png", -1)
img = cv2.inpaint(img, sensor_marks_mask, 5, cv2.INPAINT_TELEA)
img = img[:,:,0]
flank_mask = cv2.imread("Data/ReventadorFlankMask.png", -1)

mask = mask[0,:,:] + mask[1,:,:]
#Following tutorial: https://enjoymachinelearning.com/blog/multivariate-polynomial-regression-python/#:~:text=Multivariate%20polynomial%20regression%20is%20used%20to
def polyfit(image, degree):
    '''Takes in an image as a 2D numpy array and returns a multivariate polynomial regression
    fit of degree specified.'''

    # horizontal pixel coordinates
    x_range = np.arange(start=0, stop=image.shape[1], step=1)
    x_coords = np.tile(x_range, reps=image.shape[0])

    # vertical pixel coordinates
    y_coords = np.zeros(shape=(image.shape[1]))
    for y_coord in range(1, image.shape[0]):
        y_coords_to_add = np.ones(shape=(image.shape[1])) * y_coord
        y_coords = np.concatenate((y_coords, y_coords_to_add))

    # pixel_values
    pixel_values = image.flatten()
    flank_mask_linear = flank_mask.flatten()

    linear_mask = np.where(mask>0, 1, 0).flatten()
    x_masked = np.ma.masked_where(np.logical_or(linear_mask>0, flank_mask_linear==0), x_coords).compressed()
    y_masked = np.ma.masked_where(np.logical_or(linear_mask>0, flank_mask_linear==0), y_coords).compressed()
    pixel_vals_masked = np.ma.masked_where(np.logical_or(linear_mask>0, flank_mask_linear==0), pixel_values).compressed()

    #Now store these in a dataframe:
    data_list = {'x_coords': x_coords,
            'y_coords': y_coords,
            'pixel_vals': pixel_values}

    masked_data_list = {'x_coords': x_masked,
            'y_coords': y_masked,
            'pixel_vals': pixel_vals_masked}

    # Create DataFrame
    image_data = pd.DataFrame(data_list)
    masked_image_data = pd.DataFrame(masked_data_list)

    poly_model = PolynomialFeatures(degree=degree)
    #poly_model_masked = PolynomialFeatures(degree=degree)
    poly_indep_vars_masked = poly_model.fit_transform(masked_image_data[['x_coords', 'y_coords']])
    poly_indep_vars = poly_model.transform(image_data[['x_coords', 'y_coords']])
    regression = LinearRegression()

    regression.fit(poly_indep_vars_masked, masked_image_data[['pixel_vals']])

    y_pred = regression.predict(poly_indep_vars)

    #print(regression.coef_)

    #rmse = mean_squared_error(image_data[['pixel_vals']], y_pred, squared=False)
    #print(rmse)

    background_est = y_pred.reshape((486,648)).astype('uint16')
    return background_est

bg_est = polyfit(img, 4)
bg_img = np.where(flank_mask==0, img, bg_est)

plt.imshow(img, cmap="gray")
plt.show()