import numpy as np
import tkinter as tk
from tkinter import filedialog
import cv2
from scipy.stats import pearsonr
from skimage.color import rgb2gray

def image_selector():  # returns path to selected image
    path = "NULL"
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename()
    if path != "NULL":
        print("Image loaded!")
    else:
        print("Error Image not loaded!")
    return path
my_im = image_selector()
image = cv2.imread(my_im)

image_gray = rgb2gray(image)
# Convert the image to a numpy array
image_array = np.array(image)
# Get the shape of the image
height, width, channels = image_array.shape


def correlation_coefficient(image):
    # Convert the image to a 2D array
    image_array = np.array(image)

    # Calculate the horizontal correlation coefficient
    horiz_corr, _ = pearsonr(image_array[:, :-1].flatten(), image_array[:, 1:].flatten())

    # Calculate the vertical correlation coefficient
    vert_corr, _ = pearsonr(image_array[:-1, :].flatten(), image_array[1:, :].flatten())

    # Calculate the diagonal correlation coefficient
    diag_corr, _ = pearsonr(image_array[:-1, :-1].flatten(), image_array[1:, 1:].flatten())

    return (vert_corr,horiz_corr,  diag_corr)

a,b,c=correlation_coefficient(image_gray)
print(a,b,c)