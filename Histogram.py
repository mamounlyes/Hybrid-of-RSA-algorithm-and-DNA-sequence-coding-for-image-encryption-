
import tkinter as tk
from tkinter import filedialog
import dippykit as dip
import matplotlib.pyplot as plt
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
image = dip.im_read(my_im)

X = dip.float_to_im(image)

# Load the image
#image = cv2.imread(my_im)

# Extract the red, green, and blue channels
red = X[:,:,0]
green = X[:,:,1]
blue = X[:,:,2]

b_flat = blue.flatten()
g_flat = green.flatten()
r_flat = red.flatten()

plt.subplot(3, 1, 1)

# Plot the histograms
#plt.title('RGB Histogram')

#plt.hist(red.flatten(), bins=256, color='red', alpha=0.5)
#plt.hist(green.flatten(), bins=256, color='green', alpha=0.5)
#plt.hist(blue.flatten(), bins=256, color='blue', alpha=0.5)
plt.hist(b_flat, bins=256, range=(0, 256), color='b')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')
plt.subplot(3, 1, 2)
plt.hist(g_flat, bins=256, range=(0, 256), color='g')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')
plt.subplot(3, 1, 3)
plt.hist(r_flat, bins=256, range=(0, 256), color='r')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')


plt.savefig('encrypted DNA.jpg')
x=plt.show()


# Saving the image
