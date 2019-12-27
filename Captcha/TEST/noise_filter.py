import os
import matplotlib.pyplot as plt
import cv2
from Captcha.TEST.Conservative_Filter import conservative_smoothing_gray

"""
    Start by opening the dataset 
    From that we get a test image and use it's complementary
    The complementary is used to match what we have seen so far in the data
"""
path = os.getcwd()
path = path.replace("\TEST","")
print(path)

path = path + "/ADVERSARIAL_DATABASE_V2/"

listing = os.listdir(path)
test_image = cv2.imread(path + listing[10])
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

plt.imshow(test_image, cmap='gray')
plt.show()

print(test_image.shape)

# new_image = cv2.Laplacian(test_image, cv2.CV_64F)
# new_image = conservative_smoothing_gray(test_image, 4)
# new_image = cv2.blur(test_image,(3,3))
# new_image = cv2.GaussianBlur(test_image, (3,3), 0)
new_image = cv2.medianBlur(test_image, 3)

print(new_image.shape)

plt.imshow(new_image, cmap='gray')
plt.show()