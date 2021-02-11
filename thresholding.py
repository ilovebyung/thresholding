import cv2
import numpy as np
import matplotlib.pyplot as plt

# read image in black and white
image = cv2.imread('image.jpg', 0)
plt.imshow(image, cmap='gray')

# 1.threshold
# if pixel is > 80, pixel value = 255 else pixel value 0
(T, threshold) = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
thresh_with_blur = cv2.medianBlur(threshold, 15, 0)
plt.imshow(thresh_with_blur, cmap='gray')

# extract forground
mask = cv2.bitwise_and(image, image, mask=thresh_with_blur)
plt.imshow(mask, cmap='gray')

# concatanate image Horizontally
Horizontal = np.concatenate((image, thresh_with_blur, mask), axis=1)
plt.imshow(Horizontal, cmap='gray')

# read video
cap = cv2.VideoCapture('video.mp4')
while True:
    ret, frame = cap.read()
    # hue saturation value
    video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # cv2.imshow('frame', frame)
    # cv2.imshow('mask', mask)
    cv2.imshow('result', result)

    key = cv2.waitKey(1)
    if key == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# convert back to color
RGB = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
plt.imshow(RGB)
