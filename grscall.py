import cv2

path = 'users/farhad/sample.jpg'

# Load color image (BGR) and convert to gray
img = cv2.imread(path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load in grayscale mode
img_gray_mode = cv2.imread(path, 0)

# diff = img_gray_mode - img_gray
diff = cv2.bitwise_xor(img_gray,img_gray_mode)

cv2.imshow('diff', diff)
cv2.waitKey()