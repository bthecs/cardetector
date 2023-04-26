import cv2
import test

lpr = test.LPR()



img = cv2.imread(f"auto3.png")
txt = lpr.read_license(img)
print(txt)
