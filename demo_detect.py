from paddleocr import PaddleOCR
import cv2
from align import align
# Also switch the language by modifying the lang parameter
ocr = PaddleOCR(lang="en") # The model file will be downloaded automatically when executed for the first time
img_path ='/media/sang/UBUNTU/ANHHAI-VIENNGHIENCUU/KQ sieu am/KQ sieu am/IMG_4337.JPG'
result = ocr.ocr(img_path,rec=False)
# Recognition and detection can be performed separately through parameter control
# result = ocr.ocr(img_path, det=False)  Only perform recognition
# result = ocr.ocr(img_path, rec=False)  Only perform detection
# Print detection frame and recognition result
for line in result:
    print(line)

# Visualization
mat = cv2.imread(img_path)

boxes = [line for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]

# for box in boxes:    
#     top_left     = (int(box[0][0]), int(box[0][1]))
#     bottom_right = (int(box[2][0]), int(box[2][1]))

#     cv2.rectangle(mat, top_left, bottom_right, (0, 255, 0), 2)
# mat=cv2.resize(mat,(1500,900))
for box in boxes:
    me = align(mat,box)
    cv2.imshow("result", me)
    cv2.waitKey(0)
