from paddleocr import PaddleOCR
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from align import align
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
# load model ocr
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = './vietocr/weights/transformerocr.pth'
# config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['cnn']['pretrained']=False
config['device'] = 'cuda:0'
config['predictor']['beamsearch']=False
detector = Predictor(config)
# Also switch the language by modifying the lang parameter
ocr = PaddleOCR(lang="en") # The model file will be downloaded automatically when executed for the first time
img_path ='/media/sang/UBUNTU/ANHHAI-VIENNGHIENCUU/KQ sieu am/KQ sieu am/IMG_4337.JPG'
result = ocr.ocr(img_path,rec=False)
# Visualization
mat = cv2.imread(img_path)

boxes = [line for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]

for box in boxes:    
    crop_img = align(mat,box)
    convert_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(convert_img)
    s = detector.predict(im_pil)
    print(s)
    cv2.imshow("image",crop_img)
    cv2.waitKey(0)
