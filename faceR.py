import cv2 as cv
import numpy as np

def face_detect_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #以下分别是HAAR和LBP特征数据，任意选择一种即可，注意：路径中的‘/’和‘\’是有要求的
    # 通过级联检测器 cv.CascadeClassifier，加载特征数据
    face_detector = cv.CascadeClassifier("./lbpcascades/lbpcascade_frontalcatface.xml")
    faces = face_detector.detectMultiScale(gray, 1.02, 5)
    for x, y, w, h in faces:
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv.imshow("result", image)


capture = cv.VideoCapture(0)
cv.namedWindow("result", cv.WINDOW_AUTOSIZE)
while (True):
    ret, frame = capture.read()
    # cv.flip函数表示图像翻转，沿y轴翻转, 0: 沿x轴翻转, <0: x、y轴同时翻转
    frame = cv.flip(frame, 1)
    face_detect_demo(frame)
    c = cv.waitKey()
    if c == 27:#当键盘按下‘ESC’退出程序
        break

#cv.waitKey(0)参数为0，如cv2.waitKey(0)只显示当前帧图像，相当于视频暂停,；
cv.waitKey(0)
cv.destroyAllWindows()#作用是能正常关闭绘图窗口
