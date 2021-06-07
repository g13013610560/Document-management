import cv2
import dlib

path = "./fish.jpg"
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#人脸检测画框
detector = dlib.get_frontal_face_detector()
# 获取人脸关键点检测器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#获取人脸框位置信息
dets = detector(gray, 1)#1表示采样（upsample）次数  0识别的人脸少点,1识别的多点,2识别的更多,小脸也可以识别
for face in dets:
    shape = predictor(img, face)  # 寻找人脸的68个标定点
    # 遍历所有点，打印出其坐标，并圈出来
    for pt in shape.parts():
        pt_pos = (pt.x, pt.y)
        cv2.circle(img, pt_pos, 2, (0, 0, 255), 1)#img, center, radius, color, thickness

    cv2.imshow("image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()