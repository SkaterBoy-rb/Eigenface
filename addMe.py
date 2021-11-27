import cv2 as cv

def save():
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)  # 计算机自带的摄像头为0，外部设备为1
    i = 1
    while True:
        ret, frame = cap.read()  # ret:True/False,代表有没有读到图片  frame:当前截取一帧的图片
        cv.imshow("capture", frame)

        if (cv.waitKey(1) & 0xFF) == ord('s'):  # 不断刷新图像，这里是1ms 返回值为当前键盘按键值
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # RGB图像转为单通道的灰度图像
            gray = cv.resize(gray, (92, 112))  # 图像大小为92*112
            cv.imwrite('./picture/s41/%d.pgm' % i, gray)
            i += 1
        if (i > 10 or cv.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

save()
