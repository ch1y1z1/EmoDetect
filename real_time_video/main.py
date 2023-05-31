import onnx_predict
import cv2
import time

# open the camera
cap = cv2.VideoCapture(0)

time1 = time.time()
fps = 0
fps_tmp = 0

# 获取每帧
while True:
    ret, frame = cap.read()
    if ret:
        # 识别
        output = onnx_predict.onnx_predict(frame)
        # 画框
        print(output)

        # 最大的
        max_key = max(output, key=output.get)

        for idx, (i, j) in enumerate(output.items()):
            cv2.putText(frame, i + ':', (10, 30 * (idx+1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if i == max_key else (0,255,0), 2)
            cv2.putText(frame, str(j), (200, 30* (idx+1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if i == max_key else (0,255,0), 2)
        # 显示
        # fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 实时帧率
        fps_tmp += 1
        if time.time() - time1 >= 1:
            fps = fps_tmp
            fps_tmp = 0
            time1 = time.time()
        


        cv2.putText(frame, 'FPS: ' + str(fps), (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('frame', frame)
    # 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()