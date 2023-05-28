import onnx
import onnxruntime
import numpy as np
import face_roi
from PIL import Image, ImageOps

EmoDict = {
    0: 'angry',
    1: 'disgusted',
    2: 'fearful',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprised'
}

# 反转字典
EmoDict_rev = {v: k for k, v in EmoDict.items()}


onnx_model = onnx.load("emo.onnx")
ort_session = onnxruntime.InferenceSession("emo.onnx")


def transform(img):
    gray_image = img.convert('L')
    img = np.array(gray_image)
    h, w = img.shape
    if h > w:
        img = img[(h - w) // 2:(h + w) // 2, :]
    else:
        img = img[:, (w - h) // 2:(w + h) // 2]
    img = Image.fromarray(img)
    img = img.resize((48, 48))
    img = np.array(img)
    img = img.astype(np.float32)
    img = img / 255.0
    img = (img - 0.5) / 0.5
    return img


def onnx_predict(img):
    img = face_roi.get_face(img)
    img = transform(img)
    outputs = ort_session.run(None, {"input": img.reshape(1, 1, 48, 48)})
    return {EmoDict[i]: conf for i, conf in enumerate(outputs[0][0].tolist())}