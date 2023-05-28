import face_recognition
from PIL import Image
import numpy as np

# resize = T.Compose([
#     T.Resize(224),
#     T.CenterCrop(224),
# ])
def resize(img):
    # print(type(img), img.shape)
    h, w, _ = img.shape
    if h > w:
        img = img[(h - w) // 2:(h + w) // 2, :,:]
    else:
        img = img[:, (w - h) // 2:(w + h) // 2,:]
    img = Image.fromarray(img)
    img = img.resize((224, 224))
    # print(type(img), img.shape)
    return img

def get_face(image):
    # transform PIL Image to numpy array
    img = image.copy()
    # img = np.array(img)
    image = np.array(image)
    # resize the image
    image = resize(image)

    image = np.array(image)

    face_locations = face_recognition.face_locations(image)
    # cut the face
    if len(face_locations) == 0:
        print("No face found")
        return img
    else :
        face_location = face_locations[0]
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        return pil_image
