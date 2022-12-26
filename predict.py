import numpy as np
from tensorflow import keras

from config import MODEL
from detected import get_letter_list
from train import model_train

model_train()


# предсказание
def mnist_predict_img(model, img):
    img_arr = 1 - img / 255
    x = np.expand_dims(img_arr, axis=0)
    res = model.predict(x)
    return str(np.argmax(res))


# проход по массиву изображений
def img_to_str(model, letter_list):
    output = []
    for j in range(len(letter_list)):
        result = ''
        letters = letter_list[j]

        for i in range(len(letters)):
            dn = letters[i + 1][0] - letters[i][0] - letters[i][1] if i < len(letters) - 1 else 0
            listb = letters[i][2]
            result += mnist_predict_img(model, listb)

            if dn > letters[i][1] / 3:
                result += ' '

        output.append(result)

    return output


model = keras.models.load_model(MODEL)
predict_result = img_to_str(model, get_letter_list())
print(predict_result)
