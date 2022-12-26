# handw_yolo
Предназначен для поиска и распознования рукописных цифр на изображении.
В папке source расположены веса для yolov5(weight.pt),а также custon_data.yaml и detect.py.

!python /content/yolov5/detect.py —weights /content/hw_yolo/source/weight.pt —source /content/hw_yolo/images/train —data

/content/hw_yolo/source/custom_data.yaml —save-txt.

В этой же папке находится модель машинного обучения - mnist_letters.h5.

Для распознования необходимо с помощью detect.py найти координаты рамок найденых цифр. 
Затем с помощью opencv, разделить исходдное изображение по координатам рамок.
Далее загрузить модель и вызвать predict для каждого изображения.
