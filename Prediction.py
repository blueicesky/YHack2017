import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model


def predict(file):
  img_width, img_height = 96, 96
  model_path = './models/model.h5'
  model_weights_path = './models/weights.h5'
  model = load_model(model_path)
  model.load_weights(model_weights_path)
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  answer = np.argmax(result)
  proba = model.predict_proba(x)
  '''if answer == 0:
    print("Label: healthy")
  elif answer == 1:
    print("Label: unhealthy")'''

  return answer

healthy_t = 0
healthy_f = 0
unhealthy_t = 0
unhealthy_f = 0

for i, ret in enumerate(os.walk('./test-data/healthy')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: healthy")
    result = predict(ret[0] + '/' + filename)
    if result == 0:
      healthy_t += 1
    else:
      healthy_f += 1

for i, ret in enumerate(os.walk('./test-data/unhealthy')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: unhealthy")
    result = predict(ret[0] + '/' + filename)
    if result == 1:
      unhealthy_t += 1
    else:
      unhealthy_f += 1


"""
Check metrics
"""
print("True healthy: ", healthy_t)
print("False healthy: ", healthy_f)
print("True unhealthy: ", unhealthy_t)
print("False unhealthy: ", unhealthy_f)
