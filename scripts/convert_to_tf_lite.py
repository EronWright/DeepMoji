""" Convert to TF Lite
"""
from __future__ import print_function
import sys
import numpy as np
from os.path import abspath, dirname
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import json
import math
import tensorflow as tf
from deepmoji.model_def import deepmoji_emojis
from deepmoji.global_variables import PRETRAINED_PATH, ROOT_PATH

maxlen = 30

SAVED_PATH = '{}/model/model/saved'.format(ROOT_PATH)
CONVERTED_PATH = '{}/model/model/model.tflite'.format(ROOT_PATH)

# EXPORT_PATH = '{}/model/exported'.format(ROOT_PATH)

# print('Loading model from {}.'.format(PRETRAINED_PATH))
# model = deepmoji_emojis(maxlen, PRETRAINED_PATH)
# model.summary()

# print('Saving model to {}'.format(EXPORT_PATH))
# tf.keras.models.save_model(model, EXPORT_PATH)

# print('Converting model from {}'.format(EXPORT_PATH))
# converter = tf.lite.TFLiteConverter.from_keras_model_file(EXPORT_PATH)

print('Converting model from {}'.format(SAVED_PATH))
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_PATH)
tflite_model = converter.convert()
open(CONVERTED_PATH, "wb").write(tflite_model)

