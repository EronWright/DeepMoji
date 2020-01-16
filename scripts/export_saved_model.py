""" Convert to TF SavedModel
"""
from __future__ import print_function
import sys
import numpy as np
from os.path import abspath, dirname
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import json
import math
from keras import backend as K
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants

from deepmoji.model_def import deepmoji_emojis
from deepmoji.global_variables import PRETRAINED_PATH, ROOT_PATH

maxlen = 30

EXPORT_PATH = '{}/model/model/exported'.format(ROOT_PATH)

print('Loading model from {}.'.format(PRETRAINED_PATH))
model = deepmoji_emojis(maxlen, PRETRAINED_PATH)
model.summary()

# print('Saving model to {}'.format(EXPORT_PATH))
# tf.keras.models.save_model(model, EXPORT_PATH)

# print('Converting model from {}'.format(EXPORT_PATH))
# converter = tf.lite.TFLiteConverter.from_keras_model_file(EXPORT_PATH)

# Set the learning phase to Test since the model is already trained.
K.set_learning_phase(0)

# Build the Protocol Buffer SavedModel at 'export_path'
builder = saved_model_builder.SavedModelBuilder(EXPORT_PATH)
# Create prediction signature to be used by TensorFlow Serving Predict API
signature = predict_signature_def(inputs={"sentences": model.input},
                                  outputs={"scores": model.output})

with K.get_session() as sess:
    # Save the meta graph and the variables
    builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                         signature_def_map={"predict": signature})

builder.save()
