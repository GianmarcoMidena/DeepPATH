# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A binary to train Inception on the ImageNet data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf

from inception import inception_train
from inception.imagenet_data import ImagenetData

FLAGS = tf.compat.v1.app.flags.FLAGS


def main(_):
  dataset = ImagenetData(subset=FLAGS.subset)
  assert dataset.data_files()
  if tf.io.gfile.exists(FLAGS.train_dir):
    tf.io.gfile.rmtree(FLAGS.train_dir)
  tf.io.gfile.makedirs(FLAGS.train_dir)
  inception_train.train(dataset)


if __name__ == '__main__':
  tf.compat.v1.app.run()
