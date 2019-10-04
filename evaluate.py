#!/usr/bin/env python
"""加载模型，评测在测试集上的表现"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import coref_model as cm
import util

if __name__ == "__main__":
    config = util.initialize_from_env()
    model = cm.CorefModel(config)
    with tf.Session() as session:
        model.restore(session)
        model.evaluate(session, official_stdout=True)
