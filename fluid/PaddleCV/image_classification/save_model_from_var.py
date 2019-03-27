from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import time
import sys
import paddle
import paddle.fluid as fluid
#import reader_cv2 as reader
import reader as reader
import argparse
import functools
from utility import add_arguments, print_arguments
import math

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('pretrained_model', str,  None,                "Whether to use pretrained model.")
add_arg('saved_model', str,  "new_model",                "Where saved model.")
add_arg('model',            str,  "SE_ResNeXt50_32x4d", "Set the network to use.")
add_arg('model_category',   str,  "models_name",            "Whether to use models_name or not, valid value:'models','models_name'." )

# yapf: enable


def set_models(model_category):
    global models
    assert model_category in ["models", "models_name"
                              ], "{} is not in lists: {}".format(
                                  model_category, ["models", "models_name"])
    if model_category == "models_name":
        import models_name as models
    else:
        import models as models


def save(args):
    # parameters from arguments
    class_dim = 1000
    model_name = args.model
    pretrained_model = args.pretrained_model
    with_memory_optimization = True
    image_shape = [3,224,224]

    model_list = [m for m in dir(models) if "__" not in m]
    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # model definition
    model = models.__dict__[model_name]()

    if model_name == "GoogleNet":
        out0, out1, out2 = model.net(input=image, class_dim=class_dim)
        cost0 = fluid.layers.cross_entropy(input=out0, label=label)
        cost1 = fluid.layers.cross_entropy(input=out1, label=label)
        cost2 = fluid.layers.cross_entropy(input=out2, label=label)
        avg_cost0 = fluid.layers.mean(x=cost0)
        avg_cost1 = fluid.layers.mean(x=cost1)
        avg_cost2 = fluid.layers.mean(x=cost2)

        avg_cost = avg_cost0 + 0.3 * avg_cost1 + 0.3 * avg_cost2
        acc_top1 = fluid.layers.accuracy(input=out0, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out0, label=label, k=5)
    else:
        out = model.net(input=image, class_dim=class_dim)
        cost, pred = fluid.layers.softmax_with_cross_entropy(
            out, label, return_softmax=True)
        avg_cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=pred, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=pred, label=label, k=5)

    test_program = fluid.default_main_program().clone(for_test=True)

    fetch_list = [avg_cost.name, acc_top1.name, acc_top5.name]
    if with_memory_optimization:
        fluid.memory_optimize(
            fluid.default_main_program(), skip_opt_set=set(fetch_list))

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    val_reader = paddle.batch(reader.val(), batch_size=1)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])


    fluid.io.save_inference_model(args.saved_model, ['image', 'label'], [test_program.current_block().var(i) for i in fetch_list], exe, test_program)
    print("Save done.")
    

def main():
    args = parser.parse_args()
    print_arguments(args)
    set_models(args.model_category)
    save(args)


if __name__ == '__main__':
    main()
