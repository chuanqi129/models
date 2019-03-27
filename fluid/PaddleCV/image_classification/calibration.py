from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function
import numpy as np
import time
import sys
import paddle
import paddle.fluid as fluid
import reader
import argparse
import functools
from utility import add_arguments, print_arguments
import paddle.fluid.core as core
import os
sys.path.append('..')
import int8_supporting.utility as ut
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  32,                 "Minibatch size.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('class_dim',        int,  1000,                "Class number.")
add_arg('image_shape',      str,  "3,224,224",         "Input image size")
add_arg('out',              str,  "calibration_out",   "Output INT8 model")
add_arg('with_mem_opt',     bool, True,                "Whether to use memory optimization or not.")
add_arg('use_train_data',   bool, False,               "Whether to use train data for sampling or not.")
add_arg('pretrained_model', str,  None,                "Whether to use pretrained model.")
add_arg('model',            str, "SE_ResNeXt50_32x4d", "Set the network to use.")
add_arg('iterations',       int, 1, "Sampling iteration")
add_arg('algo',             str, 'direct', "calibration algo")
add_arg('debug',            bool, False, "print program and save the dot file")
add_arg('first_conv_int8',  bool, False, "enable the first convolution int8")
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


def dropout_opt(test_program):
    dropout_op_index = [index for index, value in enumerate(test_program.global_block().ops) if value.type == 'dropout']
    dropout_op_index.reverse()
    for i in dropout_op_index:
        dropout_op = test_program.current_block().ops[i]
        input_name = dropout_op.input_arg_names[0]
        output_names = dropout_op.output_arg_names
        output_name = output_names[1]

        for n in range(i, len(test_program.current_block().ops)):
            op_n = test_program.current_block().ops[n]
            if output_name not in op_n.input_arg_names:
                continue
            for k in op_n.input_names:
                op_n_inputs = op_n.input(k)
                if len(op_n_inputs) > 0 and output_name in op_n_inputs:
                    op_n_new_inputs = [
                        input_name
                        if op_n_input == output_name else op_n_input
                        for op_n_input in op_n_inputs
                    ]
                    op_n.desc.set_input(k, op_n_new_inputs)

        test_program.current_block()._remove_op(i)
        for output_name in output_names:
            test_program.current_block()._remove_var(output_name)


    return test_program


def eval(args):
    # parameters from arguments
    class_dim = args.class_dim
    model_name = args.model
    pretrained_model = args.pretrained_model
    with_memory_optimization = args.with_mem_opt
    image_shape = [int(m) for m in args.image_shape.split(",")]

    model_list = [m for m in dir(models) if "__" not in m]
    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    int8_model = os.path.join(os.getcwd(), args.out)
    print("Start calibration for {}...".format(model_name))

    tmp_scale_folder = ".tmp_scale"

    if os.path.exists(
            int8_model):  # Not really need to execute below operations
        os.system("rm -rf " + int8_model)
        os.system("mkdir " + int8_model)

    if not os.path.exists(tmp_scale_folder):
        os.system("mkdir {}".format(tmp_scale_folder))

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
        cost = fluid.layers.cross_entropy(input=out, label=label)

        avg_cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

    test_program = fluid.default_main_program().clone(for_test=True)

    if with_memory_optimization:
        fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    t = fluid.transpiler.InferenceTranspiler()
    t.transpile(test_program,
                fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace())

    # prune dropout op
    test_program = dropout_opt(test_program)

    sampling_reader = paddle.batch(
        reader.train() if args.use_train_data else reader.val(),
        batch_size=args.batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
    fetch_list = [avg_cost.name, acc_top1.name, acc_top5.name]

    calibrator = ut.Calibrator(
        program=test_program,
        pretrained_model=pretrained_model,
        iterations=args.iterations,
        debug=args.debug,
        first_conv_int8=args.first_conv_int8,
        algo=args.algo)

    sampling_data = {}

    calibrator.generate_sampling_program()
    feeded_var_names = None
    for batch_id, data in enumerate(sampling_reader()):
        _, _, _ = exe.run(calibrator.sampling_program,
                          fetch_list=fetch_list,
                          feed=feeder.feed(data))
        for i in calibrator.sampling_program.list_vars():
            if i.name in calibrator.sampling_vars:
                np_data = np.array(fluid.global_scope().find_var(i.name)
                                   .get_tensor())
                if i.name not in sampling_data:
                    sampling_data[i.name] = []
                sampling_data[i.name].append(np_data)

        if batch_id != args.iterations - 1:
            continue
        feeded_var_names = feeder.feed(data).keys()
        break

    calibrator.generate_quantized_data(sampling_data)

    fluid.io.save_inference_model(int8_model, feeded_var_names, [
        calibrator.sampling_program.current_block().var(i) for i in fetch_list
    ], exe, calibrator.sampling_program)
    print(
        "Calibration is done and the corresponding files were generated at {}".
        format(os.path.abspath(args.out)))


def main():
    args = parser.parse_args()
    print_arguments(args)
    set_models(args.model_category)
    eval(args)


if __name__ == '__main__':
    main()
