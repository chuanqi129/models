import numpy as np
import sys
import os
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import argparse
import functools
from utility import add_arguments, print_arguments
sys.path.append('..')
import int8_supporting.utility as ut

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

add_arg('input',		str,	'',	"complete program and weights dir.")
add_arg('op',			str,	'',	"prune the program from the specified.")
add_arg('dot',			str,	'',	"output dot file name.")
add_arg('output',		str,	'pruned_out',	"pruned model dir.")
add_arg('with_transpiler',	bool,	True,	"whether open transpiler.")

def prune(args):
    model = args.input
    dot_name = args.dot
    prune_op = args.op
    pruned_dir = args.output
    
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    [test_program, feed_dict,
         fetch_targets] = fluid.io.load_inference_model(model, exe)
    
    if args.with_transpiler:
        t = fluid.transpiler.InferenceTranspiler()
        t.transpile(test_program, place)
    
    if prune_op:
        prune_index = -1
        for op_index, op in enumerate(test_program.current_block().ops):
	    # Todo hardcode for prune pass
            #if op.type == prune_op:
            if op.type in ["softmax", "cross_entropy", "cross_entropy2", "softmax_with_cross_entropy"]:
                prune_fetch_list = []
                for input_name in op.input_names:
		    if input_name != 'X':
			continue
                    prune_fetch_list.append(op.input(input_name)[0])
                prune_index = op_index
                break

        for index in range(len(test_program.current_block().ops) - 1, prune_index - 1, -1):
            test_program.current_block()._remove_op(index)
        label_index = feed_dict.index('label')
        feed_dict.pop(label_index)
        fetch_targets = [test_program.current_block().var(i) for i in prune_fetch_list]
        fluid.io.save_inference_model(pruned_dir, feed_dict, fetch_targets, exe, test_program)

    if args.dot:
    	ut.dot(test_program, args.dot)

if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    prune(args)
