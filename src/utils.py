'''
utils.py

Utility functions for running COPILOT metasurface optimization.
These configure GPU / CPU device usage for both the TensorFlow
and PyTorch implementations.
'''

import os
from tensorflow.config.experimental import set_memory_growth
from tensorflow.config import list_physical_devices
from torch import set_default_tensor_type
import subprocess as sp


# MB
def cpu_memory_info():
    command = "cat /proc/meminfo"
    memory_info = sp.check_output(command.split()).decode('ascii').split('\n')
    memory_total_info = memory_info[0]
    memory_total_value = int(memory_total_info.split()[1])
    
    memory_used_info = memory_info[1]
    memory_used_value = int(memory_used_info.split()[1])
    
    return memory_total_value / 1024, memory_used_value / 1024

# MiB
def gpu_memory_info():
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_total_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_total_values = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]

    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]

    return memory_total_values, memory_used_values


def gpu_memory_info_str():
    memory_total_values, memory_used_values = gpu_memory_info()
    info_list_str = ' ('
    for ind, (total_value, used_value) in enumerate(zip(memory_total_values, memory_used_values)):
        info_list_str += 'GPU ' + str(ind) + ':' + f' {used_value / total_value * 100:.2f}' + '%, '
    info_list_str = info_list_str[:-2] + ')'
    return info_list_str


def print_gpu_memory_usage():
    memory_total_values, memory_used_values = gpu_memory_info()

    print(' ')
    for ind, (total_value, used_value) in enumerate(zip(memory_total_values, memory_used_values)):
        str_total_value = str(total_value) + ' MiB'
        str_used_value = str(used_value) + ' MiB'
        str_ratio = f' ({used_value / total_value * 100:.2f}' + '%)'
        print(
            'GPU memory_usage in device ' + str(ind) + ': ' + str_used_value + ' /' + str_total_value + str_ratio +
            ' ' + str(total_value - used_value) + ' MiB available')


def config_gpu_memory_usage():
    gpus = list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                set_memory_growth(gpu, True)
                # Limit memory growth to 10 GB on each gpu.
                #tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=10*1024)])
        except RuntimeError as errmsg:
            print(errmsg)
     
    # Make PyTorch allocate tensors on the GPU by default.
    set_default_tensor_type('torch.cuda.FloatTensor')