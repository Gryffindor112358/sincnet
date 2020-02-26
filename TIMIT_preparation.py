#!/usr/bin/env python3

# TIMIT_preparation 
# Mirco Ravanelli 
# Mila - University of Montreal 

# July 2018

# Description: 
# This code prepares TIMIT for the following speaker identification experiments. 
# It removes start and end silences according to the information reported in the *.wrd files and normalizes the amplitude of each sentence.

# How to run it:
# python TIMIT_preparation.py $TIMIT_FOLDER $OUTPUT_FOLDER data_lists/TIMIT_all.scp 

import shutil
import os
import soundfile as sf
import numpy as np
import sys


def ReadList(list_file):  # list_file是一个写着要用哪些wav文件的一个文件，其实就是个列表
    f = open(list_file, "r")
    lines = f.readlines()
    list_sig = []
    for x in lines:
        list_sig.append(x.rstrip())#删除空格
    f.close()
    return list_sig  # 返回一个list


def copy_folder(in_folder, out_folder):  # 将in_folder中的文件夹结构完全复制到out_folder
    if not (os.path.isdir(out_folder)):
        shutil.copytree(in_folder, out_folder, ignore=ig_f)


def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]


in_folder = sys.argv[1]  # TIMIT数据集文件夹，命令中对应$TIMIT_FOLDER
out_folder = sys.argv[2]  # 复制后的文件夹，开始的时候先别创建，代码会自己创建，命令中对应$OUTPUT_FOLDER
list_file = sys.argv[3]  # 那个列表文件，命令中对应data_lists/TIMIT_all.scp

# Read List file
list_sig = ReadList(list_file)  # wav文件list

# Replicate input folder structure to output folder
copy_folder(in_folder, out_folder)  # 将in_folder中的文件夹结构完全复制到out_folder

# Speech Data Reverberation Loop
for i in range(len(list_sig)):  # 循环对wav文件处理
    # Open the wav file
    wav_file = in_folder + '/' + list_sig[i]
    [signal, fs] = sf.read(wav_file)
    signal = signal.astype(np.float64)

    # Signal normalization
    signal = signal / np.abs(np.max(signal))

    # Read wrd file
    wrd_file = wav_file.replace(".wav", ".wrd")  # 读与wav文件对应的wrd文件
    wrd_sig = ReadList(wrd_file)
    beg_sig = int(wrd_sig[0].split(' ')[0])  # 第一行的第一个时间点（开始
    end_sig = int(wrd_sig[-1].split(' ')[1])  # 最后一行的第二个时间点（结束

    # Remove silences
    signal = signal[beg_sig:end_sig]

    # Save normalized speech
    print(out_folder)
    print(list_sig[i])
    file_out = out_folder + '/' + list_sig[i]

    sf.write(file_out, signal, fs)   # 处理过的数据保存在OUTPUT_FOLDER中

    print("Done %s" % (file_out))
