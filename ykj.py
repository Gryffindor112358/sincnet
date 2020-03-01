import shutil
import os
import soundfile as sf
import numpy as np
import sys

wav_file = r'D:\Equipments\PyCharm Community Edition 2019.1.3\Projects\all_needed\sincnet\sincnet\TIMIT_FOLDER\TRAIN\DR1\FCJF0\SA1.WAV'
[signal, fs] = sf.read(wav_file)
print(signal.shape)