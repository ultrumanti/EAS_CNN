
# imports and stuff
import copy

import numpy as np
# !pip install scikit-image
from skimage import io
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import itertools
# Matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
import os
# ! pip install python-levenshtein
import random
import numpy as np
# from Ev_Net_2 import EV_Unet
# import torch
from collections import Counter
# from tes4 import Encoder_gene
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import difflib
# import Levenshtein
from copy import deepcopy
# import torch
from torch import nn
from torchvision import models
# import torch.nn.functional as F
from torch.nn import Module, Conv2d, Parameter, Softmax
from torch.nn import BatchNorm2d
# import random
from difflib import SequenceMatcher

from IPython.display import clear_output
import datetime


'''
    CONV: "AC_Block,Multi_Block"
    encoder_2_decoder_link: "att_pa,skip_att"
    to_decoder: "single,both,more_two"
    to_decoder_sin_att: "att"
    to_decoder_both_position: "after","behind"
    to_decoder_both_Connection mode: "addition,concatence"
    to_decoder_both_Connection tensor': "one,two,three,four"
    to_decoder_both_Connection att': "PAM_CAM,Channel"
    decoder_output_att: "att_pa,att_aam,att_aem,no_att"
'''
# <editor-fold desc=" decoder_1 ">
# </editor-fold>
# from Ev_Net_2 import EV_Unet
# import torch

model_settings = {
                    'Encoder_1': "",
                    'Encoder_2': "",
                    'Encoder_3': "",
                    'Encoder_4': "",
                    'Encoder_5': "",
                    'encoder_2_decoder_5': "",

                    'to_decoder_4_conv': "",
                    'to_decoder_4_more_two_tensor': "",
                    'to_decoder_4': "",
                    'to_decoder_4_sin_att': "",
                    'to_decoder_4_both_position': "",
                    'to_decoder_4_both_Connection_mode': "",
                    'to_decoder_4_both_Connection_tensor': "",
                    'to_decoder_4_both_Connection_att': "",

                    'to_decoder_3_conv': "",
                    'to_decoder_3_more_two_tensor': "",
                    'to_decoder_3': "",
                    'to_decoder_3_sin_att': "",
                    'to_decoder_3_both_position': "",
                    'to_decoder_3_both_Connection_mode': "",
                    'to_decoder_3_both_Connection_tensor': "",
                    'to_decoder_3_both_Connection_att': "",

                    'to_decoder_2_conv': "",
                    'to_decoder_2_more_two_tensor': "",
                    'to_decoder_2': "",
                    'to_decoder_2_sin_att': "",
                    'to_decoder_2_both_position': "",
                    'to_decoder_2_both_Connection_mode': "",
                    'to_decoder_2_both_Connection_tensor': "",
                    'to_decoder_2_both_Connection_att': "",

                    'to_decoder_1_conv': "",
                    'to_decoder_1_more_two_tensor': "",
                    'to_decoder_1': "",
                    'to_decoder_1_sin_att': "",
                    'to_decoder_1_both_position': "",
                    'to_decoder_1_both_Connection_mode': "",
                    'to_decoder_1_both_Connection_tensor': "",
                    'to_decoder_1_both_Connection_att': "",

                    'decoder_output_conv': "",
                    'decoder_output_att': "",
                    'layer_num': "",
                    # 'to_decoder_4_conv': "AC_Block",
                  }



model_struc_num = {
'decoder_4_skip_single' : 0,
'decoder_4_skip_both' : 0,
'decoder_4_skip_more_two' : 0,
'decoder_4_att_no' : 0,
'decoder_4_att_pc' : 0,
}

# import random


def Encoder_gene(num_begin):
    key = num_begin[0]
    key_1 = num_begin[1]
    if key == '0' and key_1 == '0':
        model_settings['Encoder_1'] = "AC_Block_3"
    elif key == '0' and key_1 == '1':
        model_settings['Encoder_1'] = "CNN_Block_3"
    if key == '1' and key_1 == '0':
        model_settings['Encoder_1'] = "AC_Block_5"
    elif key == '1' and key_1 == '1':
        model_settings['Encoder_1'] = "CNN_Block_5"


    key = num_begin[2]
    key_1 = num_begin[3]
    if key == '0' and key_1 == '0':
        model_settings['Encoder_2'] = "AC_Block_3"
    elif key == '0' and key_1 == '1':
        model_settings['Encoder_2'] = "CNN_Block_3"
    if key == '1' and key_1 == '0':
        model_settings['Encoder_2'] = "AC_Block_5"
    elif key == '1' and key_1 == '1':
        model_settings['Encoder_2'] = "CNN_Block_5"

    key = num_begin[4]
    key_1 = num_begin[5]
    if key == '0' and key_1 == '0':
        model_settings['Encoder_3'] = "AC_Block_3"
    elif key == '0' and key_1 == '1':
        model_settings['Encoder_3'] = "CNN_Block_3"
    if key == '1' and key_1 == '0':
        model_settings['Encoder_3'] = "AC_Block_5"
    elif key == '1' and key_1 == '1':
        model_settings['Encoder_3'] = "CNN_Block_5"

    key = num_begin[6]
    key_1 = num_begin[7]
    if key == '0' and key_1 == '0':
        model_settings['Encoder_5'] = "AC_Block_3"
    elif key == '0' and key_1 == '1':
        model_settings['Encoder_5'] = "CNN_Block_3"
    if key == '1' and key_1 == '0':
        model_settings['Encoder_5'] = "AC_Block_5"
    elif key == '1' and key_1 == '1':
        model_settings['Encoder_5'] = "CNN_Block_5"

    key = num_begin[8]
    if key == '0':
        model_settings['encoder_2_decoder_5'] = "att_pa"
    elif key == '1':
        model_settings['encoder_2_decoder_5'] = "skip_att"

    key = num_begin[9]
    key_1 = num_begin[10]
    if key == '0' and key_1 == '0':
        model_settings['to_decoder_4_conv'] = "AC_Block_3"
    elif key == '0' and key_1 == '1':
        model_settings['to_decoder_4_conv'] = "CNN_Block_3"
    if key == '1' and key_1 == '0':
        model_settings['to_decoder_4_conv'] = "AC_Block_5"
    elif key == '1' and key_1 == '1':
        model_settings['to_decoder_4_conv'] = "CNN_Block_5"

    model_settings['to_decoder_4_more_two_tensor'] = ""
    index_to_decoder_4_more_two_tensor = 0
    key = num_begin[11]
    if key == '0':
        model_settings['to_decoder_4_more_two_tensor'] = "con]--["
        model_settings['to_decoder_4_both_Connection_tensor'] = "one"
        index_to_decoder_4_more_two_tensor = index_to_decoder_4_more_two_tensor + 1
    elif key == '1':
        model_settings['to_decoder_4_more_two_tensor'] = "no]--["

    key = num_begin[12]
    if key == '0':
        model_settings['to_decoder_4_more_two_tensor'] = model_settings['to_decoder_4_more_two_tensor'] + "con]--["
        model_settings['to_decoder_4_both_Connection_tensor'] = "two"
        index_to_decoder_4_more_two_tensor = index_to_decoder_4_more_two_tensor + 1
    elif key == '1':
        model_settings['to_decoder_4_more_two_tensor'] = model_settings['to_decoder_4_more_two_tensor'] + "no]--["
    key = num_begin[13]
    if key == '0':
        model_settings['to_decoder_4_more_two_tensor'] = model_settings['to_decoder_4_more_two_tensor'] + "con]--["
        model_settings['to_decoder_4_both_Connection_tensor'] = "three"
        index_to_decoder_4_more_two_tensor = index_to_decoder_4_more_two_tensor + 1
    elif key == '1':
        model_settings['to_decoder_4_more_two_tensor'] = model_settings['to_decoder_4_more_two_tensor'] + "no]--["
    key = num_begin[14]
    if key == '0':
        model_settings['to_decoder_4_more_two_tensor'] = model_settings['to_decoder_4_more_two_tensor'] + "con"
        model_settings['to_decoder_4_both_Connection_tensor'] = "four"
        index_to_decoder_4_more_two_tensor = index_to_decoder_4_more_two_tensor + 1
    elif key == '1':
        model_settings['to_decoder_4_more_two_tensor'] = model_settings['to_decoder_4_more_two_tensor'] + "no"

    if index_to_decoder_4_more_two_tensor == 0:
        model_settings['to_decoder_4'] = "single"
    elif index_to_decoder_4_more_two_tensor == 1:
        model_settings['to_decoder_4'] = "both"
    else:
        model_settings['to_decoder_4'] = "more_two"

    model_settings['index_to_decoder_4_more_two_tensor'] = str(index_to_decoder_4_more_two_tensor)
    key = num_begin[15]
    if key == '0':
        model_settings['to_decoder_4_sin_att'] = "att"
    elif key == '1':
        model_settings['to_decoder_4_sin_att'] = "no_att"

    key = num_begin[16]
    key_1 = num_begin[17]
    if key == '0' and key_1 == '0':
        model_settings['to_decoder_4_both_Connection_att'] = "PAM_CAM"
    elif key == '0' and key_1 == '1':
        model_settings['to_decoder_4_both_Connection_att'] = "Channel"

    elif key == '1' and key_1 == '0':
        model_settings['to_decoder_4_both_Connection_att'] = "att_aam"
    else:
        model_settings['to_decoder_4_both_Connection_att'] = "no_att"

    key = num_begin[18]
    if key == '0':
        model_settings['to_decoder_4_both_position'] = "after"
    elif key == '1':
        model_settings['to_decoder_4_both_position'] = "behind"

    key = num_begin[19]
    if key == '0':
        model_settings['to_decoder_4_both_Connection_mode'] = "addition"
    elif key == '1':
        model_settings['to_decoder_4_both_Connection_mode'] = "concatence"

    key = num_begin[20]
    key_1 = num_begin[21]
    if key == '0' and key_1 == '0':
        model_settings['to_decoder_3_conv'] = "AC_Block_3"
    elif key == '0' and key_1 == '1':
        model_settings['to_decoder_3_conv'] = "CNN_Block_3"
    if key == '1' and key_1 == '0':
        model_settings['to_decoder_3_conv'] = "AC_Block_5"
    elif key == '1' and key_1 == '1':
        model_settings['to_decoder_3_conv'] = "CNN_Block_5"

    model_settings['to_decoder_3_more_two_tensor'] = ""
    index_to_decoder_3_more_two_tensor = 0
    key = num_begin[22]
    if key == '0':
        model_settings['to_decoder_3_more_two_tensor'] = "con]--["
        model_settings['to_decoder_3_both_Connection_tensor'] = "one"
        index_to_decoder_3_more_two_tensor = index_to_decoder_3_more_two_tensor + 1
    elif key == '1':
        model_settings['to_decoder_3_more_two_tensor'] = "no]--["
    key = num_begin[23]
    if key == '0':
        model_settings['to_decoder_3_more_two_tensor'] = model_settings['to_decoder_3_more_two_tensor'] + "con]--["
        model_settings['to_decoder_3_both_Connection_tensor'] = "two"
        index_to_decoder_3_more_two_tensor = index_to_decoder_3_more_two_tensor + 1
    elif key == '1':
        model_settings['to_decoder_3_more_two_tensor'] = model_settings['to_decoder_3_more_two_tensor'] + "no]--["
    key = num_begin[24]
    if key == '0':
        model_settings['to_decoder_3_more_two_tensor'] = model_settings['to_decoder_3_more_two_tensor'] + "con]--["
        model_settings['to_decoder_3_both_Connection_tensor'] = "three"
        index_to_decoder_3_more_two_tensor = index_to_decoder_3_more_two_tensor + 1
    elif key == '1':
        model_settings['to_decoder_3_more_two_tensor'] = model_settings['to_decoder_3_more_two_tensor'] + "no]--["
    key = num_begin[25]
    if key == '0':
        model_settings['to_decoder_3_more_two_tensor'] = model_settings['to_decoder_3_more_two_tensor'] + "con"
        model_settings['to_decoder_3_both_Connection_tensor'] = "four"
        index_to_decoder_3_more_two_tensor = index_to_decoder_3_more_two_tensor + 1
    elif key == '1':
        model_settings['to_decoder_3_more_two_tensor'] = model_settings['to_decoder_3_more_two_tensor'] + "no"

    if index_to_decoder_3_more_two_tensor == 0:
        model_settings['to_decoder_3'] = "single"
        model_struc_num['decoder_3_skip_single'] = 1
    elif index_to_decoder_3_more_two_tensor == 1:
        model_settings['to_decoder_3'] = "both"
        model_struc_num['decoder_3_skip_both'] = 1
    else:
        model_settings['to_decoder_3'] = "more_two"
        model_struc_num['decoder_3_skip_more_two'] = 1

    key = num_begin[26]
    if key == '0':
        model_settings['to_decoder_3_sin_att'] = "att"
        if(model_settings['to_decoder_3'] == "single"):
            model_struc_num['decoder_3_single_att_no'] = 1
    elif key == '1':
        model_settings['to_decoder_3_sin_att'] = "no_att"
        if (model_settings['to_decoder_3'] == "single"):
            model_struc_num['decoder_3_single_att_no'] = 1

    key = num_begin[27]
    key_1 = num_begin[28]
    if key == '0' and key_1 == '0':
        model_settings['to_decoder_3_both_Connection_att'] = "PAM_CAM"
        model_struc_num['decoder_3_PAM_CAM'] = 1

    elif key == '0' and key_1 == '1':
        model_settings['to_decoder_3_both_Connection_att'] = "Channel"
        model_struc_num['decoder_3_Channel'] = 1

    elif key == '1' and key_1 == '0':
        model_settings['to_decoder_3_both_Connection_att'] = "att_aam"
        model_struc_num['decoder_3_att_aam'] = 1
    else:
        model_settings['to_decoder_3_both_Connection_att'] = "no_att"
        model_struc_num['decoder_3_no_att'] = 1

    key = num_begin[29]
    if key == '0':
        model_settings['to_decoder_3_both_position'] = "after"
        if (model_settings['to_decoder_3'] == "both"):
            model_struc_num['decoder_3_after'] = 1

    elif key == '1':
        model_settings['to_decoder_3_both_position'] = "behind"
        if (model_settings['to_decoder_3'] == "both"):
            model_struc_num['decoder_3_behind'] = 1

    key = num_begin[30]
    if key == '0':
        model_settings['to_decoder_3_both_Connection_mode'] = "addition"
        if (model_settings['to_decoder_3'] == "both"):
            model_struc_num['decoder_3_behind'] = 1

    elif key == '1':
        model_settings['to_decoder_3_both_Connection_mode'] = "concatence"

    key = num_begin[31]
    key_1 = num_begin[32]
    if key == '0' and key_1 == '0':
        model_settings['to_decoder_2_conv'] = "AC_Block_3"
    elif key == '0' and key_1 == '1':
        model_settings['to_decoder_2_conv'] = "CNN_Block_3"
    if key == '1' and key_1 == '0':
        model_settings['to_decoder_2_conv'] = "AC_Block_5"
    elif key == '1' and key_1 == '1':
        model_settings['to_decoder_2_conv'] = "CNN_Block_5"

    model_settings['to_decoder_2_more_two_tensor'] = ""
    index_to_decoder_2_more_two_tensor = 0
    key = num_begin[33]
    if key == '0':
        model_settings['to_decoder_2_more_two_tensor'] = "con]--["
        model_settings['to_decoder_2_both_Connection_tensor'] = "one"
        index_to_decoder_2_more_two_tensor = index_to_decoder_2_more_two_tensor + 1
    elif key == '1':
        model_settings['to_decoder_2_more_two_tensor'] = "no]--["

    key = num_begin[34]
    if key == '0':
        model_settings['to_decoder_2_more_two_tensor'] = model_settings['to_decoder_2_more_two_tensor'] + "con]--["
        model_settings['to_decoder_2_both_Connection_tensor'] = "two"
        index_to_decoder_2_more_two_tensor = index_to_decoder_2_more_two_tensor + 1
    elif key == '1':
        model_settings['to_decoder_2_more_two_tensor'] = model_settings['to_decoder_2_more_two_tensor'] + "no]--["

    key = num_begin[35]
    if key == '0':
        model_settings['to_decoder_2_more_two_tensor'] = model_settings['to_decoder_2_more_two_tensor'] + "con]--["
        model_settings['to_decoder_2_both_Connection_tensor'] = "three"
        index_to_decoder_2_more_two_tensor = index_to_decoder_2_more_two_tensor + 1
    elif key == '1':
        model_settings['to_decoder_2_more_two_tensor'] = model_settings['to_decoder_2_more_two_tensor'] + "no]--["

    key = num_begin[36]
    if key == '0':
        model_settings['to_decoder_2_more_two_tensor'] = model_settings['to_decoder_2_more_two_tensor'] + "con"
        model_settings['to_decoder_2_both_Connection_tensor'] = "four"
        index_to_decoder_2_more_two_tensor = index_to_decoder_2_more_two_tensor + 1
    elif key == '1':
        model_settings['to_decoder_2_more_two_tensor'] = model_settings['to_decoder_2_more_two_tensor'] + "no"

    if index_to_decoder_2_more_two_tensor == 0:
        model_settings['to_decoder_2'] = "single"
    elif index_to_decoder_2_more_two_tensor == 1:
        model_settings['to_decoder_2'] = "both"
    else:
        model_settings['to_decoder_2'] = "more_two"

    key = num_begin[37]
    if key == '0':
        model_settings['to_decoder_2_sin_att'] = "att"
    elif key == '1':
        model_settings['to_decoder_2_sin_att'] = "no_att"

    key = num_begin[38]
    key_1 = num_begin[39]
    if key == '0' and key_1 == '0':
        model_settings['to_decoder_2_both_Connection_att'] = "PAM_CAM"
    elif key == '0' and key_1 == '1':
        model_settings['to_decoder_2_both_Connection_att'] = "Channel"

    elif key == '1' and key_1 == '0':
        model_settings['to_decoder_2_both_Connection_att'] = "att_aam"
    else:
        model_settings['to_decoder_2_both_Connection_att'] = "no_att"

    key = num_begin[40]
    if key == '0':
        model_settings['to_decoder_2_both_position'] = "after"
    elif key == '1':
        model_settings['to_decoder_2_both_position'] = "behind"

    key = num_begin[41]
    if key == '0':
        model_settings['to_decoder_2_both_Connection_mode'] = "addition"
    elif key == '1':
        model_settings['to_decoder_2_both_Connection_mode'] = "concatence"

    key = num_begin[42]
    key_1 = num_begin[43]
    if key == '0' and key_1 == '0':
        model_settings['to_decoder_1_conv'] = "AC_Block_3"
    elif key == '0' and key_1 == '1':
        model_settings['to_decoder_1_conv'] = "CNN_Block_3"
    if key == '1' and key_1 == '0':
        model_settings['to_decoder_1_conv'] = "AC_Block_5"
    elif key == '1' and key_1 == '1':
        model_settings['to_decoder_1_conv'] = "CNN_Block_5"

    model_settings['to_decoder_1_more_two_tensor'] = ""
    index_to_decoder_1_more_two_tensor = 0
    key = num_begin[44]
    if key == '0':
        model_settings['to_decoder_1_more_two_tensor'] = "con]--["
        model_settings['to_decoder_1_both_Connection_tensor'] = "one"
        index_to_decoder_1_more_two_tensor = index_to_decoder_1_more_two_tensor + 1
    elif key == '1':
        model_settings['to_decoder_1_more_two_tensor'] = "no]--["

    key = num_begin[45]
    if key == '0':
        model_settings['to_decoder_1_more_two_tensor'] = model_settings['to_decoder_1_more_two_tensor'] + "con]--["
        model_settings['to_decoder_1_both_Connection_tensor'] = "two"
        index_to_decoder_1_more_two_tensor = index_to_decoder_1_more_two_tensor + 1
    elif key == '1':
        model_settings['to_decoder_1_more_two_tensor'] = model_settings['to_decoder_1_more_two_tensor'] + "no]--["

    key = num_begin[46]
    if key == '0':
        model_settings['to_decoder_1_more_two_tensor'] = model_settings['to_decoder_1_more_two_tensor'] + "con]--["
        model_settings['to_decoder_1_both_Connection_tensor'] = "three"
        index_to_decoder_1_more_two_tensor = index_to_decoder_1_more_two_tensor + 1
    elif key == '1':
        model_settings['to_decoder_1_more_two_tensor'] = model_settings['to_decoder_1_more_two_tensor'] + "no]--["

    key = num_begin[47]
    if key == '0':
        model_settings['to_decoder_1_more_two_tensor'] = model_settings['to_decoder_1_more_two_tensor'] + "con"
        model_settings['to_decoder_1_both_Connection_tensor'] = "four"
        index_to_decoder_1_more_two_tensor = index_to_decoder_1_more_two_tensor + 1
    elif key == '1':
        model_settings['to_decoder_1_more_two_tensor'] = model_settings['to_decoder_1_more_two_tensor'] + "no"

    if index_to_decoder_1_more_two_tensor == 0:
        model_settings['to_decoder_1'] = "single"
    elif index_to_decoder_1_more_two_tensor == 1:
        model_settings['to_decoder_1'] = "both"
    else:
        model_settings['to_decoder_1'] = "more_two"

    key = num_begin[48]
    if key == '0':
        model_settings['to_decoder_1_sin_att'] = "att"
    elif key == '1':
        model_settings['to_decoder_1_sin_att'] = "no_att"

    key = num_begin[49]
    key_1 = num_begin[50]
    if key == '0' and key_1 == '0':
        model_settings['to_decoder_1_both_Connection_att'] = "PAM_CAM"
    elif key == '0' and key_1 == '1':
        model_settings['to_decoder_1_both_Connection_att'] = "Channel"

    elif key == '1' and key_1 == '0':
        model_settings['to_decoder_1_both_Connection_att'] = "att_aam"
    else:
        model_settings['to_decoder_1_both_Connection_att'] = "no_att"

    key = num_begin[51]
    if key == '0':
        model_settings['to_decoder_1_both_position'] = "after"
    elif key == '1':
        model_settings['to_decoder_1_both_position'] = "behind"

    key = num_begin[52]
    if key == '0':
        model_settings['to_decoder_1_both_Connection_mode'] = "addition"
    elif key == '1':
        model_settings['to_decoder_1_both_Connection_mode'] = "concatence"

    key = num_begin[53]
    if key == '0':
        model_settings['decoder_output_att'] = "PAM_CAM"
    elif key == '1':
        model_settings['decoder_output_att'] = "no_att"

    key = num_begin[54]
    if key == '0':
        model_settings['layer_num'] = "full"
    elif key == '1':
        model_settings['layer_num'] = "reduce_one"

    return model_settings