import warnings
import ecco
import os
from transformers import set_seed
import torch
import json
import numpy as np
import re
import h5py

warnings.filterwarnings('ignore')
#os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import numpy as np

def append_to_hdf5(file_path, dataset_name, new_data):
    new_data = np.expand_dims(new_data, axis=0)
    with h5py.File(file_path, 'a') as f:
        if dataset_name in f:
            dset = f[dataset_name]
            current_size = dset.shape[0]
            new_size = current_size + new_data.shape[0]
            dset.resize((new_size, *new_data.shape[1:]))
            dset[current_size:] = new_data
        else:
            f.create_dataset(dataset_name, data=new_data, maxshape=(None, *new_data.shape[1:]))

# 假设你有以下的分类列表 (True为正确，False为错误)
classification_list = json.load(open("halueval_qa_baseline_result.json", "r"))["judge_result"]
model_name = "Llama-2-7b-hf"
file_name="HaluEval_qa"

# 打开store_data.h5并读取数据
with h5py.File('./features/Llama-2-7b-hf/HaluEval_qa_dataset/store_data.h5', 'r') as f:
    activation_values = f['activation_values'][:]
    final_output_rank = f['final_output_rank'][:]
    word_id_topk_rank = f['word_id_topk_rank'][:]
    topk_rank_prob = f['topk_rank_prob'][:]

# 根据classification_list将数据分割
for i, is_correct in enumerate(classification_list):
    if is_correct:
        # 存储到 correct_data.h5
        append_to_hdf5('./features/'+model_name+'/'+file_name+'_dataset/correct_data.h5'.format(model_name, file_name), 'correct_activation_values', activation_values[i])
        append_to_hdf5('./features/'+model_name+'/'+file_name+'_dataset/correct_data.h5'.format(model_name, file_name), 'correct_final_output_rank', final_output_rank[i])
        append_to_hdf5('./features/'+model_name+'/'+file_name+'_dataset/correct_data.h5'.format(model_name, file_name), 'correct_word_id_topk_rank', word_id_topk_rank[i])
        append_to_hdf5('./features/'+model_name+'/'+file_name+'_dataset/correct_data.h5'.format(model_name, file_name), 'correct_topk_rank_prob', topk_rank_prob[i])
    else:
        # 存储到 false_data.h5
        append_to_hdf5('./features/'+model_name+'/'+file_name+'_dataset/false_data.h5'.format(model_name, file_name), 'false_activation_values', activation_values[i])
        append_to_hdf5('./features/'+model_name+'/'+file_name+'_dataset/false_data.h5'.format(model_name, file_name), 'false_final_output_rank', final_output_rank[i])
        append_to_hdf5('./features/'+model_name+'/'+file_name+'_dataset/false_data.h5'.format(model_name, file_name), 'false_word_id_topk_rank', word_id_topk_rank[i])
        append_to_hdf5('./features/'+model_name+'/'+file_name+'_dataset/false_data.h5'.format(model_name, file_name), 'false_topk_rank_prob', topk_rank_prob[i])