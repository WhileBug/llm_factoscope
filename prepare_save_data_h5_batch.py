import numpy as np
import h5py
import os
import torch
import ecco
import json

random_seed = 0

def append_to_hdf5(file_path, dataset_name, new_data):
    with h5py.File(file_path, 'a') as f:
        if dataset_name in f:
            dset = f[dataset_name]
            current_size = dset.shape[0]
            new_size = current_size + new_data.shape[0]
            dset.resize((new_size, *new_data.shape[1:]))
            dset[current_size:] = new_data
        else:
            f.create_dataset(dataset_name, data=new_data, maxshape=(None, *new_data.shape[1:]))

def load_single_batch(model_name, data_name, batch_index, batch_size=10):
    correct_file_path = f'./features/{model_name}/{data_name}_dataset/correct_data.h5'
    if not os.path.exists(correct_file_path):
        print(correct_file_path)
        return None
    
    with h5py.File(correct_file_path, 'r') as f:
        correct_data = f['correct_activation_values'][:]
        correct_rank = f['correct_final_output_rank'][:]
        correct_word_id_topk_rank = f['correct_word_id_topk_rank'][:]
        correct_topk_rank_prob = f['correct_topk_rank_prob'][:]

    false_file_path = f'./features/{model_name}/{data_name}_dataset/false_data.h5'
    if not os.path.exists(false_file_path):
        print(false_file_path)
        return None

    with h5py.File(false_file_path, 'r') as f:
        false_data = f['false_activation_values'][:]
        false_rank = f['false_final_output_rank'][:]
        false_word_id_topk_rank = f['false_word_id_topk_rank'][:]
        false_topk_rank_prob = f['false_topk_rank_prob'][:]

    # Calculate total length and split data into batches
    total_len = min(correct_data.shape[0], false_data.shape[0])
    indices = np.array_split(np.arange(total_len), batch_size)
    
    # Select the specific batch based on batch_index
    if batch_index >= len(indices):
        raise IndexError(f"Batch index {batch_index} is out of range for {len(indices)} total batches.")
    
    batch_indices = indices[batch_index]
    
    # Return the batch data
    correct_batch_data = correct_data[batch_indices]
    correct_batch_rank = correct_rank[batch_indices]
    correct_batch_word_id_topk_rank = correct_word_id_topk_rank[batch_indices]
    correct_batch_topk_rank_prob = correct_topk_rank_prob[batch_indices]

    false_batch_data = false_data[batch_indices]
    false_batch_rank = false_rank[batch_indices]
    false_batch_word_id_topk_rank = false_word_id_topk_rank[batch_indices]
    false_batch_topk_rank_prob = false_topk_rank_prob[batch_indices]
    
    return (correct_batch_data, correct_batch_rank, correct_batch_word_id_topk_rank, correct_batch_topk_rank_prob,
            false_batch_data, false_batch_rank, false_batch_word_id_topk_rank, false_batch_topk_rank_prob)

def process_activation_data(all_data, mean, std):
    return (all_data - mean) / std

def process_rank_data(all_rank):
    a = -1
    return 1 / (a * (all_rank - 1) + 1 + 1e-7)

def process_word_id_topk_rank_data(all_word_id_topk_rank, model_emb, file_path):
    batch, layer, n_words = all_word_id_topk_rank.shape
    print(batch, layer, n_words)
    
    for b in range(batch):
        layer_distance = None
        for l in range(layer - 1):
            words0 = torch.tensor(all_word_id_topk_rank[b, l, :]).unsqueeze(0)
            words1 = torch.tensor(all_word_id_topk_rank[b, l + 1, :]).unsqueeze(0)

            emb0 = model_emb(words0)
            emb1 = model_emb(words1)

            distances = torch.cosine_similarity(emb0, emb1, dim=2)

            if layer_distance is None:
                layer_distance = distances
            else:
                layer_distance = torch.cat((layer_distance, distances), dim=0)

        append_to_hdf5(file_path, 'all_word_id_topk_rank', layer_distance.unsqueeze(0).detach().cpu().numpy())  

def main():
    np.random.seed(random_seed)
    dataset_name = ["multi"]
    llama2_model_config = {    
        'embedding': "model.embed_tokens.weight",
        'type': 'causal',
        'activations': ['mlp\.act_fn'],
        'token_prefix': 'Ġ',
        'partial_token_prefix': ''
    }

    MODELS_DIR = "./llm_models"
    model_name = "Llama-2-7b-hf"
    lm = ecco.from_pretrained(os.path.join(MODELS_DIR, model_name), model_config=llama2_model_config, activations=False)
    lm.model.config._name_or_path = model_name

    sum_val = 0
    counter = 0

    # Set the number of batches
    batch_size = 10

    # Calculate mean and std
    for current_dataset_name in dataset_name:
        for batch_index in range(batch_size):
            batch_data = load_single_batch(model_name, current_dataset_name, batch_index, batch_size)
            correct_data, _, _, _, false_data, _, _, _ = batch_data
            sum_val += np.sum(correct_data) + np.sum(false_data)
            counter += correct_data.size + false_data.size

    mean = sum_val / counter

    sum_val = 0
    for current_dataset_name in dataset_name:
        for batch_index in range(batch_size):
            batch_data = load_single_batch(model_name, current_dataset_name, batch_index, batch_size)
            correct_data, _, _, _, false_data, _, _, _ = batch_data
            sum_val += np.sum(np.square(correct_data - mean)) + np.sum(np.square(false_data - mean))

    std = np.sqrt(sum_val / counter)
    print('mean: {}, std: {}'.format(mean, std))

    with open(f'./features/{model_name}/mean_std.json', 'w') as f:
        json.dump({'mean': mean, 'std': std}, f)

    for current_dataset_name in dataset_name:
        file_path = f'./features/{model_name}/all_data_{current_dataset_name}.h5'
        for batch_index in range(batch_size):
            batch_data = load_single_batch(model_name, current_dataset_name, batch_index, batch_size)
            (correct_data, correct_rank, correct_word_id_topk_rank, correct_topk_rank_prob,
             false_data, false_rank, false_word_id_topk_rank, false_topk_rank_prob) = batch_data

            print('[batch length] dataset: {}, batch: {}, correct: {}, false: {}'.format(current_dataset_name, batch_index, correct_data.shape[0], false_data.shape[0]))

            all_data = np.concatenate((correct_data, false_data), axis=0)
            all_rank = np.concatenate((correct_rank, false_rank), axis=0)
            all_word_id_topk_rank = np.concatenate((correct_word_id_topk_rank, false_word_id_topk_rank), axis=0)
            all_topk_rank_prob = np.concatenate((correct_topk_rank_prob, false_topk_rank_prob), axis=0)
            all_label = np.concatenate((np.ones(correct_data.shape[0]), np.zeros(false_data.shape[0])), axis=0)

            all_data = process_activation_data(all_data, mean, std)
            all_rank = process_rank_data(all_rank)

            append_to_hdf5(file_path, 'all_activation_values', all_data)
            append_to_hdf5(file_path, 'all_final_output_rank', all_rank)
            append_to_hdf5(file_path, 'all_topk_rank_prob', all_topk_rank_prob)
            
            del all_data
            del all_rank
            del all_topk_rank_prob
            
            process_word_id_topk_rank_data(all_word_id_topk_rank, lm.model.model.embed_tokens.cpu(), file_path)
            append_to_hdf5(file_path, 'all_label', all_label)
            del all_word_id_topk_rank  

if __name__ == "__main__":
    main()
