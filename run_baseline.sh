mlx worker launch --cpu=1 --gpu=1 --memory=32 --type=v100-32g -- python3 construct_data.py
python3 prepare_save_data_h5.py
mlx worker launch --cpu=1 --gpu=1 --memory=32 --type=v100-32g -- python3 train_nn_tri_test.py --model_name Llama-2-7b-hf --support_size 100