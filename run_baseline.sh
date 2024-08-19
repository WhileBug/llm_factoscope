#export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
#export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
#export https_proxy=http://sys-proxy-rd-relay.byted.org:8118

#mlx worker launch --cpu=1 --gpu=1 --memory=64 --type=v100-32g -- python3 construct_data.py
#mlx worker launch --cpu=1 --gpu=1 --memory=64 --type=v100-32g python3 split_data.py
mlx worker launch --cpu=1 --gpu=1 --memory=64 --type=v100-32g python3 prepare_save_data_h5.py
mlx worker launch --cpu=1 --gpu=1 --memory=64 --type=v100-32g -- python3 train_nn_tri_test.py --model_name Llama-2-7b-hf --support_size 100