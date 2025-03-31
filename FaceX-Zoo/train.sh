mkdir 'log'
python train.py \
    --data_root './trainingData' \
    --train_file './trainingData/train_file.txt' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file './training_mode/backbone_conf.yaml' \
    --head_type 'MagFace' \
    --head_conf_file './training_mode/head_conf.yaml' \
    --lr 0.1 \
    --out_dir 'out_dir' \
    --epoches 18 \
    --step '10, 13, 16' \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 512 \
    --momentum 0.9 \
    --log_dir 'log' \
    --tensorboardx_logdir 'mv-hrnet' \
    2>&1 | tee log/log.log
