mkdir -p 'log'
python train.py \
    --data_root 'c:/Users/Lemon/OneDrive/Dokumenty/GitHub/projekt-z-rozpoznawania-i-przetwarzania-obraz-w/FaceX-Zoo/train_data/faces_with_glasses' \
    --train_file 'c:/Users/Lemon/OneDrive/Dokumenty/GitHub/projekt-z-rozpoznawania-i-przetwarzania-obraz-w/FaceX-Zoo/train_data/train_list_glasses.txt' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file '../backbone_conf.yaml' \
    --head_type 'SST_Prototype' \
    --head_conf_file '../head_conf.yaml' \
    --lr 0.01 \
    --out_dir 'out_dir_glasses' \
    --epoches 100 \
    --step '50,75,90' \
    --print_freq 50 \
    --batch_size 128 \
    --momentum 0.9 \
    --alpha 0.999 \
    --log_dir 'log' \
    --tensorboardx_logdir 'sst_mobileface_glasses' \
    --save_freq 5 \
    --evaluate \
    --test_set 'GlassesTest' \
    --test_data_conf_file 'c:/Users/Lemon/OneDrive/Dokumenty/GitHub/projekt-z-rozpoznawania-i-przetwarzania-obraz-w/FaceX-Zoo/test_protocol/data_conf.yaml' \
    2>&1 | tee log/log_glasses.log