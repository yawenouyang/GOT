dataset='clinc150'
label_num=150
sample_num=100
threshold=200
gpu=6
seed=1111

python train_clm.py --gpu $gpu --dataset $dataset --seed $seed --label_num $label_num
python locating_module.py --gpu $gpu --dataset $dataset --label_num $label_num --threshold $threshold
python generating_module.py --gpu $gpu --dataset $dataset --label_num $label_num --sample_num $sample_num