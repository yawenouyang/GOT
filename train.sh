seeds=(1111 2222 3333 4444 5555)
for seed in ${seeds[@]}; do
	python train.py --gpu 1 --seed $seed --al_ratio 0  # without OOD
	python train.py --gpu 1 --seed $seed --al_ratio 0.1 --weight 0  # without weighting
	python train.py --gpu 1 --seed $seed --al_ratio 0.1 --weight 1  # with weighting
done