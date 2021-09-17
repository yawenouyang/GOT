# Before you run the weighting module, you should train a classifier without OOD weights first.
# This is the path of the classifier parameter.
param_path='output/params/clinc150/0.9092808120592845/params.pkl'

gpu=1
# For parallel, because preparing s is slow.
python weighting_module.py --gpu $gpu --param_path $param_path --ith 0 --split_num 4 &
python weighting_module.py --gpu $gpu --param_path $param_path --ith 1 --split_num 4 &
python weighting_module.py --gpu $gpu --param_path $param_path --ith 2 --split_num 4 &
python weighting_module.py --gpu $gpu --param_path $param_path --ith 3 --split_num 4
# End
python weighting_module.py --gpu $gpu --param_path $param_path --option cal_inf