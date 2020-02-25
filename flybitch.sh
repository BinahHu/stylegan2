python run_training.py \
  --num-gpus=4 --data-dir=/datasets --config=config-f \
  --dataset=church --content-dataset=conntent --style-dataset=style \
  --total-kimg 1 --gamma=100