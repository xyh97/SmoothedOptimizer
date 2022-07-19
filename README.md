# Learning to Learn with Smooth Regularization
This is our PyTorch implementation of learned RNN optimizers with regularzation to enforce smoothness. Experiments are conducted on one GeForce GTX 1080Ti.

## Prerequisites
You can find all prerequisites in requirements.txt and install them by
```
pip install -r requirements.txt
```

## Data
- MNIST and CIFAR-10 can be downloaded automatically by torchvision.
- Mini-Imagenet as described [here](https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet)
  - You can download it from [here](https://drive.google.com/file/d/1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR/view?usp=sharing) (~2.7GB, google drive link)


## Run
### Optimization of neural networks
Codes for this task can be found in the folder ``optimization``. Default values for both lambda and epsilon are 1.0. You can train the RNN optimizer with your own lambda and epsilon by
```shell
python main.py --lamb [your lambda] --eps [your epsilon]
```
For more details about hyperparameters, you can check ``main.py``. You can tune these hyperparemeters to obtain better performance.

In addition, to evaluate the learned optimizer, you can run
```shell
python test.py --batch_size [your batch size] --optimizer_steps [your optimizer steps]
```

### Few-Shot Learning
Codes for few-shot learning can be found in the folder ``few-shot-learning``. Here are details for running experiments.


For 5-shot, 5-way training, run
```bash
bash scripts/train_5s_5c.sh
```
Hyper-parameters are referred to [this repo](https://github.com/twitter/meta-learning-lstm).

For 5-shot, 5-way evaluation, run *(remember to change `--resume` and `--seed` arguments)*
```bash
bash scripts/eval_5s_5c.sh
```

In addition, for 1-shot, 5-way training and evaluation, run respectively
```bash 
bash scripts/train_1s_5c.sh
```
and
```bash
bash scripts/eval_1s_5c.sh
```