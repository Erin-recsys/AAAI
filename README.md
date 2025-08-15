# AAAI
source code for paper of AAAI26
## Running Steps

### 1. Training Model
```bash
python main.py --dataset beauty
```

### 2. Inference Testing
```bash
python main.py --inference 1 --model_path ./code/checkpoints/best_model_beauty_seed2020_ii2.pth --dataset beauty
```

**Note:** Due to file size limitations, only the `beauty` dataset is included in this repository. Other datasets can be obtained from the baseline papers or official dataset websites.
