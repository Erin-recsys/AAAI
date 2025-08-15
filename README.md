# AAAI
source code for paper of AAAI26
## Running Steps

### 1. Training Model
```bash
python main.py --dataset beauty --epochs 200 --lr 0.001 --recdim 32 --layer 3
```

### 2. Inference Testing
```bash
python main.py --inference 1 --model_path ./code/checkpoints/best_model_beauty_seed2024_ii2.pth --dataset beauty
```
