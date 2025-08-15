import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
import os
from os.path import join
# ===================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.Loss(Recmodel, world.config)

Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")
best_recall_100 = 0.0

if world.INFERENCE:
    if world.MODEL_PATH and os.path.exists(world.MODEL_PATH):
        checkpoint = torch.load(world.MODEL_PATH, map_location=world.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model_dict = Recmodel.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        Recmodel.load_state_dict(filtered_dict, strict=False)
        print(f"Loaded model: {world.MODEL_PATH}")
        print(f"Model epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Best recall@100: {checkpoint.get('best_recall_100', 'unknown')}")
        # Start inference testing
        print("Starting inference testing...")
        results = Procedure.Test(dataset, Recmodel, 0, w, world.config['multicore'])
        exit()  # Exit after inference
    else:
        print(f"Model file does not exist: {world.MODEL_PATH}")
        exit()


try:
    for epoch in range(world.TRAIN_epochs):
        epoch_start_time = time.time()

        if epoch % 1 == 0:  
            if epoch % 1 == 0 and epoch > 0:  
                start_time = time.time()
                cprint("[TEST ON ALL TEST SETS]")
                results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])

                if results is not None:
                    
                    recall_100_idx = world.topks.index(100)
                    current_recall_100 = results['recall'][recall_100_idx]
                    
                    if current_recall_100 > best_recall_100:
                        best_recall_100 = current_recall_100
                        
                        save_path = os.path.join(world.FILE_PATH, f"best_model_{world.dataset}_seed{world.seed}_ii{world.config['ii_layers']}.pth")
                        if os.path.exists(save_path):
                            old_checkpoint = torch.load(save_path, map_location='cpu')
                            old_best_recall = old_checkpoint.get('best_recall_100', 0.0)
                            print(f"Found existing model, best recall@100: {old_best_recall:.4f}")
                        else:
                            old_best_recall = 0.0
                            print("No existing model file found")

                        if current_recall_100 > old_best_recall:
                            print(f"New best recall@100: {best_recall_100:.4f}, saving model...")
                            torch.save({
                                'model_state_dict': Recmodel.state_dict(),
                                'epoch': epoch,
                                'best_recall_100': best_recall_100,
                                'config': world.config
                            }, save_path)
                            print(f"Model saved to: {save_path}")

            output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
            epoch_time = time.time() -  epoch_start_time 

            print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
            print("EPOCH Time: {:.4f}s".format(epoch_time))
finally:
    if world.tensorboard:
        w.close()