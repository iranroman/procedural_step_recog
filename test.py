import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from step_recog.datasets import procedure_videos
from step_recog.models import StepNet
from torch.utils.data import DataLoader

def main(cfg):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define datasets
    test_data = procedure_videos.ProceduralStepsDataset(cfg['DATASET']['DIR'], split='test', val_fold=cfg['DATASET']['VAL_FOLD'],augment_images=False, time_augment=False)

    test_loader = DataLoader(test_data, batch_size=cfg['DATASET']['BATCH_SIZE'], shuffle=False, num_workers=cfg['DATASET']['NWORKERS'])

    model = StepNet(cfg,device).to(device)

    model.eval()
    model.load_state_dict(torch.load(cfg['CHECKPOINT']), strict=False)
    total_accuracy_steps = 0
    os.makedirs('test_outputs')
    with torch.no_grad():
        for i, (images, step_labels, states) in enumerate(test_loader):
            print(test_data.video_names[i])
            images, step_labels, states = images.to(device), step_labels.to(device), states.to(device)
            yhat_steps, yhat_state, yhat_omnivore, skill_steps = model(images,test_data.video_skill[i])
            preds_steps = torch.argmax(yhat_steps, dim=2)
            total_accuracy_steps += (preds_steps == step_labels).sum().item() / step_labels.shape[-1]
            np.save(f'test_outputs/{i}_step_labels',step_labels.cpu().numpy())
            np.save(f'test_outputs/{i}_step_preds',yhat_steps.cpu().numpy())
            print(test_data.video_skill[i],i, 'accuracy', ((preds_steps == step_labels)).sum().item() / step_labels.shape[-1])
            print('running accuracy:',total_accuracy_steps/(i+1))
    accuracy_steps = total_accuracy_steps / len(test_loader)
    print("FINAL_TEST_accuracy_steps", accuracy_steps)


if __name__ == "__main__":

    VAL_FOLD = 5

    cfg = {
        "MODEL": {
            "VID_BACKBONE": "omnivore_swinB_epic",
            "VID_NFRAMES": 32,
            "VID_MEAN": [0.485, 0.456, 0.406],
            "VID_STD": [0.229, 0.224, 0.225],
            "VID_EMBED_SIZE": 1024,
            "GRU_INPUT_SIZE": 1024,
            "GRU_DROPOUT": 0.5,
            "GRU_NUM_LAYERS": 2,
        },
        'DATASET': {
            'DIR' : '/vast/irr2020/BBN',
            'VAL_FOLD' : VAL_FOLD,
            'NSTEPS': 19, #including NO STEP
            'NMACHINESTATES':3,
            'BATCH_SIZE': 1,
            'NWORKERS': 0,
        },
        'TRAIN': {
            'LR': 0.001,
            'EPOCHS': 50,
        },
        'CHECKPOINT':'wandb/run-20240417_160138-ioy4g76e/files/best_model.pth', 
    }

    main(cfg)
