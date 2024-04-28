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
    test_data = procedure_videos.ProceduralStepsDataset(cfg['DATASET']['DIR'], split='training', val_fold=cfg['DATASET']['VAL_FOLD'],augment_images=False, augment_time=False,fold_split='val',using_kfold_cross_validation=True)
    print(test_data.video_names)
    input()
    test_data = procedure_videos.ProceduralStepsDataset(cfg['DATASET']['DIR'], split='test', val_fold=cfg['DATASET']['VAL_FOLD'],augment_images=False, augment_time=False)

    test_loader = DataLoader(test_data, batch_size=cfg['DATASET']['BATCH_SIZE'], shuffle=False, num_workers=cfg['DATASET']['NWORKERS'])

    model = StepNet(cfg,device).to(device)

    model.eval()
    model.load_state_dict(torch.load(cfg['CHECKPOINT']))
    total_accuracy_steps = 0
    total_accuracy_states = 0
    total_accuracy_M2 = 0
    total_M2 = 0
    total_accuracy_M3 = 0
    total_M3 = 0
    total_accuracy_M5 = 0
    total_M5 = 0
    total_accuracy_R18 = 0
    total_R18 = 0
    os.makedirs('test_outputs')
    with torch.no_grad():
        for i, (images, step_labels, states, skill_id) in enumerate(test_loader):
            #print(test_data.video_names[i])
            images, step_labels, states, skill_id = images.to(device), step_labels.to(device), states.to(device), skill_id.to(device)
            yhat_steps, yhat_state, yhat_omnivore, skill_steps = model(images,test_data.video_skills[i])
            preds_steps = torch.argmax(yhat_steps, dim=2)
            preds_states = torch.argmax(yhat_state, dim=3)
            total_accuracy_steps += (preds_steps == step_labels).float().mean().item()
            total_accuracy_states += (preds_states == states).float().mean().item()

            preds_in_skill = torch.isin(preds_steps[0],skill_steps)
            indices_false = torch.where(preds_in_skill == False)[0]
            max_indices = yhat_steps[0][indices_false].max(dim=1).indices
            # Swap the max values to the last column
            for j, row_idx in enumerate(indices_false):
                max_idx = max_indices[j]
                # Move max value to the last column
                yhat_steps[0][row_idx, -1] = yhat_steps[0][row_idx, max_idx]
                yhat_steps[0][row_idx, max_idx] = -float("inf")
            preds_steps = torch.argmax(yhat_steps, dim=2)
            assert torch.isin(preds_steps[0],skill_steps).all()
            if test_data.video_skills[i] == 'M2':
                total_M2 += 1
                total_accuracy_M2 += (preds_steps == step_labels).float().mean().item()
            if test_data.video_skills[i] == 'M3':
                total_M3 += 1
                total_accuracy_M3 += (preds_steps == step_labels).float().mean().item()
            if test_data.video_skills[i] == 'M5':
                total_M5 += 1
                total_accuracy_M5 += (preds_steps == step_labels).float().mean().item()
            if test_data.video_skills[i] == 'R18':
                total_R18 += 1
                total_accuracy_R18 += (preds_steps == step_labels).float().mean().item()
            print(i,test_data.video_names[i],test_data.video_skills[i], 'accuracy', ((preds_steps == step_labels)).float().mean().item())
            print('running accuracy:',total_accuracy_steps/(i+1))
            np.save(f'test_outputs/{test_data.video_names[i]}_step_labels',step_labels.cpu().numpy())
            np.save(f'test_outputs/{test_data.video_names[i]}_step_preds',yhat_steps.cpu().numpy())
            np.save(f'test_outputs/{test_data.video_names[i]}_skill_steps',skill_steps.cpu().numpy())
    accuracy_steps = total_accuracy_steps / len(test_loader)
    print("FINAL_TEST_accuracy_steps", accuracy_steps)


if __name__ == "__main__":

    VAL_FOLD = 1

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
            'NSTEPS': 20, #including NO STEP
            'NMACHINESTATES':3,
            'BATCH_SIZE': 1,
            'NWORKERS': 0,
        },
        'TRAIN': {
            'LR': 0.001,
            'EPOCHS': 50,
        },
        #'CHECKPOINT':'wandb/run-20240417_160138-ioy4g76e/files/best_model.pth', 
        'CHECKPOINT':'wandb/run-20240421_121000-dwyc8fqs/files/best_model1.pth',
        'STEPS':{
            'Apply dressing.': 0, 
            'Apply pressure.': 1, 
            'Apply strap.': 2, 
            'Insert applicator.': 3, 
            'Insert plunger.': 4, 
            'Lock windless.': 5, 
            'Mark time.': 6, 
            'Open packaging.': 7, 
            'Peel seal.': 8, 
            'Place chest seal.': 9, 
            'Place tourniquet.': 10, 
            'Pull strap.': 11, 
            'Pull tourniquet.': 12, 
            'Push plunger.': 13, 
            'Secure device.': 14, 
            'Turn windless.': 15, 
            'Wipe blood.': 16, 
            'Wrap dressing.': 17,
            'No step.':18,
        }
    }

    main(cfg)
