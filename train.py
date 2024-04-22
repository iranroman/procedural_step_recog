import torch
import torch.nn as nn
import torch.optim as optim
from step_recog.datasets import procedure_videos
from step_recog.datasets.dataloader import collate_fn_truncate
from step_recog.models import StepNet
from torch.utils.data import DataLoader
import wandb


def main(cfg):
    wandb.init(project="bbn_ptg_stepnet", config=cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define datasets
    tr_data = procedure_videos.ProceduralStepsDataset(cfg['DATASET']['DIR'], fold_split='train', val_fold=cfg['DATASET']['VAL_FOLD'], augment_images=True, augment_time=True, using_kfold_cross_validation=True)
    vl_data = procedure_videos.ProceduralStepsDataset(cfg['DATASET']['DIR'], fold_split='val', val_fold=cfg['DATASET']['VAL_FOLD'], using_kfold_cross_validation=True)

    tr_loader = DataLoader(tr_data, collate_fn=collate_fn_truncate, batch_size=cfg['DATASET']['BATCH_SIZE'], shuffle=True, num_workers=cfg['DATASET']['NWORKERS'])
    vl_loader = DataLoader(vl_data, batch_size=1, num_workers=cfg['DATASET']['NWORKERS'])

    model = StepNet(cfg,device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['TRAIN']['LR'])
    criterion_steps = nn.CrossEntropyLoss(reduction='none')
    criterion_states = nn.CrossEntropyLoss(reduction='none')

    best_val_accuracy = 0

    __ID_SKILL__ = cfg['MODEL']['SKILL_ID']

    for epoch in range(cfg['TRAIN']['EPOCHS']):
        model.train()
        for images, step_labels, states, skill_id in tr_loader:
            images, step_labels, states = images.to(device), step_labels.to(device), states.to(device)
            optimizer.zero_grad()
            yhat_steps, yhat_state_machine, yhat_omnivore, h = model(images)
            loss_steps = criterion_steps(yhat_steps.permute(0, 2, 1), step_labels)
            loss_states = criterion_states(yhat_state_machine.permute(0, 3, 1, 2), states)
            loss = loss_steps.mean() + loss_states.mean()
            loss.backward()
            optimizer.step()
            wandb.log({"train_loss": loss.item()})
            print('training loss',loss.item())

        model.eval()
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
        with torch.no_grad():
            for images, step_labels, states, skill_id in vl_loader:
                images, step_labels, states, skill_id = images.to(device), step_labels.to(device), states.to(device), skill_id.to(device)
                yhat_steps, yhat_state_machine, yhat_omnivore, h = model(images)
                skill_steps = model.SKILL_STEPS[__ID_SKILL__[skill_id[0].item()]]
                preds_steps = torch.argmax(yhat_steps, dim=2)
                preds_states = torch.argmax(yhat_state_machine, dim=3)
                total_accuracy_steps += (preds_steps == step_labels).float().mean().item()
                total_accuracy_states += (preds_states == states).float().mean().item()

                preds_in_skill = torch.isin(preds_steps[0],skill_steps)
                indices_false = torch.where(preds_in_skill == False)[0]
                max_indices = yhat_steps[0][indices_false].max(dim=1).indices
                # Swap the max values to the last column
                for i, row_idx in enumerate(indices_false):
                    max_idx = max_indices[i]
                    # Move max value to the last column
                    yhat_steps[0][row_idx, -1] = yhat_steps[0][row_idx, max_idx]
                    yhat_steps[0][row_idx, max_idx] = -float("inf")
                preds_steps = torch.argmax(yhat_steps, dim=2)
                assert torch.isin(preds_steps[0],skill_steps).all()
                if __ID_SKILL__[skill_id[0].item()] == 'M2':
                    total_M2 += 1
                    total_accuracy_M2 += (preds_steps == step_labels).float().mean().item()
                if __ID_SKILL__[skill_id[0].item()] == 'M3':
                    total_M3 += 1
                    total_accuracy_M3 += (preds_steps == step_labels).float().mean().item()
                if __ID_SKILL__[skill_id[0].item()] == 'M5':
                    total_M5 += 1
                    total_accuracy_M5 += (preds_steps == step_labels).float().mean().item()
                if __ID_SKILL__[skill_id[0].item()] == 'R18':
                    total_R18 += 1
                    total_accuracy_R18 += (preds_steps == step_labels).float().mean().item()

        accuracy_steps = total_accuracy_steps / len(vl_loader)
        accuracy_states = total_accuracy_states / len(vl_loader)
        wandb.log({"val_accuracy_steps": accuracy_steps, "val_accuracy_states": accuracy_states,
            "M2_step_accuracy": total_accuracy_M2 / total_M2,
            "M3_step_accuracy": total_accuracy_M3 / total_M3,
            "M5_step_accuracy": total_accuracy_M5 / total_M5,
            "R18_step_accuracy": total_accuracy_R18 / total_R18,
        })
        print("val_accuracy_steps", accuracy_steps)
        print(" M2_step_accuracy", total_accuracy_M2 / total_M2)
        print(" M3_step_accuracy", total_accuracy_M3 / total_M3)
        print(" M5_step_accuracy", total_accuracy_M5 / total_M5)
        print(" R18_step_accuracy", total_accuracy_R18 / total_R18)
        print("val_accuracy_states", accuracy_states)

        if accuracy_steps > best_val_accuracy:
            best_val_accuracy = accuracy_steps
            torch.save(model.state_dict(), 'best_model.pth')
            wandb.save('best_model.pth')

    wandb.finish()


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
            "USE_STATE_HEAD": False,
            "SKILL_ID": ['M2', 'M3', 'M5', 'R18'],
            "SKILL_STEPS": {
                "M2": [11,12,2,15,6,12,14,7,18],
                "M3": [1,8,0,17,14,18],
                "M5": [8,4,5,13,1,18],
                "R18": [3,8,16,9,10,18],
            }
        },
        'DATASET': {
            'DIR' : '/vast/irr2020/BBN',
            'VAL_FOLD' : VAL_FOLD,
            'NSTEPS': 20, #including NO STEP
            'NMACHINESTATES':3,
            'BATCH_SIZE': 32,
            'NWORKERS': 0,
        },
        'TRAIN': {
            'LR': 0.001,
            'EPOCHS': 50,
        },
    }

    main(cfg)
