import torch
from torch import nn

class Decapitvore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.omni = torch.hub.load("facebookresearch/omnivore:main", model=cfg['MODEL']['VID_BACKBONE'])
        self.heads = self.omni.heads
        self.omni.heads = nn.Identity() # decapitate omnivore
        self.embedding_size = cfg['MODEL']['VID_EMBED_SIZE']

    def forward(self, x):
        with torch.no_grad():
            shoul = self.omni(x, input_type="video")
            y_raw = self.heads(shoul)
        return shoul, y_raw

class StepNet(nn.Module):
    def __init__(self,cfg, device):
        super().__init__()

        self.device = device
        
        self.dropout = nn.Dropout(cfg['MODEL']['GRU_DROPOUT'])

        self.video_backbone = Decapitvore(cfg)
        self.video_dense = nn.Linear(self.video_backbone.embedding_size,cfg['MODEL']['GRU_INPUT_SIZE'])
        self.vid_nframes = cfg['MODEL']['VID_NFRAMES']
        self.vid_mean = torch.tensor(cfg['MODEL']['VID_MEAN']).to(self.device)
        self.vid_std = torch.tensor(cfg['MODEL']['VID_STD']).to(self.device)
        self.vid_mean = self.vid_mean[None,:,None,None,None]
        self.vid_std = self.vid_std[None,:,None,None,None]

        self.gru = nn.GRU(cfg['MODEL']['GRU_INPUT_SIZE'],cfg['MODEL']['GRU_INPUT_SIZE'],cfg['MODEL']['GRU_NUM_LAYERS'],dropout=cfg['MODEL']['GRU_DROPOUT'])
        self.gru_dense_steps = nn.Linear(cfg['MODEL']['GRU_INPUT_SIZE'],cfg['DATASET']['NSTEPS'])
        self.gru_states = nn.GRU(cfg['DATASET']['NSTEPS'],cfg['MODEL']['GRU_INPUT_SIZE'],1,dropout=cfg['MODEL']['GRU_DROPOUT'])
        self.gru_dense_state_machine = nn.Linear(cfg['MODEL']['GRU_INPUT_SIZE'],(cfg['DATASET']['NSTEPS']-1)*cfg['DATASET']['NMACHINESTATES'])

        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(-1)


    def forward(self, x):

        vid_embeds = []
        omni_outs = []
        x = x.permute(0,2,1,3,4)
        x -= self.vid_mean
        x /= self.vid_std
        for t in range(x.shape[2]-self.vid_nframes+1):
            x_vid = x[:,:,t:t+self.vid_nframes]
            assert x_vid.shape[2] == self.vid_nframes
            vid_emb, vid_y_raw = self.video_backbone(x_vid)
            vid_embeds.append(vid_emb)
            omni_outs.append(vid_y_raw)
        vid_embeds = torch.stack(vid_embeds)
        omni_outs = torch.stack(omni_outs)
        vid_embeds = self.dropout(self.relu(self.video_dense(vid_embeds)))
        gru_out,hidden = self.gru(vid_embeds)
        y_hat_steps = self.gru_dense_steps(gru_out).permute(1,0,2)
        gru_out,hidden = self.gru_states(y_hat_steps)
        y_hat_state_machine = self.gru_dense_state_machine(gru_out).permute(1,0,2)
        y_hat_state_machine = y_hat_state_machine.reshape(y_hat_steps.shape[0],y_hat_steps.shape[1],y_hat_steps.shape[2]-1,-1)
        return y_hat_steps, y_hat_state_machine, omni_outs


if __name__ == "__main__":

    config = {
        "MODEL": {
            "VID_BACKBONE": "omnivore_swinB_epic",
        },
        "DATASET": {
        }
    }

    model = Omnivore(config)
    print(model)
