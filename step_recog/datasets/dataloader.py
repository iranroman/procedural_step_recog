import torch
import torch.nn.functional as F

def collate(images, steps, states, length, method,NFRAMES=32):
    ims = []
    sp = []
    st = []
    m = []
    for i in range(len(images)):
        if len(steps[i]) == length:
            ims.append(images[i])
            sp.append(steps[i])
            st.append(states[i])
            m.append(torch.ones_like(steps[i]))
        else:
            if method == 'truncate':
                idx = torch.randint(0,len(steps[i])-length,(1,)).item()
                ims.append(images[i][idx:idx+length+NFRAMES-1])
                sp.append(steps[i][idx:idx+length])
                st.append(states[i][idx:idx+length])
                m.append(torch.ones_like(sp[-1]))
            elif method == 'pad':
                ims.append(F.pad(images[i],(0,0,0,0,0,0,0,length-steps[i].shape[0])))
                sp.append(F.pad(steps[i],(0,length-steps[i].shape[0])))
                st.append(F.pad(states[i],(0,0,0,length-steps[i].shape[0])))
                m.append(torch.cat((torch.ones_like(steps[i]),torch.zeros(length-steps[i].shape[0])),dim=0))
    return torch.stack(ims), torch.stack(sp), torch.stack(st), torch.stack(m)

def collate_fn_truncate(data):
    images, steps, states = zip(*data)
    global_len = min([len(s) for s in steps])
    return collate(images, steps, states, global_len,'truncate')

def collate_fn_pad(data):
    images, steps, states = zip(*data)
    global_len = max([len(s) for s in steps])
    return collate(images, steps, states, global_len,'pad')
