import torch
import torch.nn.functional as F

def collate(images, steps, states, length, skills, method,NFRAMES=32):
    ims = []
    sp = []
    st = []
    k = []
    for i in range(len(images)):
        k.append(skills[i])
        if len(steps[i]) == length:
            ims.append(images[i])
            sp.append(steps[i])
            st.append(states[i])
        else:
            idx = torch.randint(0,len(steps[i])-length,(1,)).item()
            ims.append(images[i][idx:idx+length+NFRAMES-1])
            sp.append(steps[i][idx:idx+length])
            st.append(states[i][idx:idx+length])
    return torch.stack(ims), torch.stack(sp), torch.stack(st), torch.stack(k)

def collate_fn_truncate(data):
    images, steps, states, skills = zip(*data)
    global_len = min([len(s) for s in steps])
    return collate(images, steps, states, global_len,skills,'truncate')

def collate_fn_pad(data):
    images, steps, states = zip(*data)
    global_len = max([len(s) for s in steps])
    return collate(images, steps, states, global_len,'pad')
