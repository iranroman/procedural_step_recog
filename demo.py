import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
from PIL import Image
import tqdm
import yaml
import cv2
import torch
import supervision as sv
import numpy as np
from step_recog.models import StepNet
from step_recog.statemachine import ProcedureStateMachine
import matplotlib.pyplot as plt

STATES = [
    (128, 128, 128), ##unobserved = grey
    (217, 217, 38),  ##current = blue
    (38, 217, 38)    ##done = green
]

device = 'cuda' if torch.cuda.is_available() else 'cpu' #'mps' if torch.backends.mps.is_available() else 'cpu'

import ipdb
@ipdb.iex
@torch.no_grad()
def main(video_path, skill, checkpoint_path, cfg_file="config/without_state_head.yaml", output_path='output.mp4', fps=10):
    '''Visualize the outputs of the model on a video.

    '''
    # create video reader and video writer
    video_info = sv.VideoInfo.from_video_path(video_path)
    og_fps = video_info.fps
    video_info.fps = fps

    # define model
    cfg = yaml.load(open(cfg_file), Loader=yaml.CLoader)
    model = StepNet(cfg, device).eval().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    sm = ProcedureStateMachine(len(model.SKILL_STEPS[skill]))

    STEPS = [f'{i}' for i in range(len(model.SKILL_STEPS[skill]))]

    y_outputs = []
    state_outputs = []

    h = None
    buffer = None
    skip_n = max(int(og_fps / fps), 1)
    print(output_path, video_info, skip_n)
    with sv.VideoSink(output_path, video_info=video_info) as sink:
        # iterate over video frames
        pbar = tqdm.tqdm(enumerate(sv.get_video_frames_generator(video_path)), total=video_info.total_frames)
        for idx, frame in pbar:
            if idx % int(og_fps / fps) != 0:
               continue
            # if idx > 100: break
            
            x = cv2.resize(frame, (224, 224))
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = x.astype(np.float32) / 255.0
            x = torch.as_tensor(x).permute(2, 0, 1)[None, None]

            # x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # x = model.transform_image(PIL.Image.fromarray(x))[None, None]

            buffer = model.insert_image_buffer(x, buffer)
            assert buffer.shape == (1, 32, 3, 224, 224)
            assert 0.3 < buffer.max() <= 1
            assert 0 <= buffer.min() < 1

            # for i, a in enumerate(buffer[0]):
            #     cv2.imwrite(f'asdf{i}.png', (a.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:,:,::-1])
            # input()

            y_hat_steps, y_hat_state_machine, omni_outs, h = model(buffer.clone(), h)

            prob_step, prob_state = model.skill_step_proba(y_hat_steps, y_hat_state_machine, skill=skill)

            prob_step = prob_step[0, -1].cpu().numpy()
            if prob_state is not None:
                prob_state = prob_state[0, -1].cpu().numpy()
                idx_state = np.argmax(prob_state,-1)
            else:
                sm.process_timestep(prob_step)
                idx_state = sm.current_state

            # np.round(prob_state.astype(float), 2).tolist()
            tqdm.tqdm.write(f'{idx_state}  {np.round(prob_step.astype(float), 2).tolist()}')
            step_idx = (np.where(idx_state==1)[0].tolist() or [len(STEPS)])[0]
            step_desc = "No step" if step_idx >= len(STEPS) else STEPS[step_idx]
            pbar.set_description(step_desc)         

            # draw the prediction (could be your bar chart) on the frame
            plot_graph(frame, prob_step, step_desc, idx_state)
            sink.write_frame(frame)

            y_outputs.append(prob_step)
            state_outputs.append(idx_state.copy())

    y_outputs = np.array(y_outputs)
    state_outputs = np.array(state_outputs)
    plt.figure(figsize=(16, 8))
    plt.subplot(311)
    plt.title(f'{os.path.basename(video_path).rsplit(".",1)[0]}')
    plt.imshow(y_outputs.T, interpolation='none', aspect='auto', origin='lower')
    y_pred = np.argmax(y_outputs, axis=1)
    plt.scatter(np.arange(len(y_outputs)), y_pred, c='white', alpha=0.7)
    plt.subplot(312)
    plt.title("State")
    plt.imshow(state_outputs.T, interpolation='none', aspect='auto', origin='lower', vmin=0, vmax=2)
    plt.subplot(313)
    plt.title("Confidence")
    plt.plot(y_outputs[np.arange(len(y_pred)), y_pred])
    plt.ylim([0, 1])
    plt.savefig(f'{output_path}.png')
    plt.close()

    ##TODO: Review the offsets
    ##colors in BGR
def plot_graph(frame, prob_step, step_desc, current_state, tl=(10, 25), scale=1.0, bar_space=10, text_color=(0, 219, 219), bar_clor=(22, 22, 197), thickness=1):
    width       = 30
    height      = 100
    start_point = (50, 50)
    border_color = (0, 0, 0)
    max_desc_length = 62
    end_point   = (start_point[0] + width, start_point[1] + height)

    cv2.putText(frame, "Step: " + step_desc[:max_desc_length], (int(tl[0]), int(tl[1])), cv2.FONT_HERSHEY_COMPLEX, scale, border_color, thickness * 2) #black border
    cv2.putText(frame, "Step: " + step_desc[:max_desc_length], (int(tl[0]), int(tl[1])), cv2.FONT_HERSHEY_COMPLEX, scale, text_color, thickness)

    if prob_step is not None:
        for i, prob in enumerate(prob_step):
            current_start = (start_point[0] + i * (width + bar_space), int( height * (1 - prob) + start_point[1]) )
            current_end   = (current_start[0] + width, end_point[1])  

            cv2.rectangle(frame, tuple(a - 1 for a in current_start), tuple(a + 1 for a in current_end), border_color, -1) #black border        
            cv2.rectangle(frame, current_start, current_end, bar_clor, -1)     

            if i == len(prob_step) - 1:
                cv2.line(frame, (start_point[0], end_point[1] + bar_space), (current_end[0], end_point[1] + bar_space), border_color, 2) #black border                
                cv2.line(frame, (start_point[0], end_point[1] + bar_space), (current_end[0], end_point[1] + bar_space), text_color, 1)        

            cv2.putText(frame, str(i + 1), (current_start[0] + int(width / 2),  end_point[1] + tl[1]), cv2.FONT_HERSHEY_COMPLEX, scale / 2, border_color, thickness * 2) #black border  
            cv2.putText(frame, str(i + 1), (current_start[0] + int(width / 2),  end_point[1] + tl[1]), cv2.FONT_HERSHEY_COMPLEX, scale / 2, text_color, int(thickness / 2)) 

        ## ===================================================================================================================================================================# 
            if i < len(current_state):
                state_current_start = (current_start[0], end_point[1] + 40)
                state_current_end   = (current_end[0], end_point[1] + 55)

                cv2.rectangle(frame, tuple(a - 1 for a in state_current_start), tuple(a + 1 for a in state_current_end), border_color, -1) #black border 
                cv2.rectangle(frame, state_current_start, state_current_end, STATES[current_state[i]], -1)     
        ## ===================================================================================================================================================================#            

        cv2.line(frame, (start_point[0] - bar_space + 2, end_point[1]), (start_point[0] - bar_space + 2, start_point[1]), border_color, 2)
        cv2.line(frame, (start_point[0] - bar_space + 2, end_point[1]), (start_point[0] - bar_space + 2, start_point[1]), text_color, 1)      

        cv2.putText(frame, "100", (start_point[0] - 40,  start_point[1] + 10), cv2.FONT_HERSHEY_COMPLEX, scale / 2, border_color, thickness * 2) #black border 
        cv2.putText(frame, "100", (start_point[0] - 40,  start_point[1] + 10), cv2.FONT_HERSHEY_COMPLEX, scale / 2, text_color, int(thickness / 2)) 

        cv2.putText(frame, "0",   (start_point[0] - 20,  end_point[1]), cv2.FONT_HERSHEY_COMPLEX, scale / 2, border_color, thickness * 2) #black border        
        cv2.putText(frame, "0",   (start_point[0] - 20,  end_point[1]), cv2.FONT_HERSHEY_COMPLEX, scale / 2, text_color, int(thickness / 2))   


if __name__ == '__main__':
    import fire
    fire.Fire(main)
