import tqdm
import cv2
import torch
import supervision as sv
import numpy as np

from model import StepPredictor

import ipdb
@ipdb.iex
@torch.no_grad()
def main(video_path, output_path='output.mp4', cfg_file=""):
    '''Visualize the outputs of the model on a video.

    '''
    # define model
    model = StepPredictor(cfg_file).to("cuda")

    # create video reader and video writer
    video_info = sv.VideoInfo.from_video_path(video_path)

    step_process = 30
    prob_step = np.zeros(model.cfg.MODEL.OUTPUT_DIM + 1)
    prob_step[-1] = 1.0
    step_desc = "No step"

    with sv.VideoSink(output_path, video_info=video_info) as sink:
        # iterate over video frames
        for idx, frame in tqdm.tqdm(enumerate(sv.get_video_frames_generator(video_path))):
            frame_aux = model.prepare(frame)

            if model.has_omni_maxlen() and idx % step_process == 0:
              # take in a queue frame and make the next prediction
              prob_step = model(frame_aux, queue_omni_frame = False).cpu().squeeze().numpy()
              step_idx  = np.argmax(prob_step)
              step_desc = "No step" if step_idx >= len(model.STEPS) else model.STEPS[step_idx]              
              
            model.queue_frame(frame_aux)  

            # draw the prediction (could be your bar chart) on the frame
            plot_graph(frame, prob_step, step_desc)
            sink.write_frame(frame)

##TODO: Review the offsets
##colors in BGR
def plot_graph(frame, prob_step, step_desc, tl=(10, 25), scale=1.0, bar_space=10, text_color=(0, 219, 219), bar_clor=(22, 22, 197), thickness=1):
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

    cv2.line(frame, (start_point[0] - bar_space + 2, end_point[1]), (start_point[0] - bar_space + 2, start_point[1]), border_color, 2)
    cv2.line(frame, (start_point[0] - bar_space + 2, end_point[1]), (start_point[0] - bar_space + 2, start_point[1]), text_color, 1)      

    cv2.putText(frame, "100", (start_point[0] - 40,  start_point[1] + 10), cv2.FONT_HERSHEY_COMPLEX, scale / 2, border_color, thickness * 2) #black border 
    cv2.putText(frame, "100", (start_point[0] - 40,  start_point[1] + 10), cv2.FONT_HERSHEY_COMPLEX, scale / 2, text_color, int(thickness / 2)) 

    cv2.putText(frame, "0",   (start_point[0] - 20,  end_point[1]), cv2.FONT_HERSHEY_COMPLEX, scale / 2, border_color, thickness * 2) #black border        
    cv2.putText(frame, "0",   (start_point[0] - 20,  end_point[1]), cv2.FONT_HERSHEY_COMPLEX, scale / 2, text_color, int(thickness / 2))   


if __name__ == '__main__':
    import fire
    fire.Fire(main)