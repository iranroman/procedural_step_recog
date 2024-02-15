import tqdm
import cv2
import torch
import supervision as sv

from .model import StepPredictor


import ipdb
@ipdb.iex
@torch.no_grad()
def main(video_path, output_path='output.mp4', skill="M3"):
    '''Visualize the outputs of the model on a video.

    '''
    # define model
    model = StepPredictor(skill)

    # create video reader and video writer
    video_info = sv.VideoInfo.from_video_path(video_path)
    with sv.VideoSink(output_path, video_info=video_info) as sink:
        # iterate over video frames
        for frame in tqdm.tqdm(sv.get_video_frames_generator(video_path)):
            
            # take in a frame and make the next prediction
            prob_step = model(frame)
            # draw the prediction (could be your bar chart) on the frame
            output_frame = draw_step_prediction(frame, prob_step.cpu().numpy().tolist(), model.STEPS)
            sink.write_frame(output_frame)


def draw_step_prediction(frame, prob_step, step_labels):
    # reuse your opencv bar chart code?
    print([
        f'{label} ({p})'
        for p, label in sorted(zip(prob_step, step_labels))
    ])
    draw_text_list(frame, [
        f'{label} ({p})'
        for p, label in sorted(zip(prob_step, step_labels))
    ])
    return frame


def draw_text_list(img, texts, i=0, tl=(10, 20), scale=0.4, space=50, color=(255, 255, 255), thickness=1):
    for i, txt in enumerate(texts, i+1):
        cv2.putText(
            img, txt, (int(tl[0]), int(tl[1]+scale*space*(i-1))), 
            cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    return img, i


if __name__ == '__main__':
    import fire
    fire.Fire(main)