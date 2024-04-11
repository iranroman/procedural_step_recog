
## Dataset structrue

```
/path/to/dataset
    training_video_info.csv
    training_video_names_shuffled.txt
    |_ procedure_descriptions
        |_ task1.txt
        |_ task2.txt
        |_ task3.txt
        ...
    |_ training_videos
        |_ T1-1
            |_ T1_1.mp4
            |_ T1_1_step_labels_by_frame.txt
            |_ frames
                |_frame_0000000001.png
                |_frame_0000000002.png
                |_frame_0000000003.png

        |_ T2-1
        ...
    |_ testing_videos
        ...
```

where each `procedure_description/taskX.txt` file looks like:

```
Procedure Name

1. Step 1 is lorem ipsum
2. Step 2 is ipsum lorem
...
```

and the `training_videos/T1-1/T1_1_step_labels_by_frame.txt` looks like (first value is start frame, second is end frame):

```
276     784     Step 1.
337     556     Step 2.
...
```

`training_video_info.csv` has the columns `video_id,video_fps,step_labels_by_frame,video_duration,total_number_of_frames` where `video_fps` and `video_duration` are floating point, `step_labels_by_frame` is a boolean indicating whether there exist step labels for the video.

`training_video_names_shuffled.txt` is a file where each line is a video name, but the video names appear in shuffled order, which is convenient to consistently define video splits for cross-validation.

## Video Frame Extraction

If you need to extract the frames, use this command (inside `training_videos`, for example):

```
find . -type f -name "*.mp4" -exec sh -c '
  for video in "$@"; do
    dirname=$(dirname "$video")
    mkdir -p "$dirname/frames"
    ffmpeg -i "$video" -vf "scale=-1:224" "$dirname/frames/frame_%010d.jpg"
  done
' sh {} +
```
