
## Dataset structrue

```
/path/to/dataset
    |_ procedure_descriptions
        |_ task1.txt
        |_ task2.txt
        |_ task3.txt
        ...
    |_ training_videos
        |_ TR-1
            |_ TR_1.mp4
            |_ TR_1_step_labels_by_frame.txt
            |_ frames
                |_TR_1_frame_0000000001.png
                |_TR_1_frame_0000000002.png
                |_TR_1_frame_0000000003.png

```

where each `procedure_description/taskX.txt` file looks like:

```
Procedure Name

1. Step 1 is lorem ipsum
2. Step 2 is ipsum lorem
...
```

and the `training_videos/TR-1/TR_1_step_labels_by_frame.txt` looks like (first value is start frame, second is end frame):

```
276     784     Step 1.
337     556     Step 2.
...
```

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
