# procedural_step_recog

Implements (and trains) a GRU model that can process representations of human actions to infer the current step in a procedure (i.e. step in a recipe, or in a medical procedure).

## Steps to follow:
1. extract the video frames:
    ```
    bash scripts/extract_frames.sh "/path/to/videos" "/path/to/output/rgb_frames"
    ```
    example: `bash scripts/extract_frames.sh "/Users/iranroman/datasets/M2_Tourniquet/Data" "Users/iranroman/datasets/BBN_0p52/M2_Tourniquet/rgb_frames"`

2. extract the video embeddings using Omnivore:
