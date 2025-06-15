# Video Utils pipeline nodes

The Repository contains 2 main application:
1. `[Video Utils] - Splitting` 
2. `[Video Utils] - Stitching`


## [Video Utils] - Splitting

The Application provides the following splitting functions:

1. `video_to_frames` - Splits a video into individual frames with the following options:
   - Split Types:
     - `frames_interval`: Extract frames at regular frame intervals (e.g., every N frames)
     - `time_interval`: Extract frames at regular time intervals (e.g., every N seconds)
     - `num_splits`: Split video into specified number of frames (evenly distributed)
   - Configuration Parameters:
     - `output_dir`: Directory where frames will be saved
     - `splitter_arg`: Value depends on split type:
       - For `frames_interval`: Number of frames between extractions
       - For `time_interval`: Time in seconds between extractions
       - For `num_splits`: Number of desired frame splits
   - Output Metadata:
     - `origin_video_name`: Original video filename
     - `created_time`: Timestamp of frame extraction

2. `smart-frames-splitting` - Splits a video into frames using similarity-based sampling:
   - Split Types:
     - `structural_similarity_sampling`: Uses structural similarity to detect significant changes between frames
     - `embedding_similarity_sampling`: Uses ResNet50 embeddings to detect semantic changes between frames
   - Configuration Parameters:
     - `output_dir`: Directory where frames will be saved
     - `min_interval`: Minimum time (in seconds) between selected frames (default: 10)
     - `threshold`: Similarity threshold for frame selection (default: 0.13)
     - `window_size`: Size of the sliding window for structural similarity (default: 7, range: 3-25)
   - Output Metadata:
     - `origin_video_name`: Original video filename
     - `created_time`: Timestamp of frame extraction

3. `video_to_videos` - Splits a video into multiple sub-videos with the following options:
   - Split Types:
     - `num_frames`: Split by number of frames per sub-video
     - `num_splits`: Split by number of desired sub-videos
     - `out_length`: Split by desired length of each sub-video in seconds
   - Configuration Parameters:
     - `output_dir`: Directory where sub-videos will be saved
     - `splitter_arg`: Value depends on split type:
       - For `num_frames`: Number of frames per sub-video
       - For `num_splits`: Number of desired sub-videos
       - For `out_length`: Length of each sub-video in seconds
     - `n_overlap`: Number of overlapping frames between consecutive sub-videos
   - Output Metadata:
     - `origin_video_name`: Original video filename with extension
     - `time`: Timestamp of video creation
     - `sub_videos_intervals`: List of frame intervals for each sub-video

All functions preserve the original video's annotations and metadata, creating new items with proper references to the source video. The metadata helps track the relationship between the original video and its derived items, which is particularly useful for stitching operations.


## [Video Utils] - Stitching

The Application provides the following stitching functions:

1. `frames_to_vid` - Merges frames into a video with the following options:
   - Configuration Parameters:
     - `output_video_type`: Output video format (webm or mp4)
     - `output_dir`: Directory where the merged video will be saved
     - `input_dir`: Directory containing the input frames (default: "input")
     - `fps`: Frames per second for the output video (default: 20)
     - `tracker`: Object tracking method:
       - `ByteTrack` (default): Fast and efficient tracking
       - `DeepSORT`: Deep learning-based tracking
     - Tracker Configuration:
       - `min_box_area`: Minimum area for tracked boxes (default: 10)
       - `track_thresh`: Minimum confidence score for detections (default: 0.5)
       - `track_buffer`: Max frames a track can go unmatched (default: 30)
       - `match_thresh`: Minimum similarity score to link detections (default: 0.8)
   - Item Filtering:
     - First filters items by the specified `input_dir` path
     - Then filters by metadata:
       - `origin_video_name`: Matches frames from the same original video
       - `created_time`: Matches frames created in the same batch
     - Items are sorted by name to ensure correct frame order

2. `videos_to_video` - Merges multiple videos into a single video with the following options:
   - Configuration Parameters:
     - `output_dir`: Directory where the merged video will be saved
     - `input_dir`: Directory containing the input videos (default: "input")
     - `tracker`: Object tracking method:
       - `ByteTrack` (default): Fast and efficient tracking
       - `DeepSORT`: Deep learning-based tracking
   - Item Filtering:
     - First filters items by the specified `input_dir` path
     - Then filters by metadata:
       - `origin_video_name`: Matches videos from the same original source
       - `time`: Matches videos created in the same batch
       - `sub_videos_intervals`: Used to determine if videos are from the same split
     - For videos from the same split:
       - Uses `sub_videos_intervals` to merge in the correct order
       - Preserves frame intervals and annotations
     - For unrelated videos:
       - Merges videos sequentially
       - Maintains original annotations

Both functions preserve the original annotations and can optionally track objects across frames using either ByteTrack or DeepSORT algorithms.


## Pipeline Template

