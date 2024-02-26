# Video Utils

The Repository contains 2 main application:
1. `[Video Utils] - Stitching` 
2. `[Video Utils] - Trimming`

## [Video Utils] - Stitching

The Application provides the following stitching functions:
1. `frames_to_vid` - to connect frames annotations to the original video item (using IoU logic).
2. `videos_to_video` - to connect videos segments annotations to the original video item (using IoU logic).


## [Video Utils] - Trimming

The Application provides the following trimming functions:
1. `video_to_frames` - to split the original video item to frames and clone the annotations to them from the original video.
2. `video_to_videos` - to split the original video item to video segments and clone the annotations to them from the original video.
3. `video_to_frames_smart_subsampling` - to split the original video to segments using `structural_similarity` logic 
   from `skimage` library. 

## Requirements

`dtlpy` \
`opencv-python`
