import dtlpy as dl
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from PIL import Image as PILImage
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from scipy.spatial.distance import cosine
import torch
import numpy as np
import cv2
from tqdm import tqdm
import os
import torch.nn.functional as F
import time
import shutil
import tempfile

logger = logging.getLogger("[VideoFrameExtractor]")


class VideoFrameExtractor(dl.BaseServiceRunner):
    """A service for extracting representative frames from videos using ResNet50 embeddings.

    This class provides functionality to:
    1. Extract key frames from videos based on content similarity
    2. Generate embeddings for frames using ResNet50
    3. Save extracted frames to disk

    Attributes:
        device (torch.device): The device (CPU/GPU) used for processing
        stream (bool): Flag indicating if streaming mode is enabled
        embedder (ResNet50): Pre-trained ResNet50 model for generating frame embeddings
    """

    def __init__(self):
        """Initialize the VideoFrameExtractor with ResNet50 model and device setup."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing VideoFrameExtractor using device: {self.device}")
        self.stream = False
        self.embedder = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        logger.info("Successfully loaded ResNet50 model")

    def extract_only_frames(self, video_path: str) -> str:
        """Extract frames from a video file.

        Args:
            video_path (str): Path to the input video file

        Returns:
            str: Path to the output video file
        """
        name, _ = os.path.splitext((os.path.basename(video_path)))
        output_folder = f"output/{name}"
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Output folder: {output_folder}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video properties - FPS: {fps}, Total frames: {n_frames}")

        frame_count = 0

        with tqdm(total=n_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                frame_count += 1
                if not ret:
                    break

                filename = f"{output_folder}/frame_{frame_count:04d}.jpg"
                cv2.imwrite(filename, frame)
                logger.debug(f"Saved frame {frame_count}")

                pbar.update(1)

        cap.release()
        return output_folder

    def extract_video_frames(self, item: dl.Item, context: dl.Context) -> List[dl.Item]:
        """Extract representative frames from a video item and combine them into a new video.

        Args:
            item (dl.Item): The input video item from Dataloop platform

        Returns:
            List[dl.Item]: The output video item containing the representative frames

        Note:
            - Frames are saved to an 'output' directory with the video name as a subdirectory
            - Output video maintains the same format and FPS as input video
        """
        node_context = context.node
        upload_format = node_context.metadata.get('customNodeConfig', dict()).get('uploadFormat', "frames") # video or frames
        use_subsampling = node_context.metadata.get('customNodeConfig', dict()).get('useSubsampling', True)

        frame_items = []
        to_upload = None
        logger.info(f"Starting frame extraction for video item: {item.name}")

        # Download video file
        video_filepath = item.download()

        # Get video properties
        cap = cv2.VideoCapture(video_filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        cap.release()

        start_time = time.time()
        
        # if use_subsampling is False, extract all frames
        if use_subsampling is False:
            frames_path = self.extract_only_frames(video_path=video_filepath)
            remote_path = f"{os.path.splitext(item.filename)[0]}/{upload_format}"
            output_item = item.dataset.items.upload(local_path=frames_path, remote_path=remote_path)
            logger.info(f"Uploaded {len(output_item)} frames to {remote_path}")
        
        # if use_subsampling is True, subsample the video
        else:
            frames = self.subsample_video(video_path=video_filepath, threshold=0.17)
            logger.info(f"Video subsampling completed in {time.time() - start_time:.2f} seconds")

            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"Using temporary directory: {temp_dir}")

                # Create video writer
                name, ext = os.path.splitext(os.path.basename(video_filepath))
                output_video = os.path.join(temp_dir, f"summarized_{name}{ext}")
                out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

                start_time = time.time()
                # Sort frames by index to maintain temporal order
                sorted_frames = sorted(frames.items(), key=lambda x: x[0])

                for i_frame, frame in sorted_frames:
                    try:
                        logger.debug(f"Processing frame {i_frame}")
                        filename = os.path.join(temp_dir, f"frame_{i_frame:04d}.jpg")
                        cv2.imwrite(filename, frame)
                        frame_items.append(filename)
                        # Write frame to video
                        out.write(frame)
                        logger.debug(f"Frame {i_frame} saved at {filename}")
                    except Exception as e:
                        logger.error(f"Failed processing frame {i_frame}: {str(e)}")
                        continue

                out.release()

                if upload_format == "video":
                    # Check if the output video file was created
                    if not os.path.exists(output_video):
                        logger.error(f"Output video file not found: {output_video}")
                        output_item = None

                    logger.info(f"Processed {len(frame_items)} frames successfully")
                    logger.info(f"Created summarized video at {output_video}")

                    # Upload the summarized video to platform
                    to_upload = output_video
                else:
                    # Upload the frames to the platform
                    to_upload = frame_items
                
                remote_path = f"{os.path.splitext(item.filename)[0]}/{upload_format}"
                output_item = item.dataset.items.upload(local_path=to_upload, remote_path=remote_path)
            
        # Uploader returns generator or a single item, or None
        if output_item is None:
            output_item = list()
        elif isinstance(output_item, dl.Item):
            output_item = [output_item]
        else:
            output_item = [item for item in output_item]


        return output_item

    def get_embedding(self, frame: np.ndarray) -> np.ndarray:
        """Generate embedding for a video frame using ResNet50.

        Args:
            frame (np.ndarray): Input frame as a numpy array in BGR format

        Returns:
            np.ndarray: Flattened embedding vector for the frame
        """
        # Resize and preprocess the frame
        img = cv2.resize(frame, (224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Get the embedding
        return self.embedder.predict(img).flatten()

    def subsample_video(self, video_path: str, threshold: float) -> Dict[int, np.ndarray]:
        """Subsample video frames based on content similarity using ResNet50 embeddings.

        Args:
            video_path (str): Path to the input video file
            threshold (float): Cosine distance threshold for frame similarity

        Returns:
            Dict[int, np.ndarray]: Dictionary mapping frame indices to frame arrays

        Note:
            - Saves frames that differ significantly (based on threshold) from previous keyframe
            - Always includes first and last frames
            - Maintains minimum interval of 2 seconds between saved frames
        """
        # Open the video file
        name, _ = os.path.splitext((os.path.basename(video_path)))
        output_folder = f"output/{name}"
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Output folder: {output_folder}")

        cap = cv2.VideoCapture(video_path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Video properties - FPS: {fps}, Total frames: {n_frames}")

            # Read the first frame
            ret, prev_frame = cap.read()
            if not ret:
                logger.error("Failed to read the first frame")
                return {}

            prev_embedding = self.get_embedding(prev_frame)
            frame_count = 0
            i_saved = 0
            min_interval = int(fps * 2)  # once in 2 sec
            frames = dict()

            # Save first frame
            filename = f"{output_folder}/frame_{frame_count:04d}.jpg"
            cv2.imwrite(filename, prev_frame)
            frames[frame_count] = prev_frame
            logger.info("Saved first frame")

            with tqdm(total=n_frames, desc="Processing frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    frame_count += 1
                    if not ret:
                        break
                    if (frame_count - i_saved) < min_interval:
                        pbar.update(1)
                        continue

                    embedding = self.get_embedding(frame)
                    distance = cosine(prev_embedding, embedding)

                    if distance > threshold:
                        filename = f"{output_folder}/frame_{frame_count:04d}.jpg"
                        cv2.imwrite(filename, frame)
                        frames[frame_count] = frame
                        i_saved = frame_count
                        prev_embedding = embedding
                        logger.debug(f"Saved frame {frame_count} (distance: {distance:.3f})")

                    prev_frame = frame
                    pbar.update(1)

            # Save last frame
            frames[frame_count] = prev_frame
            logger.info(f"Processing completed. Total frames processed: {frame_count}, Frames saved: {len(frames)}")

            return frames

        finally:
            cap.release()
