import logging
import os
import tempfile
import datetime
from typing import List, Dict, Any

import cv2
import dtlpy as dl
from skimage.metrics import structural_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from scipy.spatial.distance import cosine
import numpy as np

logger = logging.getLogger('video-utils.video_to_frames')


class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self):
        self.embedder = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        self.split_type = None
        self.dl_output_folder = None
        self.splitter_arg = None
        self.threshold = None
        self.min_interval = None
        self.window_size = None

    def get_new_items_metadata(self, item: dl.Item) -> Dict[str, Any]:
        """
        Get metadata for new frame items.

        Args:
            item: Source video item

        Returns:
            Dict containing metadata for new items
        """
        received_org_name = item.metadata.get("origin_video_name", None)
        received_time = item.metadata.get("time", None)
        origin_video_name = os.path.basename(item.filename) if received_org_name is None else received_org_name
        time = datetime.datetime.now().isoformat() if received_time is None else received_time
        return {"origin_video_name": origin_video_name, "time": time}

    def get_embedding(self, frame: np.ndarray) -> np.ndarray:
        """
        Get embedding vector for a frame using ResNet50

        Args:
            frame: Input video frame

        Returns:
            numpy array: Frame embedding vector
        """
        # Resize and preprocess the frame
        img = cv2.resize(frame, (224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        # Get the embedding
        return self.embedder.predict(img).flatten()

    def embedding_similarity_sampling(self, cap: cv2.VideoCapture, fps: int) -> List[int]:
        """
        Sample frames based on embedding similarity using cosine distance.

        Args:
            cap: OpenCV video capture object
            fps: Video frames per second

        Returns:
            list: Frame indices that meet similarity threshold
        """
        logger.info(f"embedding_similarity_sampling: {self.threshold}, {self.min_interval}")
        frames_list = []
        frame_count = 0
        prev_embedding = None
        min_interval = int(fps * self.min_interval)  # convert seconds to frames
        i_saved = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            if (frame_count - i_saved) < min_interval:
                frame_count += 1
                continue

            if prev_embedding is None:
                prev_embedding = self.get_embedding(frame)
                frames_list.append(frame_count)
                i_saved = frame_count
                frame_count += 1
                continue

            embedding = self.get_embedding(frame)
            distance = cosine(prev_embedding, embedding)

            if distance > self.threshold:
                frames_list.append(frame_count)
                i_saved = frame_count
                prev_embedding = embedding

            frame_count += 1

        return frames_list

    def structural_similarity_sampling(self, cap: cv2.VideoCapture, fps: int) -> List[int]:
        """
        Sample frames based on structural similarity between consecutive frames.

        Args:
            cap: OpenCV video capture object
            fps: Video frames per second

        Returns:
            list: Frame indices that meet similarity threshold
        """
        logger.info(f"structural_similarity_sampling: {self.window_size}, {self.threshold}, {self.min_interval}")

        frames_list = []
        frame_count = 0
        reference = None
        min_interval = int(fps * self.min_interval)  # convert seconds to frames
        i_saved = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            if (frame_count - i_saved) < min_interval:
                frame_count += 1
                continue

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if reference is None:
                reference = frame_gray
                frames_list.append(frame_count)
                i_saved = frame_count
                frame_count += 1
                continue
            ssim = structural_similarity(reference, frame_gray, win_size=self.window_size)
            # Convert SSIM to distance measure (1 - SSIM) so higher values mean more different
            ssim_distance = 1 - ssim
            if ssim_distance > self.threshold:
                frames_list.append(frame_count)
                reference = frame_gray
                i_saved = frame_count
            frame_count += 1
        return frames_list

    def get_frames_list(self, cap: cv2.VideoCapture) -> List[int]:
        """
        Get list of frame indices to extract based on split type.

        Args:
            cap: OpenCV video capture object

        Returns:
            list: Frame indices to extract
        """
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        logger.info(f"split_type: {self.split_type}")
        if self.split_type == 'frames_interval':
            logger.info(f"frames_interval: {self.splitter_arg}")
            return list(range(0, total_frames, self.splitter_arg))

        if self.split_type == 'time_interval':
            logger.info(f"time_interval: {self.splitter_arg}")
            return list(range(0, total_frames, int(fps * self.splitter_arg)))

        if self.split_type == 'num_splits':
            logger.info(f"num_splits: {self.splitter_arg}")
            if self.splitter_arg <= 1:
                # If num_splits is 1 or less, just return the first frame
                return [0]
            if total_frames <= 1:
                # If video has 0 or 1 frames, return all frames
                return list(range(total_frames))

            # Calculate divisor ensuring it's never zero
            divisor = max(1, (self.splitter_arg - 1) - (0 if total_frames % (self.splitter_arg - 1) else 1))
            frames_interval = total_frames // divisor
            logger.info(f"frames_interval: {frames_interval}")
            return list(range(0, total_frames, frames_interval))

        if self.split_type == 'embedding_similarity_sampling':
            return self.embedding_similarity_sampling(cap, fps)
        if self.split_type == 'structural_similarity_sampling':
            return self.structural_similarity_sampling(cap, fps)

        raise ValueError(f"Invalid split type: {self.split_type}")

    def upload_frames(
        self, item: dl.Item, frames_list: List[int], cap: cv2.VideoCapture, temp_dir: str
    ) -> List[dl.Item]:
        """
        Upload extracted frames as new items with annotations.

        Args:
            item: Input video item
            frames_list: List of frame indices to extract
            cap: OpenCV video capture object
            temp_dir: Temporary directory path for storing frames

        Returns:
            List of uploaded frame items
        """
        if len(frames_list) == 0:
            return
        num_digits = len(str(max(frames_list)))
        item_dataset = item.dataset
        annotations = item.annotations.list()
        logger.info(f"frames_dir: {self.dl_output_folder}")
        frames_dir = os.path.join(temp_dir, os.path.basename(self.dl_output_folder.rstrip('/')))
        os.makedirs(frames_dir)
        for frame_idx in frames_list:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if not success:
                break
            frame_path = os.path.join(
                frames_dir,
                f"{os.path.splitext(os.path.basename(item.filename))[0]}_{str(frame_idx).zfill(num_digits)}.jpg",
            )
            cv2.imwrite(frame_path, frame)

        logger.info(f"uploading frames to {self.dl_output_folder}")
        # batch upload
        frames_items_generator = item_dataset.items.upload(
            local_path=frames_dir,
            remote_path="/" + os.path.dirname(self.dl_output_folder.rstrip('/')).lstrip('/'),
            item_metadata=self.get_new_items_metadata(item),
        )
        frames_items_list = sorted(list(frames_items_generator), key=lambda x: x.name)

        # add index to frame items
        for i, frame_item in enumerate(frames_items_list):
            frame_item.metadata["splliting_frame_index"] = i
            frame_item.update()

        logger.info("upload frames annotations")
        if annotations:
            for frame_item in frames_items_list:
                frame_idx = int(frame_item.name.split('_')[-1].split('.')[0])
                frame_annotation = annotations.get_frame(frame_num=frame_idx)
                builder = frame_item.annotations.builder()
                for ann in frame_annotation.annotations:
                    if ann.object_visible:
                        builder.add(
                            annotation_definition=dl.Box(
                                top=ann.top, left=ann.left, bottom=ann.bottom, right=ann.right, label=ann.label
                            )
                        )
                frame_item.annotations.upload(builder)
        return frames_items_list

    def video_to_frames(self, item: dl.Item, context: dl.Context) -> List[dl.Item]:
        """
        Split video into frames based on configured split type and parameters.

        Args:
            item (dl.Item): Input video item
            context (dl.Context): Pipeline context containing node configuration

        Returns:
            List[dl.Item]: List of extracted frames
        """
        cap = None
        items = []
        try:
            logger.info('Running service Video To Frames')

            node = context.node
            self.split_type = node.metadata['customNodeConfig'].get('split_type', 'structural_similarity_sampling')
            self.dl_output_folder = node.metadata['customNodeConfig']['output_dir']
            logger.info(f"split_type: {self.split_type}")
            if self.split_type not in {'structural_similarity_sampling', 'embedding_similarity_sampling'}:
                self.splitter_arg = node.metadata['customNodeConfig']['splitter_arg']
            else:
                self.threshold = node.metadata['customNodeConfig']['threshold']
                self.min_interval = node.metadata['customNodeConfig']['min_interval']
                if self.split_type == 'structural_similarity_sampling':
                    self.window_size = node.metadata['customNodeConfig']['window_size']

            temp_dir = tempfile.mkdtemp()
            logger.info(f"start downloading video to tmp dir {temp_dir}")
            input_video = item.download(local_path=temp_dir)
            cap = cv2.VideoCapture(input_video)
            if not cap.isOpened():
                raise RuntimeError("Failed to open video file")

            frames_list = self.get_frames_list(cap)
            logger.info(f"frames_list: {frames_list}")
            items = self.upload_frames(item, frames_list, cap, temp_dir)

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise
        finally:
            if cap is not None:
                cap.release()

        return items
