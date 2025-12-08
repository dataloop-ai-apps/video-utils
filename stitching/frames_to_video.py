import datetime
import logging
import os
import tempfile
from typing import List

import cv2
import dtlpy as dl
import numpy as np
from stitching.trackers_adapters import ByteTrackTracker, TrackerConfig

logger = logging.getLogger('video-utils.frames_to_vid')


class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self):
        self.fps: int = 0
        self.dl_output_dir: str = ""
        self.output_video_type: str = ""
        self.dl_input_dir: str = ""
        self.tracker = None
        self.dataset = None
        self.trackers_config = None

    def set_config_params(self, node: dl.PipelineNode) -> None:
        """
        Sets configuration parameters from pipeline node metadata.

        Args:
            node: Pipeline node containing configuration
        """
        self.fps = node.metadata['customNodeConfig']['fps']
        self.dl_output_dir = node.metadata['customNodeConfig']['output_dir']
        self.output_video_type = node.metadata['customNodeConfig']['output_video_type']
        self.dl_input_dir = node.metadata['customNodeConfig']['input_dir']
        self.trackers_config = TrackerConfig(
            min_box_area=node.metadata['customNodeConfig']['min_box_area'],
            track_thresh=node.metadata['customNodeConfig']['track_thresh'],
            track_buffer=node.metadata['customNodeConfig']['track_buffer'],
            match_thresh=node.metadata['customNodeConfig']['match_thresh'],
        )
        logger.info(f"customNodeConfig: {node.metadata['customNodeConfig']}")

    def get_input_items(self, item: dl.Item) -> List[dl.Item]:
        """
        Gets list of input frame items from specified directory.

        Args:
            item (dl.Item): Reference item used to determine dataset and directory

        Returns:
            List[dl.Item]: List of frame items sorted by name
        """
        logger.info(f"input_dir: {self.dl_input_dir}")
        input_dir = self.dl_input_dir if self.dl_input_dir != "" else os.path.dirname(item.filename)
        filters = dl.Filters(field='dir', values="/" + input_dir.lstrip('/'))
        # if origin video name is set on the received item, then use it to filter the frames
        original_video_name = item.metadata.get('origin_video_name', None)
        if original_video_name is not None:
            filters.add(field='metadata.origin_video_name', values=original_video_name)
        # if created_time is set on the received item, then use it to filter the frames
        created_time = item.metadata.get('created_time', None)
        if created_time is not None:
            filters.add(field='metadata.created_time', values=created_time)
        items = self.dataset.items.get_all_items(filters=filters)
        logger.info(f"get_input_items number of items: {len(items)}")
        if not items or len(items) == 0:
            logger.error("No images found in specified directory")
            return []
        
        return sorted(items, key=lambda x: x.name)

    def stitch_and_upload(self, cv_frames: List[np.ndarray], local_output_folder: str) -> dl.Item:
        """
        Stitches frames into a video and uploads it to the dataset.

        Args:
            cv_frames: List of OpenCV frames to stitch together
            local_output_folder (str): Local directory to save the output video

        Returns:
            dl.Item: The uploaded video item
        """
        output_video_path = os.path.join(
            local_output_folder,
            f"merge_{datetime.datetime.now().isoformat().replace('.', '').replace(':', '_')}.{self.output_video_type}",
        )
        logger.info(f"Stitching frames into video at {output_video_path}")
        fourcc = cv2.VideoWriter_fourcc(*("VP80" if self.output_video_type.lower() == "webm" else "mp4v"))
        writer = cv2.VideoWriter(output_video_path, fourcc, self.fps, (cv_frames[0].shape[1], cv_frames[0].shape[0]))
        try:
            # Loop through each input image file and write it to the output video
            logger.info("Writing frames to video")
            for frame in cv_frames:
                writer.write(frame)

            # Release the VideoWriter object
            writer.release()
            logger.info("Uploading video to dataset")
            video_item = self.dataset.items.upload(
                local_path=output_video_path, remote_path="/" + self.dl_output_dir.lstrip('/')
            )
            video_item.fps = self.fps
            video_item.update()
            return video_item
        finally:
            # Ensure VideoWriter is released even if an error occurs
            if writer is not None:
                writer.release()

    def frames_to_vid(self, item: dl.Item, context: dl.Context) -> dl.Item:
        """
        Converts a sequence of frames into a video with optional object tracking.

        Args:
            item (dl.Item): The Dataloop item to process
            context (dl.Context): Pipeline context containing configuration

        Returns:
            dl.Item: The uploaded video item
        """

        logger.info('Running service Frames To Video - Demo1')

        self.set_config_params(context.node)
        self.dataset = item.dataset
        logger.info(f"dataset: {self.dataset.name}")

        local_input_folder = tempfile.mkdtemp(suffix="_input")
        local_output_folder = tempfile.mkdtemp(suffix="_output")

        items = self.get_input_items(item)
        self.dataset.items.download(local_path=local_input_folder, items=items)
        images_files = [os.path.join(local_input_folder, "items", item.filename.lstrip("/")) for item in items]
        logger.info("convert to cv frames")
        cv_frames = [cv2.imread(img_path) for img_path in images_files]
        video_item = self.stitch_and_upload(cv_frames, local_output_folder)
        builder = video_item.annotations.builder()
        self.tracker = ByteTrackTracker(
            annotations_builder=builder, frame_rate=self.fps, config=self.trackers_config
        )
        logger.info("Tracking frames")
        for i, (frame_i, item_i) in enumerate(zip(cv_frames, items)):
            frame_annotations = item_i.annotations.list().annotations
            self.tracker.update(frame_i, i, frame_annotations)
        logger.info("Uploading annotations to video")
        video_item.annotations.upload(annotations=builder)
        return video_item
