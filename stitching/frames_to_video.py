import datetime
import logging
import os
import tempfile
from typing import List

import cv2
import dtlpy as dl
import numpy as np
from dotenv import load_dotenv
from trackings.utils import load_opt
from trackers import ByteTrackTracker, BoTSORTTracker, DeepSORTTracker

logger = logging.getLogger('video-utils.frames_to_vid')


class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self):
        self.fps = None
        self.output_dir = None
        self.output_video_type = None
        self.input_dir = None
        self.trackerName = None
        self.tracker = None

    def set_config_params(self, node: dl.PipelineNode) -> None:
        """
        Sets configuration parameters from pipeline node metadata.

        Args:
            node: Pipeline node containing configuration
        """
        self.fps = node.metadata['customNodeConfig']['fps']
        self.output_dir = node.metadata['customNodeConfig']['output_dir']
        self.output_video_type = node.metadata['customNodeConfig']['output_video_type']
        self.input_dir = node.metadata['customNodeConfig']['input_dir']
        self.trackerName = node.metadata['customNodeConfig']['tracker']
        logger.info(f"customNodeConfig: {node.metadata['customNodeConfig']}")

    def get_input_items(self, dataset: dl.Dataset) -> List[dl.Item]:
        """
        Gets all items from dataset matching input directory path.

        Args:
            dataset: Dataloop dataset to get items from

        Returns:
            list: List of dataset items matching input path
        """
        filters = dl.Filters(field='dir', values=self.input_dir)
        filters.sort_by(field='name')
        items = dataset.items.get_all_items(filters=filters)
        if not items or len(items) == 0:
            logger.error("No images match to merge")
            return []
        # TODO : check if there a batch download
        logger.info(f"get_input_items number of items: {len(items)}")
        return items

    def stitch_and_upload(self, dataset: dl.Dataset, cv_frames: List[np.ndarray]) -> dl.Item:
        """
        Stitches frames into a video and uploads it to the dataset.

        Args:
            dataset: Dataloop dataset to upload video to
            cv_frames: List of OpenCV frames to stitch together

        Returns:
            dl.Item: The uploaded video item
        """
        output_video_path = os.path.join(
            self.local_output_folder,
            f"merge_{datetime.datetime.now().isoformat().replace('.', '').replace(':', '_')}.{self.output_video_type}",
        )
        logger.info(f"Stitching frames into video at {output_video_path}")
        fourcc = cv2.VideoWriter_fourcc(*("VP80" if self.output_video_type.lower() == "webm" else "mp4v"))
        writer = cv2.VideoWriter(output_video_path, fourcc, self.fps, (cv_frames[0].shape[1], cv_frames[0].shape[0]))
        # Loop through each input image file and write it to the output video
        logger.info(f"Writing frames to video")
        for frame in cv_frames:
            writer.write(frame)

        # Release the VideoWriter object
        writer.release()
        logger.info(f"Uploading video to dataset")
        video_item = dataset.items.upload(local_path=output_video_path, remote_path=self.output_dir)
        video_item.fps = self.fps
        video_item.update()
        return video_item

    def frames_to_vid(self, item: dl.Item, context: dl.Context) -> None:
        """
        Converts a sequence of frames into a video with optional object tracking.

        Args:
            item (dl.Item): The Dataloop item to process
            context (dl.Context): Pipeline context containing configuration

        Returns:
            None
        """

        logger.info('Running service Frames To Video')

        self.set_config_params(context.node)

        self.local_input_folder = tempfile.mkdtemp(suffix="_input")
        self.local_output_folder = tempfile.mkdtemp(suffix="_output")

        items = self.get_input_items(item.dataset)
        cv_frames = [cv2.imread(item.download(local_path=self.local_input_folder)) for item in items]
        video_item = self.stitch_and_upload(item.dataset, cv_frames)
        builder = video_item.annotations.builder()
        if self.trackerName == "ByteTrack":
            self.tracker = ByteTrackTracker(opts=load_opt(), annotations_builder=builder)
        elif self.trackerName == "DeepSORT":
            self.tracker = DeepSORTTracker(opts=load_opt(), annotations_builder=builder)
        elif self.trackerName == "BoTSORT":
            self.tracker = BoTSORTTracker(opts=load_opt(), annotations_builder=builder)
        logger.info("Tracking frames")
        for i, (frame_i, item_i) in enumerate(zip(cv_frames, items)):
            frame_annotations = item_i.annotations.list().annotations
            self.tracker.update(frame_i, i, frame_annotations)
        logger.info("Uploading annotations to video")
        video_item.annotations.upload(annotations=builder)


if __name__ == "__main__":
    if dl.token_expired():
        dl.login()

    load_dotenv()
    api_key = os.getenv('DATALOOP_API_KEY')
    dl.login_api_key(api_key=api_key)
    runner = ServiceRunner()
    context = dl.Context()
    context.pipeline_id = "682069122afb795bc3c41d59"
    context.node_id = "bd1dc151-6067-4197-85aa-1b65394e2077"
    context.node.metadata["customNodeConfig"] = {
        "fps": 20,
        "output_dir": "/tmp_stitching_frames_to_video_deep_22",
        "input_dir": "/split_5_sec_to_one_frame",
        "output_video_type": "webm",
        "tracker": "DeepSORT",
    }

    # context.node.metadata["customNodeConfig"] = {"window_size": 7, "threshold": 0.13, "output_dir": "/testing_238"}
    runner.frames_to_vid(item=dl.items.get(item_id="682a250fb188d7f0a74d53ed"), context=context)
