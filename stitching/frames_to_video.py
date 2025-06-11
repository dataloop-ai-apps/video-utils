import datetime
import logging
import os
import tempfile
from typing import List

import cv2
import dtlpy as dl
import numpy as np
from dotenv import load_dotenv
from trackers_adapters import ByteTrackTracker, DeepSORTTracker

logger = logging.getLogger('video-utils.frames_to_vid')


class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self):
        self.fps: int = 0
        self.dl_output_dir: str = ""
        self.output_video_type: str = ""
        self.dl_input_dir: str = ""
        self.trackerName: str = ""
        self.tracker = None
        self.local_input_folder: str = ""
        self.local_output_folder: str = ""
        self.dataset = None

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
        self.trackerName = node.metadata['customNodeConfig']['tracker']
        logger.info(f"customNodeConfig: {node.metadata['customNodeConfig']}")

    def get_input_items(self, items: List[dl.Item]) -> List[dl.Item]:
        """
        Gets input items either from provided list or from remote directory.
        If input_dir is specified, fetches items from that remote directory.
        Otherwise uses the provided items list.

        Args:
            items: List of input items

        Returns:
            List[dl.Item]: Filtered and sorted list of items
        """
        items = sorted(items, key=lambda x: x.name)
        logger.info(f"received items length: {len(items)}")
        print(f"received items length: {len(items)}")
        if self.dl_input_dir is not None and self.dl_input_dir.strip():
            logger.info(f"input_dir: {self.dl_input_dir}")
            print(f"input_dir: {self.dl_input_dir}")
            filters = dl.Filters(field='dir', values="/" + self.dl_input_dir)
            filters.sort_by(field='name')
            items = self.dataset.items.get_all_items(filters=filters)
            logger.info(f"get_input_items number of items: {len(items)}")
            print(f"get_input_items number of items: {len(items)}")
        if not items or len(items) == 0:
            logger.error("No images match to merge")
            return []
        # TODO : check if there a batch download
        logger.info(f"get_input_items number of items: {len(items)}")
        return items

    def stitch_and_upload(self, cv_frames: List[np.ndarray]) -> dl.Item:
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
        try:
            # Loop through each input image file and write it to the output video
            logger.info(f"Writing frames to video")
            for frame in cv_frames:
                writer.write(frame)

            # Release the VideoWriter object
            writer.release()
            logger.info(f"Uploading video to dataset")
            video_item = self.dataset.items.upload(local_path=output_video_path, remote_path="/" + self.dl_output_dir)
            video_item.fps = self.fps
            video_item.update()
            return video_item
        finally:
            # Ensure VideoWriter is released even if an error occurs
            if writer is not None:
                writer.release()

    def frames_to_vid(self, items: List[dl.Item], context: dl.Context) -> dl.Item:
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
        self.dataset = items[0].dataset
        logger.info(f"dataset: {self.dataset.name}")
        self.local_input_folder = tempfile.mkdtemp(suffix="_input")
        self.local_output_folder = tempfile.mkdtemp(suffix="_output")

        items = self.get_input_items(items)
        # cv_frames = [cv2.imread(item.download(local_path=self.local_input_folder)) for item in items]
        images_files = self.dataset.items.download(local_path=self.local_input_folder, items=items)
        images_files = sorted(images_files, reverse=False)  # Sort filenames in descending order
        cv_frames = [cv2.imread(img_path) for img_path in images_files]
        video_item = self.stitch_and_upload(cv_frames)
        builder = video_item.annotations.builder()
        if self.trackerName == "ByteTrack":
            self.tracker = ByteTrackTracker(annotations_builder=builder, frame_rate=self.fps)
        elif self.trackerName == "DeepSORT":
            self.tracker = DeepSORTTracker(annotations_builder=builder)
        else:
            raise ValueError(f"Invalid tracker name: {self.trackerName}")
        logger.info("Tracking frames")
        for i, (frame_i, item_i) in enumerate(zip(cv_frames, items)):
            frame_annotations = item_i.annotations.list().annotations
            self.tracker.update(frame_i, i, frame_annotations)
        logger.info("Uploading annotations to video")
        video_item.annotations.upload(annotations=builder)
        return video_item


if __name__ == "__main__":
    if dl.token_expired():
        dl.login()

    load_dotenv()
    api_key = os.getenv('DATALOOP_API_KEY')
    dl.login_api_key(api_key=api_key)
    runner = ServiceRunner()
    context = dl.Context()
    context.pipeline_id = "68483366590d2be8798fdf40"
    context.node_id = "73683df0-43f2-4347-81b4-2ca97ea5c3f8"
    context.node.metadata["customNodeConfig"] = {
        "fps": 20,
        "output_dir": "tmp_22332887",
        "input_dir": "split_5_sec_to_one_frame",
        "output_video_type": "webm",
        "tracker": "DeepSORT",
    }

    # context.node.metadata["customNodeConfig"] = {"window_size": 7, "threshold": 0.13, "output_dir": "/testing_238"}
    runner.frames_to_vid(items=[dl.items.get(item_id="682a25d33ec9c74d876a4550")], context=context)
