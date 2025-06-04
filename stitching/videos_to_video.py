import datetime
import os
import tempfile
import copy
import logging
from typing import List

import cv2
import dtlpy as dl
from dotenv import load_dotenv

from trackers_adapters import ByteTrackTracker, DeepSORTTracker

logger = logging.getLogger('video-utils.videos_to_video')


class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self):
        self.output_dir = None
        self.input_dir = None
        self.trackerName = None
        self.local_input_folder = None
        self.local_output_folder = None

    @staticmethod
    def clone_annotation(ann: dl.Annotation):
        """
        Clone an existing Annotation object, returning a new Annotation instance
        with the same content but no platform ID.
        """
        return ann.__class__.new(
            item=ann.item,
            annotation_definition=ann.annotation_definition,
            object_id=ann.object_id,
            automated=ann.automated,
            metadata=copy.deepcopy(ann.metadata),
            frame_num=ann.start_frame,
            parent_id=ann.parent_id,
            start_time=ann.start_time,
            item_height=ann.item_height,
            item_width=ann.item_width,
            end_time=ann.end_time,
        )

    @staticmethod
    def is_items_from_same_split(items):
        """
        Check if all items are from the same video split.

        Args:
            items: List of Dataloop items to check

        Returns:
            bool: True if all items are from same split, False otherwise
        """
        if not all(
            key in item.metadata for item in items for key in ('origin_video_name', 'sub_videos_intervals', 'time')
        ):
            return False
        original_name = items[0].metadata["origin_video_name"]
        sub_videos_intervals = items[0].metadata["sub_videos_intervals"]
        time = items[0].metadata["time"]
        for item in items:
            if (
                item.metadata["origin_video_name"] != original_name
                or item.metadata["sub_videos_intervals"] != sub_videos_intervals
                or item.metadata["time"] != time
            ):
                return False
        return True

    def merge_by_sub_videos_intervals(self, writer, input_files, sub_videos_intervals, items):
        """
        Merges multiple sub-videos based on frame intervals.

        Args:
            writer: VideoWriter object for output video
            input_files: List of input video file paths
            sub_videos_intervals: List of frame intervals for each sub-video
            items: List of Dataloop items corresponding to input videos

        Returns:
            Tuple containing merged annotations and frames
        """
        merged_video_frames = []
        merged_video_annotations = []
        total_frames_count = sub_videos_intervals[-1][1] + 1
        # Loop through each input video file and write its frames to the output video
        for i, (input_file, item) in enumerate(zip(input_files, items)):
            annotations = item.annotations.list()
            next_interval_start_frame = (
                sub_videos_intervals[i + 1][0] if i < len(sub_videos_intervals) - 1 else total_frames_count
            )
            start_frame, end_frame = sub_videos_intervals[i]
            logger.info(
                f"input file index {i}, "
                f"start frame {start_frame}, "
                f"end frame {end_frame}, "
                f"next interval start frame {next_interval_start_frame}"
            )

            # Open the input video file
            cap = cv2.VideoCapture(input_file)

            for frame_index, j in enumerate(range(start_frame, next_interval_start_frame)):
                frame_annotations = annotations.get_frame(frame_num=frame_index).annotations
                merged_video_annotations.append([ServiceRunner.clone_annotation(ann) for ann in frame_annotations])
                ret, frame = cap.read()
                if ret:
                    merged_video_frames.append(frame)
                    writer.write(frame)
                else:
                    break

            # Release the input video file
            cap.release()

        return merged_video_annotations, merged_video_frames

    def regular_merge(self, writer, input_files, items):
        """
        Merges videos sequentially and returns merged annotations and frames.

        Args:
            writer: VideoWriter object for output video
            input_files: List of input video file paths
            items: List of Dataloop items corresponding to input videos

        Returns:
            Tuple containing merged annotations and frames
        """
        merged_video_annotations = []
        merged_video_frames = []
        # Loop through each input video file and write its frames to the output video
        for i, (input_file, item) in enumerate(zip(input_files, items)):
            annotations = item.annotations.list()
            cap = cv2.VideoCapture(input_file)
            frame_index = 0
            ret, frame = cap.read()
            while ret:
                frame_annotations = annotations.get_frame(frame_num=frame_index).annotations
                merged_video_annotations.append([ServiceRunner.clone_annotation(ann) for ann in frame_annotations])
                merged_video_frames.append(frame)
                writer.write(frame)
                ret, frame = cap.read()
                frame_index += 1
            logger.info(f"regular merge input file {i} , number of frames {frame_index}")
            # Release the input video file
            cap.release()

        return merged_video_annotations, merged_video_frames

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
        if self.input_dir is not None and self.input_dir.strip():
            dataset = items[0].dataset
            filters = dl.Filters(field='dir', values=self.input_dir)
            filters.sort_by(field='name')
            items = dataset.items.get_all_items(filters=filters)
        if not items or len(items) == 0:
            logger.error("No images match to merge")
            return []
        # TODO : check if there a batch download
        logger.info(f"get_input_items number of items: {len(items)}")
        return items

    def set_config_params(self, node: dl.PipelineNode):
        """
        Sets configuration parameters from pipeline node metadata.

        Args:
            node: Pipeline node containing configuration
        """
        self.output_dir = node.metadata['customNodeConfig']['output_dir']
        self.input_dir = node.metadata['customNodeConfig']['input_dir']
        self.trackerName = node.metadata['customNodeConfig']['tracker']

    def upload_annotations(self, video_item, merged_video_annotations, merged_video_frames):
        """
        Uploads annotations to video item using specified tracker.

        Args:
            video_item: Dataloop video item to upload annotations to
            merged_video_annotations: List of annotations per frame
            merged_video_frames: List of video frames
        """
        if self.trackerName == "ByteTrack":
            tracker = ByteTrackTracker(annotations_builder=video_item.annotations.builder(), frame_rate=video_item.fps)
        elif self.trackerName == "DeepSORT":
            tracker = DeepSORTTracker(annotations_builder=video_item.annotations.builder(), frame_rate=video_item.fps)
        else:
            raise ValueError(f"Invalid tracker: {self.trackerName}")
        for i, (frame, frame_annotations) in enumerate(zip(merged_video_frames, merged_video_annotations)):
            tracker.update(frame, i, frame_annotations)
        video_item.annotations.upload(annotations=tracker.annotations_builder)

    def get_video_writer(self, first_input_file, first_item, is_same_split):
        """
        Creates and configures video writer based on first input video.

        Args:
            first_input_file: Path to first input video file
            first_item: First Dataloop video item
            is_same_split: Whether videos are from same split

        Returns:
            tuple: VideoWriter object, output path, and fps
        """
        video_type = os.path.splitext(os.path.basename(first_input_file))[1].replace(".", "")
        logger.info(f"video_type: {video_type}")
        fourcc = cv2.VideoWriter_fourcc(*("VP80" if video_type.lower() == "webm" else "mp4v"))
        cap = cv2.VideoCapture(first_input_file)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        logger.info(f"fps: {fps}")
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        logger.info(f"frame_size: {frame_size}")
        cap.release()
        output_video_path = os.path.join(
            self.local_output_folder,
            f"{first_item.metadata['origin_video_name'].replace(f'.{video_type}', '') + '_' if is_same_split else ''}merge_{datetime.datetime.now().isoformat().replace('.', '').replace(':', '_')}.{video_type}",
        )
        logger.info(f"output_video_path: {output_video_path}")

        # Create a VideoWriter object to write the merged video to a file
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
        return writer, output_video_path, fps

    def videos_to_video(self, items: List[dl.Item], context: dl.Context):
        """
        Merges multiple videos into a single video with annotations.

        Args:
            items: List of input video items
            context: Pipeline context containing configuration

        Returns:
            dl.Item: The merged video item
        """
        self.set_config_params(context.node)

        self.local_input_folder = tempfile.mkdtemp(suffix="_input")
        self.local_output_folder = tempfile.mkdtemp(suffix="_output")

        items = self.get_input_items(items)
        if not items or len(items) == 0:
            logger.error("No videos match to merge")
            return

        is_same_split = ServiceRunner.is_items_from_same_split(items)
        logger.info(f"is_same_split: {is_same_split}")
        input_files = [item.download(local_path=self.local_input_folder) for item in items]
        logger.info(f"input_files length: {len(input_files)}")
        # Create a VideoWriter object to write the merged video to a file
        writer, output_video_path, fps = self.get_video_writer(input_files[0], items[0], is_same_split)

        if is_same_split:
            logger.info("merge by sub videos intervals")
            sub_videos_intervals = items[0].metadata["sub_videos_intervals"]
            merged_video_annotations, merged_video_frames = self.merge_by_sub_videos_intervals(
                writer, input_files, sub_videos_intervals, items
            )
        else:
            logger.info("regular merge")
            merged_video_annotations, merged_video_frames = self.regular_merge(writer, input_files, items)

        # Release the VideoWriter object
        writer.release()

        video_item = items[0].dataset.items.upload(local_path=output_video_path, remote_path=self.output_dir)
        video_item.fps = fps
        video_item.update()
        logger.info("uploading annotations to video")
        self.upload_annotations(video_item, merged_video_annotations, merged_video_frames)

        return video_item


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
        "output_dir": "/videos_to_video_2805_29",
        "input_dir": "/merge_videos_testcase",
        "tracker": "ByteTrack",
    }

    # export PYTHONPATH=/app/BoT_SORT

    #    16  python app/tmp.py

    #    17  docker ps

    #    18  exit

    #    19  python app/stitching/frames_to_video.py

    #    20  export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

    # export CUDA_HOME=/usr/local/cuda-11.8
    # export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    # export PYTHONPATH=/app/BoT_SORT

    # context.node.metadata["customNodeConfig"] = {"window_size": 7, "threshold": 0.13, "output_dir": "/testing_238"}

    # pip install cython
    # pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    # pip install cython_bbox

    runner.videos_to_video(items=[dl.items.get(item_id="682c716f3bf48ff6189a3e57")], context=context)
