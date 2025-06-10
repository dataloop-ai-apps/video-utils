import datetime
import os
import tempfile
import copy
import logging
from typing import List, Optional, Tuple

import cv2
import dtlpy as dl
from dotenv import load_dotenv
import argparse
import numpy as np
import torch

from trackers.deep_sort_pytorch.deep_sort.deep_sort import DeepSort
from trackers.ByteTrack.yolox.tracker.byte_tracker import BYTETracker

logger = logging.getLogger('video-utils.videos_to_video')


class BaseTracker:
    """Base class for object tracking implementations.

    This class provides common functionality for different tracking algorithms,
    including annotation handling and box size filtering.

    Args:
        min_box_area (float): Minimum area threshold for bounding boxes. Boxes smaller
            than this will be filtered out.
        annotations_builder (dl.AnnotationBuilder, optional): Dataloop annotation builder
            instance for creating annotations. Defaults to None.

    Attributes:
        min_box_area (float): Minimum box area threshold
        label_to_id_map (dict): Maps label strings to numeric IDs
        id_to_label_map (dict): Maps numeric IDs back to label strings
        annotations_builder (dl.AnnotationBuilder): Builder for creating annotations
    """

    def __init__(self, min_box_area: float, annotations_builder: Optional[dl.AnnotationCollection] = None) -> None:
        """Initialize base tracker with minimum box area and optional annotation builder.

        Args:
            min_box_area (float): Minimum area threshold for bounding boxes
            annotations_builder (dl.AnnotationBuilder, optional): Dataloop annotation builder
        """
        self.min_box_area = min_box_area
        self.label_to_id_map = {}
        self.id_to_label_map = {}
        self.annotations_builder = annotations_builder

    def update(
        self, frame: np.ndarray, fn: int, frame_annotations: List[dl.Annotation]
    ) -> Optional[dl.AnnotationCollection]:
        """Update the tracker with a new frame and its annotations.

        Args:
            frame: The current video frame to process
            fn (int): Frame number
            frame_annotations: List of annotations for the current frame

        Returns:
            dl.AnnotationCollection: Updated annotations collection with tracking results

        Raises:
            NotImplementedError: This is a pure virtual method that must be implemented by subclasses
        """
        raise NotImplementedError("update() must be implemented by subclass")

    def add_annotation(
        self,
        box_size: float,
        fn: int,
        label_id: int,
        top: float,
        left: float,
        bottom: float,
        right: float,
        object_id: int,
        label: Optional[str] = None,
    ) -> None:
        """Add annotation if box size exceeds minimum area threshold.

        Args:
            box_size (float): Area of bounding box
            fn (int): Frame number
            label_id (int): Class label ID
            top (float): Top coordinate
            left (float): Left coordinate
            bottom (float): Bottom coordinate
            right (float): Right coordinate
            object_id (int): Unique object ID
            label (str, optional): Class label string
        """
        if box_size <= self.min_box_area:
            return

        if fn == 0:
            fixed = True
        else:
            fixed = False
        if label is None:
            label = self.id_to_label_map.get(label_id, None)
        if label is None:
            print(f"label is None for object_id: {object_id}")
            return

        if self.annotations_builder is not None:
            self.annotations_builder.add(
                annotation_definition=dl.Box(top=top, left=left, bottom=bottom, right=right, label=label),
                fixed=fixed,
                frame_num=fn,
                end_frame_num=fn,
                object_id=object_id,
            )


class ByteTrackTracker(BaseTracker):
    @staticmethod
    def iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        """Calculate intersection over union between two bounding boxes.

        Args:
            boxA: First box coordinates (top, left, bottom, right)
            boxB: Second box coordinates (top, left, bottom, right)

        Returns:
            float: IoU score between 0 and 1
        """
        # box: (l, t, r, b)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def __init__(self, annotations_builder: dl.AnnotationCollection, frame_rate: float) -> None:
        super().__init__(min_box_area=0, annotations_builder=annotations_builder)
        opts = argparse.Namespace(track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False)
        self.tracker = BYTETracker(args=opts, frame_rate=frame_rate)

    def update(
        self, frame: np.ndarray, fn: int, frame_annotations: List[dl.Annotation]
    ) -> Optional[dl.AnnotationCollection]:
        """Update the tracker with a new frame and its annotations.

        Args:
            frame: The current video frame to process
            fn (int): Frame number
            frame_annotations: List of annotations for the current frame

        Returns:
            dl.AnnotationCollection: Updated annotations collection with tracking results
        """
        tracker_annotations = np.zeros((len(frame_annotations), 5))
        # Store input boxes for later matching
        input_boxes = []  # (left, top, right, bottom, label, ann object)

        for i, ann in enumerate(frame_annotations):
            if ann.type != 'box':
                continue
            l, t, r, b = ann.left, ann.top, ann.right, ann.bottom
            tracker_annotations[i, :4] = [t, l, b, r]
            try:
                tracker_annotations[i, 4] = ann.metadata['user']['model']['confidence']
            except KeyError:
                tracker_annotations[i, 4] = 1.0
            input_boxes.append((t, l, b, r, getattr(ann, "label", None), ann))

        height, width = frame.shape[:2]
        img_info = (height, width, fn)
        img_size = (height, width)
        output_results_tensor = torch.from_numpy(tracker_annotations).float()
        online_targets = self.tracker.update(output_results_tensor, img_info, img_size)

        for t in online_targets:
            tlwh = t.tlwh
            tlbr = t.tlbr  # (left, top, right, bottom)
            tid = t.track_id

            # Find best match in input_boxes by calculating IoU (Intersection over Union)
            # We use IoU to match tracked boxes with original input boxes to preserve the original labels,
            # since ByteTracker doesn't maintain label information during tracking
            best_iou = 0
            best_label = None
            for t, l, b, r, label, ann in input_boxes:
                curr_iou = ByteTrackTracker.iou((t, l, b, r), tlbr)
                if curr_iou > best_iou:
                    best_iou = curr_iou
                    best_label = label

            self.add_annotation(
                box_size=tlwh[2] * tlwh[3],
                fn=fn,
                label_id=0,
                top=tlbr[0],
                left=tlbr[1],
                bottom=tlbr[2],
                right=tlbr[3],
                object_id=tid,
                label=best_label,
            )

        return self.annotations_builder


class DeepSORTTracker(BaseTracker):
    def __init__(self, annotations_builder: dl.AnnotationCollection) -> None:
        super().__init__(min_box_area=0, annotations_builder=annotations_builder)
        model_path = os.path.join(os.path.dirname(__file__), 'deep_sort_checkpoint', 'ckpt.t7')
        if not os.path.exists(model_path):
            model_path = '/trackers/deep_sort_pytorch_ckpt/ckpt.t7'
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    "DeepSORT model checkpoint not found at either 'deep_sort_checkpoint/ckpt.t7' or '/trackers/ckpt.t7'"
                )
        self.tracker = DeepSort(model_path=model_path, use_cuda=torch.cuda.is_available())

    def update(
        self, frame: np.ndarray, fn: int, frame_annotations: List[dl.Annotation]
    ) -> Optional[dl.AnnotationCollection]:
        """Update the tracker with a new frame and its annotations.

        Args:
            frame: The current video frame to process
            fn (int): Frame number
            frame_annotations: List of annotations for the current frame

        Returns:
            dl.AnnotationCollection: Updated annotations collection with tracking results
        """
        dets = []
        confs = []
        clss = []

        for ann in frame_annotations:
            if ann.type != 'box':
                continue
            x1, y1, x2, y2 = ann.left, ann.top, ann.right, ann.bottom
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w / 2.0, y1 + h / 2.0
            dets.append([cx, cy, w, h])
            try:
                confs.append(ann.metadata['user']['model']['confidence'])
            except KeyError:
                confs.append(1.0)
            label_id = self.label_to_id_map.get(ann.label, None)
            if label_id is None:
                label_id = len(self.label_to_id_map)
                self.id_to_label_map[label_id] = ann.label
                self.label_to_id_map[ann.label] = label_id
            clss.append(label_id)

        if len(dets) == 0:
            return self.annotations_builder

        dets = np.array(dets)
        confs = np.array(confs)
        clss = np.array(clss)

        outputs, _ = self.tracker.update(dets, confs, clss, frame)

        for t in outputs:
            x1, y1, x2, y2, tcls, tid = t
            box_size = (x2 - x1) * (y2 - y1)
            self.add_annotation(
                box_size=box_size, fn=fn, label_id=int(tcls), top=y1, left=x1, bottom=y2, right=x2, object_id=int(tid)
            )

        return self.annotations_builder


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
