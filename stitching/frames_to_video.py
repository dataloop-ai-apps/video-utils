import datetime
import logging
import os
import tempfile
import argparse
from typing import List, Optional, Tuple

import cv2
import dtlpy as dl
import numpy as np
import torch
from dotenv import load_dotenv

from trackers.deep_sort_pytorch.deep_sort.deep_sort import DeepSort
from trackers.ByteTrack.yolox.tracker.byte_tracker import BYTETracker


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


logger = logging.getLogger('video-utils.frames_to_vid')


class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self):
        self.fps: int = 0
        self.output_dir: str = ""
        self.output_video_type: str = ""
        self.input_dir: str = ""
        self.trackerName: str = ""
        self.tracker = None
        self.local_input_folder: str = ""
        self.local_output_folder: str = ""

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
        if self.input_dir is not None and self.input_dir.strip():
            logger.info(f"input_dir: {self.input_dir}")
            dataset = items[0].dataset
            filters = dl.Filters(resource=dl.FiltersResource.ITEM, field='dir', values=("/" + self.input_dir))
            filters.sort_by(field='name')
            # items = dataset.items.get_all_items(filters=filters)
            items = list(dataset.items.list(filters=filters).all())
            logger.info(f"get_input_items number of items: {len(items)}")
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
        try:
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

        self.local_input_folder = tempfile.mkdtemp(suffix="_input")
        self.local_output_folder = tempfile.mkdtemp(suffix="_output")

        items = self.get_input_items(items)
        # cv_frames = [cv2.imread(item.download(local_path=self.local_input_folder)) for item in items]
        images_files = items[0].dataset.items.download(local_path=self.local_input_folder, items=items)
        images_files = sorted(images_files, reverse=True)  # Sort filenames in descending order
        cv_frames = [cv2.imread(img_path) for img_path in images_files]
        video_item = self.stitch_and_upload(items[0].dataset, cv_frames)
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
    context.pipeline_id = "682069122afb795bc3c41d59"
    context.node_id = "bd1dc151-6067-4197-85aa-1b65394e2077"
    context.node.metadata["customNodeConfig"] = {
        "fps": 20,
        "output_dir": "/tmp_stitching_frames_to_video_byte_track_2488",
        "input_dir": "/split_5_sec_to_one_frame",
        "output_video_type": "webm",
        "tracker": "ByteTrack",
    }

    # context.node.metadata["customNodeConfig"] = {"window_size": 7, "threshold": 0.13, "output_dir": "/testing_238"}
    runner.frames_to_vid(items=[dl.items.get(item_id="682a250fb188d7f0a74d53ed")], context=context)
