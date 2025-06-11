import os
import argparse
from typing import Optional, List, Tuple
import numpy as np
import torch
import dtlpy as dl

from trackers.deep_sort_pytorch.deep_sort.deep_sort import DeepSort
from trackers.ByteTrack.yolox.tracker.byte_tracker import BYTETracker


class TrackerConfig:
    """Configuration class for all tracker parameters.

    This class centralizes all configuration parameters used by different trackers.

    Args:
        min_box_area (float): Minimum area threshold for bounding boxes. Boxes smaller
            than this will be filtered out.
        track_thresh (float): Detection confidence threshold for tracking.
        track_buffer (int): Number of frames to keep track of lost objects.
        match_thresh (float): IoU threshold for matching detections to tracks.
        mot20 (bool): Whether to use MOT20 evaluation metrics.
    """

    def __init__(
        self,
        min_box_area: float = 0,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        mot20: bool = False,
    ):
        self.min_box_area = min_box_area
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.mot20 = mot20


class BaseTracker:
    """Base class for object tracking implementations.

    This class provides common functionality for different tracking algorithms,
    including annotation handling and box size filtering.

    Args:
        config (TrackerConfig): Configuration object containing all tracker parameters
        annotations_builder (dl.AnnotationBuilder, optional): Dataloop annotation builder
            instance for creating annotations. Defaults to None.

    Attributes:
        config (TrackerConfig): Configuration object containing all tracker parameters
        label_to_id_map (dict): Maps label strings to numeric IDs
        id_to_label_map (dict): Maps numeric IDs back to label strings
        annotations_builder (dl.AnnotationBuilder): Builder for creating annotations
    """

    def __init__(self, annotations_builder: dl.AnnotationCollection, config: TrackerConfig = TrackerConfig()) -> None:
        """Initialize base tracker with configuration and optional annotation builder.

        Args:
            annotations_builder (dl.AnnotationBuilder): Dataloop annotation builder
            config (TrackerConfig): Configuration object containing all tracker parameters
        """
        self.config = config
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
        if box_size <= self.config.min_box_area:
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

    def __init__(
        self,
        annotations_builder: dl.AnnotationCollection,
        frame_rate: float = 20,
        config: TrackerConfig = TrackerConfig(),
    ) -> None:
        super().__init__(annotations_builder=annotations_builder, config=config)
        self.tracker = BYTETracker(args=config, frame_rate=frame_rate)

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
    def __init__(self, annotations_builder: dl.AnnotationCollection, config: TrackerConfig = TrackerConfig()) -> None:
        super().__init__(annotations_builder=annotations_builder, config=config)
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
