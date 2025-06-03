import os
import argparse
import numpy as np
import torch
import dtlpy as dl

# Import BoTSORT from local path
from trackers.BoT_SORT.tracker.mc_bot_sort import BoTSORT

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
        annotations_list (list): List to store annotation dictionaries
    """

    def __init__(self, min_box_area, annotations_builder=None):
        """Initialize base tracker with minimum box area and optional annotation builder.

        Args:
            min_box_area (float): Minimum area threshold for bounding boxes
            annotations_builder (dl.AnnotationBuilder, optional): Dataloop annotation builder
        """
        self.min_box_area = min_box_area
        self.label_to_id_map = {}
        self.id_to_label_map = {}
        self.annotations_builder = annotations_builder
        self.annotations_list = []

    def update(self, frame, fn, frame_annotations): ...

    def add_annotation(self, box_size, fn, label_id, top, left, bottom, right, object_id, label=None):
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

        annotation = {
            'annotation_definition': dl.Box(top=top, left=left, bottom=bottom, right=right, label=label),
            'fixed': fixed,
            'frame_num': fn,
            'end_frame_num': fn,
            'object_id': object_id,
        }

        self.annotations_list.append(annotation)

        if self.annotations_builder is not None:
            self.annotations_builder.add(
                annotation_definition=annotation['annotation_definition'],
                fixed=annotation['fixed'],
                frame_num=annotation['frame_num'],
                end_frame_num=annotation['end_frame_num'],
                object_id=annotation['object_id'],
            )


class ByteTrackTracker(BaseTracker):
    @staticmethod
    def iou(boxA, boxB):
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

    def __init__(self, annotations_builder, frame_rate):
        super().__init__(annotations_builder)
        opts = argparse.Namespace(track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False)
        self.tracker = BYTETracker(args=opts, frame_rate=frame_rate)

    def update(self, frame, fn, frame_annotations):
        """Update the tracker with a new frame and its annotations.

        Args:
            frame: The current video frame to process
            fn (int): Frame number
            frame_annotations: List of annotations for the current frame

        Returns:
            dl.AnnotationBuilder: Updated annotations builder with tracking results
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
                area=tlwh[2] * tlwh[3],
                frame_number=fn,
                confidence=0,
                left=tlbr[0],
                top=tlbr[1],
                right=tlbr[2],
                bottom=tlbr[3],
                track_id=tid,
                label=best_label,
            )

        return self.annotations_builder


class BoTSORTTracker(BaseTracker):
    def __init__(self, annotations_builder, frame_rate):
        super().__init__(annotations_builder)
        opts = argparse.Namespace(
            track_high_thresh=0.11,
            track_low_thresh=0.1,
            new_track_thresh=0.2,
            conf_thres=0.09,
            iou_thres=0.7,
            agnostic_nms=True,
            name='exp',
            track_thresh=0.6,
            track_buffer=30,
            match_thresh=0.8,
            aspect_ratio_thresh=1.6,
            min_box_area=10,
            mot20=True,
            cmc_method="sparseOptFlow",
            ablation=False,
            with_reid=False,
            proximity_thresh=0.5,
            appearance_thresh=0.25,
        )
        self.tracker = BoTSORT(opts, frame_rate=frame_rate)

    def update(self, frame, fn, frame_annotations):
        """Update the tracker with a new frame and its annotations.

        Args:
            frame: The current video frame to process
            fn (int): Frame number
            frame_annotations: List of annotations for the current frame

        Returns:
            dl.AnnotationBuilder: Updated annotations builder with tracking results
        """
        tracker_annotations = np.zeros((len(frame_annotations), 6))
        for i, ann in enumerate(frame_annotations):
            if ann.type != 'box':
                continue
            tracker_annotations[i, :4] = [ann.top, ann.left, ann.bottom, ann.right]
            try:
                tracker_annotations[i, 4] = ann.metadata['user']['model']['confidence']
            except KeyError:
                tracker_annotations[i, 4] = 1
            label_id = self.label_to_id_map.get(ann.label, None)
            if label_id is None:
                label_id = len(self.label_to_id_map)
                self.id_to_label_map[label_id] = ann.label
                self.label_to_id_map[ann.label] = label_id
            tracker_annotations[i, 5] = label_id
        online_targets = self.tracker.update(tracker_annotations, frame.copy())
        for t in online_targets:
            tlwh = t.tlwh
            tlbr = t.tlbr
            tid = t.track_id
            tcls = t.cls
            self.add_annotation(
                box_size=tlwh[2] * tlwh[3],
                fn=fn,
                label_id=tcls,
                top=tlbr[0],
                left=tlbr[1],
                bottom=tlbr[2],
                right=tlbr[3],
                object_id=tid,
            )
        return self.annotations_builder


class DeepSORTTracker(BaseTracker):
    def __init__(self, annotations_builder):
        super().__init__(annotations_builder)
        model_path = os.path.join(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'trackers',
                'deep_sort_pytorch',
                'deep_sort',
                'deep',
                'checkpoint',
                'ckpt.t7',
            )
        )
        self.tracker = DeepSort(model_path=model_path, use_cuda=torch.cuda.is_available())

    def update(self, frame, fn, frame_annotations):
        """Update the tracker with a new frame and its annotations.

        Args:
            frame: The current video frame to process
            fn (int): Frame number
            frame_annotations: List of annotations for the current frame

        Returns:
            dl.AnnotationBuilder: Updated annotations builder with tracking results
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
            print(f"-HHH- t: {t}")
            print(f"-HHH- x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, tcls: {tcls}, tid: {tid}")
            box_size = (x2 - x1) * (y2 - y1)
            print(f"-HHH- box_size: {box_size}")
            self.add_annotation(
                box_size=box_size, fn=fn, label_id=int(tcls), top=y1, left=x1, bottom=y2, right=x2, object_id=int(tid)
            )

        return self.annotations_builder
