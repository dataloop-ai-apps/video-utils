import numpy as np
import cv2
import os
import sys
import torch
import dtlpy as dl
from trackings.utils import plot_one_box, load_opt

# Add ByteTrack to Python path
byte_track_path = os.path.join(os.path.dirname(__file__), 'ByteTrack')
if byte_track_path not in sys.path:
    sys.path.insert(0, byte_track_path)
from yolox.tracker.byte_tracker import BYTETracker

# Add BoT_SORT to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'BoT_SORT'))
from tracker.mc_bot_sort import BoTSORT

# Add DeepSORT to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'deep_sort_pytorch'))
from deep_sort.deep_sort import DeepSort


class BaseTracker:
    def __init__(self, min_box_area, annotations_builder):
        self.min_box_area = min_box_area
        self.label_to_id_map = {}
        self.id_to_label_map = {}
        self.annotations_builder = annotations_builder

    def update(self, frame, fn, frame_item): ...

    def add_annotation(self, box_size, fn, label_id, top, left, bottom, right, object_id, label=None):
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
        self.annotations_builder.add(
            annotation_definition=dl.Box(top=top, left=left, bottom=bottom, right=right, label=label),
            fixed=fixed,
            frame_num=fn,
            end_frame_num=fn,
            object_id=object_id,
        )


class ByteTrackTracker(BaseTracker):
    @staticmethod
    def iou(boxA, boxB):
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

    def __init__(self, opts, annotations_builder):
        super().__init__(opts.min_box_area, annotations_builder)
        self.opts = opts
        self.opts.track_thresh = 0.5
        self.opts.track_buffer = 30
        self.opts.match_thresh = 0.8
        self.tracker = BYTETracker(args=self.opts, frame_rate=20.0)

    def update(self, frame, fn, frame_item):
        frame_annotation = frame_item.annotations.list().annotations
        tracker_annotations = np.zeros((len(frame_annotation), 5))
        # Store input boxes for later matching
        input_boxes = []  # (left, top, right, bottom, label, ann object)

        for i, ann in enumerate(frame_annotation):
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

            # Find best match in input_boxes
            best_iou = 0
            best_label = None
            for t, l, b, r, label, ann in input_boxes:
                curr_iou = ByteTrackTracker.iou((t, l, b, r), tlbr)
                if curr_iou > best_iou:
                    best_iou = curr_iou
                    best_label = label

            self.add_annotation(tlwh[2] * tlwh[3], fn, 0, tlbr[0], tlbr[1], tlbr[2], tlbr[3], tid, label=best_label)

        return self.annotations_builder


class BoTSORTTracker(BaseTracker):
    def __init__(self, opts, annotations_builder):
        super().__init__(opts.min_box_area, annotations_builder)
        self.opts = opts
        self.tracker = BoTSORT(self.opts, frame_rate=20.0)
        self.tracker.track_high_thresh = 0.11
        self.tracker.args.track_high_thresh = 0.11
        self.tracker.new_track_thresh = 0.2
        self.tracker.args.new_track_thresh = 0.2

    def update(self, frame, fn, frame_item):
        frame_annotation = frame_item.annotations.list().annotations
        tracker_annotations = np.zeros((len(frame_annotation), 6))
        for i, ann in enumerate(frame_annotation):
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
            self.add_annotation(tlwh[2] * tlwh[3], fn, tcls, tlbr[0], tlbr[1], tlbr[2], tlbr[3], tid)
        return self.annotations_builder


class DeepSORTTracker(BaseTracker):
    def __init__(self, opts, annotations_builder):
        super().__init__(opts.min_box_area, annotations_builder)
        self.opts = opts
        model_path = os.path.join(
            os.path.dirname(__file__), 'deep_sort_pytorch', 'deep_sort', 'deep', 'checkpoint', 'ckpt.t7'
        )
        self.tracker = DeepSort(
            model_path=model_path,
            max_dist=0.2,
            min_confidence=0.5,
            nms_max_overlap=0.5,
            max_iou_distance=0.7,
            max_age=70,
            n_init=3,
            nn_budget=100,
            use_cuda=opts.use_cuda if hasattr(opts, 'use_cuda') else True,
        )

    def update(self, frame, fn, frame_item):
        frame_annotation = frame_item.annotations.list().annotations
        dets = []
        confs = []
        clss = []

        for ann in frame_annotation:
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
            self.add_annotation(box_size, fn, int(tcls), y1, x1, y2, x2, int(tid))

        return self.annotations_builder
