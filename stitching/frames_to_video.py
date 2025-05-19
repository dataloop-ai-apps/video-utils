import numpy as np
from trackings.utils import plot_one_box, load_opt
from numpy import random
import cv2
import sys
import dtlpy as dl
import os
from dotenv import load_dotenv
import os
import logging
import datetime
import tempfile
import torch

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

    def update(self, frame, fn, frame_item, video_item): ...

    def add_annotation(self, box_size, fn, label_id, top, left, bottom, right, object_id):
        if box_size <= self.min_box_area:
            return

        if fn == 0:
            fixed = True
        else:
            fixed = False

        label = self.id_to_label_map[label_id]
        self.annotations_builder.add(
            annotation_definition=dl.Box(top=top, left=left, bottom=bottom, right=right, label=label),
            fixed=fixed,
            frame_num=fn,
            end_frame_num=fn,
            object_id=object_id,
        )


class BoTSORTTracker(BaseTracker):
    def __init__(self, opts, annotations_builder):
        super().__init__(opts.min_box_area, annotations_builder)
        self.opts = opts
        self.tracker = BoTSORT(self.opts, frame_rate=20.0)
        self.tracker.track_high_thresh = 0.11
        self.tracker.args.track_high_thresh = 0.11
        self.tracker.new_track_thresh = 0.2
        self.tracker.args.new_track_thresh = 0.2

    def update(self, frame, fn, frame_item, video_item):
        annotations_builder = video_item.annotations.builder()
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
        return annotations_builder


class ByteTrackTracker(BaseTracker):
    def __init__(self, opts, annotations_builder):
        super().__init__(opts.min_box_area, annotations_builder)
        self.opts = opts
        # Add required parameters for ByteTrack
        self.opts.track_thresh = 0.5
        self.opts.track_buffer = 30
        self.opts.match_thresh = 0.8
        self.tracker = BYTETracker(args=self.opts, frame_rate=20.0)

    def update(self, frame, fn, frame_item, video_item):
        annotations_builder = video_item.annotations.builder()
        frame_annotation = frame_item.annotations.list().annotations
        tracker_annotations = np.zeros((len(frame_annotation), 5))
        for i, ann in enumerate(frame_annotation):
            if ann.type != 'box':
                continue
            tracker_annotations[i, :4] = [ann.top, ann.left, ann.bottom, ann.right]  # XYXY order!
            try:
                tracker_annotations[i, 4] = ann.metadata['user']['model']['confidence']
            except KeyError:
                tracker_annotations[i, 4] = 1.0
            # label_id = self.label_to_id_map.get(ann.label, None)
            # if label_id is None:
            #     label_id = len(self.label_to_id_map)
            #     self.id_to_label_map[label_id] = ann.label
            #     self.label_to_id_map[ann.label] = label_id
            # tracker_annotations[i, 5] = label_id

        # Get image info and size
        height, width = frame.shape[:2]
        # Use frame index or 0 if not available
        img_info = (height, width, fn)
        img_size = (height, width)

        # ByteTrack expects [x1, y1, x2, y2, score, class_id]
        # Make sure tracker_annotations columns are in this order!
        # If your ann is (top, left, bottom, right), convert to (left, top, right, bottom)
        # If so, use:
        # tracker_annotations[i, :4] = [ann.left, ann.top, ann.right, ann.bottom]
        output_results_tensor = torch.from_numpy(tracker_annotations).float()
        online_targets = self.tracker.update(output_results_tensor, img_info, img_size)
        print(f"-HHH-online_targets: {online_targets}")
        for t in online_targets:
            print(f"-HHH-t: {t}")
            tlwh = t.tlwh
            tlbr = t.tlbr
            tid = t.track_id
            # tcls = t.cls
            self.add_annotation(tlwh[2] * tlwh[3], fn, 0, tlbr[0], tlbr[1], tlbr[2], tlbr[3], tid)
        return annotations_builder


class DeepSORTTracker(BaseTracker):
    def __init__(self, opts, annotations_builder):
        super().__init__(opts.min_box_area, annotations_builder)
        self.opts = opts
        # Initialize your DeepSORT object here.
        # Adjust the following line to match your library's DeepSORT initialization.
        # For ZQPei/deep_sort_pytorch, you might need to initialize with build_tracker(...) or similar.
        model_path = os.path.join(
            os.path.dirname(__file__), 'deep_sort_pytorch', 'deep_sort', 'deep', 'checkpoint', 'ckpt.t7'
        )
        self.tracker = DeepSort(
            model_path=model_path,  # Path to the appearance model
            max_dist=0.2,  # Maximum cosine distance threshold
            min_confidence=0.5,  # Detection confidence threshold
            nms_max_overlap=0.5,  # Non-maxima suppression threshold
            max_iou_distance=0.7,  # Maximum IOU distance for track matching
            max_age=70,  # Maximum number of frames to keep track alive
            n_init=3,  # Number of frames for track initialization
            nn_budget=100,  # Maximum size of feature database
            use_cuda=opts.use_cuda if hasattr(opts, 'use_cuda') else True,
        )

    def update(self, frame, fn, frame_item, video_item):
        annotations_builder = video_item.annotations.builder()
        frame_annotation = frame_item.annotations.list().annotations
        dets = []
        confs = []
        clss = []

        for ann in frame_annotation:
            if ann.type != 'box':
                continue
            # DeepSORT expects xywh format: (center_x, center_y, width, height)
            x1, y1, x2, y2 = ann.left, ann.top, ann.right, ann.bottom
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w / 2.0, y1 + h / 2.0
            dets.append([cx, cy, w, h])
            try:
                confs.append(ann.metadata['user']['model']['confidence'])
            except KeyError:
                confs.append(1.0)
            # For class id mapping if needed
            label_id = self.label_to_id_map.get(ann.label, None)
            if label_id is None:
                label_id = len(self.label_to_id_map)
                self.id_to_label_map[label_id] = ann.label
                self.label_to_id_map[ann.label] = label_id
            clss.append(label_id)

        if len(dets) == 0:
            # No detections for this frame
            return annotations_builder

        dets = np.array(dets)
        confs = np.array(confs)
        clss = np.array(clss)

        # DeepSORT update: adjust as needed for your implementation!
        # For ZQPei/deep_sort_pytorch it may look like:
        # outputs, _ = self.tracker.update(dets, confs, clss, frame)
        # If your tracker expects [N, 4] (xywh), [N], [N], and frame (RGB or BGR as needed).
        outputs, _ = self.tracker.update(dets, confs, clss, frame)

        for t in outputs:
            # ZQPei/deep_sort_pytorch returns [x1, y1, x2, y2, class, id]
            x1, y1, x2, y2, tcls, tid = t
            box_size = (x2 - x1) * (y2 - y1)
            self.add_annotation(box_size, fn, int(tcls), y1, x1, y2, x2, int(tid))

        return annotations_builder


class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self): ...

    @staticmethod
    def get_image_size(image_path):
        """
        computes the shape of an image
        :param image_path: the path of the image
        :return: the shape of the image
        """
        image = cv2.imread(image_path)
        if image is not None:
            height, width, _ = image.shape
            return width, height
        return None

    @staticmethod
    def infinite_sequence():
        """
        makes an infinite sequence
        :return: an infinite sequence one by one
        """
        num = 1
        while True:
            yield num
            num += 1

    @staticmethod
    def get_iou(bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        bb1-2 : list
            [top, left, bottom, right)
        """
        top1, left1, bottom1, right1 = bb1[0], bb1[1], bb1[2], bb1[3]
        top2, left2, bottom2, right2 = bb2[0], bb2[1], bb2[2], bb2[3]

        x_left = max(left1, left2)
        y_top = max(top1, top2)
        x_right = min(right1, right2)
        y_bottom = min(bottom1, bottom2)
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bb1_area = (right1 - left1) * (bottom1 - top1)
        bb2_area = (right2 - left2) * (bottom2 - top2)
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        return iou

    @staticmethod
    def match_annotation_object_id(current_annotations_data, prev_annotations_data, object_unique_id):
        """
        match annotations object id
        :param current_annotations_data: the current annotations data
        :param prev_annotations_data: the previous annotations data
        :param object_unique_id: an infinite sequence of ints
        """
        ann_matches = []
        best_match = (-1, 0)
        matched_prev_anns = []
        for i, current_annotation_data in enumerate(current_annotations_data):
            for j, prev_annotation_data in enumerate(prev_annotations_data):
                if j in matched_prev_anns or prev_annotation_data[1][4] != current_annotation_data[1][4]:
                    continue
                iou = ServiceRunner.get_iou(current_annotation_data[1], prev_annotation_data[1])
                if iou > best_match[1]:
                    best_match = (j, iou)
            ann_matches.append([i, best_match[0]])
            matched_prev_anns.append(best_match[0])
            best_match = (-1, 0)
        for curr_ann_idx, prev_ann_idx in ann_matches:
            if prev_ann_idx == -1:
                current_annotations_data[curr_ann_idx][0] = next(object_unique_id)
            else:
                current_annotations_data[curr_ann_idx][0] = prev_annotations_data[prev_ann_idx][0]

    @staticmethod
    def upload_annotations(video_item, frames_annotations):
        """
        uploads the annotations to Dataloop video item
        :param video_item: the video item
        :param frames_annotations: the annotations data per frame
        """
        object_unique_id = ServiceRunner.infinite_sequence()
        current_annotations_data = []
        prev_annotations_data = []
        builder = video_item.annotations.builder()

        for i, frame_annotations in enumerate(frames_annotations):
            if not frame_annotations:
                prev_annotations_data = []
                continue

            for ann in frame_annotations:
                current_annotations_data.append(
                    [
                        next(object_unique_id) if not prev_annotations_data else None,
                        [ann["top"], ann["left"], ann["bottom"], ann["right"], ann["label"]],
                    ]
                )
            if prev_annotations_data:
                ServiceRunner.match_annotation_object_id(
                    current_annotations_data, prev_annotations_data, object_unique_id
                )
            for ann in current_annotations_data:
                builder.add(
                    annotation_definition=dl.Box(
                        top=ann[1][0], left=ann[1][1], bottom=ann[1][2], right=ann[1][3], label=ann[1][4]
                    ),
                    frame_num=i,
                    # need to input the element id to create the connection between frames
                    object_id=ann[0],
                )
            prev_annotations_data = current_annotations_data.copy()
            current_annotations_data = []

        video_item.annotations.upload(annotations=builder)

    def set_config_params(self, node: dl.PipelineNode):
        self.fps = node.metadata['customNodeConfig']['fps']
        self.output_dir = node.metadata['customNodeConfig']['output_dir']
        self.output_video_type = node.metadata['customNodeConfig']['output_video_type']
        self.input_dir = node.metadata['customNodeConfig']['input_dir']

    def get_input_files(self, dataset):
        filters = dl.Filters(field='dir', values=self.input_dir)
        filters.sort_by(field='name')
        items = dataset.items.get_all_items(filters=filters)
        if not items or len(items) == 0:
            print("No images match to merge")
            return []
        # TODO : check if there a batch download
        return items

    def old_get_items_annotations(self, items):
        frame_annotations_data = []
        for item in items:
            frame_annotations_data.append(
                [
                    {"top": ann.top, "left": ann.left, "bottom": ann.bottom, "right": ann.right, "label": ann.label}
                    for ann in item.annotations.list().annotations
                ]
            )
        return frame_annotations_data

    def stitch_and_upload(self, dataset, cv_frames):
        output_video_path = os.path.join(
            self.local_output_folder,
            f"merge_{datetime.datetime.now().isoformat().replace('.', '').replace(':', '_')}.{self.output_video_type}",
        )
        fourcc = cv2.VideoWriter_fourcc(*("VP80" if self.output_video_type.lower() == "webm" else "mp4v"))
        writer = cv2.VideoWriter(output_video_path, fourcc, self.fps, (cv_frames[0].shape[1], cv_frames[0].shape[0]))
        # Loop through each input image file and write it to the output video
        for frame in cv_frames:
            # Write the image to the output video
            writer.write(frame)

        # Release the VideoWriter object
        writer.release()
        video_item = dataset.items.upload(local_path=output_video_path, remote_path=self.output_dir)
        video_item.fps = self.fps
        video_item.update()
        return video_item

    def stitch_items_annotations(self, stitched_item, cv_frames, items):
        annotation_builder = stitched_item.annotations.builder()
        for fn, frame in enumerate(cv_frames):
            self.update_tracker(frame, fn, items[fn], annotation_builder)
        stitched_item.annotations.upload(annotations=annotation_builder)

    def frames_to_vid(self, item: dl.Item, context: dl.Context):
        logger = logging.getLogger('video-utils.frames_to_vid')
        logger.info('Running service Frames To Video')

        self.set_config_params(context.node)

        self.local_input_folder = tempfile.mkdtemp(suffix="_input")
        self.local_output_folder = tempfile.mkdtemp(suffix="_output")

        items = self.get_input_files(item.dataset)
        cv_frames = [cv2.imread(item.download(local_path=self.local_input_folder)) for item in items]
        video_item = self.stitch_and_upload(item.dataset, cv_frames)
        # self.tracker = ByteTrackTracker(opts=load_opt(), annotations_builder=video_item.annotations.builder())
        # self.tracker = DeepSORTTracker(opts=load_opt(), annotations_builder=video_item.annotations.builder())
        self.tracker = BoTSORTTracker(opts=load_opt(), annotations_builder=video_item.annotations.builder())

        for i, (frame_i, item_i) in enumerate(zip(cv_frames, items)):
            self.tracker.update(frame_i, i, item_i, video_item)
        video_item.annotations.upload(annotations=self.tracker.annotations_builder)


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
        "output_dir": "/white_dancers_frames_bot_sort",
        "input_dir": "/white_dancers_frames",
        "output_video_type": "webm",
    }

    # context.node.metadata["customNodeConfig"] = {"window_size": 7, "threshold": 0.13, "output_dir": "/testing_238"}
    runner.frames_to_vid(item=dl.items.get(item_id="6821ec8fb188d7f242334661"), context=context)
