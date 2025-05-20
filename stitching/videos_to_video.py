import datetime
import os
import shutil
import threading
import cv2
import dtlpy as dl
import numpy as np
from trackings.utils import load_opt
from .trackers import ByteTrackTracker


class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self): ...

    @staticmethod
    def create_folder(folder):
        """
        creates empty folder
        :param folder: the path of the folder
        """
        if os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)
        os.mkdir(folder)

    @staticmethod
    def is_items_from_same_split(items):
        """
        checks if all of the items are from the same video
        :param items: the items to check
        :return: true if all of the items are from the same video, false otherwise
        """
        if not all(
            [
                all(key in item.metadata for key in ("origin_video_name", "sub_videos_intervals", "time"))
                for item in items
            ]
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

    @staticmethod
    def merge_by_sub_videos_intervals(writer, input_files, sub_videos_intervals, items):
        """
        merges videos by sub videos intervals
        :param writer: handler of write video
        :param input_files: the sub videos to merge
        :param sub_videos_intervals: the sub videos intervals of the videos
        :param items: the items of the videos
        """
        sub_videos_annotations_data = []
        sub_videos_frames = []
        curr_sub_video_frames = []
        curr_sub_video_annotations = []
        total_frames_count = sub_videos_intervals[-1][1] + 1
        # Loop through each input video file and write its frames to the output video
        for i, input_file in enumerate(input_files):
            item = items[i]
            annotations = item.annotations.list()
            next_interval_start_frame = (
                sub_videos_intervals[i + 1][0] if i < len(sub_videos_intervals) - 1 else total_frames_count
            )
            start_frame, end_frame = sub_videos_intervals[i]
            # Open the input video file
            cap = cv2.VideoCapture(input_file)

            for frame_index, j in enumerate(range(start_frame, next_interval_start_frame)):
                frame_annotations = annotations.get_frame(frame_num=frame_index).annotations
                curr_sub_video_annotations.append(
                    [
                        {
                            "top": ann.top,
                            "left": ann.left,
                            "bottom": ann.bottom,
                            "right": ann.right,
                            "label": ann.label,
                            "object_visible": ann.object_visible,
                            "object_id": int(ann.id, 16),
                        }
                        for ann in frame_annotations
                    ]
                )
                ret, frame = cap.read()
                if ret:
                    curr_sub_video_frames.append(frame)
                    writer.write(frame)
                else:
                    break

            # TODO : this copy is not needed
            sub_videos_annotations_data.append(curr_sub_video_annotations.copy())
            curr_sub_video_annotations = []
            sub_videos_frames.append(curr_sub_video_frames.copy())
            curr_sub_video_frames = []
            # Release the input video file
            cap.release()
        # Release the VideoWriter object
        writer.release()
        return sub_videos_annotations_data, sub_videos_frames

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
    def get_bb_from_ann(ann):
        """
        get bounding box from annotation
        :param ann: an annotation
        :return: the bounding box of the annotation
        """
        return [ann["top"], ann["left"], ann["bottom"], ann["right"]]

    @staticmethod
    def match_annotation_object_id(prev_sub_video_last_frame_annotations, sub_video_annotations):
        """
        match the object id between annotation from different frames
        :param prev_sub_video_last_frame_annotations: the last frame annotations of the prev sub video
        :param sub_video_annotations: the current sub video annotations
        """
        cur_sub_video_first_frame_annotations = sub_video_annotations[0]
        ann_matches = []
        best_match = (-1, 0)
        matched_prev_anns = []
        for i, current_annotation in enumerate(cur_sub_video_first_frame_annotations):
            for j, prev_annotation in enumerate(prev_sub_video_last_frame_annotations):
                if j in matched_prev_anns or prev_annotation["label"] != current_annotation["label"]:
                    continue
                iou = ServiceRunner.get_iou(
                    ServiceRunner.get_bb_from_ann(current_annotation), ServiceRunner.get_bb_from_ann(prev_annotation)
                )
                if iou > best_match[1]:
                    best_match = (j, iou)
            ann_matches.append([i, best_match[0]])
            matched_prev_anns.append(best_match[0])
            best_match = (-1, 0)

        for curr_ann_idx, prev_ann_idx in ann_matches:
            if prev_ann_idx == -1:
                continue
            else:
                new_object_id = prev_sub_video_last_frame_annotations[prev_ann_idx]["object_id"]
                object_id_to_replace = cur_sub_video_first_frame_annotations[curr_ann_idx]["object_id"]
                for frame_annotations in sub_video_annotations:
                    for annotation in frame_annotations:
                        if annotation["object_id"] == object_id_to_replace:
                            annotation["object_id"] = new_object_id

    @staticmethod
    def upload_annotations(video_item, sub_videos_annotations_data):
        """
        uploads the annotations to Dataloop video item
        :param video_item: the video item
        :param sub_videos_annotations_data: the annotations data sub video
        """
        ServiceRunner.merge_annotations_id(sub_videos_annotations_data)
        builder = video_item.annotations.builder()
        frame_index = 0
        for sub_video_annotations_data in sub_videos_annotations_data:
            for frame_annotations_data in sub_video_annotations_data:
                for ann in frame_annotations_data:
                    builder.add(
                        annotation_definition=dl.Box(
                            top=ann["top"],
                            left=ann["left"],
                            bottom=ann["bottom"],
                            right=ann["right"],
                            label=ann["label"],
                        ),
                        object_visible=ann["object_visible"],
                        frame_num=frame_index,
                        # need to input the element id to create the connection between frames
                        object_id=ann["object_id"],
                    )
                frame_index += 1

        video_item.annotations.upload(annotations=builder)

    @staticmethod
    def videos_to_video(item, dql_filter, output_folder):
        """
        merges sub videos to one video
        :param item: an item to trigger this service
        :param dql_filter: a DQL filter of the sub videos to merge
        :param output_folder: the remote output folder
        """
        dataset = item.dataset
        local_input_folder = "input_folder" + str(threading.get_native_id())
        local_output_folder = "output_folder" + str(threading.get_native_id())
        ServiceRunner.create_folder(local_input_folder)
        ServiceRunner.create_folder(local_output_folder)
        filters = dl.Filters(custom_filter=dql_filter)
        filters.sort_by(field='name')
        items = dataset.items.get_all_items(filters=filters)
        if not items or len(items) == 0:
            print("No videos match to merge")
            return
        is_same_split = ServiceRunner.is_items_from_same_split(items)
        input_files = [item.download(local_path=local_input_folder) for item in items]
        first_input_file = input_files[0]
        video_type = os.path.splitext(os.path.basename(first_input_file))[1].replace(".", "")

        fourcc = cv2.VideoWriter_fourcc(*("VP80" if video_type.lower() == "webm" else "mp4v"))
        cap = cv2.VideoCapture(first_input_file)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        output_video_path = os.path.join(
            local_output_folder,
            f"{items[0].metadata['origin_video_name'].replace(f'.{video_type}', '') + '_' if is_same_split else ''}merge_{datetime.datetime.now().isoformat().replace('.', '').replace(':', '_')}.{video_type}",
        )

        # Create a VideoWriter object to write the merged video to a file
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
        if is_same_split:
            sub_videos_intervals = items[0].metadata["sub_videos_intervals"]
            sub_videos_annotations_data = ServiceRunner.merge_by_sub_videos_intervals(
                writer, input_files, sub_videos_intervals, items
            )
        else:
            sub_videos_annotations_data = ServiceRunner.regular_merge(writer, input_files, items)
        video_item = dataset.items.upload(local_path=output_video_path, remote_path=output_folder)
        video_item.fps = fps
        ServiceRunner.upload_annotations(video_item, sub_videos_annotations_data)
        shutil.rmtree(local_input_folder, ignore_errors=True)
        shutil.rmtree(local_output_folder, ignore_errors=True)

    @staticmethod
    def old_match_annotation_object_id_with_tracker(
        prev_sub_video_last_frames_annotations, sub_video_annotations, prev_frames
    ):
        """
        Match object IDs between sub-videos using ByteTrack tracker
        :param prev_sub_video_last_frames_annotations: List of annotations from last N frames of previous sub-video
        :param sub_video_annotations: List of annotations from current sub-video
        :param prev_frames: List of actual frame images from previous sub-video
        """
        if not prev_sub_video_last_frames_annotations or not sub_video_annotations or not prev_frames:
            return

        # Initialize tracker with default options
        opts = load_opt()
        opts.track_buffer = 30  # Keep track of objects for 30 frames
        opts.track_thresh = 0.5  # Confidence threshold for tracking
        opts.match_thresh = 0.8  # IoU threshold for matching

        # Initialize tracker without annotations builder
        tracker = ByteTrackTracker(opts=opts, annotations_builder=None)

        # Process previous frames to initialize tracking
        for i, (frame_anns, frame) in enumerate(zip(prev_sub_video_last_frames_annotations, prev_frames)):
            # Update tracker with previous frame
            tracker.update(frame, i, frame_anns)

        # Process first frame of current sub-video
        current_frame_anns = sub_video_annotations[0]
        # Use the last frame from previous sub-video as reference
        frame = prev_frames[-1]

        # Update tracker with current frame
        tracker.update(frame, len(prev_sub_video_last_frames_annotations), current_frame_anns)

        # Get tracking results and update object IDs
        for frame_annotations in sub_video_annotations:
            for annotation in frame_annotations:
                # Find matching track ID from tracker's state
                for track in tracker.tracker.tracked_stracks:
                    if track.is_activated:
                        # Calculate IoU between annotation and track
                        ann_box = [annotation["top"], annotation["left"], annotation["bottom"], annotation["right"]]
                        track_box = [track.tlbr[0], track.tlbr[1], track.tlbr[2], track.tlbr[3]]
                        iou = ServiceRunner.get_iou(ann_box, track_box)

                        # If IoU is high enough, update the object ID
                        if iou > opts.match_thresh:
                            annotation["object_id"] = track.track_id
                            break

    ########################################################
    ########################################################

    def match_annotation_objects_ids(
        self, prev_sub_video_anns, prev_sub_video_frames, sub_videos_anns_data, sub_videos_frames
    ):

        # Initialize tracker with default options
        opts = load_opt()
        opts.track_buffer = 30  # Keep track of objects for 30 frames
        opts.track_thresh = 0.5  # Confidence threshold for tracking
        opts.match_thresh = 0.8  # IoU threshold for matching

        # Initialize tracker without annotations builder
        tracker = ByteTrackTracker(opts=opts, annotations_builder=None)

        # get last 30 freames of prev sub video
        for i, (frame_anns, frame) in enumerate(zip(prev_sub_video_anns[-30:], prev_sub_video_frames[-30:])):
            tracker.update(frame, i, frame_anns)

        # and the firest frame in current sub video
        tracker.update(sub_videos_frames[0], 30, sub_videos_anns_data[0])

        # match object ids based on tracker results
        # we need to match annotations from prev sub video last frame to annotations in the current sub video
        # this will be done by tracker results, so we need a map from

    def merge_by_sub_videos_intervals(self, writer, input_files, sub_videos_intervals, items):
        """
        merges videos by sub videos intervals
        :param writer: handler of write video
        :param input_files: the sub videos to merge
        :param sub_videos_intervals: the sub videos intervals of the videos
        :param items: the items of the videos
        """
        merged_video_frames = []
        merged_video_annotations = []
        total_frames_count = sub_videos_intervals[-1][1] + 1
        # Loop through each input video file and write its frames to the output video
        for i, input_file in enumerate(input_files):
            item = items[i]
            annotations = item.annotations.list()
            next_interval_start_frame = (
                sub_videos_intervals[i + 1][0] if i < len(sub_videos_intervals) - 1 else total_frames_count
            )
            start_frame, end_frame = sub_videos_intervals[i]
            # Open the input video file
            cap = cv2.VideoCapture(input_file)

            for frame_index, j in enumerate(range(start_frame, next_interval_start_frame)):
                frame_annotations = annotations.get_frame(frame_num=frame_index).annotations
                merged_video_annotations.append(
                    [
                        {
                            "top": ann.top,
                            "left": ann.left,
                            "bottom": ann.bottom,
                            "right": ann.right,
                            "label": ann.label,
                            "object_visible": ann.object_visible,
                            "object_id": int(ann.id, 16),
                        }
                        for ann in frame_annotations
                    ]
                )
                ret, frame = cap.read()
                if ret:
                    merged_video_frames.append(frame)
                    writer.write(frame)
                else:
                    break

            # Release the input video file
            cap.release()
        # Release the VideoWriter object
        writer.release()
        return merged_video_annotations, merged_video_frames

    def merge_annotations_id(self, sub_videos_annotations_data, sub_videos_frames):
        """
        merge the object id between annotation of the sub videos
        :param sub_videos_annotations_data: the the sub videos annotations data to merge object id
        """
        prev_sub_video_last_frame_annotations_data = None
        prev_sub_video_frames = None
        for sub_video_annotations_data, sub_video_frames in zip(sub_videos_annotations_data, sub_videos_frames):
            if sub_video_annotations_data is None or len(sub_video_annotations_data) == 0:
                continue
            if prev_sub_video_last_frame_annotations_data is None:
                prev_sub_video_last_frame_annotations_data = sub_video_annotations_data[-1].copy()
                prev_sub_video_frames = sub_video_frames
                continue
            ServiceRunner.match_annotation_object_id(
                prev_sub_video_last_frame_annotations_data, sub_video_annotations_data
            )
            prev_sub_video_last_frame_annotations_data = sub_video_annotations_data[-1].copy()

    def regular_merge(self, writer, input_files, items):
        """
        merge between sub videos one by one
        :param writer: handler of write video
        :param input_files: the sub videos to merge
        :param items: the items of the videos
        """
        merged_video_annotations = []
        merged_video_frames = []
        # Loop through each input video file and write its frames to the output video
        for i, input_file in enumerate(input_files):
            item = items[i]
            annotations = item.annotations.list()
            cap = cv2.VideoCapture(input_file)
            frame_index = 0
            ret, frame = cap.read()
            while ret:
                frame_annotations = annotations.get_frame(frame_num=frame_index).annotations
                merged_video_annotations.append(
                    [
                        {
                            "top": ann.top,
                            "left": ann.left,
                            "bottom": ann.bottom,
                            "right": ann.right,
                            "label": ann.label,
                            "object_visible": ann.object_visible,
                            "object_id": int(ann.id, 16),
                        }
                        for ann in frame_annotations
                    ]
                )
                merged_video_frames.append(frame)
                writer.write(frame)
                ret, frame = cap.read()
                frame_index += 1
            # Release the input video file
            cap.release()
        # Release the VideoWriter object
        writer.release()
        return merged_video_annotations, merged_video_frames

    # TODO : move this to base class
    def get_input_files(self, dataset):
        filters = dl.Filters(field='dir', values=self.input_dir)
        filters.sort_by(field='name')
        items = dataset.items.get_all_items(filters=filters)
        if not items or len(items) == 0:
            print("No images match to merge")
            return []
        # TODO : check if there a batch download
        return items

    def set_config_params(self, node: dl.PipelineNode):
        self.fps = node.metadata['customNodeConfig']['fps']
        self.output_dir = node.metadata['customNodeConfig']['output_dir']
        self.output_video_type = node.metadata['customNodeConfig']['output_video_type']
        self.input_dir = node.metadata['customNodeConfig']['input_dir']

    def upload_annotations(self, video_item, merged_video_annotations, merged_video_frames):
        tracker = DeepSORTTracker(opts=load_opt(), annotations_builder=video_item.annotations.builder())
        for i, (frame, frame_annotations) in enumerate(zip(merged_video_frames, merged_video_annotations)):
            tracker.update(frame, i, frame_annotations)
        video_item.annotations.upload(annotations=self.tracker.annotations_builder)

    def videos_to_video(self, item: dl.Item, context: dl.Context):

        self.set_config_params(context.node)

        self.local_input_folder = tempfile.mkdtemp(suffix="_input")
        self.local_output_folder = tempfile.mkdtemp(suffix="_output")

        items = self.get_input_files(item.dataset)
        input_files = [item.download(local_path=self.local_input_folder) for item in items]
        first_input_file = input_files[0]
        video_type = os.path.splitext(os.path.basename(first_input_file))[1].replace(".", "")

        fourcc = cv2.VideoWriter_fourcc(*("VP80" if video_type.lower() == "webm" else "mp4v"))
        cap = cv2.VideoCapture(first_input_file)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        output_video_path = os.path.join(
            local_output_folder,
            f"{items[0].metadata['origin_video_name'].replace(f'.{video_type}', '') + '_' if is_same_split else ''}merge_{datetime.datetime.now().isoformat().replace('.', '').replace(':', '_')}.{video_type}",
        )

        # Create a VideoWriter object to write the merged video to a file
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
        if is_same_split:
            sub_videos_intervals = items[0].metadata["sub_videos_intervals"]
            merged_video_annotations, merged_video_frames = self.merge_by_sub_videos_intervals(
                writer, input_files, sub_videos_intervals, items
            )
        else:
            merged_video_annotations, merged_video_frames = self.regular_merge(writer, input_files, items)

        video_item = item.dataset.items.upload(local_path=output_video_path, remote_path=self.output_folder)
        video_item.fps = fps
        video_item.update()
        self.upload_annotations(video_item, merged_video_annotations, merged_video_frames)
