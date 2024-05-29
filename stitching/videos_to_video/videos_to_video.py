import datetime
import os
import shutil
import threading
import cv2
import dtlpy as dl


class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self):
        ...

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
                [all(key in item.metadata for key in ("origin_video_name", "sub_videos_intervals", "time")) for item in
                 items]):
            return False
        original_name = items[0].metadata["origin_video_name"]
        sub_videos_intervals = items[0].metadata["sub_videos_intervals"]
        time = items[0].metadata["time"]
        for item in items:
            if item.metadata["origin_video_name"] != original_name or item.metadata[
                "sub_videos_intervals"] != sub_videos_intervals or item.metadata["time"] != time:
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
        sub_video_annotations_data = []
        total_frames_count = sub_videos_intervals[-1][1] + 1
        # Loop through each input video file and write its frames to the output video
        for i, input_file in enumerate(input_files):
            item = items[i]
            annotations = item.annotations.list()
            next_interval_start_frame = sub_videos_intervals[i + 1][0] if i < len(
                sub_videos_intervals) - 1 else total_frames_count
            start_frame, end_frame = sub_videos_intervals[i]
            # Open the input video file
            cap = cv2.VideoCapture(input_file)

            for frame_index, j in enumerate(range(start_frame, next_interval_start_frame)):
                frame_annotations = annotations.get_frame(frame_num=frame_index).annotations
                sub_video_annotations_data.append([{"top": ann.top,
                                                    "left": ann.left,
                                                    "bottom": ann.bottom,
                                                    "right": ann.right,
                                                    "label": ann.label,
                                                    "object_visible": ann.object_visible,
                                                    "object_id": int(ann.id, 16)} for ann in frame_annotations])
                ret, frame = cap.read()
                if ret:
                    writer.write(frame)
                else:
                    break
            sub_videos_annotations_data.append(sub_video_annotations_data.copy())
            sub_video_annotations_data = []
            # Release the input video file
            cap.release()
        # Release the VideoWriter object
        writer.release()
        return sub_videos_annotations_data

    @staticmethod
    def regular_merge(writer, input_files, items):
        """
        merge between sub videos one by one
        :param writer: handler of write video
        :param input_files: the sub videos to merge
        :param items: the items of the videos
        """
        sub_videos_annotations_data = []
        sub_video_annotations_data = []
        # Loop through each input video file and write its frames to the output video
        for i, input_file in enumerate(input_files):
            item = items[i]
            annotations = item.annotations.list()
            cap = cv2.VideoCapture(input_file)
            frame_index = 0
            ret, frame = cap.read()
            while ret:
                frame_annotations = annotations.get_frame(frame_num=frame_index).annotations
                sub_video_annotations_data.append([{"top": ann.top,
                                                    "left": ann.left,
                                                    "bottom": ann.bottom,
                                                    "right": ann.right,
                                                    "label": ann.label,
                                                    "object_visible": ann.object_visible,
                                                    "object_id": int(ann.id, 16)} for ann in frame_annotations])
                writer.write(frame)
                ret, frame = cap.read()
                frame_index += 1
            sub_videos_annotations_data.append(sub_video_annotations_data.copy())
            sub_video_annotations_data = []
            # Release the input video file
            cap.release()
        # Release the VideoWriter object
        writer.release()
        return sub_videos_annotations_data

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
                iou = ServiceRunner.get_iou(ServiceRunner.get_bb_from_ann(current_annotation),
                                            ServiceRunner.get_bb_from_ann(prev_annotation))
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
    def merge_annotations_id(sub_videos_annotations_data):
        """
        merge the object id between annotation of the sub videos
        :param sub_videos_annotations_data: the the sub videos annotations data to merge object id
        """
        prev_sub_video_last_frame_annotations_data = []
        for sub_video_annotations_data in sub_videos_annotations_data:
            if not sub_video_annotations_data:
                continue
            if not prev_sub_video_last_frame_annotations_data:
                prev_sub_video_last_frame_annotations_data = sub_video_annotations_data[-1].copy()
                continue
            ServiceRunner.match_annotation_object_id(prev_sub_video_last_frame_annotations_data,
                                                     sub_video_annotations_data)
            prev_sub_video_last_frame_annotations_data = sub_video_annotations_data[-1].copy()

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
                    builder.add(annotation_definition=dl.Box(top=ann["top"],
                                                             left=ann["left"],
                                                             bottom=ann["bottom"],
                                                             right=ann["right"],
                                                             label=ann["label"]),
                                object_visible=ann["object_visible"],
                                frame_num=frame_index,
                                # need to input the element id to create the connection between frames
                                object_id=ann["object_id"])
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
        output_video_path = os.path.join(local_output_folder,
                                         f"{items[0].metadata['origin_video_name'].replace(f'.{video_type}', '') + '_' if is_same_split else ''}merge_{datetime.datetime.now().isoformat().replace('.', '').replace(':', '_')}.{video_type}")

        # Create a VideoWriter object to write the merged video to a file
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
        if is_same_split:
            sub_videos_intervals = items[0].metadata["sub_videos_intervals"]
            sub_videos_annotations_data = ServiceRunner.merge_by_sub_videos_intervals(writer, input_files,
                                                                                      sub_videos_intervals,
                                                                                      items)
        else:
            sub_videos_annotations_data = ServiceRunner.regular_merge(writer, input_files, items)
        video_item = dataset.items.upload(local_path=output_video_path, remote_path=output_folder)
        video_item.fps = fps
        ServiceRunner.upload_annotations(video_item, sub_videos_annotations_data)
        shutil.rmtree(local_input_folder, ignore_errors=True)
        shutil.rmtree(local_output_folder, ignore_errors=True)
        return video_item
