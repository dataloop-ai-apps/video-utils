import datetime
import shutil
import threading

import dtlpy as dl
import cv2
import os


class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self):
        ...

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
    def create_folder(folder):
        """
        creates empty folder
        :param folder: the path of the folder
        """
        if os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)
        os.mkdir(folder)

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
                    [next(object_unique_id) if not prev_annotations_data else None, [ann["top"],
                                                                                     ann["left"],
                                                                                     ann["bottom"],
                                                                                     ann["right"],
                                                                                     ann["label"]]])
            if prev_annotations_data:
                ServiceRunner.match_annotation_object_id(current_annotations_data, prev_annotations_data, object_unique_id)
            for ann in current_annotations_data:
                builder.add(annotation_definition=dl.Box(top=ann[1][0],
                                                         left=ann[1][1],
                                                         bottom=ann[1][2],
                                                         right=ann[1][3],
                                                         label=ann[1][4]),
                            frame_num=i,
                            # need to input the element id to create the connection between frames
                            object_id=ann[0])
            prev_annotations_data = current_annotations_data.copy()
            current_annotations_data = []

        video_item.annotations.upload(annotations=builder)

    @staticmethod
    def frames_to_vid(item, dql_filter, output_folder, output_video_type, fps):
        """
        merges frames to one video
        :param item: an item to trigger this service
        :param dql_filter: a DQL filter of the frames to merge
        :param output_folder: the output folder
        :param output_video_type: the type of the new video
        :param fps: the fps of the new video
        """
        assert fps > 0, "fps must be greater then 0"
        assert output_video_type.lower() in ["webm", "mp4", "avi"]
        dataset = item.dataset
        local_input_folder = "input_folder" + str(threading.get_native_id())
        local_output_folder = "output_folder" + str(threading.get_native_id())
        ServiceRunner.create_folder(local_input_folder)
        ServiceRunner.create_folder(local_output_folder)

        if dql_filter is None:
            filters = dl.Filters()
            filters.add(field='metadata.user.parentItemId', values=item.id)
        else:
            filters = dl.Filters(custom_filter=dql_filter)
        filters.sort_by(field='name')
        items = dataset.items.get_all_items(filters=filters)
        if not items or len(items) == 0:
            print("No images match to merge")
            return
        input_files = [item.download(local_path=local_input_folder) for item in items]

        fourcc = cv2.VideoWriter_fourcc(*("VP80" if output_video_type.lower() == "webm" else "mp4v"))

        frame_size = ServiceRunner.get_image_size(input_files[0])
        output_video_path = os.path.join(local_output_folder,
                                         f"merge_{datetime.datetime.now().isoformat().replace('.', '').replace(':', '_')}.{output_video_type}")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
        frames_annotations = []
        # Loop through each input image file and write it to the output video
        for i, input_file in enumerate(input_files):
            # Read the input image file
            frame = cv2.imread(input_file)

            # Write the image to the output video
            writer.write(frame)
            frame_annotations_data = [{"top": ann.top,
                                       "left": ann.left,
                                       "bottom": ann.bottom,
                                       "right": ann.right,
                                       "label": ann.label} for ann in items[i].annotations.list().annotations]
            frames_annotations.append(frame_annotations_data)

        # Release the VideoWriter object
        writer.release()
        video_item = dataset.items.upload(local_path=output_video_path, remote_path=output_folder)
        video_item.fps = fps
        ServiceRunner.upload_annotations(video_item, frames_annotations)
        shutil.rmtree(local_input_folder, ignore_errors=True)
        shutil.rmtree(local_output_folder, ignore_errors=True)
        return video_item
