import datetime
import shutil
import threading

import cv2
import os
import dtlpy as dl


class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self):
        ...

    @staticmethod
    def get_sub_videos_intervals_by_num_frames(num_frames_per_split, total_frames, overlap):
        """
        compute the sub videos intervals by number of frames per sub video
        :param num_frames_per_split: the number of frames per sub video
        :param total_frames: the total number of frames in the original video
        :param overlap: the number of frames to overlap between sub videos
        :return: sub videos intervals by number of frames per sub video
        """
        assert overlap >= 0, "overlap must be greater than or equal to 0"
        sub_videos_intervals = []
        start_frame = 0
        if isinstance(num_frames_per_split, list):
            for num_frames in num_frames_per_split:
                assert num_frames > 0, "number of frames per split must be greater then 0"
                end_frame = start_frame + num_frames - 1
                sub_videos_intervals.append([start_frame, min(end_frame, total_frames - 1)])
                start_frame += max(num_frames - overlap, 0)
                if end_frame >= total_frames - 1:
                    break
        else:
            assert num_frames_per_split > overlap >= 0, "number of frames per split must be greater then overlap "
            while total_frames - start_frame >= num_frames_per_split:
                end_frame = start_frame + num_frames_per_split - 1
                sub_videos_intervals.append([start_frame, min(end_frame, total_frames - 1)])
                start_frame += (num_frames_per_split - overlap)
                if end_frame >= total_frames - 1:
                    break

        if end_frame < total_frames - 1:
            sub_videos_intervals.append([start_frame, total_frames - 1])
        return sub_videos_intervals

    @staticmethod
    def get_sub_videos_intervals_by_num_splits(num_splits, total_frames, overlap):
        """
        compute the sub videos intervals by number of splits
        :param num_splits: the number of splits
        :param total_frames: the total number of frames in the original video
        :param overlap: the number of frames to overlap between sub videos
        :return: sub videos intervals by number of splits
        """
        assert num_splits > 0, "number of splits must be greater then 0"
        assert total_frames > overlap >= 0, "overlap must be greater than or equal to 0 and less than total_frames"
        sub_videos_intervals = []
        total_frames_with_overlap = total_frames + (num_splits - 1) * overlap
        sub_video_frames_count = total_frames_with_overlap // num_splits
        sub_videos_with_extra_frame = total_frames_with_overlap % num_splits
        start_frame = 0
        extra_frame = False
        for i in range(num_splits):
            if sub_videos_with_extra_frame > 0:
                extra_frame = True
            end_frame = start_frame + sub_video_frames_count + (1 if extra_frame else 0) - 1
            sub_videos_intervals.append([start_frame, end_frame])
            start_frame += sub_video_frames_count + (1 if extra_frame else 0) - overlap
            sub_videos_with_extra_frame -= 1
            extra_frame = False
        return sub_videos_intervals

    @staticmethod
    def get_sub_videos_intervals_by_length(out_length, total_frames, fps, overlap):
        """
        compute the sub videos intervals by length of each sub video
        :param out_length: the length of each sub video
        :param total_frames: the total number of frames in the original video
        :param overlap: the number of frames to overlap between sub videos
        :return: sub videos intervals by length of each sub video
        """
        if isinstance(out_length, list):
            num_frames_per_split = []
            for ltime in out_length:
                assert ltime > 0, "length of sub video must be greater then 0"
                num_frames_per_split.append(ltime * fps)
        else:
            assert out_length > 0, "length of sub video must be greater then 0"
            num_frames_per_split = out_length * fps
        return ServiceRunner.get_sub_videos_intervals_by_num_frames(num_frames_per_split, total_frames, overlap)

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
    def upload_annotations(sub_videos_annotations_info, sub_videos_items, fps):
        """
        uploads the annotations to Dataloop video items
        :param sub_videos_annotations_info: the annotations data per video
        :param sub_videos_items: the video items
        :param fps: the fps of the videos
        """
        for i, sub_video_item in enumerate(sub_videos_items):
            sub_video_item.fps = fps
            builder = sub_video_item.annotations.builder()
            sub_video_annotations_info = sub_videos_annotations_info[sub_video_item.name]
            for frame_index, frame_annotation in sub_video_annotations_info:
                for ann in frame_annotation:
                    builder.add(annotation_definition=dl.Box(top=ann["top"],
                                                             left=ann["left"],
                                                             bottom=ann["bottom"],
                                                             right=ann["right"],
                                                             label=ann["label"]),
                                object_visible=ann["object_visible"],
                                # set the frame for the annotation
                                frame_num=frame_index,
                                # need to input the element id to create the connection between frames
                                object_id=ann["object_id"])

            # Upload the annotations to platform
            sub_video_item.annotations.upload(annotations=builder)

    @staticmethod
    def video_to_videos(item, output_folder, mode, splitter_arg, n_overlap):
        """
        splits video to sub videos by given mode
        :param item: the video item to split
        :param output_folder: the remote output folder
        :param mode: the mode to split by
        :param splitter_arg: an argument to split by
        :param n_overlap: the number of frames to overlap between sub videos
        """
        local_input_folder = "input_folder" + str(threading.get_native_id())
        local_output_folder = "output_folder" + str(threading.get_native_id())
        item_dataset = item.dataset
        ServiceRunner.create_folder(local_input_folder)
        ServiceRunner.create_folder(local_output_folder)

        input_video = item.download(local_path=local_input_folder)
        # Open the input video file
        cap = cv2.VideoCapture(input_video)
        input_base_name = os.path.splitext(os.path.basename(input_video))[0]
        video_type = os.path.splitext(os.path.basename(input_video))[1].replace(".", "")
        fourcc = cv2.VideoWriter_fourcc(*("VP80" if video_type.lower() == "webm" else "mp4v"))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # Get the total number of frames in the input video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_fc_len = len(str(total_frames))

        sub_videos_intervals = []

        if mode == "num_frames":
            sub_videos_intervals = ServiceRunner.get_sub_videos_intervals_by_num_frames(splitter_arg, total_frames,
                                                                                        n_overlap)
        elif mode == "num_splits":
            sub_videos_intervals = ServiceRunner.get_sub_videos_intervals_by_num_splits(splitter_arg, total_frames,
                                                                                        n_overlap)
        elif mode == "out_length":
            sub_videos_intervals = ServiceRunner.get_sub_videos_intervals_by_length(splitter_arg, total_frames, fps,
                                                                                    n_overlap)
        print(sub_videos_intervals)
        annotations = item.annotations.list()
        sub_videos_annotations_info = {}
        # Loop through each split
        for i, (start_frame, end_frame) in enumerate(sub_videos_intervals):
            sub_video_name = f"{input_base_name}_{str(i).zfill(max_fc_len)}_{datetime.datetime.now().isoformat().replace('.', '').replace(':', '_')}.{video_type}"
            output_video = os.path.join(local_output_folder, sub_video_name)
            # Create a VideoWriter object to write the split video to a file
            sub_videos_annotations_info[sub_video_name] = []
            writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

            # Set the current frame to the start frame of the split
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Loop through each frame in the split and write it to the output file
            for sub_video_frame_count, frame_index in enumerate(range(start_frame, end_frame + 1)):
                frame_annotation = annotations.get_frame(frame_num=frame_index).annotations
                frame_annotation_data = [{"top": ann.top,
                                          "left": ann.left,
                                          "bottom": ann.bottom,
                                          "right": ann.right,
                                          "label": ann.label,
                                          "object_visible": ann.object_visible,
                                          "object_id": int(ann.id, 16)} for ann in frame_annotation]
                sub_videos_annotations_info[sub_video_name].append([sub_video_frame_count, frame_annotation_data])
                ret, frame = cap.read()
                if ret:
                    writer.write(frame)
                else:
                    break

            # Release the VideoWriter object
            writer.release()
        # Release the input video file
        cap.release()
        sub_videos_items = item_dataset.items.upload(local_path=os.path.join(local_output_folder, "*"),
                                                     remote_path=output_folder,
                                                     item_metadata={
                                                         "origin_video_name": f"{input_base_name}.{video_type}",
                                                         "time": datetime.datetime.now().isoformat(),
                                                         "sub_videos_intervals": sub_videos_intervals})
        ServiceRunner.upload_annotations(sub_videos_annotations_info, sub_videos_items, item.fps)
        shutil.rmtree(local_input_folder, ignore_errors=True)
        shutil.rmtree(local_output_folder, ignore_errors=True)
