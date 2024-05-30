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
    def video_split_by_frames_interval(video_path, video_item, output_directory, dl_output_folder, total_frames,
                                       frames_interval):
        """
        splits video by frames interval
        :param video_path: the path of the video to split
        :param video_item: the video item to split
        :param output_directory: the local output folder
        :param dl_output_folder: the remote output folder
        :param total_frames: the total frames number of the video
        :param frames_interval: the frames interval to split by
        """
        assert frames_interval > 0, "frames_interval must be greater then 0"
        frame_items = list()
        item_dataset = video_item.dataset
        annotations = video_item.annotations.list()

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        input_base_name = os.path.splitext(os.path.basename(video_path))[0]
        max_fc_len = len(str(total_frames))
        frame_count = 0
        success, frame = cap.read()

        while success:

            # Check if the frame is sampled according to the interval
            if frame_count % frames_interval == 0:
                # Save the frame as an image
                frame_path = os.path.join(output_directory,
                                          f"{input_base_name}_{str(frame_count).zfill(max_fc_len)}_{datetime.datetime.now().isoformat().replace('.', '').replace(':', '_')}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_item = item_dataset.items.upload(local_path=frame_path,
                                                       remote_path=dl_output_folder,
                                                       item_metadata={
                                                           "user": {"parentItemId": video_item.id}
                                                       })
                frame_items.append(frame_item)
                if annotations:
                    frame_annotation = annotations.get_frame(frame_num=frame_count)
                    builder = frame_item.annotations.builder()
                    for ann in frame_annotation.annotations:
                        if ann.object_visible:
                            builder.add(annotation_definition=dl.Box(top=ann.top,
                                                                     left=ann.left,
                                                                     bottom=ann.bottom,
                                                                     right=ann.right,
                                                                     label=ann.label))
                    # Upload box to the item
                    frame_item.annotations.upload(builder)

            frame_count += 1
            # Read a frame from the video
            success, frame = cap.read()

        # Release the video file
        cap.release()
        return frame_items

    @staticmethod
    def video_split_by_seconds_interval(video_path, item, output_directory, dl_output_folder, total_frames,
                                        seconds_interval):
        """
       splits video by seconds interval
       :param video_path: the path of the video to split
       :param item: the video item to split
       :param output_directory: the local output folder
       :param dl_output_folder: the remote output folder
       :param total_frames: the total frames number of the video
       :param seconds_interval: the seconds interval to split by
        """
        assert seconds_interval > 0, "seconds_interval must be greater then 0"
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        frame_items = ServiceRunner.video_split_by_frames_interval(video_path, item, output_directory, dl_output_folder,
                                                                   total_frames, fps * seconds_interval)
        return frame_items

    @staticmethod
    def video_split_by_num_splits(video_path, item, output_directory, dl_output_folder, total_frames, num_splits):
        """
       splits video by number of splits
       :param video_path: the path of the video to split
       :param item: the video item to split
       :param output_directory: the local output folder
       :param dl_output_folder: the remote output folder
       :param total_frames: the total frames number of the video
       :param num_splits: the number of splits to split by
        """
        assert num_splits > 1, "num_splits must be greater then 1"
        assert num_splits <= total_frames, "total frames count must be greater then num_splits"
        frames_interval = total_frames // (num_splits - 1) - (0 if total_frames % (num_splits - 1) else 1)
        frame_items = ServiceRunner.video_split_by_frames_interval(video_path, item, output_directory, dl_output_folder,
                                                                   total_frames, frames_interval)
        return frame_items

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
    def video_to_frames(item, output_folder, mode, splitter_arg, context: dl.Context = None):
        """
        splits video by given mode
        :param item: the video item to split
        :param output_folder: the remote output folder
        :param mode: the mode to split by
        :param splitter_arg: an argument to split by
        """
        if context is not None and context.node is not None:
            config = context.node.metadata.get("customNodeConfig", {})
            output_folder = config.get("output_folder", output_folder)
            mode = config.get("mode", mode)
            splitter_arg = config.get("splitter_arg", splitter_arg)

        local_input_folder = "input_folder" + str(threading.get_native_id())
        local_output_folder = "output_folder" + str(threading.get_native_id())
        ServiceRunner.create_folder(local_input_folder)
        ServiceRunner.create_folder(local_output_folder)

        input_video = item.download(local_path=local_input_folder)
        cap = cv2.VideoCapture(input_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if mode == "frames_interval":
            frame_items = ServiceRunner.video_split_by_frames_interval(input_video, item, local_output_folder,
                                                                       output_folder, total_frames, splitter_arg)
        elif mode == "num_splits":
            frame_items = ServiceRunner.video_split_by_num_splits(input_video, item, local_output_folder, output_folder,
                                                                  total_frames, splitter_arg)
        elif mode == "seconds_interval":
            frame_items = ServiceRunner.video_split_by_seconds_interval(input_video, item, local_output_folder,
                                                                        output_folder, total_frames, splitter_arg)
        else:
            assert False, "mode can only be frames_interval or num_splits or seconds_interval"
        shutil.rmtree(local_input_folder, ignore_errors=True)
        shutil.rmtree(local_output_folder, ignore_errors=True)
        return item, frame_items
