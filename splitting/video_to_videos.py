import datetime
import logging
import os
import tempfile
from typing import List, Tuple, Dict, Union, Any

import cv2
import dtlpy as dl

logger = logging.getLogger('video-utils.video_to_videos')


class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self):
        self.input_base_name = None
        self.video_type = None
        self.fourcc = None
        self.fps = None
        self.frame_size = None
        self.total_frames = None
        self.max_fc_len = None
        self.split_type = None
        self.dl_output_folder = None
        self.splitter_arg = None
        self.n_overlap = None
        self.local_input_folder = None
        self.local_output_folder = None

    def get_sub_videos_intervals_by_num_frames(self, num_frames_per_split: Union[int, List[int]]) -> List[List[int]]:
        """
        Compute sub video intervals based on number of frames per split.

        Args:
            num_frames_per_split (int or list): Number of frames per split or list of frame counts.

        Returns:
            list: List of [start_frame, end_frame] intervals for each sub video.
        """
        assert self.n_overlap >= 0, "overlap must be greater than or equal to 0"
        sub_videos_intervals = []
        start_frame = 0
        if isinstance(num_frames_per_split, list):
            for num_frames in num_frames_per_split:
                assert num_frames > 0, "number of frames per split must be greater then 0"
                end_frame = start_frame + num_frames - 1
                sub_videos_intervals.append([start_frame, min(end_frame, self.total_frames - 1)])
                start_frame += max(num_frames - self.n_overlap, 0)
                if end_frame >= self.total_frames - 1:
                    break
        else:
            assert (
                num_frames_per_split > self.n_overlap >= 0
            ), "number of frames per split must be greater then overlap "
            while self.total_frames - start_frame >= num_frames_per_split:
                end_frame = start_frame + num_frames_per_split - 1
                sub_videos_intervals.append([start_frame, min(end_frame, self.total_frames - 1)])
                start_frame += num_frames_per_split - self.n_overlap
                if end_frame >= self.total_frames - 1:
                    break

        if end_frame < self.total_frames - 1:
            sub_videos_intervals.append([start_frame, self.total_frames - 1])
        return sub_videos_intervals

    def get_sub_videos_intervals_by_num_splits(self, num_splits: int) -> List[List[int]]:
        """
        Compute sub video intervals based on desired number of splits.

        Args:
            num_splits (int): Number of sub videos to create.

        Returns:
            list: List of [start_frame, end_frame] intervals for each sub video.
        """
        assert num_splits > 0, "number of splits must be greater then 0"
        assert (
            self.total_frames > self.n_overlap >= 0
        ), "overlap must be greater than or equal to 0 and less than total_frames"
        sub_videos_intervals = []
        total_frames_with_overlap = self.total_frames + (num_splits - 1) * self.n_overlap
        sub_video_frames_count = total_frames_with_overlap // num_splits
        sub_videos_with_extra_frame = total_frames_with_overlap % num_splits
        start_frame = 0
        extra_frame = False
        for i in range(num_splits):
            if sub_videos_with_extra_frame > 0:
                extra_frame = True
            end_frame = start_frame + sub_video_frames_count + (1 if extra_frame else 0) - 1
            sub_videos_intervals.append([start_frame, end_frame])
            start_frame += sub_video_frames_count + (1 if extra_frame else 0) - self.n_overlap
            sub_videos_with_extra_frame -= 1
            extra_frame = False
        return sub_videos_intervals

    def get_sub_videos_intervals_by_length(self, out_length: Union[float, List[float]]) -> List[List[int]]:
        """
        Compute sub video intervals based on desired output length in seconds.

        Args:
            out_length (Union[float, List[float]]): Length of each sub video in seconds.

        Returns:
            list: List of [start_frame, end_frame] intervals for each sub video.
        """
        if isinstance(out_length, list):
            num_frames_per_split = []
            for ltime in out_length:
                assert ltime > 0, "length of sub video must be greater then 0"
                num_frames_per_split.append(int(ltime * self.fps))
        else:
            assert out_length > 0, "length of sub video must be greater then 0"
            num_frames_per_split = int(out_length * self.fps)
        return self.get_sub_videos_intervals_by_num_frames(num_frames_per_split)

    def set_config_params(self, node: dl.PipelineNode) -> None:
        """
        Set configuration parameters from pipeline node metadata.

        Args:
            node (dl.PipelineNode): Pipeline node containing configuration
        """
        logger.info(f"node custom config: {node.metadata['customNodeConfig']}")
        self.split_type = node.metadata['customNodeConfig']['split_type']
        self.dl_output_folder = node.metadata['customNodeConfig']['output_dir']
        self.splitter_arg = node.metadata['customNodeConfig']['splitter_arg']
        self.n_overlap = node.metadata['customNodeConfig']['n_overlap']

    def initialize_video_params(self, cap: cv2.VideoCapture, input_video: str) -> None:
        """
        Initialize video parameters from OpenCV capture object.

        Args:
            cap (cv2.VideoCapture): Video capture object
            input_video (str): Path to input video file
        """
        self.input_base_name = os.path.splitext(os.path.basename(input_video))[0]
        self.video_type = os.path.splitext(os.path.basename(input_video))[1].replace(".", "")
        # Use appropriate codec based on video type
        self.fourcc = cv2.VideoWriter_fourcc(*("VP80" if self.video_type.lower() == "webm" else "mp4v"))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # Get the total number of frames in the input video
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.max_fc_len = len(str(self.total_frames))

        logger.info(
            f"Video params initialized: input_base_name={self.input_base_name}, video_type={self.video_type}, "
            f"fourcc={self.fourcc}, fps={self.fps}, frame_size={self.frame_size}, "
            f"total_frames={self.total_frames}, max_fc_len={self.max_fc_len}"
        )

    def write_video_segment(
        self, cap: cv2.VideoCapture, annotations: Any, start_frame: int, end_frame: int, i: int
    ) -> Tuple[str, List[List[Any]]]:
        """
        Extract a segment from source video and collect its annotations.

        Args:
            cap: OpenCV video capture object
            annotations: Video annotations
            start_frame: Starting frame index
            end_frame: Ending frame index
            i: Split index

        Returns:
            Tuple of (sub video filename, annotations list)
        """
        sub_video_name = f"{self.input_base_name}_{str(i).zfill(self.max_fc_len)}_{datetime.datetime.now().isoformat().replace('.', '').replace(':', '_')}.{self.video_type}"
        output_video = os.path.join(self.local_output_folder, self.dl_output_folder, sub_video_name)
        sub_video_annotations = []

        writer = cv2.VideoWriter(output_video, self.fourcc, self.fps, self.frame_size)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for sub_video_frame_count, frame_index in enumerate(range(start_frame, end_frame + 1)):
            frame_annotation = annotations.get_frame(frame_num=frame_index).annotations
            frame_annotation_data = [
                {
                    "top": ann.top,
                    "left": ann.left,
                    "bottom": ann.bottom,
                    "right": ann.right,
                    "label": ann.label,
                    "object_visible": ann.object_visible,
                    "object_id": int(ann.id, 16),
                }
                for ann in frame_annotation
            ]
            sub_video_annotations.append([sub_video_frame_count, frame_annotation_data])
            ret, frame = cap.read()
            if ret:
                writer.write(frame)
            else:
                break

        writer.release()
        return sub_video_name, sub_video_annotations

    def upload_sub_videos_and_annotations(
        self,
        item: dl.Item,
        sub_videos_annotations_info: Dict[str, List[List[Any]]],
        sub_videos_intervals: List[List[int]],
    ) -> List[dl.Item]:
        """
        Upload sub videos and their annotations to the platform.

        Args:
            item: Source video item
            sub_videos_annotations_info: Dict mapping sub video names to their annotations
            sub_videos_intervals: List of frame intervals for each sub video

        Returns:
            List of uploaded sub video items
        """
        logger.info(f"Uploading sub videos to {self.dl_output_folder}")
        sub_videos_items = item.dataset.items.upload(
            local_path=os.path.join(self.local_output_folder, self.dl_output_folder),
            remote_path='/',
            item_metadata={
                "origin_video_name": f"{self.input_base_name}.{self.video_type}",
                "time": datetime.datetime.now().isoformat(),
                "sub_videos_intervals": sub_videos_intervals,
            },
        )
        logger.info("start uploading sub videos and annotations")
        for sub_video_item in sub_videos_items:
            sub_video_item.fps = item.fps
            builder = sub_video_item.annotations.builder()
            sub_video_annotations_info = sub_videos_annotations_info[sub_video_item.name]
            for frame_index, frame_annotation in sub_video_annotations_info:
                for ann in frame_annotation:
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
                        object_id=ann["object_id"],
                    )

            # Upload the annotations to platform
            sub_video_item.annotations.upload(annotations=builder)
        return sub_videos_items

    def video_to_videos(self, item: dl.Item, context: dl.Context) -> None:
        """
        Split a video into multiple sub-videos based on specified parameters.

        Args:
            item (dl.Item): Input video item to split
            context (dl.Context): Function context containing configuration

        Returns:
            List[dl.Item]: List of created sub-video items
        """
        logger.info('Running service Video To Videos')
        cap = None

        try:
            self.set_config_params(node=context.node)

            self.local_input_folder = tempfile.mkdtemp(suffix="_input")
            self.local_output_folder = tempfile.mkdtemp(suffix="_output")

            logger.info(f"Downloading video to {self.local_input_folder}")
            input_video = item.download(local_path=self.local_input_folder)

            cap = cv2.VideoCapture(input_video)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {input_video}")

            self.initialize_video_params(cap, input_video)

            logger.info(f"split type: {self.split_type}")
            sub_videos_intervals = []
            if self.split_type == "num_frames":
                sub_videos_intervals = self.get_sub_videos_intervals_by_num_frames(self.splitter_arg)
            elif self.split_type == "num_splits":
                sub_videos_intervals = self.get_sub_videos_intervals_by_num_splits(self.splitter_arg)
            elif self.split_type == "out_length":
                sub_videos_intervals = self.get_sub_videos_intervals_by_length(self.splitter_arg)
            else:
                raise ValueError(f"Invalid split type: {self.split_type}")

            logger.info(f"sub_videos_intervals: {sub_videos_intervals}")

            # Process each video split
            annotations = item.annotations.list()
            sub_videos_annotations_info = {}
            for i, (start_frame, end_frame) in enumerate(sub_videos_intervals):
                sub_video_name, sub_video_annotations = self.write_video_segment(
                    cap=cap, annotations=annotations, start_frame=start_frame, end_frame=end_frame, i=i
                )
                sub_videos_annotations_info[sub_video_name] = sub_video_annotations

            logger.info(f"Uploading sub videos and annotations to {self.dl_output_folder}")
            return self.upload_sub_videos_and_annotations(item, sub_videos_annotations_info, sub_videos_intervals)

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise
        finally:
            if cap is not None:
                cap.release()
                logger.info("Released video capture resources")


if __name__ == "__main__":
    use_rc_env = False

    if dl.token_expired():
        dl.login()

    if use_rc_env:
        dl.setenv('rc')
    else:
        dl.setenv('prod')
    if dl.token_expired():
        dl.login()

    runner = ServiceRunner()
    context = dl.Context()
    context.pipeline_id = "682069122afb795bc3c41d59"
    context.node_id = "bd1dc151-6067-4197-85aa-1b65394e2077"
    context.node.metadata["customNodeConfig"] = {
        "split_type": "out_length",
        "splitter_arg": 3,
        "output_dir": "/3_sec_videos_2",
        "n_overlap": 0,
    }

    # context.node.metadata["customNodeConfig"] = {"window_size": 7, "threshold": 0.13, "output_dir": "/testing_238"}
    runner.video_to_videos(item=dl.items.get(item_id="6823077f82b17749dc98a968"), context=context)
