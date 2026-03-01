import datetime
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Union, Any

import cv2
import dtlpy as dl

logger = logging.getLogger('video-utils.video_to_videos')


@dataclass
class VideoContext:
    """Per-call state bundled together to avoid storing anything on the service instance."""

    split_type: str
    dl_output_folder: str
    splitter_arg: Union[int, float, List]
    n_overlap: int
    input_base_name: str
    video_type: str
    fps: int
    total_frames: int
    max_fc_len: int
    frame_size: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    fourcc: int = 0


class ServiceRunner(dl.BaseServiceRunner):

    # ── Context builder ──────────────────────────────────────

    @staticmethod
    def _build_context(config: Dict[str, Any], item: dl.Item) -> VideoContext:
        """Build VideoContext from pipeline config and Dataloop item metadata."""
        system_meta = item.metadata.get('system', {})
        fps = int(system_meta.get('fps', 30))
        duration = float(system_meta.get('duration', 0))
        total_frames = int(duration * fps)
        width = int(system_meta.get('width', 0))
        height = int(system_meta.get('height', 0))

        base_name = os.path.splitext(os.path.basename(item.filename))[0]
        video_type = os.path.splitext(os.path.basename(item.filename))[1].lstrip(".")

        ctx = VideoContext(
            split_type=config['split_type'],
            dl_output_folder=config['output_dir'],
            splitter_arg=config['splitter_arg'],
            n_overlap=config.get('n_overlap', 0),
            input_base_name=base_name,
            video_type=video_type,
            fps=fps,
            total_frames=total_frames,
            max_fc_len=len(str(total_frames)),
            frame_size=(width, height),
            fourcc=cv2.VideoWriter_fourcc(*("VP80" if video_type.lower() == "webm" else "mp4v")),
        )
        logger.info(
            f"Video context: base_name={ctx.input_base_name}, type={ctx.video_type}, "
            f"fps={ctx.fps}, size={ctx.frame_size}, frames={ctx.total_frames}"
        )
        return ctx

    # ── Shared helpers ────────────────────────────────────────

    @staticmethod
    def get_new_items_metadata(item: dl.Item, sub_videos_intervals: List[List[int]]) -> Dict[str, Any]:
        """
        Get metadata for new sub-video items.

        Args:
            item: Source video item
            sub_videos_intervals: List of frame intervals for each sub video

        Returns:
            Dict containing metadata for new items
        """
        user_meta = item.metadata.get('user', {})
        origin_video_name = user_meta.get(
            'origin_video_name',
            item.metadata.get('origin_video_name', os.path.basename(item.filename))
        )
        time = user_meta.get(
            'time',
            item.metadata.get('time', datetime.datetime.now().isoformat())
        )
        return {"origin_video_name": origin_video_name, "time": time, "sub_videos_intervals": sub_videos_intervals}

    @staticmethod
    def _compute_intervals(ctx: VideoContext) -> List[List[int]]:
        """Dispatch to the appropriate interval method based on split_type."""
        if ctx.split_type == "num_frames":
            return ServiceRunner.get_sub_videos_intervals_by_num_frames(ctx, ctx.splitter_arg)
        elif ctx.split_type == "num_splits":
            return ServiceRunner.get_sub_videos_intervals_by_num_splits(ctx, ctx.splitter_arg)
        elif ctx.split_type == "out_length":
            return ServiceRunner.get_sub_videos_intervals_by_length(ctx, ctx.splitter_arg)
        else:
            raise ValueError(f"Invalid split type: {ctx.split_type}")

    @staticmethod
    def get_sub_videos_intervals_by_num_frames(
        ctx: VideoContext, num_frames_per_split: Union[int, List[int]]
    ) -> List[List[int]]:
        """
        Compute sub video intervals based on number of frames per split.

        Args:
            ctx: Per-call video context
            num_frames_per_split (int or list): Number of frames per split or list of frame counts.

        Returns:
            list: List of [start_frame, end_frame] intervals for each sub video.
        """
        assert ctx.n_overlap >= 0, "overlap must be greater than or equal to 0"
        sub_videos_intervals = []
        start_frame = 0
        end_frame = 0
        if isinstance(num_frames_per_split, list):
            for num_frames in num_frames_per_split:
                assert num_frames > 0, "number of frames per split must be greater then 0"
                end_frame = start_frame + num_frames - 1
                sub_videos_intervals.append([start_frame, min(end_frame, ctx.total_frames - 1)])
                start_frame += max(num_frames - ctx.n_overlap, 0)
                if end_frame >= ctx.total_frames - 1:
                    break
        else:
            assert (
                num_frames_per_split > ctx.n_overlap >= 0
            ), "number of frames per split must be greater then overlap "
            while ctx.total_frames - start_frame >= num_frames_per_split:
                end_frame = start_frame + num_frames_per_split - 1
                sub_videos_intervals.append([start_frame, min(end_frame, ctx.total_frames - 1)])
                start_frame += num_frames_per_split - ctx.n_overlap
                if end_frame >= ctx.total_frames - 1:
                    break

        if end_frame < ctx.total_frames - 1:
            sub_videos_intervals.append([start_frame, ctx.total_frames - 1])
        return sub_videos_intervals

    @staticmethod
    def get_sub_videos_intervals_by_num_splits(ctx: VideoContext, num_splits: int) -> List[List[int]]:
        """
        Compute sub video intervals based on desired number of splits.

        Args:
            ctx: Per-call video context
            num_splits (int): Number of sub videos to create.

        Returns:
            list: List of [start_frame, end_frame] intervals for each sub video.
        """
        assert num_splits > 0, "number of splits must be greater then 0"
        assert (
            ctx.total_frames > ctx.n_overlap >= 0
        ), "overlap must be greater than or equal to 0 and less than total_frames"
        sub_videos_intervals = []
        total_frames_with_overlap = ctx.total_frames + (num_splits - 1) * ctx.n_overlap
        sub_video_frames_count = total_frames_with_overlap // num_splits
        sub_videos_with_extra_frame = total_frames_with_overlap % num_splits
        start_frame = 0
        extra_frame = False
        for i in range(num_splits):
            if sub_videos_with_extra_frame > 0:
                extra_frame = True
            end_frame = start_frame + sub_video_frames_count + (1 if extra_frame else 0) - 1
            sub_videos_intervals.append([start_frame, end_frame])
            start_frame += sub_video_frames_count + (1 if extra_frame else 0) - ctx.n_overlap
            sub_videos_with_extra_frame -= 1
            extra_frame = False
        return sub_videos_intervals

    @staticmethod
    def get_sub_videos_intervals_by_length(
        ctx: VideoContext, out_length: Union[float, List[float]]
    ) -> List[List[int]]:
        """
        Compute sub video intervals based on desired output length in seconds.

        Args:
            ctx: Per-call video context
            out_length (Union[float, List[float]]): Length of each sub video in seconds.

        Returns:
            list: List of [start_frame, end_frame] intervals for each sub video.
        """
        if isinstance(out_length, list):
            num_frames_per_split = []
            for ltime in out_length:
                assert ltime > 0, "length of sub video must be greater then 0"
                num_frames_per_split.append(int(ltime * ctx.fps))
        else:
            assert out_length > 0, "length of sub video must be greater then 0"
            num_frames_per_split = int(out_length * ctx.fps)
        return ServiceRunner.get_sub_videos_intervals_by_num_frames(ctx, num_frames_per_split)

    # ── Segment creation strategies ───────────────────────────

    @staticmethod
    def _create_opencv_segments(
        ctx: VideoContext,
        input_video: str,
        sub_videos_dir: str,
        sub_videos_intervals: List[List[int]],
    ) -> None:
        """Create sub-video segments via OpenCV frame-by-frame re-encoding."""
        cap = cv2.VideoCapture(input_video)
        try:
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {input_video}")
            for i, (start_frame, end_frame) in enumerate(sub_videos_intervals):
                out_name = f"{ctx.input_base_name}_{str(i).zfill(ctx.max_fc_len)}.{ctx.video_type}"
                out_path = os.path.join(sub_videos_dir, out_name)
                writer = cv2.VideoWriter(out_path, ctx.fourcc, ctx.fps, ctx.frame_size)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                for _ in range(start_frame, end_frame + 1):
                    ret, frame = cap.read()
                    if ret:
                        writer.write(frame)
                    else:
                        break
                writer.release()
        finally:
            cap.release()

    @staticmethod
    def _create_ffmpeg_segments(
        ctx: VideoContext,
        input_video: str,
        sub_videos_dir: str,
        sub_videos_intervals: List[List[int]],
    ) -> None:
        """Create sub-video segments via FFmpeg stream-copy (preserves all streams)."""
        for i, (start_frame, end_frame) in enumerate(sub_videos_intervals):
            start_sec = start_frame / ctx.fps
            duration_sec = (end_frame - start_frame + 1) / ctx.fps
            out_name = f"{ctx.input_base_name}_{str(i).zfill(ctx.max_fc_len)}.{ctx.video_type}"
            out_path = os.path.join(sub_videos_dir, out_name)
            ServiceRunner._ffmpeg_segment(input_video, out_path, start_sec, duration_sec)

    @staticmethod
    def _ffmpeg_segment(input_path: str, output_path: str, start: float, duration: float):
        """Extract a single segment with FFmpeg using stream-copy."""
        cmd = [
            'ffmpeg', '-y',
            '-ss', f'{start:.3f}',
            '-t', f'{duration:.3f}',
            '-i', input_path,
            '-c', 'copy',
            '-avoid_negative_ts', 'make_zero',
            output_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg segment failed: {e.stderr}")
            raise

    # ── Annotation carry-over ─────────────────────────────────

    @staticmethod
    def _carry_annotations(
        item: dl.Item,
        sub_videos_items: List[dl.Item],
        sub_videos_intervals: List[List[int]],
    ) -> None:
        """Copy frame-level annotations from the source video to each sub-video."""
        annotations = item.annotations.list()
        for sub_video_item, (start_frame, end_frame) in zip(sub_videos_items, sub_videos_intervals):
            sub_video_item.fps = item.fps
            builder = sub_video_item.annotations.builder()
            for local_idx, orig_idx in enumerate(range(start_frame, end_frame + 1)):
                for ann in annotations.get_frame(frame_num=orig_idx).annotations:
                    builder.add(
                        annotation_definition=dl.Box(
                            top=ann.top,
                            left=ann.left,
                            bottom=ann.bottom,
                            right=ann.right,
                            label=ann.label,
                        ),
                        object_visible=ann.object_visible,
                        frame_num=local_idx,
                        object_id=int(ann.id, 16),
                    )
            sub_video_item.annotations.upload(annotations=builder)

    # ── Entry point ───────────────────────────────────────────

    def video_to_videos(self, item: dl.Item, context: dl.Context) -> List[dl.Item]:
        """
        Split a video into multiple sub-videos based on specified parameters.

        Args:
            item (dl.Item): Input video item to split
            context (dl.Context): Function context containing configuration

        Returns:
            List[dl.Item]: List of created sub-video items
        """
        logger.info('Running service Video To Videos')

        config = context.node.metadata['customNodeConfig']
        use_ffmpeg = config.get('use_ffmpeg', False)
        carry_annotations = config.get('carry_annotations', True)

        local_input_folder = tempfile.mkdtemp(suffix="_input")
        local_output_folder = tempfile.mkdtemp(suffix="_output")

        logger.info(f"Downloading video to {local_input_folder}")
        input_video = item.download(local_path=local_input_folder)

        try:
            ctx = self._build_context(config, item)
            sub_videos_intervals = self._compute_intervals(ctx)
            logger.info(f"sub_videos_intervals: {sub_videos_intervals}")

            sub_videos_dir = os.path.join(
                local_output_folder, os.path.basename(ctx.dl_output_folder.rstrip('/'))
            )
            os.makedirs(sub_videos_dir)

            create_segments = self._create_ffmpeg_segments if use_ffmpeg else self._create_opencv_segments
            create_segments(ctx, input_video, sub_videos_dir, sub_videos_intervals)

            item_metadata = self.get_new_items_metadata(item, sub_videos_intervals)
            remote_path = "/" + os.path.dirname(ctx.dl_output_folder.rstrip('/')).lstrip('/')

            sub_videos_items = item.dataset.items.upload(
                local_path=sub_videos_dir,
                remote_path=remote_path,
                item_metadata=item_metadata,
            )
            if isinstance(sub_videos_items, dl.Item):
                sub_videos_items = [sub_videos_items]
            sub_videos_items = sorted(list(sub_videos_items), key=lambda x: x.name)

            if carry_annotations:
                logger.info("Carrying annotations to sub-videos")
                self._carry_annotations(item, sub_videos_items, sub_videos_intervals)

            logger.info(f"Uploaded {len(sub_videos_items)} sub-videos to {ctx.dl_output_folder}")
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise

        return sub_videos_items
