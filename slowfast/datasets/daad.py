#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""DAAD-X multi-view video loader for ADM/M2MVT reproduction."""

import csv
import os
import random

import numpy as np
import torch
import torch.utils.data

import slowfast.utils.logging as logging
from slowfast.utils.env import pathmgr

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY


logger = logging.get_logger(__name__)

_VIEWS = ("driver", "front", "left", "right", "rear", "aria_gaze")
_NUM_CLASSES = 7


def _with_mp4_suffix(video_id):
    return video_id if os.path.splitext(video_id)[1] else "{}.mp4".format(video_id)


def _rng_state():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state()


def _set_rng_state(state):
    py_state, np_state, torch_state = state
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.random.set_rng_state(torch_state)


@DATASET_REGISTRY.register()
class Daad(torch.utils.data.Dataset):
    """
    DAAD-X video loader.

    Each sample contains six synchronized videos with matching filenames under
    the driver/front/left/right/rear/ariagaze folders. The loader returns a
    dict of tensors keyed by view name, plus the maneuver class target.
    """

    def __init__(self, cfg, mode, num_retries=30):
        assert mode in ["train", "val", "test"], "Split '{}' not supported for DAAD".format(
            mode
        )
        self.mode = mode
        self.cfg = cfg
        self._num_retries = num_retries
        self._video_meta = {}

        if self.mode in ["train", "val"]:
            self._num_clips = 1
        else:
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS

        logger.info("Constructing DAAD {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        annotation_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "annotations", "{}.csv".format(self.mode)
        )
        assert pathmgr.exists(annotation_file), "{} not found".format(annotation_file)

        self._video_ids = []
        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []

        with pathmgr.open(annotation_file, "r") as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader):
                if len(row) == 0 or all(cell.strip() == "" for cell in row):
                    continue

                video_id = row[0].strip()
                if row_idx == 0 and video_id.lower() in ["video_id", "video", "filename"]:
                    continue

                try:
                    label = int(row[1])
                except (IndexError, ValueError) as err:
                    logger.warning(
                        "Skipping malformed DAAD row {} in {}: {} ({})".format(
                            row_idx + 1, annotation_file, row, err
                        )
                    )
                    continue

                if label < 0 or label >= _NUM_CLASSES:
                    logger.warning(
                        "Skipping DAAD row {} with invalid maneuver class {}.".format(
                            row_idx + 1, label
                        )
                    )
                    continue

                filename = _with_mp4_suffix(video_id)
                paths = {
                    view: os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, view, filename)
                    for view in _VIEWS
                }
                missing = [path for path in paths.values() if not pathmgr.exists(path)]
                if missing:
                    logger.warning(
                        "Skipping DAAD video {} because files are missing: {}".format(
                            video_id, missing
                        )
                    )
                    continue

                for clip_idx in range(self._num_clips):
                    self._video_ids.append(video_id)
                    self._path_to_videos.append(paths)
                    self._labels.append(label)
                    self._spatial_temporal_idx.append(clip_idx)
                    self._video_meta[len(self._path_to_videos) - 1] = {
                        view: {} for view in _VIEWS
                    }

        assert len(self._path_to_videos) > 0, "Failed to load DAAD split from {}".format(
            annotation_file
        )
        logger.info(
            "DAAD dataloader constructed (size: {}) from {}".format(
                len(self._path_to_videos), annotation_file
            )
        )

    def _sample_params(self, index, short_cycle_idx=None):
        if self.mode in ["train", "val"]:
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                min_scale = int(
                    round(float(min_scale) * crop_size / self.cfg.MULTIGRID.DEFAULT_S)
                )
        elif self.mode == "test":
            temporal_sample_index = (
                self._spatial_temporal_idx[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            spatial_sample_index = (
                self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        return temporal_sample_index, spatial_sample_index, min_scale, max_scale, crop_size

    def _decode_view(
        self,
        index,
        view,
        temporal_sample_index,
        num_frames,
        sampling_rate,
        min_scale,
        rng_state,
    ):
        path_to_video = self._path_to_videos[index][view]
        video_container = container.get_video_container(
            path_to_video,
            self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
            self.cfg.DATA.DECODING_BACKEND,
        )
        _set_rng_state(rng_state)
        target_fps = self.cfg.DATA.TARGET_FPS
        if self.cfg.DATA.TRAIN_JITTER_FPS > 0.0 and self.mode == "train":
            target_fps += random.uniform(0.0, self.cfg.DATA.TRAIN_JITTER_FPS)

        frames, time_idx, _ = decoder.decode(
            video_container,
            [sampling_rate],
            [num_frames],
            temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index][view],
            target_fps=target_fps,
            backend=self.cfg.DATA.DECODING_BACKEND,
            use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
            max_spatial_scale=min_scale,
            temporally_rnd_clips=False,
        )
        if frames is None or None in frames:
            raise RuntimeError("Failed to decode {}".format(path_to_video))
        return frames[0], time_idx

    def _transform_view(self, frames, spatial_sample_index, min_scale, max_scale, crop_size):
        frames = frames.float() / 255.0
        frames = utils.tensor_normalize(frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
        frames = frames.permute(3, 0, 1, 2)

        scl = self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE
        asp = self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE
        relative_scales = None if self.mode != "train" or len(scl) == 0 else scl
        relative_aspect = None if self.mode != "train" or len(asp) == 0 else asp
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=(
                self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT if self.mode == "train" else False
            ),
        )
        if self.cfg.DATA.REVERSE_INPUT_CHANNEL:
            frames = frames[[2, 1, 0], :, :, :]
        return frames

    def __getitem__(self, index):
        short_cycle_idx = None
        if isinstance(index, tuple):
            index, _ = index
            if self.cfg.MULTIGRID.SHORT_CYCLE:
                index, short_cycle_idx = index

        (
            temporal_sample_index,
            spatial_sample_index,
            min_scale,
            max_scale,
            crop_size,
        ) = self._sample_params(index, short_cycle_idx)

        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )

        for i_try in range(self._num_retries):
            decode_rng = _rng_state()
            spatial_rng = _rng_state()
            try:
                decoded = {}
                time_idx = None
                for view in _VIEWS:
                    frames, view_time_idx = self._decode_view(
                        index,
                        view,
                        temporal_sample_index,
                        self.cfg.DATA.NUM_FRAMES,
                        sampling_rate,
                        min_scale,
                        decode_rng,
                    )
                    decoded[view] = frames
                    if time_idx is None:
                        time_idx = view_time_idx

                inputs = {}
                for view in _VIEWS:
                    _set_rng_state(spatial_rng)
                    inputs[view] = self._transform_view(
                        decoded[view],
                        spatial_sample_index,
                        min_scale,
                        max_scale,
                        crop_size,
                    )

                label = self._labels[index]
                metadata = {"video_id": self._video_ids[index]}
                return inputs, label, index, metadata

            except Exception as err:
                logger.warning(
                    "Failed to load DAAD video {} (idx {}, trial {}) with error: {}".format(
                        self._video_ids[index], index, i_try, err
                    )
                )
                if self.mode != "test":
                    index = random.randint(0, len(self._path_to_videos) - 1)
                    (
                        temporal_sample_index,
                        spatial_sample_index,
                        min_scale,
                        max_scale,
                        crop_size,
                    ) = self._sample_params(index, short_cycle_idx)
                continue

        raise RuntimeError(
            "Failed to fetch DAAD video after {} retries.".format(self._num_retries)
        )

    def __len__(self):
        return self.num_videos

    @property
    def num_videos(self):
        return len(self._path_to_videos)
