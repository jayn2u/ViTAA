# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os

from vitaa.utils.paths import cuhk_pedes_paths, load_project_env


class DatasetCatalog(object):
    DATASETS = {
        "cuhkpedes_train": "train.json",
        "cuhkpedes_val": "val.json",
        "cuhkpedes_test": "test.json",
    }

    @staticmethod
    def get(name):
        if "cuhkpedes" in name:
            load_project_env()
            paths = cuhk_pedes_paths()
            split_file = DatasetCatalog.DATASETS.get(name)
            if split_file is None:
                raise RuntimeError("Dataset not available: {}".format(name))
            args = dict(
                root=str(paths["vitaa_data_root"]),
                img_dir=str(paths["img_dir"]),
                seg_dir=str(paths["segs_dir"]),
                ann_file=str(paths["annotations_dir"] / split_file),
            )
            return dict(
                factory="CUHKPEDESDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))