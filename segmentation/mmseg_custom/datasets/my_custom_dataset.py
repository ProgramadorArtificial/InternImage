# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class MyCustomDataset(CustomDataset):
    """Custom dataset.
    """
    CLASSES = ('Background', 'Macaco', 'Gato')

    PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0]]

    def __init__(self, **kwargs):
        super(MyCustomDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            #reduce_zero_label=False,
            **kwargs)