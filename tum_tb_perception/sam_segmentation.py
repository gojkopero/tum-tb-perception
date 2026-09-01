#!/usr/bin/env python3

"""
Wrapper around MobileSAM / Segment-Anything for taskboard segmentation.

Provides a class that loads the SAM model and segments the taskboard
from an RGB image given a bounding-box or point prompt.
"""

import numpy as np


class SAMSegmenter:
    """
    Segment-Anything wrapper that supports MobileSAM (preferred) and
    the standard SAM ViT-B as fallback.

    Parameters
    ----------
    model_type : str
        SAM model registry key ('vit_t' for MobileSAM, 'vit_b' / 'vit_h' for SAM).
    checkpoint_path : str
        Path to the SAM/MobileSAM checkpoint (.pt file).
    device : str
        Torch device string ('cpu' or 'cuda').
    """

    def __init__(self, model_type='vit_t', checkpoint_path=None, device='cpu'):
        self.device = device
        self.model_type = model_type
        self.name = self.__class__.__name__

        # Try MobileSAM first, then fall back to segment-anything:
        try:
            from mobile_sam import sam_model_registry, SamPredictor
            print(f'[INFO] [{self.name}] Using mobile_sam backend')
        except ImportError:
            try:
                from segment_anything import sam_model_registry, SamPredictor
                print(f'[INFO] [{self.name}] Using segment_anything backend')
            except ImportError:
                raise ImportError(
                    'Neither mobile_sam nor segment_anything is installed. '
                    'Install with: pip install mobile-sam  OR  pip install segment-anything'
                )

        if checkpoint_path is None:
            raise ValueError('SAM checkpoint_path must be provided.')

        print(f'[INFO] [{self.name}] Loading SAM model (type={model_type}) '
              f'from {checkpoint_path} on {device}')

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device)
        sam.eval()

        self.predictor = SamPredictor(sam)
        print(f'[INFO] [{self.name}] SAM model loaded successfully')

    def segment_from_bbox(self, image_rgb, bbox):
        """
        Segment the taskboard using a bounding-box prompt.

        Parameters
        ----------
        image_rgb : ndarray (H, W, 3)
            RGB image (uint8).
        bbox : array-like (4,)
            Bounding box [xmin, ymin, xmax, ymax] in pixel coordinates.

        Returns
        -------
        mask : ndarray (H, W), bool
            Binary segmentation mask.
        score : float
            Model confidence score for the mask.
        """
        self.predictor.set_image(image_rgb)

        box_np = np.array(bbox, dtype=np.float32)
        masks, scores, _ = self.predictor.predict(
            box=box_np,
            multimask_output=False,
        )
        return masks[0], float(scores[0])

    def segment_from_points(self, image_rgb, point_coords, point_labels=None):
        """
        Segment the taskboard using point prompts.

        Parameters
        ----------
        image_rgb : ndarray (H, W, 3)
            RGB image (uint8).
        point_coords : ndarray (N, 2)
            Prompt points in pixel coordinates (x, y).
        point_labels : ndarray (N,), optional
            1 = foreground, 0 = background. Defaults to all foreground.

        Returns
        -------
        mask : ndarray (H, W), bool
            Binary segmentation mask.
        score : float
            Model confidence score for the mask.
        """
        self.predictor.set_image(image_rgb)

        coords = np.array(point_coords, dtype=np.float32)
        if point_labels is None:
            point_labels = np.ones(len(coords), dtype=np.int32)

        masks, scores, _ = self.predictor.predict(
            point_coords=coords,
            point_labels=point_labels,
            multimask_output=False,
        )
        return masks[0], float(scores[0])
