#!/usr/bin/env python3

"""
YOLOv8 object detector using ONNX Runtime.

Port of the C++ inference pipeline from
DesignLearnRG_euROBIN/vision/Detection to pure Python.
Runs a YOLOv8 ONNX model and returns bounding-box detections
with class IDs, confidences, bounding boxes, and centre points.
"""

import os
import time

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError(
        "onnxruntime is required. Install with: pip install onnxruntime"
    )


class YOLODetector:
    """
    YOLOv8 detector backed by ONNX Runtime.

    Parameters
    ----------
    model_path : str
        Path to the .onnx model file.
    class_names : list[str]
        Ordered list of class names matching the model output indices.
    conf_threshold : float
        Minimum confidence for a detection to be kept.
    iou_threshold : float
        IoU threshold for non-maximum suppression.
    img_size : int
        Square input size expected by the model (default 640).
    """

    def __init__(
        self,
        model_path: str,
        class_names: list,
        conf_threshold: float = 0.2,
        iou_threshold: float = 0.5,
        img_size: int = 640,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.resize_scale = 1.0

        # Create ONNX Runtime session:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, sess_options=opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Warm-up:
        dummy = np.zeros((1, 3, img_size, img_size), dtype=np.float32)
        self.session.run(self.output_names, {self.input_name: dummy})

    # ------------------------------------------------------------------
    # Pre-processing (letterbox, matches C++ PreProcess)
    # ------------------------------------------------------------------

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """Letterbox resize + BGR→RGB + HWC→CHW normalisation."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]
        if w >= h:
            self.resize_scale = w / float(self.img_size)
            new_w = self.img_size
            new_h = int(h / self.resize_scale)
        else:
            self.resize_scale = h / float(self.img_size)
            new_h = self.img_size
            new_w = int(w / self.resize_scale)

        resized = cv2.resize(img_rgb, (new_w, new_h))

        # Pad to square canvas:
        canvas = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w, :] = resized

        # HWC → CHW, float32, normalise to [0, 1]:
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # CHW
        blob = np.expand_dims(blob, axis=0)  # NCHW
        return blob

    # ------------------------------------------------------------------
    # Post-processing (NMS, matches C++ TensorProcess for YOLO_DETECT_V8)
    # ------------------------------------------------------------------

    def _postprocess(self, output: np.ndarray) -> list:
        """
        Parse YOLOv8 detection output and apply NMS.

        Parameters
        ----------
        output : ndarray of shape (1, 4+num_classes, num_anchors)

        Returns
        -------
        detections : list of dict
            Each dict: {class_id, class_name, confidence, bbox, center}
        """
        # Shape: (1, signal_result_num, stride_num) → squeeze batch dim:
        raw = output[0]  # (signal_result_num, stride_num)

        # YOLOv8 transposes: need (stride_num, signal_result_num)
        raw = raw.T  # (stride_num, 4+num_classes)

        num_classes = len(self.class_names)
        boxes = []
        confidences = []
        class_ids = []

        for row in raw:
            cx, cy, w, h = row[:4]
            class_scores = row[4 : 4 + num_classes]
            max_score = float(class_scores.max())
            class_id = int(class_scores.argmax())

            if max_score < self.conf_threshold:
                continue

            # Convert from letterbox coords to original image coords:
            left = int((cx - 0.5 * w) * self.resize_scale)
            top = int((cy - 0.5 * h) * self.resize_scale)
            width = int(w * self.resize_scale)
            height = int(h * self.resize_scale)

            boxes.append([left, top, width, height])
            confidences.append(float(max_score))
            class_ids.append(class_id)

        if len(boxes) == 0:
            return []

        # NMS:
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences,
            self.conf_threshold, self.iou_threshold,
        )

        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                x, y, w, h = boxes[idx]
                cx = x + w / 2.0
                cy = y + h / 2.0
                detections.append({
                    "class_id": class_ids[idx],
                    "class_name": self.class_names[class_ids[idx]],
                    "confidence": confidences[idx],
                    "bbox": (x, y, w, h),
                    "center": (cx, cy),
                })

        return detections

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, img_bgr: np.ndarray) -> list:
        """
        Run YOLOv8 detection on a BGR image.

        Returns
        -------
        detections : list of dict
            Each dict has keys:
            - class_id (int)
            - class_name (str)
            - confidence (float)
            - bbox (x, y, w, h) in original image pixels
            - center (cx, cy) bounding-box centre in original image pixels
        """
        blob = self._preprocess(img_bgr)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        return self._postprocess(outputs[0])


def load_class_names_from_yaml(yaml_path: str) -> list:
    """
    Load class names from a YOLO-style YAML file.

    Expected format::

        names:
          0: blue_button
          1: door_handle
          ...

    Returns an ordered list of class name strings.
    """
    names = {}
    in_names_section = False

    with open(yaml_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if stripped == "names:":
                in_names_section = True
                continue
            if in_names_section and ":" in stripped:
                parts = stripped.split(":", 1)
                try:
                    idx = int(parts[0].strip())
                    name = parts[1].strip()
                    names[idx] = name
                except ValueError:
                    break

    return [names[i] for i in sorted(names.keys())]
