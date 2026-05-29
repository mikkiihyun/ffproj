#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Optional

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


def _order_points(pts):
    pts = np.asarray(pts, dtype="float32").reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(d)]
    bottom_left = pts[np.argmax(d)]

    return top_left, top_right, bottom_right, bottom_left


def _border_dark_score(image, quad):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    pts = np.asarray(quad, dtype=np.float32)
    scores = []

    for i in range(4):
        p = pts[i]
        q = pts[(i + 1) % 4]
        samples = max(30, int(np.linalg.norm(q - p)))
        xs = np.linspace(p[0], q[0], samples)
        ys = np.linspace(p[1], q[1], samples)
        xi = np.clip(np.round(xs).astype(np.int32), 0, w - 1)
        yi = np.clip(np.round(ys).astype(np.int32), 0, h - 1)
        scores.append(np.mean(gray[yi, xi] < 120))

    return float(np.mean(scores))


def _inner_brightness_score(image, quad):
    src = np.asarray(quad, dtype=np.float32)
    dst = np.array([[0, 0], [79, 0], [79, 44], [0, 44]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(src, dst)
    patch = cv2.warpPerspective(image, transform, (80, 45))
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    inner = gray[5:40, 5:75]
    return float(np.mean(inner) / 255.0)


def detect_monitor(image):
    """
    Detect the monitor corners in the input BGR image.

    Return:
      top_left, top_right, bottom_right, bottom_left

    Each point should be an (x, y) pair in the original image coordinate system.
    Return (None, None, None, None) if detection fails.
    """
    if image is None:
        return None, None, None, None

    orig_h, orig_w = image.shape[:2]
    max_work_size = 1200
    resize_scale = 1.0

    if max(orig_h, orig_w) > max_work_size:
        resize_scale = max_work_size / float(max(orig_h, orig_w))
        image_work = cv2.resize(
            image,
            (int(round(orig_w * resize_scale)), int(round(orig_h * resize_scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        image_work = image

    h, w = image_work.shape[:2]
    gray = cv2.cvtColor(image_work, cv2.COLOR_BGR2GRAY)

    k = max(3, int(min(h, w) * 0.006))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

    best_pts = None
    best_score = -1.0
    image_area = h * w

    def evaluate_mask(mask):
        nonlocal best_pts, best_score

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < image_area * 0.01:
                continue

            peri = cv2.arcLength(cnt, True)
            candidates = []

            for eps in (0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    candidates.append((approx.reshape(4, 2), False))
                    break

            if not candidates:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                if cv2.contourArea(box.astype(np.float32)) > 0:
                    candidates.append((box, True))

            for pts, is_fallback in candidates:
                tl, tr, br, bl = _order_points(pts)
                ordered = np.array([tl, tr, br, bl], dtype=np.float32)
                poly_area = cv2.contourArea(ordered)

                if poly_area <= 0 or poly_area > image_area * 0.85:
                    continue

                x, y, bw, bh = cv2.boundingRect(ordered.astype(np.int32))
                touches_edge = x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2
                edge_vertices = np.sum(
                    (ordered[:, 0] <= 2)
                    | (ordered[:, 1] <= 2)
                    | (ordered[:, 0] >= w - 2)
                    | (ordered[:, 1] >= h - 2)
                )

                if edge_vertices >= 2:
                    continue
                if (bw > w * 0.96 and touches_edge) or (
                    bw > w * 0.96 and bh > h * 0.80
                ):
                    continue

                width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) * 0.5
                height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) * 0.5
                if width < 1 or height < 1:
                    continue

                ratio = width / height
                ratio_error = min(abs(ratio - 16 / 9), abs((1 / ratio) - 16 / 9))
                if ratio_error > 1.6:
                    continue

                border_score = _border_dark_score(image_work, ordered)
                if border_score < 0.05:
                    continue

                inner_score = _inner_brightness_score(image_work, ordered)
                ratio_score = max(0.0, 1.0 - ratio_error / 1.6)
                score = (
                    poly_area
                    * (0.15 + border_score)
                    * (0.25 + ratio_score)
                    * max(0.08, inner_score * inner_score)
                )

                if is_fallback:
                    score *= 0.1

                if score > best_score:
                    best_score = score
                    best_pts = (tl, tr, br, bl)

    for thr in (35, 50, 70, 90):
        evaluate_mask(cv2.inRange(gray, 0, thr))

    hsv = cv2.cvtColor(image_work, cv2.COLOR_BGR2HSV)
    evaluate_mask(cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 70, 210])))

    mask = cv2.inRange(gray, 0, 85)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    evaluate_mask(mask)

    if best_pts is None:
        return None, None, None, None

    if resize_scale != 1.0:
        inv_scale = 1.0 / resize_scale
        best_pts = tuple(np.array(p, dtype=np.float32) * inv_scale for p in best_pts)

    return best_pts


def rectify_monitor(image, top_left, top_right, bottom_right, bottom_left):
    """
    Perspective-transform the detected monitor into a front-facing view.

    Return:
      rectified BGR image

    Return None if rectification fails.
    """
    if (
        image is None
        or top_left is None
        or top_right is None
        or bottom_right is None
        or bottom_left is None
    ):
        return None

    pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    ordered = list(_order_points(pts))
    edge_lengths = [
        np.linalg.norm(ordered[(i + 1) % 4] - ordered[i])
        for i in range(4)
    ]

    horizontal_pair_avg = (edge_lengths[0] + edge_lengths[2]) * 0.5
    vertical_pair_avg = (edge_lengths[1] + edge_lengths[3]) * 0.5

    if vertical_pair_avg > horizontal_pair_avg:
        ordered = [ordered[1], ordered[2], ordered[3], ordered[0]]
        edge_lengths = edge_lengths[1:] + edge_lengths[:1]

    max_width = max(1, int(max(edge_lengths[0], edge_lengths[2])))
    target_ratio = 16.0 / 9.0
    max_height = max(1, int(round(max_width / target_ratio)))

    max_rectified_width = 1000
    if max_width > max_rectified_width:
        scale = max_rectified_width / float(max_width)
        max_width = max_rectified_width
        max_height = max(1, int(round(max_height * scale)))

    src = np.array(ordered, dtype="float32")
    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    transform = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, transform, (max_width, max_height))


def _extend_line_on_edges(edges, line):
    x1, y1, x2, y2 = line
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    length = math.hypot(dx, dy)

    if length < 1:
        return line

    direction = np.array([dx / length, dy / length], dtype=np.float32)
    p0 = np.array([float(x1), float(y1)], dtype=np.float32)
    ys, xs = np.where(edges > 0)

    if len(xs) == 0:
        return line

    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    rel = pts - p0
    proj = rel[:, 0] * direction[0] + rel[:, 1] * direction[1]
    perp = np.abs(rel[:, 0] * direction[1] - rel[:, 1] * direction[0])

    h, w = edges.shape
    dist_tol = max(2.5, min(h, w) * 0.006)
    near_proj = np.sort(proj[perp <= dist_tol])

    if len(near_proj) < 2:
        return line

    raw_min = min(0.0, length)
    raw_max = max(0.0, length)
    max_gap = max(6.0, min(h, w) * 0.025)
    clusters = []
    start = near_proj[0]
    prev = near_proj[0]

    for value in near_proj[1:]:
        if value - prev > max_gap:
            clusters.append((start, prev))
            start = value
        prev = value

    clusters.append((start, prev))
    best_cluster = None
    best_span = -1.0

    for start, end in clusters:
        overlaps_raw = end >= raw_min - max_gap and start <= raw_max + max_gap
        span = end - start
        if overlaps_raw and span > best_span:
            best_span = span
            best_cluster = (start, end)

    if best_cluster is None or best_span <= length * 0.8:
        return line

    start, end = best_cluster
    pt1 = p0 + direction * start
    pt2 = p0 + direction * end

    return (
        int(round(np.clip(pt1[0], 0, w - 1))),
        int(round(np.clip(pt1[1], 0, h - 1))),
        int(round(np.clip(pt2[0], 0, w - 1))),
        int(round(np.clip(pt2[1], 0, h - 1))),
    )


def detect_line(rectified):
    """
    Detect the longest line inside the rectified monitor image.

    Return:
      (x1, y1, x2, y2)

    Return None if line detection fails.
    """
    if rectified is None:
        return None

    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    margin_x = max(3, int(w * 0.08))
    margin_y = max(3, int(h * 0.08))
    roi = rectified[margin_y : h - margin_y, margin_x : w - margin_x]

    if roi.size == 0:
        return None

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    edges_gray = cv2.Canny(roi_blur, 20, 90)

    bg_color = np.median(roi.reshape(-1, 3), axis=0).astype(np.float32)
    color_dist = np.linalg.norm(roi.astype(np.float32) - bg_color, axis=2)

    if color_dist.max() > 1e-6:
        color_dist = np.clip(
            color_dist * (255.0 / color_dist.max()),
            0,
            255,
        ).astype(np.uint8)
    else:
        color_dist = np.zeros_like(roi_gray)

    color_dist = cv2.GaussianBlur(color_dist, (3, 3), 0)
    edges_color = cv2.Canny(color_dist, 12, 50)
    edges = cv2.bitwise_or(edges_gray, edges_color)

    min_len = max(12, int(min(w, h) * 0.04))
    max_gap = max(1, int(min(w, h) * 0.006))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=max(8, int(min(w, h) * 0.018)),
        minLineLength=min_len,
        maxLineGap=max(2, max_gap),
    )

    if lines is None:
        return None

    candidates = []
    for item in lines.reshape(-1, 4):
        x1, y1, x2, y2 = item
        raw = (int(x1), int(y1), int(x2), int(y2))
        extended = _extend_line_on_edges(edges, raw)
        ex1, ey1, ex2, ey2 = extended
        extended_len = float(np.hypot(ex2 - ex1, ey2 - ey1))
        angle = calculate_angle((ex1, ey1, ex2, ey2))

        if angle is None:
            continue

        candidates.append((extended_len, abs(angle), extended))

    if not candidates:
        return None

    non_cardinal = [item for item in candidates if 10 <= item[1] <= 80]

    if non_cardinal:
        best_len = max(item[0] for item in candidates)
        best_non_cardinal = max(non_cardinal, key=lambda item: item[0])
        if best_non_cardinal[0] >= best_len * 0.35:
            best = best_non_cardinal[2]
        else:
            best = max(candidates, key=lambda item: item[0])[2]
    else:
        best = max(candidates, key=lambda item: item[0])[2]

    x1, y1, x2, y2 = best

    return (
        int(x1 + margin_x),
        int(y1 + margin_y),
        int(x2 + margin_x),
        int(y2 + margin_y),
    )


def calculate_angle(line) -> Optional[float]:
    """
    Calculate the line angle in degrees.

    The angle must be expressed in the rectified monitor coordinate system.
    Return None if the angle cannot be calculated.
    """
    if line is None:
        return None

    x1, y1, x2, y2 = line

    if x1 == x2 and y1 == y2:
        return None

    if y1 <= y2:
        top_x, top_y, bottom_x, bottom_y = x1, y1, x2, y2
    else:
        top_x, top_y, bottom_x, bottom_y = x2, y2, x1, y1

    dx = bottom_x - top_x
    dy = bottom_y - top_y
    angle = math.degrees(math.atan2(dx, dy))

    if angle >= 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    return angle


class LineDetector(Node):
    def __init__(self) -> None:
        super().__init__("line_detector_node")

        self.declare_parameter("topic_image", "/camera/camera/color/image_raw")
        self.declare_parameter("topic_student", "/student/angle")

        topic_image = str(self.get_parameter("topic_image").value)
        topic_student = str(self.get_parameter("topic_student").value)

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            topic_image,
            self.image_callback,
            10,
        )

        self.angle_pub = self.create_publisher(
            Float32,
            topic_student,
            10,
        )

        self.line_pub = self.create_publisher(
            Image,
            "/debug/line",
            10,
        )

        self.get_logger().info(
            f"Line detector started. Subscribing to {topic_image!r}, "
            f"publishing to {topic_student!r}."
        )

    def image_callback(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Failed to convert image: {exc!r}")
            return

        # 1. Detect monitor.
        top_left, top_right, bottom_right, bottom_left = detect_monitor(image)
        if any(p is None for p in (top_left, top_right, bottom_right, bottom_left)):
            self.get_logger().warning("Monitor not detected.")
            return

        # 2. Rectify monitor.
        rectified = rectify_monitor(image, top_left, top_right, bottom_right, bottom_left)
        if rectified is None:
            self.get_logger().warning("Monitor not rectified.")
            return

        # 3. Detect line.
        line = detect_line(rectified)
        if line is None:
            self.get_logger().warning("Line not detected.")
            return
        self._debug_line(msg, rectified, line)

        # 4. Calculate and publish angle.
        angle = calculate_angle(line)
        if angle is None:
            self.get_logger().warning("Angle not calculated.")
            return

        angle_msg = Float32()
        angle_msg.data = float(angle)
        self.angle_pub.publish(angle_msg)

        self.get_logger().info(f"Line angle: {float(angle):.2f} deg")

    def _debug_line(self, msg, rectified, line) -> None:
        debug_line = rectified.copy()

        x1, y1, x2, y2 = line
        cv2.line(
            debug_line,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            6,
        )

        debug_line_msg = self.bridge.cv2_to_imgmsg(debug_line, encoding="bgr8")
        debug_line_msg.header = msg.header
        self.line_pub.publish(debug_line_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LineDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
