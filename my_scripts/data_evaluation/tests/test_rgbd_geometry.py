import tempfile
import unittest
import zipfile

import numpy as np

from my_scripts.data_evaluation.rgbd_geometry import (
    DepthZipReader,
    compute_metric,
    lift_points_2d,
    metric_needs_reference,
    metric_pose_points,
    mmpose_points_for_frame,
    normalize_points,
    torso_frame_from_points,
)


class RGBDGeometryTests(unittest.TestCase):
    def setUp(self):
        self.points = {
            "left_shoulder": np.array([-0.2, 0.0, 0.0]),
            "right_shoulder": np.array([0.2, 0.0, 0.0]),
            "left_hip": np.array([-0.2, 1.0, 0.0]),
            "right_hip": np.array([0.2, 1.0, 0.0]),
            "nose": np.array([0.0, 0.0, 0.2]),
            "right_elbow": np.array([1.2, 1.0, 0.0]),
            "right_wrist": np.array([2.2, 0.0, 0.0]),
            "left_elbow": np.array([-1.2, 1.0, 0.0]),
            "left_wrist": np.array([-2.2, 0.0, 0.0]),
        }

    def test_supported_metric_conventions(self):
        torso = torso_frame_from_points(self.points)
        self.assertIsNotNone(torso)
        self.assertAlmostEqual(np.linalg.norm(torso["inferior"]), 1.0)
        self.assertAlmostEqual(np.dot(torso["inferior"], torso["right"]), 0.0)
        self.assertAlmostEqual(
            compute_metric(self.points, "right_elbow_flexion"), 90.0
        )
        self.assertAlmostEqual(
            compute_metric(self.points, "left_elbow_flexion"), 90.0
        )
        # Plane-angle ROM tests report the limb elevation from the body's down
        # axis (down = side_hip - side_shoulder). The arm here points (out, down)
        # so the elevation is 45 deg; the test label (flex/extension/abduction)
        # distinguishes the direction.
        for metric in (
            "right_shoulder_adduction",
            "left_shoulder_adduction",
            "right_shoulder_flexion",
            "left_shoulder_flexion",
            "right_shoulder_abduction",
            "left_shoulder_abduction",
        ):
            self.assertAlmostEqual(compute_metric(self.points, metric), 45.0)

    def test_lower_limb_metrics(self):
        # Standing, then knee flexed ~90.
        pts = {
            "left_shoulder": np.array([0.0, 0.0, 1.5]),
            "left_hip": np.array([0.0, 0.0, 0.9]),
            "left_knee": np.array([0.0, 0.0, 0.45]),
            "left_ankle": np.array([0.0, 0.0, 0.10]),
            "left_big_toe": np.array([0.0, 0.25, 0.10]),
        }
        # Neutral: knee straight -> 0, hip hanging -> 0, ankle ~neutral.
        self.assertAlmostEqual(compute_metric(pts, "left_knee_flexion"), 0.0, places=5)
        self.assertAlmostEqual(compute_metric(pts, "left_hip_flexion"), 0.0, places=5)
        # Flex the knee: shank rotates back so knee->ankle is horizontal.
        pts["left_ankle"] = np.array([0.0, 0.4, 0.45])
        self.assertAlmostEqual(compute_metric(pts, "left_knee_flexion"), 90.0, places=3)
        # Raise the thigh forward to horizontal (hip flexion ~90).
        pts = {
            "left_shoulder": np.array([0.0, 0.0, 1.5]),
            "left_hip": np.array([0.0, 0.0, 0.9]),
            "left_knee": np.array([0.0, 0.4, 0.9]),
            "left_ankle": np.array([0.0, 0.4, 0.45]),
            "left_big_toe": np.array([0.0, 0.65, 0.4]),
        }
        self.assertAlmostEqual(compute_metric(pts, "left_hip_flexion"), 90.0, places=3)

    def test_metric_reference_helpers(self):
        self.assertTrue(metric_needs_reference("right_shoulder_flexion"))
        self.assertTrue(metric_needs_reference("left_hip_flexion"))
        self.assertFalse(metric_needs_reference("right_elbow_flexion"))
        self.assertFalse(metric_needs_reference("left_knee_flexion"))
        self.assertFalse(metric_needs_reference("right_ankle_plantarflexion"))
        self.assertEqual(
            metric_pose_points("right_shoulder_flexion"),
            (["right_shoulder", "right_hip"], ["right_shoulder", "right_elbow"]),
        )
        self.assertEqual(
            metric_pose_points("left_hip_flexion"),
            (["left_shoulder", "left_hip"], ["left_hip", "left_knee"]),
        )
        self.assertEqual(
            metric_pose_points("right_elbow_flexion"),
            (None, ["right_shoulder", "right_elbow", "right_wrist"]),
        )

    def test_reference_down_neutralizes_lean(self):
        # The neutral (T1) pose provides a fixed down axis. Even if the peak
        # (T2) pose leans to the side, the ROM uses the neutral reference, so the
        # same arm elevation reports the same ROM regardless of lean.
        neutral = {
            "right_shoulder": np.array([0.0, 0.0, 0.0]),
            "right_hip": np.array([0.0, 0.0, -0.8]),
        }
        reference_down = neutral["right_hip"] - neutral["right_shoulder"]
        peak_upright = {
            "right_shoulder": np.array([0.0, 0.0, 0.0]),
            "right_elbow": np.array([1.0, 0.0, 0.0]),
            "right_hip": np.array([0.0, 0.0, -0.8]),
        }
        peak_lean = {
            "right_shoulder": np.array([0.0, 0.0, 0.0]),
            "right_elbow": np.array([1.0, 0.0, 0.0]),
            "right_hip": np.array([1.0, 0.0, -0.8]),
        }
        self.assertAlmostEqual(
            compute_metric(
                peak_upright, "right_shoulder_flexion", reference_down=reference_down
            ),
            90.0,
            places=3,
        )
        self.assertAlmostEqual(
            compute_metric(
                peak_lean, "right_shoulder_flexion", reference_down=reference_down
            ),
            90.0,
            places=3,
        )
        # Without a reference, the same-frame down axis leans, so the ROM drops.
        self.assertLess(
            compute_metric(peak_lean, "right_shoulder_flexion"), 90.0
        )

    def test_invalid_required_point_returns_nan(self):
        points = dict(self.points)
        points.pop("right_wrist")
        self.assertTrue(
            np.isnan(compute_metric(points, "right_elbow_flexion"))
        )

    def test_depth_reader_and_lifting(self):
        depth = np.array([[1.0, 2.0], [3.0, np.nan]], dtype="<f4")
        header = (
            "width:2\nheight:2\ndata_length:16\n"
            "timestamp:12.5\nfx:100\nfy:100\nox:1\noy:1\n\n"
        ).encode()
        with tempfile.NamedTemporaryFile(suffix=".zip") as temporary:
            with zipfile.ZipFile(temporary.name, "w") as archive:
                archive.writestr("frame_1.bin", header + depth.tobytes())
            reader = DepthZipReader(temporary.name)
            try:
                self.assertEqual(reader.read(0).shape, (2, 2))
                self.assertEqual(reader.timestamp(0), 12.5)
                lifted = lift_points_2d(
                    {"point": {"x": 0.0, "y": 0.0}},
                    reader,
                    0,
                    rgb_width=4,
                    rgb_height=4,
                    depth_sample_radius=0,
                )
                self.assertTrue(np.isfinite(lifted.points["point"]).all())
                self.assertEqual(lifted.frame_index, 0)
            finally:
                reader.close()

    def test_normalize_points(self):
        normalized = normalize_points(
            {"a": {"x": 1, "y": 2}, "b": [3, 4], "bad": {"x": 1}}
        )
        self.assertEqual(normalized, {"a": {"x": 1.0, "y": 2.0}, "b": {"x": 3.0, "y": 4.0}})

    def test_mmpose_confidence_threshold(self):
        labels = ["right_elbow", "left_elbow"]
        keypoints = np.asarray([[[10.0, 20.0], [30.0, 40.0]]])
        scores = np.asarray([[0.49, 0.50]])
        points = mmpose_points_for_frame(
            labels, keypoints, scores, 0, score_threshold=0.5
        )
        self.assertNotIn("right_elbow", points)
        self.assertEqual(points["left_elbow"], {"x": 30.0, "y": 40.0})


if __name__ == "__main__":
    unittest.main()
