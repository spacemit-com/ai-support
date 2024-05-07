import os
import platform
import unittest

import cv2
try:
    # python wheel package end2end test
    import bianbuai
except ImportError:
    # local cmake project build output test
    import bianbuai_pybind11_state as bianbuai

target_arch = ["AMD64", "x86_64", "riscv64"]

class TestInferenceSession(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestInferenceSession, self).__init__(*args, **kwargs)
        self.addTypeEqualityFunc(bianbuai.Box, 'assertBoxEqual')


    def for_arch(arch = target_arch):
        def decorator(func):
            def wrapper(self):
                if self._arch in arch:
                    func(self)
                else:
                    self.skipTest(f"Test is only for {arch} platform")
            return wrapper
        return decorator


    def assertBoxEqual(self, first, second, msg=None, delta=None, label_only=False):
        """Assert that two Object Detection Box are equal."""
        self.assertIsInstance(first, bianbuai.Box, 'First argument is not a Box')
        self.assertIsInstance(second, bianbuai.Box, 'Second argument is not a Box')

        diff = ''
        if first != second:
            if (first.x1 != second.x1 or
                first.y1 != second.y1 or
                first.x2 != second.x2 or
                first.y2 != second.y2) and not label_only:
                diff += '%s != %s\n' % ([first.x1, first.y1, first.x2, first.y2],
                                     [second.x1, second.y1, second.x2, second.y2])
            if delta and delta < abs(first.score - second.score) and not label_only:
                diff += '%s < abs(%s - %s)\n' % (delta, first.score, second.score)
            if first.label_text != second.label_text:
                diff += '%s != %s\n' % (first.label_text, second.label_text)
            if first.label != second.label:
                diff += '%s != %s\n' % (first.label, second.label)
            standardMsg = self._truncateMessage("", diff)
        if diff:
            self.fail(self._formatMessage(msg, standardMsg))


    def setUp(self):
        "Method for setting up the test fixture before exercising it."
        self._data = "data"
        self._arch = platform.machine()


    def tearDown(self):
        "Method for deconstructing the test fixture after testing it."
        pass


    @for_arch(target_arch)
    def testTaskImageClassification(self):
        """
        Test task image classification
        """
        image_path = os.path.join(self._data, "imgs", "dog.jpg")
        model_path = os.path.join(self._data, "models", "squeezenet1.1-7.onnx")
        label_path = os.path.join(self._data, "models", "synset.txt")
        intra_thread_num, inter_thread_num = 2, 2
        option = bianbuai.ImageClassificationOption(model_path, label_path, intra_thread_num, inter_thread_num)
        task = bianbuai.ImageClassificationTask(option)
        result = task.Classify(cv2.imread(image_path))
        self.assertEqual(task.getInitFlag(), 0)
        self.assertEqual(result.label_text, "n02113023 Pembroke, Pembroke Welsh corgi")
        self.assertEqual(result.label_index, 263)
        self.assertAlmostEqual(result.score, 0.8347830176353455, delta=1e-5)


    @for_arch(target_arch)
    def testTaskObjectDetectionImage(self):
        """
        Test task object detection with input image
        """
        image_path = os.path.join(self._data, "imgs", "person.jpg")
        option = bianbuai.ObjectDetectionOption()
        option.model_path = os.path.join(self._data, "models", "yolov6p5_n.q.onnx")
        option.label_path = os.path.join(self._data, "models", "coco.txt")
        option.intra_threads_num, option.inter_threads_num = 2, 2
        task = bianbuai.ObjectDetectionTask(option)
        self.assertEqual(task.getInitFlag(), 0)
        output_a = task.Detect(cv2.imread(image_path))
        # TODO: add expected outputs to support test with (e.g. Windows)
        output_e = [
            bianbuai.Box(1349, 408, 1513, 790, 0.673, "person", 0),
            bianbuai.Box( 581, 439,  682, 704, 0.565, "person", 0) if self._arch != "riscv64" else bianbuai.Box( 581, 439,  677, 709, 0.565, "person", 0),
            bianbuai.Box(1476, 421, 1592, 783, 0.522, "person", 0) if self._arch != "riscv64" else bianbuai.Box(1476, 418, 1594, 783, 0.522, "person", 0),
            bianbuai.Box( 462, 439,  557, 711, 0.673, "person", 0) if self._arch != "riscv64" else bianbuai.Box( 462, 441,  559, 711, 0.522, "person", 0),
        ]
        self.assertEqual(len(output_a), len(output_e))
        for i in range(len(output_a)):
            self.assertBoxEqual(output_a[i], output_e[i], delta=1e-3, label_only=(self._arch == "AMD64"))


    @for_arch(target_arch)
    def testTaskPoseEstimationImage(self):
        """
        Test task pose estimation with input image
        """
        image_path = os.path.join(self._data, "imgs", "person.jpg")
        # object detection
        option_1 = bianbuai.ObjectDetectionOption()
        option_1.model_path = os.path.join(self._data, "models", "yolov6p5_n.q.onnx")
        option_1.label_path = os.path.join(self._data, "models", "coco.txt")
        option_1.intra_threads_num, option_1.inter_threads_num = 2, 2
        task_1 = bianbuai.ObjectDetectionTask(option_1)
        self.assertEqual(task_1.getInitFlag(), 0)
        output_1 = task_1.Detect(cv2.imread(image_path))
        # pose estimation
        option_2 = bianbuai.PoseEstimationOption()
        option_2.model_path = os.path.join(self._data, "models", "rtmpose-t.q.onnx")
        option_2.intra_threads_num, option_2.inter_threads_num = 2, 2
        task_2 = bianbuai.PoseEstimationTask(option_2)
        self.assertEqual(task_2.getInitFlag(), 0)
        output_2 = [
            task_2.Estimate(cv2.imread(image_path), box) for box in output_1
        ]
        # TODO: add expected outputs to support test with (e.g. Windows)
        output_e = [
            [
                bianbuai.PosePoint(1417, 461, 0.245499),
                bianbuai.PosePoint(1444, 451, 0.355974),
                bianbuai.PosePoint(1403, 438, 0.368249),
                bianbuai.PosePoint(1382, 447, 0.306874),
                bianbuai.PosePoint(1380, 451, 0.306874),
                bianbuai.PosePoint(1421, 486, 0.257774),
                bianbuai.PosePoint(1384, 509, 0.282324),
                bianbuai.PosePoint(1492, 562, 0.147300),
                bianbuai.PosePoint(1384, 564, 0.152968),
                bianbuai.PosePoint(1392, 680, 0.098200),
                bianbuai.PosePoint(1382, 670, 0.159575),
                bianbuai.PosePoint(1440, 602, 0.191210),
                bianbuai.PosePoint(1421, 602, 0.152968),
                bianbuai.PosePoint(1452, 608, 0.140221),
                bianbuai.PosePoint(1448, 684, 0.171850),
                bianbuai.PosePoint(1473, 763, 0.355974),
                bianbuai.PosePoint(1473, 765, 0.319149),
            ],
            [
                bianbuai.PosePoint( 596, 446, 0.484399),
                bianbuai.PosePoint( 598, 447, 0.503274),
                bianbuai.PosePoint( 592, 449, 0.515549),
                bianbuai.PosePoint( 609, 451, 0.552374),
                bianbuai.PosePoint( 592, 443, 0.589199),
                bianbuai.PosePoint( 617, 466, 0.478724),
                bianbuai.PosePoint( 593, 467, 0.540099),
                bianbuai.PosePoint( 640, 505, 0.368249),
                bianbuai.PosePoint( 596, 503, 0.380524),
                bianbuai.PosePoint( 605, 521, 0.233224),
                bianbuai.PosePoint( 605, 524, 0.282324),
                bianbuai.PosePoint( 624, 546, 0.380524),
                bianbuai.PosePoint( 614, 541, 0.380524),
                bianbuai.PosePoint( 648, 601, 0.441899),
                bianbuai.PosePoint( 622, 601, 0.380524),
                bianbuai.PosePoint( 641, 664, 0.601474),
                bianbuai.PosePoint( 635, 664, 0.552374),
            ],
            [
                bianbuai.PosePoint(1532, 458, 0.233224),
                bianbuai.PosePoint(1557, 455, 0.233224),
                bianbuai.PosePoint(1506, 446, 0.343699),
                bianbuai.PosePoint(1478, 431, 0.208674),
                bianbuai.PosePoint(1473, 440, 0.245499),
                bianbuai.PosePoint(1513, 495, 0.110475),
                bianbuai.PosePoint(1499, 491, 0.171850),
                bianbuai.PosePoint(1574, 583, 0.073650),
                bianbuai.PosePoint(1423, 590, 0.085925),
                bianbuai.PosePoint(1565, 835, 0.061375),
                bianbuai.PosePoint(1460, 677, 0.089231),
                bianbuai.PosePoint(1521, 589, 0.089231),
                bianbuai.PosePoint(1504, 596, 0.101979),
                bianbuai.PosePoint(1556, 607, 0.135025),
                bianbuai.PosePoint(1488, 609, 0.135025),
                bianbuai.PosePoint(1583, 768, 0.331424),
                bianbuai.PosePoint(1576, 756, 0.306874),
            ],
            [
                bianbuai.PosePoint( 493, 475, 0.687398),
                bianbuai.PosePoint( 492, 470, 0.724223),
                bianbuai.PosePoint( 484, 471, 0.736498),
                bianbuai.PosePoint( 499, 471, 0.724223),
                bianbuai.PosePoint( 477, 475, 0.773323),
                bianbuai.PosePoint( 514, 497, 0.724223),
                bianbuai.PosePoint( 468, 505, 0.711948),
                bianbuai.PosePoint( 532, 528, 0.589199),
                bianbuai.PosePoint( 459, 543, 0.589199),
                bianbuai.PosePoint( 536, 537, 0.331431),
                bianbuai.PosePoint( 468, 537, 0.343699),
                bianbuai.PosePoint( 518, 572, 0.527824),
                bianbuai.PosePoint( 489, 579, 0.497146),
                bianbuai.PosePoint( 528, 637, 0.564649),
                bianbuai.PosePoint( 496, 637, 0.626023),
                bianbuai.PosePoint( 536, 689, 0.589199),
                bianbuai.PosePoint( 500, 697, 0.687398),
            ],
        ] if self._arch == "x86_64" else [
            [
                bianbuai.PosePoint(1409, 562, 0.306874),
                bianbuai.PosePoint(1415, 457, 0.282324),
                bianbuai.PosePoint(1401, 550, 0.319149),
                bianbuai.PosePoint(1448, 475, 0.319149),
                bianbuai.PosePoint(1382, 473, 0.280441),
                bianbuai.PosePoint(1456, 502, 0.140221),
                bianbuai.PosePoint(1390, 502, 0.147300),
                bianbuai.PosePoint(1527, 525, 0.101979),
                bianbuai.PosePoint(1359, 573, 0.127473),
                bianbuai.PosePoint(1492, 351, 0.073650),
                bianbuai.PosePoint(1374, 351, 0.114726),
                bianbuai.PosePoint(1415, 428, 0.127473),
                bianbuai.PosePoint(1423, 444, 0.135025),
                bianbuai.PosePoint(1450, 616, 0.184125),
                bianbuai.PosePoint(1398, 639, 0.184125),
                bianbuai.PosePoint(1522, 722, 0.355974),
                bianbuai.PosePoint(1481, 728, 0.294599),
            ],
            [
                bianbuai.PosePoint( 611, 502, 0.257774),
                bianbuai.PosePoint( 615, 491, 0.405074),
                bianbuai.PosePoint( 605, 496, 0.355974),
                bianbuai.PosePoint( 605, 490, 0.380524),
                bianbuai.PosePoint( 587, 491, 0.319149),
                bianbuai.PosePoint( 623, 470, 0.208674),
                bianbuai.PosePoint( 594, 464, 0.159575),
                bianbuai.PosePoint( 605, 398, 0.135025),
                bianbuai.PosePoint( 590, 398, 0.127473),
                bianbuai.PosePoint( 603, 453, 0.135025),
                bianbuai.PosePoint( 598, 398, 0.159575),
                bianbuai.PosePoint( 631, 576, 0.280441),
                bianbuai.PosePoint( 619, 572, 0.257774),
                bianbuai.PosePoint( 650, 612, 0.254947),
                bianbuai.PosePoint( 600, 615, 0.196400),
                bianbuai.PosePoint( 645, 669, 0.454174),
                bianbuai.PosePoint( 633, 668, 0.392799),
            ],
            [
                bianbuai.PosePoint(1514, 551, 0.220949),
                bianbuai.PosePoint(1512, 566, 0.245499),
                bianbuai.PosePoint(1510, 431, 0.233224),
                bianbuai.PosePoint(1488, 520, 0.245499),
                bianbuai.PosePoint(1501, 540, 0.220949),
                bianbuai.PosePoint(1584, 516, 0.147300),
                bianbuai.PosePoint(1468, 494, 0.159575),
                bianbuai.PosePoint(1579, 572, 0.089231),
                bianbuai.PosePoint(1464, 609, 0.101979),
                bianbuai.PosePoint(1560, 746, 0.063737),
                bianbuai.PosePoint(1464, 638, 0.101979),
                bianbuai.PosePoint(1522, 834, 0.147300),
                bianbuai.PosePoint(1514, 834, 0.135025),
                bianbuai.PosePoint(1551, 834, 0.110475),
                bianbuai.PosePoint(1490, 614, 0.122750),
                bianbuai.PosePoint(1581, 834, 0.306874),
                bianbuai.PosePoint(1573, 834, 0.319149),
            ],
            [
                bianbuai.PosePoint( 484, 478, 0.478724),
                bianbuai.PosePoint( 490, 477, 0.540099),
                bianbuai.PosePoint( 481, 480, 0.503274),
                bianbuai.PosePoint( 494, 477, 0.576924),
                bianbuai.PosePoint( 474, 480, 0.552374),
                bianbuai.PosePoint( 501, 506, 0.478724),
                bianbuai.PosePoint( 470, 508, 0.503274),
                bianbuai.PosePoint( 522, 532, 0.306874),
                bianbuai.PosePoint( 468, 541, 0.368249),
                bianbuai.PosePoint( 530, 536, 0.171850),
                bianbuai.PosePoint( 486, 562, 0.257774),
                bianbuai.PosePoint( 511, 567, 0.380524),
                bianbuai.PosePoint( 496, 569, 0.343699),
                bianbuai.PosePoint( 522, 612, 0.319149),
                bianbuai.PosePoint( 492, 619, 0.343699),
                bianbuai.PosePoint( 519, 679, 0.490999),
                bianbuai.PosePoint( 511, 685, 0.466449),
            ],
        ]
        self.assertEqual(len(output_2), len(output_e))
        """
        for pose_points in output_2:
            for pp in pose_points:
                print("(%4d,%4d,%9.6f)," % (pp.x, pp.y, pp.score))
        """
        if self._arch == "AMD64":
            print("Skip pose points check since expected outputs are not provided.")
            return
        for actual, expect in zip(output_2, output_e):
            for actual_pp, expect_pp in zip(actual, expect):
                self.assertEqual(actual_pp.x, expect_pp.x)
                self.assertEqual(actual_pp.y, expect_pp.y)
                self.assertAlmostEqual(actual_pp.score, expect_pp.score, delta=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=1)