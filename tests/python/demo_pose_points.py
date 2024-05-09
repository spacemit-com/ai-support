import argparse

def get_argsparser():
    """Parse commandline."""
    parser = argparse.ArgumentParser(description="Bianbu AI Python Demo for Human Pose Points.")
    parser.add_argument("--image", "-i", type=str, required=True, help="input test image path")
    parser.add_argument("--label", "-l", type=str, required=True, help="input test label path")
    parser.add_argument("--model-det", "-md", type=str, required=True, help="input detection model(*.onnx) path")
    parser.add_argument("--model-pose", "-mp", type=str, required=True, help="input pose model(*.onnx) path")
    parser.add_argument("--intra", type=int, default=2, help="intra thread number for backend(e.g. onnxruntime)")
    parser.add_argument("--inter", type=int, default=2, help="inter thread number for backend(e.g. onnxruntime)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_argsparser()

    # object detection
    from bianbuai import ObjectDetectionOption, ObjectDetectionTask
    option_detect = ObjectDetectionOption()
    option_detect.model_path = args.model_det
    option_detect.label_path = args.label
    option_detect.intra_threads_num = args.intra
    option_detect.inter_threads_num = args.inter
    task_detect = ObjectDetectionTask(option_detect)

    # pose estimation
    from bianbuai import PoseEstimationOption, PoseEstimationTask
    option_pose = PoseEstimationOption()
    option_pose.model_path = args.model_pose
    option_pose.intra_threads_num = args.intra
    option_pose.inter_threads_num = args.inter
    task_pose = PoseEstimationTask(option_pose)
    
    import cv2
    image = cv2.imread(args.image)
    output_boxes = task_detect.Detect(image)
    for i, box in enumerate(output_boxes):
        print("bbox[%2d] x1y1x2y2: (%4d,%4d,%4d,%4d), score: %5.3f, label_text: %s" % (i, box.x1, box.y1, box.x2, box.y2, box.score, box.label_text))
        pose_points = task_pose.Estimate(image, box)
        for i, point in enumerate(pose_points):
            print("  point[%2d] xy: (%4d,%4d), score: %5.3f" % (i, point.x, point.y, point.score))