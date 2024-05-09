import argparse

def get_argsparser():
    """Parse commandline."""
    parser = argparse.ArgumentParser(description="Bianbu AI Python Demo for Object Detection.")
    parser.add_argument("--image", "-i", type=str, required=True, help="input test image path")
    parser.add_argument("--model", "-m", type=str, required=True, help="input test model(*.onnx) path")
    parser.add_argument("--label", "-l", type=str, required=True, help="input test label path")
    parser.add_argument("--intra", type=int, default=2, help="intra thread number for backend(e.g. onnxruntime)")
    parser.add_argument("--inter", type=int, default=2, help="inter thread number for backend(e.g. onnxruntime)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_argsparser()

    import cv2
    from bianbuai import ObjectDetectionOption, ObjectDetectionTask
    option = ObjectDetectionOption()
    option.model_path = args.model
    option.label_path = args.label
    option.intra_threads_num = args.intra
    option.inter_threads_num = args.inter
    task = ObjectDetectionTask(option)
    outputs = task.Detect(cv2.imread(args.image))
    for i, box in enumerate(outputs):
        print("bbox[%2d] x1y1x2y2: (%4d,%4d,%4d,%4d), score: %5.3f, label_text: %s" % (i, box.x1, box.y1, box.x2, box.y2, box.score, box.label_text))
