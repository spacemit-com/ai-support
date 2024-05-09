import argparse

def get_argsparser():
    """Parse commandline."""
    parser = argparse.ArgumentParser(description="Bianbu AI Python Demo for Image Classification.")
    parser.add_argument("--image", "-i", type=str, required=True, help="input test image path")
    parser.add_argument("--model", "-m", type=str, required=True, help="input test model(*.onnx) path")
    parser.add_argument("--label", "-l", type=str, required=True, help="input test label path")
    parser.add_argument("--intra", type=int, default=2, help="intra thread number for backend(e.g. onnxruntime)")
    parser.add_argument("--inter", type=int, default=2, help="inter thread number for backend(e.g. onnxruntime)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_argsparser()

    import cv2
    from bianbuai import ImageClassificationOption, ImageClassificationTask
    option = ImageClassificationOption(args.model, args.label, args.intra, args.inter)
    task = ImageClassificationTask(option)
    output = task.Classify(cv2.imread(args.image))
    print("Label:", output.label_text)
    print("Score:", output.score)
