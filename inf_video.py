import os, cv2, torch, argparse
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from PIL.Image import fromarray
from networks import get_model
from copy import copy
from batch_face import RetinaFace


os.environ[
    "KMP_DUPLICATE_LIB_OK"
] = "True"  # https://stackoverflow.com/questions/74217717/what-does-os-environkmp-duplicate-lib-ok-actually-do


def parse_args():
    def str2bool(x):
        return x.lower() in ("true")

    parser = argparse.ArgumentParser()
    # training settings
    parser.add_argument(
        "--model_type", type=str, default="ResNet50_lgt", help="model_type"
    )
    parser.add_argument("--pretrain", type=str, default="imagenet", help="imagenet")
    parser.add_argument("--img_size", type=int, default=256, help="img size")
    parser.add_argument("--normfc", type=str2bool, default=False)
    parser.add_argument("--usebias", type=str2bool, default=True)
    parser.add_argument("--test_scale", type=float, default=0.9, help="batch size")
    parser.add_argument("--feat_loss", type=str, default="simsiam", help="supcon")

    parser.add_argument("--trans", type=str, default="p", help="different pre-process")
    return parser.parse_args()


def loadModel(args):
    model = get_model(
        args.model_type,
        max_iter=0,  # TODO
        num_classes=2,
        pretrained=True,
        normed_fc=args.normfc,
        use_bias=args.usebias,
        simsiam=True if args.feat_loss == "simsiam" else False,
    )
    #model_path = "results/0922/AOS_AOS_flossw0.1/ResNet50_lgt_pA_O_S_to_A_O_S_best.pth"
    #model_path = "results/0922/AOS_AOS_flossw0.5/ResNet50_lgt_pA_O_S_to_A_O_S_best.pth"
    model_path = "results/0718/ResNet50_lgt_pA_S_to_O_best.pth"
    ckpt = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device="cpu")
    model.eval()
    return model


def inference(args, img, model):
    if args.pretrain == "imagenet":
        normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    test_transform = transforms.Compose(
        [
#            transforms.RandomResizedCrop(
#                256, scale=(args.test_scale, args.test_scale), ratio=(1.0, 1.0)
#            ),
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            normalizer,
        ]
    )

    image_x_view1 = torch.unsqueeze(test_transform(fromarray(img)), dim=0)

    with torch.no_grad():
        penul_feat, logit = model(image_x_view1, out_type="ce", scale=1)
    return logit


def webcam(args, model):
    #cap = cv2.VideoCapture('/Users/tc/Downloads/face/20231102_140027.mp4')
    cap = cv2.VideoCapture('/Users/tc/Downloads/face/tc2_toilet.mov')
    #cap = cv2.VideoCapture('/Users/tc/Downloads/face/20231102_140128.mp4')

    # face detector
    mtcnn = MTCNN(select_largest=False, device="cpu", image_size=224, margin=0)

    # cv2.VideoWriter ans2: "macOS High Sierra 10.13.4, Python 3.6.5, OpenCV 3.4.1.": https://stackoverflow.com/questions/10605163/opencv-videowriter-under-osx-producing-no-output
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('/Users/tc/Downloads/face/viz/2out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    while True:
        (
            success,
            _frame,
        ) = (
            cap.read()
        )  # https://stackoverflow.com/questions/61979361/cannot-turn-on-mac-webcam-through-opencv-python

        if not success:
            break
        frame1, frame2 = copy(_frame), copy(_frame)
        frame = cv2.cvtColor(
            _frame, cv2.COLOR_BGR2RGB
        )  # frame1, frame2: imshow; frame: inference

        # Detect face
        batch_boxes, batch_probs, batch_points = mtcnn.detect(frame, landmarks=True)
        print(batch_boxes.shape)
        for batch_box in batch_boxes:
            if batch_box is None:
                continue

            box = batch_box.astype(int)
            cropped = frame[box[1] : box[3], box[0] : box[2]]

            try:
                logit = inference(args, cropped, model)

                # draw the label and bounding box on the frame
                if logit > 0.55:
                    label, color = "live", (0, 255, 0)
                else:
                    label, color = "spoof", (0, 0, 255)
                tag = f"{label}: {logit.cpu().numpy()[0][0]:.4f}"  # TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first

                cv2.putText(
                    frame1,
                    tag,
                    (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
                cv2.rectangle(
                    frame1, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2
                )
                out.write(frame1)
                cv2.imshow("Result", frame1)

            except:
                out.write(frame2)
                cv2.imshow("Result", frame2)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release everything if job is finished
    cap.release()
    out.release() 
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    model = loadModel(args=args)
    webcam(args, model)
