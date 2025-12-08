# make sure you have the following dependencies
import torch
import numpy as np
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox
import PIL.Image
from huggingface_hub import hf_hub_download 

@smart_inference_mode()
def predict(image_path, weights='yolov9-c.pt', imgsz=640, conf_thres=0.1, iou_thres=0.45):
    device = select_device('cpu')
    model = DetectMultiBackend(weights=weights, device="cpu", fp16=False, data='data/coco.yaml')
    stride, names, pt = model.stride, model.names, model.pt

    # Load image
    image = np.array(PIL.Image.open(image_path))
    img = letterbox(image, imgsz, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred[0][0], conf_thres, iou_thres, classes=None, max_det=1000)
    return pred


hf_hub_download("merve/yolov9", filename="yolov9-c.pt", local_dir="./")
preds = predict("micra.jpg", weights="yolov9-c.pt")
print(preds)