import torch
import os
from torchvision import transforms
from efficientnet_pytorch import efficientnet
from PIL import Image


class EfficientNetB0_Detect():
    def __init__(self, model_path="runs/weight/last_ckpt.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = efficientnet(score_thresh=0.5, num_classes = 4)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, img_path):
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
        img = Image.open(img_path).convert('RGB')
        img = tf(img)
        img = img.to(self.device)
        print(img.shape)
        out = self.model([img])
        return out


if __name__ == '__main__':
    detect = EfficientNetB0_Detect()
    out = detect.predict('dataset/image/000016.jpg')
    print(out[0]['boxes'])
