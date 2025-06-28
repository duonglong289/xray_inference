import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import hashlib
import json
import cv2
import detection_utils

classname = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis"
]

with open("dict_search_v2.json", "r") as f:
    dict_search = json.load(f)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

model_r50 = torch.jit.load('script_resnet.ts', map_location=device)
model_r50.eval()

model_FasterRCNN = torch.jit.load('model_faster_rcnn.ts', map_location=device)

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((448,448), interpolation=Image.Resampling.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Load ImageNet labels from JSON URL
labels = [
    'Ảnh bình thường',
    'Ảnh có dấu hiệu bất thường'
]

def preprocess_image(image):
    # Apply preprocessing transformations
    image = Image.fromarray(image)
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Define the prediction function
def predict_resnet(image: np.ndarray):
    '''
    image: RGB numpy array
    '''
    print("resnet run")
    if image is None:
        return "Ảnh gửi lỗi", "Không xác định"

    # hash_image = hashlib.md5(image.tobytes()).hexdigest()
    hash_image = hashlib.md5(str(image).encode()).hexdigest() # 6adf97f83acf6453d4a6a4b1070f3754
    key_img = dict_search.get(hash_image, None)
    if key_img is None:
        true_label = "Không xác định"
    else:
        if key_img['class'] == 'normal':
            true_label = labels[0]
        else:
            true_label = labels[1]
    
    # Apply preprocessing transformations
    image_tensor_cpu = preprocess_image(image)  # Add batch dimension
    image_tensor = image_tensor_cpu.to(device)
    with torch.no_grad():
        outputs = model_r50(image_tensor)  # Forward pass through the model
    # _, predicted = outputs.max(1)
    softmax_res = torch.nn.functional.softmax(outputs, dim=1)
    # return labels[predicted.item()], true_label
    if softmax_res[0][0] > 0.5:
        return labels[0], true_label
    else:
        return labels[1], true_label


def preprocess_faster_rcnn(image: np.ndarray):
    img = cv2.resize(image, (256, 256))

    # image = torch.tensor(img)
    image = img
    
    image = image.transpose((2, 0, 1)).astype("float32")
    resized_img = torch.from_numpy(image)
    processed_data = torch.stack([resized_img], dim=0)
    return processed_data


def predict_faster_rcnn(image: np.ndarray):
    '''
    image: RGB numpy array
    '''
    print("faster run")
    if image is None:
        return "Ảnh gửi lỗi", "Không xác định"
            
    
    viz_img = image.copy()
    viz_img = cv2.resize(viz_img, (448, 448))
    raw_h, raw_w = image.shape[:2]
    r_w = 448 / raw_w
    r_h = 448 / raw_h

    # Apply preprocessing transformations
    image_tensor = preprocess_faster_rcnn(image)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        results = model_FasterRCNN(image_tensor)  # Forward pass through the model
    bboxes = results[:, :6].detach()
    result_nms = detection_utils.nms(bboxes, iou_threshold=0.4)
    # print("NMS Result:", result_nms)

    # Apply WBF
    result_wbf = detection_utils.weighted_box_fusion(bboxes, iou_threshold=0.4)

    THR_SCORE = 0.4
    final_result = result_wbf[torch.where(result_wbf[:,4] > THR_SCORE)]
    
    for box in final_result:
        x1, y1, x2, y2, score, cls_id = box

        x1 = int(x1 * r_w)
        y1 = int(y1 * r_h)
        x2 = int(x2 * r_w)
        y2 = int(y2 * r_h)
        cv2.rectangle(viz_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
        name_cls = classname[int(cls_id)]
        cv2.putText(viz_img, name_cls, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
    # Get label
    hash_image = hashlib.md5(str(image).encode()).hexdigest() # 6adf97f83acf6453d4a6a4b1070f3754
    key_img = dict_search.get(hash_image, None)
    annot_img = image.copy()
    annot_img = cv2.resize(annot_img, (448, 448))
    if key_img is None:
        pass
    else:
        det_object = key_img['det_obj']
        for box in det_object:
            x1, y1, x2, y2, cls_id = box
            x1 = int(x1 * r_w)
            y1 = int(y1 * r_h)
            x2 = int(x2 * r_w)
            y2 = int(y2 * r_h)
            cv2.rectangle(annot_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            name_cls = classname[cls_id]
            cv2.putText(annot_img, name_cls, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return viz_img, annot_img




