import pickle
import numpy as np
import cv2
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import onnxruntime as ort
from config import CLASS_NAMES, HIGH_CONF, LOW_CONF, FACE_SIM_THRESHOLD

# ---------------- Face Recognition ----------------
# Load embeddings
with open("facerec_model.pkl", "rb") as f:
    face_db = pickle.load(f)

# Load FaceNet
resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(keep_all=False)

def get_face_embedding(img_cv):
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    face_aligned = mtcnn(Image.fromarray(img_rgb))
    if face_aligned is None:
        return None
    with torch.no_grad():
        emb = resnet(face_aligned.unsqueeze(0)).numpy()[0]
    return emb

def match_face(embedding):
    best_score = 0
    best_name = "Unknown"
    for entry in face_db:
        db_emb = entry['embedding']
        score = np.dot(embedding, db_emb) / (np.linalg.norm(embedding) * np.linalg.norm(db_emb))
        if score > best_score:
            best_score = score
            best_name = entry['identity']
    if best_score >= FACE_SIM_THRESHOLD:
        return best_name
    return "Unknown"

# ---------------- Weapon Detection ----------------
# Load ONNX model
session = ort.InferenceSession("best.onnx")

def preprocess_image(img_cv, input_size=640):
    orig_h, orig_w = img_cv.shape[:2]
    img_resized = cv2.resize(img_cv, (input_size, input_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_input = img_rgb.astype(np.float32)/255.0
    img_input = np.transpose(img_input, (2,0,1))  # HWC -> CHW
    img_input = np.expand_dims(img_input, 0)      # batch dim
    return img_input, orig_w, orig_h

def run_inference(img_input):
    outputs = session.run(None, {session.get_inputs()[0].name: img_input})
    return outputs[0]

def postprocess(pred):
    detected = []
    num_classes = len(CLASS_NAMES)
    for det in pred[0].T:
        # YOLO output: [x, y, w, h, obj_conf, class_probs...]
        object_conf = det[4]
        class_probs = det[5:5+num_classes]
        cls_idx = int(np.argmax(class_probs))
        cls_conf = class_probs[cls_idx]*object_conf
        if cls_conf >= HIGH_CONF and CLASS_NAMES[cls_idx]=="gun":
            detected.append("gun")
        elif cls_conf > LOW_CONF:
            detected.append("weapon")
    return detected
