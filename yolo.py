import cv2
import numpy as np
import torch

class ObjectDetector:
    def __init__(self, model_path, config_path, classes_file):
        # Initialisation de la classe ObjectDetector
        self.net = cv2.dnn.readNet(model_path, config_path)
        self.classes = []
        with open(classes_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def detect_objects(self, image):
        # Préparation de l'image pour la détection
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.get_output_layers())
        class_ids, confidences, boxes = self.post_process(image, outs)
        return class_ids, confidences, boxes

    def get_output_layers(self):
        # Récupération des noms des couches de sortie
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers

    def post_process(self, image, outs):
        # Traitement des résultats de la détection
        Width = image.shape[1]
        Height = image.shape[0]
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Seulement les détections avec une confiance suffisante
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        final_boxes = [boxes[i[0]] for i in indices]
        final_class_ids = [class_ids[i[0]] for i in indices]
        final_confidences = [confidences[i[0]] for i in indices]
        
        return final_class_ids, final_confidences, final_boxes

def main():
    # Exemple d'utilisation de la classe ObjectDetector
    model_path = 'yolov3.weights'
    config_path = 'yolov3.cfg'
    classes_file = 'coco.names'

    detector = ObjectDetector(model_path, config_path, classes_file)
    
    image = cv2.imread('test_image.jpg')
    class_ids, confidences, boxes = detector.detect_objects(image)

    for class_id, confidence, box in zip(class_ids, confidences, boxes):
        label = str(detector.classes[class_id])
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), (0, 255, 0), 2)
        cv2.putText(image, f'{label} {confidence:.2f}', (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Object Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
