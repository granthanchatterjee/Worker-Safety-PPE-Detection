import os
import cv2
from ultralytics import YOLO

def run_inference(input_dir, output_dir, person_model_path, ppe_model_path):
    person_model = YOLO(person_model_path)
    ppe_model = YOLO(ppe_model_path)

    ppe_classes = ["hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit", "ear-protector",
                   "safety-harness"]

    os.makedirs(output_dir, exist_ok=True)

    for img_file in os.listdir(input_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {img_file}")
            continue

        person_results = person_model.predict(source=img, conf=0.25)
        for result in person_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                conf = box.conf[0]
                cv2.putText(img, f"Person {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                person_crop = img[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue
                ppe_results = ppe_model.predict(source=person_crop, conf=0.25)
                for ppe_result in ppe_results:
                    for ppe_box in ppe_result.boxes:
                        cx1, cy1, cx2, cy2 = map(int, ppe_box.xyxy[0].tolist())

                        orig_x1 = cx1 + x1
                        orig_y1 = cy1 + y1
                        orig_x2 = cx2 + x1
                        orig_y2 = cy2 + y1
                        ppe_conf = ppe_box.conf[0]
                        cls_index = int(ppe_box.cls[0])
                        label = f"{ppe_classes[cls_index]} {ppe_conf:.2f}"
                        cv2.rectangle(img, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 0, 255), 2)
                        cv2.putText(img, label, (orig_x1, orig_y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out_path = os.path.join(output_dir, img_file)
        cv2.imwrite(out_path, img)
        print(f"Inference completed: {img_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--person_det_model", type=str, required=True)
    parser.add_argument("--ppe_detection_model", type=str, required=True)
    args = parser.parse_args()

    run_inference(args.input_dir, args.output_dir, args.person_det_model, args.ppe_detection_model)