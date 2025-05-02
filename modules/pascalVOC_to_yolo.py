import os
import xml.etree.ElementTree as ET
import shutil

def convert_annotations(input_dir, images_dir, output_dir, classes_file):
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    labels_out_dir = os.path.join(output_dir, "labels")
    images_out_dir = os.path.join(output_dir, "images")
    os.makedirs(labels_out_dir, exist_ok=True)
    os.makedirs(images_out_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".xml"):
            continue

        xml_path = os.path.join(input_dir, filename)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            continue

        image_filename = root.find('filename').text
        size_elem = root.find('size')
        width = int(size_elem.find('width').text)
        height = int(size_elem.find('height').text)

        yolo_lines = []
        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            if cls_name not in classes:
                continue
            cls_index = classes.index(cls_name)
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            x_center = ((xmin + xmax) / 2.0) / width
            y_center = ((ymin + ymax) / 2.0) / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height
            yolo_lines.append(f"{cls_index} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

        base_name, _ = os.path.splitext(image_filename)
        txt_filename = base_name + ".txt"
        with open(os.path.join(labels_out_dir, txt_filename), 'w') as out_file:
            for line in yolo_lines:
                out_file.write(line + "\n")

        src_image_path = os.path.join(images_dir, image_filename)
        if os.path.exists(src_image_path):
            shutil.copy(src_image_path, os.path.join(images_out_dir, image_filename))
        else:
            print(f"Image file not found for {image_filename}")

    print("Annotation conversion completed.")