from pathlib import Path
import json
import xml.etree.ElementTree as ET
import random
from shutil import copyfile
import yaml

from utils.dataloaders import img2label_paths


def get_data_root_dir_path():
    data_root_dir_path = Path('/home/data/1173')

    if not data_root_dir_path.exists():
        data_root_dir_path = Path(__file__).parent.absolute()

    return str(data_root_dir_path)


def convert(image_width, image_height, bbox):
    # Transform the bbox co-ordinates as per the format required by YOLO v5
    b_center_x = (bbox["xmin"] + bbox["xmax"]) / 2
    b_center_y = (bbox["ymin"] + bbox["ymax"]) / 2
    b_width    = (bbox["xmax"] - bbox["xmin"])
    b_height   = (bbox["ymax"] - bbox["ymin"])

    # Normalise the co-ordinates by the dimensions of the image
    b_center_x /= image_width
    b_center_y /= image_height
    b_width    /= image_width
    b_height   /= image_height


def convert_annotation(class2id, xml_file_path):
    tree = ET.parse(str(xml_file_path))
    root = tree.getroot()

    image_file_name = root.find('filename')
    image_file_path = xml_file_path.parent / image_file_name

    assert image_file_path.exists(), 'Image file {} not exist!'.format(image_file_path)
    label_file_paths = img2label_paths([str(image_file_path)])
    label_file_path = Path(label_file_paths[0])

    size = root.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)
    labels = []

    for obj in root.iter('object'):
        name = obj.find('name').text

        if name not in class2id:
            continue

        class_id = class2id[name]
        bndbox = obj.find('bndbox')
        bbox = {
            'xmin': float(bndbox.find('xmin').text),
            'ymin': float(bndbox.find('ymin').text),
            'xmax': float(bndbox.find('xmax').text),
            'ymax': float(bndbox.find('ymax').text)
        }

        bb = convert(image_width, image_height, bbox)

        #Write the bbox details to the file
        label = '{} {:.5f} {:.5f} {:.5f} {:.5f}'.format(class_id, b_center_x, b_center_y, b_width, b_height)
        labels.append(label)

    if not labels:
        return None

    label_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(str(label_file_path), 'w') as file_stream:
        for label in labels:
            file_stream.write('{}\n'.format(label))

    return image_file_path


def main():
    dataset_config_file_path = Path(r'./dataset/vehicle_data.yaml')

    xml_dir_path = get_data_root_dir_path()
    xml_dir_path = Path(xml_dir_path)

    train_file_percent = 0.8

    xml_file_paths = list(xml_dir_path.glob('*.xml'))
    xml_file_paths.sort()

    random.seed(1997)
    random.shuffle(xml_file_paths)

    with open(str(dataset_config_file_path), 'r') as file_stream:
        dataset_config = yaml.safe_load(file_stream)

    class_names = dataset_config['names']
    class2id = {class_name: class_index for class_index, class_name in enumerate(class_names)}

    data_root_path = Path(dataset_config['path'])
    train_file_path = data_root_path / dataset_config['train']
    val_file_path = data_root_path / dataset_config['val']

    image_file_paths = []

    for xml_file_path in xml_file_paths:
        image_file_path = convert_annotation(class2id, xml_file_path)
        image_file_paths.append(image_file_path)

    train_file_count = int(len(image_file_paths) * train_file_percent)

    train_image_file_paths = image_file_paths[:train_file_count]
    val_image_file_paths = image_file_paths[train_file_count:]

    with open(str(train_file_path), 'w') as file_stream:
        for image_file_path in train_image_file_paths:
            file_stream.write('{}\n'.format(image_file_path))

    with open(str(val_file_path), 'w') as file_stream:
        for image_file_path in val_image_file_paths:
            file_stream.write('{}\n'.format(image_file_path))


if __name__ == '__main__':
    main()
