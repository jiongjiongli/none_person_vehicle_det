from pathlib import Path
import json
import xml.etree.ElementTree as ET
import cv2
import mmcv
import random
from shutil import copyfile
from cascade_mask_rcnn_non_pserson_vehicle_coco import category_list, data, get_data_root_dir_path


def get_labeled_objs(xml_file_path):
    tree = ET.parse(str(xml_file_path))
    root = tree.getroot()
    labeled_objs = []
    image_file_names = []

    for element in root:
        if element.tag == 'filename':
            image_file_name = element.text
            image_file_names.append(image_file_name)

        if element.tag == 'object':
            labeled_obj = {}

            for child_elem in element:
                # print(child_elem.tag)

                if child_elem.tag == 'name':
                    labeled_obj['name'] = child_elem.text

                if child_elem.tag == 'bndbox':
                    bndbox_elem = child_elem

                    for grandchild_elem in child_elem:
                        if grandchild_elem.tag == 'xmin':
                            labeled_obj['x_min'] = float(grandchild_elem.text)

                        if grandchild_elem.tag == 'ymin':
                            labeled_obj['y_min'] = float(grandchild_elem.text)

                        if grandchild_elem.tag == 'xmax':
                            labeled_obj['x_max'] = float(grandchild_elem.text)

                        if grandchild_elem.tag == 'ymax':
                            labeled_obj['y_max'] = float(grandchild_elem.text)

            labeled_objs.append(labeled_obj)

    assert len(image_file_names) == 1, 'No filename for: {}'.format(xml_file_path)
    assert labeled_objs, 'No labeled_objs for: {}'.format(xml_file_path)

    return image_file_names[0], labeled_objs


def filter_labeled_objs(labeled_objs, category_list, ignored_category_names):
    results = []

    for labeled_obj in labeled_objs:
        name = labeled_obj['name']

        if name not in category_list:
            ignored_category_names.add(name)
            continue

        results.append(labeled_obj)

    return results


def draw_labeled_image(image_file_path, labeled_objs, labeled_image_file_path):
    img = cv2.imread(str(image_file_path), cv2.IMREAD_COLOR)
    color_dict = {
        'none_person_auto_tricycle': (255, 0, 0), # Blue 三轮车
        'auto_tricycle': (255, 0, 255), # Magenta
        'none_person_electric_scooter': (0, 255, 0), # Lime 电瓶车
        'electric_scooter': (255, 255, 0), # Cyan
        'electric_scooter_person': (0, 255, 255), # Yellow
        'rider': (0, 0, 255), # Red 骑车的人
    }

    thickness = 3

    for labeled_obj in labeled_objs:
        name = labeled_obj['name']

        if name not in color_dict:
            continue

        color = color_dict[name]

        x_min = labeled_obj['x_min']
        y_min = labeled_obj['y_min']
        x_max = labeled_obj['x_max']
        y_max = labeled_obj['y_max']
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)

    cv2.imwrite(labeled_image_file_path, img)


def test_draw_labeled_image():
    xml_file_path = r'/home/data/1173/nonvehicles20220623_V2_train_street_1_000189.xml'
    xml_file_path = r'nonvehicles20220623_V2_train_street_1_000189.xml'
    xml_file_path = Path(xml_file_path)

    labeled_image_file_path = r'labeled.jpg'

    image_file_name, labeled_objs = get_labeled_objs(xml_file_path)

    image_file_path = xml_file_path.parent / image_file_name

    draw_labeled_image(image_file_path, labeled_objs, labeled_image_file_path)


def xml_to_coco(xml_file_paths, data_type):
    annotations = []
    images = []
    obj_count = 0
    ignored_category_names = set()

    data_config_dict = data[data_type]
    if 'dataset' in data_config_dict:
        data_config_dict = data_config_dict['dataset']

    image_dir_path = Path(data_config_dict['img_prefix'])
    image_dir_path.mkdir(parents=True, exist_ok=True)

    ann_file_path = Path(data_config_dict['ann_file'])
    ann_file_path.parent.mkdir(parents=True, exist_ok=True)

    category_dict = {category_name: category_index for category_index, category_name in enumerate(category_list)}

    for idx, xml_file_path in enumerate(xml_file_paths):
        image_file_name, labeled_objs = get_labeled_objs(xml_file_path)
        filtered_labeled_objs = filter_labeled_objs(labeled_objs, category_list, ignored_category_names)
        image_file_path = xml_file_path.parent / image_file_name

        if not filtered_labeled_objs:
            continue

        copyfile(image_file_path, str(image_dir_path / image_file_name))

        height, width = mmcv.imread(str(image_file_path)).shape[:2]

        images.append(dict(
            id=idx,
            file_name=image_file_name,
            height=height,
            width=width))

        for labeled_obj in filtered_labeled_objs:
            name = labeled_obj['name']
            category_id = category_dict[name]

            x_min = labeled_obj['x_min']
            y_min = labeled_obj['y_min']
            x_max = labeled_obj['x_max']
            y_max = labeled_obj['y_max']
            poly = [x_min, y_min, x_max, y_min, x_min, y_max, x_max, y_max]

            data_anno = dict(
                    image_id=idx,
                    id=obj_count,
                    category_id=category_id,
                    bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                    area=(x_max - x_min) * (y_max - y_min),
                    segmentation=[poly],
                    iscrowd=0)

            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id': category_index, 'name': category_name} for category_name, category_index in category_dict.items()])

    mmcv.dump(coco_format_json, str(ann_file_path), indent=4)


def main():
    # xml_dir_path = r'/Users/jiongjiongai/proj/git/none_person_vehicle_det/lab'
    # xml_dir_path = r'/home/data/1173'
    xml_dir_path = get_data_root_dir_path()
    xml_dir_path = Path(xml_dir_path)

    valid_file_count = 2

    xml_file_paths = list(xml_dir_path.glob('*.xml'))
    xml_file_paths.sort()

    random.seed(1997)
    random.shuffle(xml_file_paths)
    train_xml_file_paths = xml_file_paths
    valid_xml_file_paths = xml_file_paths[-valid_file_count:]

    xml_to_coco(train_xml_file_paths, 'train')
    xml_to_coco(valid_xml_file_paths, 'val')


if __name__ == '__main__':
    main()
