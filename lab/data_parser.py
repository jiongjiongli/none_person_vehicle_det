from pathlib import Path
import json
import xml.etree.ElementTree as ET
import cv2

xml_file_path = r'/home/data/1173/nonvehicles20220623_V2_train_street_1_000189.xml'
xml_file_path = r'nonvehicles20220623_V2_train_street_1_000189.xml'
image_file_path = r'/home/data/1173/nonvehicles20220623_V2_train_street_1_000189.jpg'
image_file_path = r'nonvehicles20220623_V2_train_street_1_000189.jpg'

labeled_image_file_path = r'labeled.jpg'

def get_labeled_objs(xml_file_path):
    tree = ET.parse(str(xml_file_path))
    root = tree.getroot()
    labeled_objs = []

    for element in root:
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
                            labeled_obj['xmin'] = float(grandchild_elem.text)

                        if grandchild_elem.tag == 'ymin':
                            labeled_obj['ymin'] = float(grandchild_elem.text)

                        if grandchild_elem.tag == 'xmax':
                            labeled_obj['xmax'] = float(grandchild_elem.text)

                        if grandchild_elem.tag == 'ymax':
                            labeled_obj['ymax'] = float(grandchild_elem.text)

            labeled_objs.append(labeled_obj)

    return labeled_objs


labeled_objs = get_labeled_objs(xml_file_path)


img = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
color_dict = {
    # 'none_person_auto_tricycle': (255, 0, 0), # Blue 三轮车
    # 'auto_tricycle': (255, 0, 255), # Magenta
    # 'none_person_electric_scooter': (0, 255, 0), # Lime 电瓶车
    # 'electric_scooter': (255, 255, 0), # Cyan
    # 'electric_scooter_person': (0, 255, 255) # Yellow
    # 'rider': (0, 0, 255), # Red 骑车的人
}

thickness = 3

for labeled_obj in labeled_objs:
    name = labeled_obj['name']

    if name not in color_dict:
        continue

    color = color_dict[name]

    xmin = labeled_obj['xmin']
    ymin = labeled_obj['ymin']
    xmax = labeled_obj['xmax']
    ymax = labeled_obj['ymax']
    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)

cv2.imwrite(labeled_image_file_path, img)

