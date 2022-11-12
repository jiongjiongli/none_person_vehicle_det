import logging as log
import json
from shapely.geometry import Polygon

log.basicConfig(level=log.DEBUG)

class_names = ["none_person_bicycle", "none_person_motorbike", "none_person_electric_scooter", "none_person_tricycle",
               "none_person_auto_tricycle"]  # 无人非机动车的类别名称（使得roi需要报警的类别名称）
other_class_names = ['person', 'rider',
                     'bicycle_person', 'motorbike_person', 'electric_scooter_person', 'tricycle_person',
                     'auto_tricycle_person']   # 人或者有人的非机动车，这些目标出现在roi区域内时，使得整个roi不需要报警
camera_id_num_contain = {}


def is_contain(r, boxs, width, height):
    '''
    输入r表示roi，输入boxs表示包含多个检测框的列表
    函数的作用是判断roi里是否包含检测框
    '''

    roi = r.replace('POLYGON((', '').replace('))', '')
    polygona = [(float(x.split(' ')[0]) * width,
                 float(x.split(' ')[1]) * height) for x in roi.split(',')]
    ploy_a = Polygon(polygona)
    flag = False
    for box in boxs:
        poly_b = Polygon([(box[0], box[1]), (box[2], box[1]),
                         (box[2], box[3]), (box[0], box[3])])
        if ploy_a.contains(poly_b):
            flag = True
            return flag
    return flag


def init():
    model = "您的深度学习模型"  # 开发者自己编写
    return model


def process_image(net, input_image, args=None):
    args = json.loads(args)  # 把字符串转成字典
    '''
    args里的信息是这样的
    {'alert_count_threshold': 3, 'is_last': False, 'polygon_1': ['POLYGON
        0.36363636363636365 0.665,0.0015151515151515152 0.885,0.14242424242424243 0.9975,0.4681818181818182 0.7225
        ', 'POLYGON
        0.7893939393939394 0.5825,0.7621212121212121 0.7,0.8575757575757575 0.7175,0.8712121212121212 0.59
        '], 'cid': 'task_non_vehicles_000003'}
    您不需要用到 is_last 字段
    '''
    log.info(f'args:{args}')
    # roi包含非机动车的连续帧数的报警阈值
    alert_count_threshold = args['alert_count_threshold']
    camera_id = args['cid']  # 摄像头的id
    # 输入图像里的roi，有可能有多个roi的，因此它是一个列表。并且一个camera_id里的每一帧的roi都是固定的，roi里的坐标值是归一化过后的
    rois = args['polygon_1']

    if camera_id not in camera_id_num_contain:
        camera_id_num_contain[camera_id] = {roi_name: 0 for roi_name in
                                            rois}  # 初始化camera_id, 给camera_id里的每个roi初始化数值，记录roi连续出现非机动车的帧数，初始值是0

    height, width = input_image.shape[:2]

    results = net.detect(input_image)
    detect_objs = []
    # 记录输入图像里检测到的 其他非报警目标 和 报警目标(无人非机动车) 的检测框坐标
    other_boxs, not_vehicle_boxs = [], []
    for k, det in enumerate(results):
        x, y, width, height, name, score = det
        obj = {
            'name': name,
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'confidence': float(score)
        }
        detect_objs.append(obj)
        if name in class_names:
            not_vehicle_boxs.append([x, y, width, height])  # 当前目标是无人非机动车(报警目标)
        elif name in other_class_names:
            other_boxs.append([x, y, width, height])  # 当前目标是人或者有人的非机动车(非报警目标)

    for roi_name in rois:
        contain_not_vehicle = is_contain(
            roi_name, not_vehicle_boxs, width, height)  # 判断当前roi里是否包含无人非机动车
        contain_others = not is_contain(
            roi_name, other_boxs, width, height)  # 判断当前roi里是否包含其他非报警目标
        if contain_not_vehicle and contain_others:  # 当前roi 包含无人非机动车 且 不包含其他非报警目标
            camera_id_num_contain[camera_id][roi_name] += 1  # 连续帧的数值加1
        else:
            camera_id_num_contain[camera_id][roi_name] = 0  # 连续帧的数值设置0

    target_info = []  # 记录报警roi
    for roi_name in camera_id_num_contain[camera_id]:
        # 如果当前roi的包含非机动的连续帧大于报警阈值
        if camera_id_num_contain[camera_id][roi_name] > alert_count_threshold:
            # 报警roi的信息添加到target_info里，其中time_elapsed不纳入精度计算里
            target_info.append({'roi': roi_name, 'time_elapsed': 1})

    result = {'algorithm_data': {'is_alert': True, 'target_info': target_info},
              'model_data': {"objects": detect_objs}}
    return json.dumps(result)

'''
测试指标 = 报警准确的roi数目 / 测试集里的总roi数目
'''
