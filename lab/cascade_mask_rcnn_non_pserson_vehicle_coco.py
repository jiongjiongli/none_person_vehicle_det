def get_data_root_dir_path():
    from pathlib import Path
    data_root_dir_path = Path('/home/data/1173')

    if not data_root_dir_path.exists():
        data_root_dir_path = Path(__file__).parent.absolute()

    return str(data_root_dir_path)


def join_paths(path_one, path_two):
    from pathlib import Path

    path_result = str(Path(path_one) / path_two)
    return path_result


data_root_dir_path = get_data_root_dir_path()

# 这个新的配置文件继承自一个原始配置文件，只需要突出必要的修改部分即可
# _base_ = 'cascade_rcnn/cascade_mask_rcnn_r101_fpn_20e_coco.py'
# _base_ = 'cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco.py'

# https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_20e_coco.py
_base_ = 'cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_20e_coco.py'

category_list = [
    'bicycle',
    'motorbike',
    'electric_scooter',
    'tricycle',
    'auto_tricycle',
    'none_person_bicycle',
    'none_person_motorbike',
    'none_person_electric_scooter',
    'none_person_tricycle',
    'none_person_auto_tricycle',
    # 'unknown_person_bicycle',
    # 'unknown_person_motorbike',
    # 'unknown_person_electric_scooter',
    # 'unknown_person_tricycle',
    # 'unknown_person_auto_tricycle',
    'person',
    'rider',
    'bicycle_person',
    'motorbike_person',
    'electric_scooter_person',
    'tricycle_person',
    'auto_tricycle_person'
]

# 我们需要对头中的类别数量进行修改来匹配数据集的标注
num_classes = len(category_list)

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=num_classes),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=num_classes),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=num_classes)],
        mask_head=dict(num_classes=num_classes)))

# 修改数据集相关设置
dataset_type = 'CocoDataset'

classes = tuple(category_list)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        # dataset=dict(
            type=dataset_type,
            classes=classes,
            img_prefix= join_paths(data_root_dir_path, 'vehicle_data/train/'),
            ann_file= join_paths(data_root_dir_path, 'vehicle_data/train/annotation_coco.json')
            # )
        ),
    val=dict(
        type=dataset_type,
        classes=classes,
        img_prefix= join_paths(data_root_dir_path, 'vehicle_data/val/'),
        ann_file= join_paths(data_root_dir_path, 'vehicle_data/val/annotation_coco.json')),
    test=dict(
        type=dataset_type,
        classes=classes,
        img_prefix= join_paths(data_root_dir_path, 'vehicle_data/val/'),
        ann_file= join_paths(data_root_dir_path, 'vehicle_data/val/annotation_coco.json')))

# 我们可以使用预训练的 Mask R-CNN 来获取更好的性能
# 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210719_210311-d3e64ba0.pth'
# load_from = 'checkpoints/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210719_210311-d3e64ba0.pth'
# https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_20e_coco/cascade_mask_rcnn_x101_32x4d_fpn_20e_coco_20200528_083917-ed1f4751.pth
load_from = 'checkpoints/cascade_mask_rcnn_x101_32x4d_fpn_20e_coco_20200528_083917-ed1f4751.pth'

optimizer = dict(_delete_=True, type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.001,
    step=[9, 11])
runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=10)
checkpoint_config=dict(interval=1, max_keep_ckpts=1)
work_dir = '/project/train/models/'
