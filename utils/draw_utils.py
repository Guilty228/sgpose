import numpy as np
import os
import cv2


def transform_coordinates_3d(coordinates, RT):
    """
    Input: 
        coordinates: [3, N]
        RT: [4, 4]
    Return 
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input: 
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return 
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates

def get_3d_bbox(scale, shift = 0):
    """
    Input: 
        scale: [3] or scalar
        shift: [3] or scalar
    Return 
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                  [scale / 2, +scale / 2, -scale / 2],
                  [-scale / 2, +scale / 2, scale / 2],
                  [-scale / 2, +scale / 2, -scale / 2],
                  [+scale / 2, -scale / 2, scale / 2],
                  [+scale / 2, -scale / 2, -scale / 2],
                  [-scale / 2, -scale / 2, scale / 2],
                  [-scale / 2, -scale / 2, -scale / 2]]) +shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d


def draw(img, imgpts, axes, color):
    imgpts = np.int32(imgpts).reshape(-1, 2)


    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7],[5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, 3)


    # draw pillars in blue color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, 3)

    
    # finally, draw top layer in color
    for i, j in zip([0, 1, 2, 3],[1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, 3)


    # draw axes
    img = cv2.line(img, tuple(axes[0]), tuple(axes[1]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(axes[0]), tuple(axes[3]), (255, 0, 0), 3)
    img = cv2.line(img, tuple(axes[0]), tuple(axes[2]), (0, 255, 0), 3) ## y last


    return img

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.
    x: [rows, columns].
    """

    pre_shape = x.shape
    assert len(x.shape) == 2, x.shape
    new_x = x[~np.all(x == 0, axis=1)]
    post_shape = new_x.shape
    assert pre_shape[0] == post_shape[0]
    assert pre_shape[1] == post_shape[1]

    return new_x

def compute_overlaps_masks(masks1, masks2):
    '''Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    '''
    # flatten masks
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps



def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    if len(gt_class_ids)==0 or len(pred_class_ids) == 0:
        return -1 * np.ones([len(gt_class_ids)]), -1 * np.ones([len(pred_class_ids)]), None, np.zeros([0])

    pre_len = len(gt_boxes)
    gt_boxes = trim_zeros(gt_boxes)
    after_len = len(gt_boxes)
    assert pre_len == after_len
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]

    pre_len = len(pred_boxes)
    pred_boxes = trim_zeros(pred_boxes)
    after_len = len(pred_boxes)
    assert pre_len == after_len
    pred_scores = pred_scores[:pred_boxes.shape[0]]

    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    
    
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps, indices


def draw_detections(image, save_dir, data_name, image_id, intrinsics,
                    gt_bbox, gt_class_ids, gt_mask, gt_RTs, gt_scales,
                    pred_bbox, pred_class_ids, pred_mask, pred_RTs, pred_scores, pred_scales,
                    draw_gt=False, draw_pred=True):

    if draw_gt:
        output_path = os.path.join(save_dir, '{}_{}_bbox_gt.png'.format(data_name, image_id))
        draw_image_bbox = image.copy()

        if gt_RTs is not None:
            for ind, RT in enumerate(gt_RTs):
                xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
                transformed_axes = transform_coordinates_3d(xyz_axis, RT)
                projected_axes = calculate_2d_projections(transformed_axes, intrinsics)


                bbox_3d = get_3d_bbox(gt_scales[ind], 0)
                transformed_bbox_3d = transform_coordinates_3d(bbox_3d, RT)
                projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
                draw_image_bbox = draw(draw_image_bbox, projected_bbox, projected_axes, (255, 0, 0))

        cv2.imwrite(output_path, draw_image_bbox[:, :, ::-1])


    if draw_pred:

        num_pred_instances = len(pred_class_ids)
        output_path = os.path.join(save_dir, '{}_{}_bbox_pred.png'.format(data_name, image_id))
        draw_image_bbox = image.copy()

        if gt_class_ids is not None:
            gt_match, pred_match, _, pred_indices = compute_matches(gt_bbox, gt_class_ids, gt_mask,
                                                                    pred_bbox, pred_class_ids, pred_scores, pred_mask,
                                                                    0.5)

            if len(pred_indices):
                pred_RTs = pred_RTs[pred_indices]


        for ind in range(num_pred_instances):
            RT = pred_RTs[ind]

            xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
            transformed_axes = transform_coordinates_3d(xyz_axis, RT)
            projected_axes = calculate_2d_projections(transformed_axes, intrinsics)

            bbox_3d = get_3d_bbox(pred_scales[ind, :], 0)
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, RT)
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            draw_image_bbox = draw(draw_image_bbox, projected_bbox, projected_axes, (255, 0, 0))

        cv2.imwrite(output_path, draw_image_bbox[:, :, ::-1])