import numpy as np
import cv2
import torch

def rasterize_bev(bboxes_dict, map_dict, size=256, res=0.5):
    """Renders bboxes and map data into a multi-channel Bird's-Eye View grid.

    Args:
        bboxes_dict (dict): Dictionary containing padded bbox data and masks.
        map_dict (dict): Dictionary containing padded map data (lanes, crosswalks, etc.) and masks.
        size (int): The width and height of the output BEV grid.
        res (float): The resolution of the grid in meters per pixel.

    Returns:
        torch.Tensor: A (C, H, W) tensor representing the BEV grid.
    """
    bev = np.zeros((5, size, size), dtype=np.float32)  # Ch0: occ, Ch1: lanes, Ch2: cross, Ch3: lights, Ch4: sem

    # Bboxes to Occupancy and Semantic channels
    if 'bboxes' in bboxes_dict and 'data' in bboxes_dict['bboxes']:
        bboxes_data = bboxes_dict['bboxes']['data'][0].squeeze(0)  # Shape: [max_objs, 8, 3]
        mask = bboxes_dict['bboxes']['mask'][0].squeeze(0)      # Shape: [max_objs]
        classes = bboxes_dict['classes']['data'][0][mask]

        valid_bboxes = bboxes_data[mask]
        for i, corners in enumerate(valid_bboxes):
            # Project 3D corners to 2D BEV plane (x, y)
            corners_2d = corners[:, :2] / res + size // 2
            corners_2d = np.clip(corners_2d, 0, size - 1).astype(int)
            # Reshape for cv2.fillPoly
            pts = corners_2d.reshape((-1, 1, 2))
            cv2.fillPoly(bev[0], [pts], 1.0)  # Occupancy channel

            # Add a simple centroid to the semantic channel
            centroid = corners_2d.mean(axis=0).astype(int)
            if 0 <= centroid[1] < size and 0 <= centroid[0] < size:
                bev[4, centroid[1], centroid[0]] = classes[i]

    # Map: Lanes (Centerlines)
    if 'centerlines' in map_dict and 'data' in map_dict['centerlines']:
        lanes = map_dict['centerlines']['data'][0] # Shape: [max_lanes, 6, 10]
        mask = map_dict['centerlines']['mask'][0]
        valid_lanes = lanes[mask]
        for lane_segment in valid_lanes:
            # Assuming the first 4 values are [x_start, y_start, x_end, y_end]
            start_pt_raw = lane_segment[0, :2]
            end_pt_raw = lane_segment[0, 2:4]
            start_pt = (start_pt_raw / res + size // 2).astype(int)
            end_pt = (end_pt_raw / res + size // 2).astype(int)
            cv2.line(bev[1], tuple(start_pt), tuple(end_pt), 1.0, thickness=2)

    # Map: Crosswalks
    if 'crosswalks' in map_dict and 'data' in map_dict['crosswalks']:
        crosswalks = map_dict['crosswalks']['data'][0]
        mask = map_dict['crosswalks']['mask'][0]
        valid_crosswalks = crosswalks[mask]
        for crosswalk_segment in valid_crosswalks:
            start_pt_raw = crosswalk_segment[0, :2]
            end_pt_raw = crosswalk_segment[0, 2:4]
            start_pt = (start_pt_raw / res + size // 2).astype(int)
            end_pt = (end_pt_raw / res + size // 2).astype(int)
            cv2.line(bev[2], tuple(start_pt), tuple(end_pt), 1.0, thickness=1)

    # Map: Traffic Lights
    if 'traffic_lights' in map_dict and 'data' in map_dict['traffic_lights']:
        lights = map_dict['traffic_lights']['data'][0]
        mask = map_dict['traffic_lights']['mask'][0]
        valid_lights = lights[mask]
        for light in valid_lights:
            xy = (light[:2] / res + size // 2).astype(int)
            if 0 <= xy[1] < size and 0 <= xy[0] < size:
                bev[3, xy[1], xy[0]] = 1.0 # Mark presence of a traffic light

    return torch.from_numpy(bev).float()
