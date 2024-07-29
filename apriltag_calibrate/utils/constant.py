import numpy as np

CALIB_POS_TO_CAM_MAP = dict(
    calib_pos1 = ['cam1', 'cam2', 'cam7', 'lidar1', 'lidar2'],
    calib_pos2 = ['cam1', 'cam2', 'cam4', 'cam5' 'cam6', 'lidar1', 'lidar2'],
    calib_pos3 = ['cam3', 'cam4', 'cam5', 'cam6', 'lidar1', 'lidar2'],
    calib_pos4 = ['cam3', 'cam4', 'cam5', 'cam6', 'lidar1', 'lidar2'],
)

# TODO: Need to correct size params
CALIB_BOARD_PARAMS = dict(
    board_size = [0.6, 0.8],        # 60cm*80cm
    tag_size = [0.12, 0.129],       # 0.6/(0.2*4 + 0.04*5) * 0.2, 0.8/(0.2*5+0.04*6) * 0.2
    spacing = [0.024, 0.0258],      # 0.6/(0.2*4 + 0.04*5) * 0.04, 0.8/(0.2*5+0.04*6) * 0.04
    tag_num = [5, 4],   # row*col
    tag16h5 = dict(
        plane_id = 'xy',
        tag_ids = [],   # 4*5
        centers = [],   # 4*5
        corners = [],   # 4*5*4
    ),
    tag25h7 = dict(
        plane_id = 'yz',
        tag_ids = [],   # 4*5
        centers = [],   # 4*5
        corners = [],   # 4*5*4
    ),
    tag36h11 = dict(
        plane_id = 'zx',
        tag_ids = [],   # 4*5
        centers = [],   # 4*5
        corners = [],   # 4*5*4
    ),
)

single_board_centers = []
single_board_corners = []
for row in range(CALIB_BOARD_PARAMS['tag_num'][0]):
    for col in range(CALIB_BOARD_PARAMS['tag_num'][1]):
        single_board_centers.append([
            (col + 1) * CALIB_BOARD_PARAMS['spacing'][0] + (col + 0.5) * CALIB_BOARD_PARAMS['tag_size'][0],
            (row + 1) * CALIB_BOARD_PARAMS['spacing'][1] + (row + 0.5) * CALIB_BOARD_PARAMS['tag_size'][1],
        ])
        single_board_corners.append([
            [
                single_board_centers[-1][0] - 0.5 * CALIB_BOARD_PARAMS['tag_size'][0],
                single_board_centers[-1][1] - 0.5 * CALIB_BOARD_PARAMS['tag_size'][1],
            ], [
                single_board_centers[-1][0] + 0.5 * CALIB_BOARD_PARAMS['tag_size'][0],
                single_board_centers[-1][1] - 0.5 * CALIB_BOARD_PARAMS['tag_size'][1],
            ], [
                single_board_centers[-1][0] + 0.5 * CALIB_BOARD_PARAMS['tag_size'][0],
                single_board_centers[-1][1] + 0.5 * CALIB_BOARD_PARAMS['tag_size'][1],
            ], [
                single_board_centers[-1][0] - 0.5 * CALIB_BOARD_PARAMS['tag_size'][0],
                single_board_centers[-1][1] + 0.5 * CALIB_BOARD_PARAMS['tag_size'][1],
            ],
        ])
single_board_centers = np.array(single_board_centers)
single_board_corners = np.array(single_board_corners)

# XY
CALIB_BOARD_PARAMS['tag16h5']['tag_ids'] = list(range(len(single_board_centers)))
CALIB_BOARD_PARAMS['tag16h5']['centers'] = np.stack([
    single_board_centers[:, 1],
    single_board_centers[:, 0],
    np.zeros([single_board_centers.shape[0]], dtype=np.float32)
], axis=-1)
CALIB_BOARD_PARAMS['tag16h5']['corners'] = np.stack([
    single_board_corners[:, :, 1],
    single_board_corners[:, :, 0],
    np.zeros([single_board_corners.shape[0], 4], dtype=np.float32)
], axis=-1)

# YZ
CALIB_BOARD_PARAMS['tag25h7']['tag_ids'] = list(range(len(single_board_centers)))
CALIB_BOARD_PARAMS['tag25h7']['centers'] = np.stack([
    np.zeros([single_board_centers.shape[0]], dtype=np.float32),
    CALIB_BOARD_PARAMS['board_size'][1] - single_board_centers[:, 1],
    CALIB_BOARD_PARAMS['board_size'][0] - single_board_centers[:, 0],
], axis=-1)
CALIB_BOARD_PARAMS['tag25h7']['corners'] = np.stack([
    np.zeros([single_board_corners.shape[0], 4], dtype=np.float32),
    CALIB_BOARD_PARAMS['board_size'][1] - single_board_corners[:, :, 1],
    CALIB_BOARD_PARAMS['board_size'][0] - single_board_corners[:, :, 0],
], axis=-1)

# ZX
CALIB_BOARD_PARAMS['tag36h11']['tag_ids'] = list(range(len(single_board_centers)))
CALIB_BOARD_PARAMS['tag36h11']['centers'] = np.stack([
    CALIB_BOARD_PARAMS['board_size'][0] - single_board_centers[:, 0],
    np.zeros([single_board_centers.shape[0]], dtype=np.float32),
    CALIB_BOARD_PARAMS['board_size'][1] - single_board_centers[:, 1],
], axis=-1)
CALIB_BOARD_PARAMS['tag36h11']['corners'] = np.stack([
    CALIB_BOARD_PARAMS['board_size'][0] - single_board_corners[:, :, 0],
    np.zeros([single_board_corners.shape[0], 4], dtype=np.float32),
    CALIB_BOARD_PARAMS['board_size'][1] - single_board_corners[:, :, 1],
], axis=-1)

# print(single_board_centers)