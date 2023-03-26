from ai2thor.controller import Controller
import numpy as np 
from scipy.spatial.transform import Rotation as R
from transformers import CLIPProcessor, CLIPModel

def get_sim_cam_mat_with_fov(h, w, fov):

    cam_mat = np.eye(3)
    cam_mat[0, 0] = cam_mat[1, 1] = w / (2.0 * np.tan(np.deg2rad(fov / 2)))
    cam_mat[0, 2] = w / 2.0
    cam_mat[1, 2] = h / 2.0
    return cam_mat

def depth2pc_ai2thor(depth, clipping_dist=0.1, fov=90):
    """
    Return 3xN array
    """

    h, w = depth.shape

    cam_mat = get_sim_cam_mat_with_fov(h, w, fov)

    cam_mat_inv = np.linalg.inv(cam_mat)

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    # x = x[int(h/2)].reshape((1, -1))
    # y = y[int(h/2)].reshape((1, -1))
    # z = depth[int(h/2)].reshape((1, -1))

    x = x.reshape((1, -1))[:, :]
    y = y.reshape((1, -1))[:, :]
    # z = depth.reshape((1, -1))[:, :] + clipping_dist
    z = depth.reshape((1, -1))[:, :]

    p_2d = np.vstack([x, y, np.ones_like(x)])
    pc = cam_mat_inv @ p_2d
    pc = pc * z
    mask = pc[2, :] > 0.1
    mask2 = pc[2, :] < 10
    # y > -0.5
    mask3 = pc[1, :] > 0
    mask = np.logical_and(mask, mask2)
    mask = np.logical_and(mask, mask3)
    #pc = pc[:, mask]
    return pc, mask

def euler_to_rotation_matrix(euler):
    x, y, z = np.deg2rad(euler['x']), np.deg2rad(euler['y']), np.deg2rad(euler['z'])
    R_x = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    R_y = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    R_z = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def transformation_matrix(position, rotation):
    R = euler_to_rotation_matrix(rotation)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [position['x'], position['y'], position['z']]
    return T

def rotate_2d(points, angle_degrees):
    angle_rad = np.deg2rad(angle_degrees)
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad),  np.cos(angle_rad)]])
    return R @ points

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    from lseg_unit_test import LSegPredictor
    lseg_predictor = LSegPredictor()
    lang = "cabinet,sink,counter,ceiling,floor,window,drawer,vegetables,wall,fridge"
    labels = lang.split(",")
    # make random color palette
    colors = np.random.rand(len(labels), 3)*255
    # ... (insert transformation_matrix and euler_to_rotation_matrix functions here)

    c = Controller(scene='FloorPlan13', gridSize=0.05, renderDepthImage=True, renderClassImage=True, renderObjectImage=True, renderImage=True, width=128, height=128, fieldOfView=90)
    # get reachable positions
    reachable_positions = c.step(action='GetReachablePositions', agentId = 0).metadata['actionReturn']
    # make it np array
    reachable_positions = np.array([[i['x'], i['z']] for i in reachable_positions])
    
    initial_position = c.last_event.metadata['agent']['position']
    initial_rotation = c.last_event.metadata['agent']['rotation']
    initial_pose_matrix = transformation_matrix(initial_position, initial_rotation)
    # adjust reachable positions relative to initial position
    reachable_positions[:, 0] -= initial_position['x']
    reachable_positions[:, 1] -= initial_position['z']

    # Rotate reachable positions 90 degrees clockwise
    angle_degrees = -90
    reachable_positions = rotate_2d(reachable_positions.T, angle_degrees).T

    action_list = ['Pass','RotateRight','RotateRight','MoveAhead','MoveAhead']

    video_list = list()
    accumulated_pc = np.empty((3, 0))
    accumulated_colors = np.empty((3, 0))
    accumulated_labels = np.empty((1, 0))

    for action in action_list:
        event = c.step(action=action, agentId=0)

        # get depth and frame
        depth = c.last_event.depth_frame
        frame = c.last_event.frame

        fram_np_float = np.array(frame, dtype=np.float32)/255.
        lseg_output_dict = lseg_predictor.predict(fram_np_float, labels)
        pix_cls = lseg_output_dict['pix_cls']

        pc, mask = depth2pc_ai2thor(depth, clipping_dist=0.1, fov=90)

        # Transformation matrix for the current pose
        current_pose_matrix = transformation_matrix(c.last_event.metadata['agent']['position'], c.last_event.metadata['agent']['rotation'])

        # Calculate the transformation matrix relative to the initial pose
        relative_pose_matrix = np.linalg.inv(initial_pose_matrix) @ current_pose_matrix

        # Transform the point cloud relative to the initial pose
        pc_homogeneous = np.vstack((pc, np.ones(pc.shape[1])))  # Convert to homogeneous coordinates
        transformed_pc = relative_pose_matrix @ pc_homogeneous
        transformed_pc = transformed_pc[:3, :]  # Convert back to non-homogeneous coordinates

        # Accumulate the transformed point cloud
        accumulated_pc = np.hstack((accumulated_pc, transformed_pc[:, mask]))

        # Get the colors for the current point cloud
        frame_reshaped = frame.reshape(-1, 3).T
        pix_cls_reshaped = pix_cls.reshape(-1)
        current_colors = frame_reshaped[:, mask]
        current_labels = pix_cls_reshaped[mask].unsqueeze(0)

        # Accumulate the colors
        accumulated_colors = np.hstack((accumulated_colors, current_colors))
        accumulated_labels = np.hstack((accumulated_labels, current_labels))

        # append to video list
        video_list.append((frame, depth, pc, mask, c.last_event.metadata['agent']['position'], c.last_event.metadata['agent']['rotation']))

    # Visualize the accumulated point cloud
    '''fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(accumulated_pc[0], accumulated_pc[1], accumulated_pc[2], s=1, c=accumulated_colors.T / 255.0)
    ax.view_init(elev=-40, azim=-110, roll=12)  # set the view to (30, 45)
    plt.show()'''

    # Lseg Visualize 
    fig = plt.figure()
    ax = fig.add_subplot(111) #projection='3d')
    colors = np.int32(colors)
    accumulated_colors_t = accumulated_colors.T
    accumulated_labels = accumulated_labels.astype(np.int32)
    colored_labels = np.take(colors, accumulated_labels[0], axis=0)

    query = 'cabinet'
    idx_label = labels.index(query)
    mask = accumulated_labels == idx_label
    mask = mask.T
    print(f'There is {np.sum(mask)} {query} in the scene.')

    # accumulated_colors_t (64004, 3) 
    # accumulated_labels (1, 64004)
    # colored_labels (64004, 3)
    # mask (64004, 1)

    print(colored_labels , accumulated_colors_t)
    alpha=0.5
    plot_color = mask*colored_labels + (1-mask)*accumulated_colors_t

    #ax.scatter(accumulated_pc[0], accumulated_pc[1], accumulated_pc[2], s=1, c=plot_color / 255.0)
    #ax.view_init(elev=-40, azim=-110, roll=12)
    plt.scatter(accumulated_pc[0], accumulated_pc[2], s=1, c=colored_labels / 255.0)
    # plot reachable positions
    plt.scatter(reachable_positions[:, 0], reachable_positions[:, 1], s=1, c='r')
    plt.show()

    '''plt.scatter(accumulated_pc[0], accumulated_pc[2], s=1, c=accumulated_colors.T / 255.0)
    # plot reachable positions
    plt.scatter(reachable_positions[:, 0], reachable_positions[:, 1], s=1, c='r')
    #plt.show()
    plt.savefig(f'/Users/seungyounshin/Downloads/{len(action_list)}.png')'''
