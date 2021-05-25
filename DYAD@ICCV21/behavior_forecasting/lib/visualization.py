import cv2
import numpy as np

COLOR_HAND_JOINTS = [[0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                     [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                     [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                     [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                     [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little

BODY_EDGES = [[0, 1], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8], [7, 9]]
BODY_COLORS_EDGES = [[1.0, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0], [0.0, 0.4, 0.4], [0.0, 0.6, 0.6],]
COLORS = {"green": (0, 255, 0), "red": (255, 0, 0), "black": (0, 0, 0), "white": (255, 255, 255)}

def draw_bbox(img, box, color, width=1):
    """
    :param image: H x W x 3
    :param box: xyxy format
    :param box: RGB color for bbox
    :return: image with bbox drawn
    """
    left, top, right, bottom = np.round(box).astype(np.int32)
    left_top = (left, top)
    right_top = (right, top)
    right_bottom = (right, bottom)
    left_bottom = (left, bottom)
    cv2.line(img, left_top, right_top, color, width, cv2.LINE_AA)
    cv2.line(img, right_top, right_bottom, color, width, cv2.LINE_AA)
    cv2.line(img, right_bottom, left_bottom, color, width, cv2.LINE_AA)
    cv2.line(img, left_bottom, left_top, color, width, cv2.LINE_AA)

def draw_face(image, landmarks, valid=True, size=1):
    """
    :param image: H x W x 3
    :param landmarks: 28 x 2
    :return: image with landmarks drawn
    """
    color = COLORS['red'] if not valid else COLORS["green"]

    if landmarks is None:
        return image
        
    assert landmarks.shape[0] == 28
    img = image.copy()
    
    for i in range(len(landmarks)):
        cv2.circle(img, (int(round(landmarks[i][0])), int(round(landmarks[i][1]))), size, color, -1)

    return img

def draw_hand(image, landmarks, valid=True, bbox=None):
    """
    :param image: H x W x 3
    :param landmarks: 20 x 2 (4x5 fingers)
    thumb_mcp, thumb_pip, thumb_dip, thumb_tip
    index_mcp, index_pip, index_dip, index_tip,
    middle_mcp, middle_pip, middle_dip, middle_tip,
    ring_mcp, ring_pip, ring_dip, ring_tip,
    little_mcp, little_pip, little_dip, little_tip
    :return: image with landmarks drawn
    """
    if landmarks is None:
        return image
        
    assert landmarks.shape[0] == 20
    skeleton_overlay = image.copy()
    marker_sz = 6
    line_wd = 3
    root_ind = 0

    if valid:
        for joint_ind in range(landmarks.shape[0]):
            joint = landmarks[joint_ind, 0].astype('int32'), landmarks[joint_ind, 1].astype('int32')
            if joint_ind % 4 != 0:
                # draw finger line
                joint_2 = landmarks[joint_ind - 1, 0].astype('int32'), landmarks[joint_ind - 1, 1].astype('int32')
                color_line = COLOR_HAND_JOINTS[joint_ind] * np.array(255)
                cv2.line(
                    skeleton_overlay, joint_2, joint,
                    color=color_line, thickness=int(line_wd),
                    lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)
            cv2.circle(
                skeleton_overlay, joint,
                radius=marker_sz, color=COLOR_HAND_JOINTS[joint_ind] * np.array(255), thickness=-1,
                lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)

    if bbox is not None:
        color = COLORS[bbox] if valid else COLORS["red"]
        margin = 10
        hand_bbox = [min(landmarks[:,0])-margin, min(landmarks[:,1])-margin, max(landmarks[:,0])+margin, max(landmarks[:, 1])+margin]
        draw_bbox(skeleton_overlay, hand_bbox, color, width=2)


    return skeleton_overlay

def draw_body(image, landmarks, valid=True, size=7, width=2, right_visible=True, left_visible=True):
    """
    :param image: H x W x 3
    :param landmarks: 10 x 2
    middle_chest, neck, left_chest, right_chest, left_shoulder, right_shoulder, 
    left_elbow, right_elbow, left_wrist and right_wrist

    :return: image with landmarks drawn
    """

    if landmarks is None:
        return image
        
    assert landmarks.shape[0] == 10
    skeleton_overlay = image.copy()

    for edge_idx, (i_start, i_end) in enumerate(BODY_EDGES):
        if (not right_visible and (i_start == 9 or i_end == 9)) or \
            (not left_visible and (i_start == 8 or i_end == 8)):
            continue
        cv2.line(
            skeleton_overlay, tuple(landmarks[i_start][:2]), tuple(landmarks[i_end][:2]),
            color=tuple(BODY_COLORS_EDGES[edge_idx] * np.array(255)) if valid else COLORS["red"], thickness=width,
            lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)

    for joint_idx, joint in enumerate(landmarks):
        if (not right_visible and joint_idx == 9) or \
            (not left_visible and joint_idx == 8):
            continue
        cv2.circle(
            skeleton_overlay, tuple(joint[:2]),
            radius=size, color=tuple(BODY_COLORS_EDGES[joint_idx] * np.array(255)) if valid else COLORS["red"], thickness=-1,
            lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)

    return skeleton_overlay

def draw_gaze_vector(image_in, vector, face_landmarks, thickness=2, color=(0, 0, 255)):
    face_landmarks = face_landmarks[:, :2]
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)

    # The gaze needs to be projected to the image space to be drawn
    # We define the approximate camera matrix used for gaze extraction
    cam_matrix = np.array(((1.2 * 720, 0, 1280./2),
                            (0., 1.2 * 720, 720./2),
                            (0., 0., 1.)))
    face_center = __compute_face_center(face_landmarks, cam_matrix)

    # We project and draw the gaze vector
    vec2d = __project_gaze(face_center.transpose(), vector, cam_matrix)
    cv2.arrowedLine(image_out, tuple(np.round(vec2d[0,:]).astype(int)),
                    tuple(np.round(vec2d[1,:]).astype(int)), color,
                          thickness, cv2.LINE_AA, tipLength=0.2)

    return image_out

def __project_points(points_3D, intrinsic_mat):
    points_2D = np.dot(points_3D, intrinsic_mat.transpose())
    points_2D = points_2D[:, :2] / (points_2D[:, 2].reshape(-1, 1))
    return points_2D

def __project_gaze(init_vector, vector, intrinsic_mat):
    offset = 200.0
    points_3D = np.empty((2,3))
    points_3D[0,:] = init_vector
    points_3D[1,:] = init_vector + vector*offset
    return __project_points(points_3D, intrinsic_mat)

def __estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)
    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)
    return rvec, tvec

def __compute_face_center(face_landmarks, cam_matrix):
    camera_distortion = np.zeros((5,))
    # 3D face model
    face_model = np.array([ [-46.5094986,  -38.32709503,  36.41600418],
                            [-17.76072121, -32.58519745,  29.07615662],
                            [ 18.04391098, -30.95682335,  29.0629673 ],
                            [ 44.73758698, -34.10787201,  36.73243713],
                            [-10.61166477,  12.95834923,  21.62276459],
                            [ 11.28962994,  13.86424446,  21.83790016]
                        ])

    ## the easy way to get head pose information, fast and simple
    facePts = face_model.reshape(6, 1, 3)
    landmarks_sub = face_landmarks[[36, 39, 42, 45, 31, 35], :]
    landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
    landmarks_sub = landmarks_sub.reshape(6, 1, 2)  # input to solvePnP requires such shape
    hr, ht = __estimateHeadPose(landmarks_sub, facePts, cam_matrix, camera_distortion)

    # we compute the face center
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht  # rotate and translate the face model
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))

    return face_center
