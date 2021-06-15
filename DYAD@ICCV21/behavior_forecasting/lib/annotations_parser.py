import h5py
import os
import numpy as np
from functools import partial

def filter_face_landmarks(l):
    """
        We only keep eyebrows + extremes and centers of the eyes + middle upper and lower mouth lines
    """
    m = partial(np.mean, axis=0)
    r_eyebrow = l[17:22]
    l_eyebrow = l[22:27]
    r_eye = np.round([l[36], m(l[[37, 38, 40, 41]]), l[39]])
    l_eye = np.round([l[42], m(l[[43, 44, 46, 47]]), l[45]])
    upper_mouth = [l[48], m(l[[49, 60]]), m(l[[50, 61]]), m(l[[51, 62]]), m(l[[52, 63]]), m(l[[53, 64]]), l[54]]
    lower_mouth = [m(l[[55, 64]]), m(l[[56, 65]]), m(l[[57, 66]]), m(l[[58, 67]]), m(l[[59, 60]])]
    
    new_landmarks = np.concatenate([r_eyebrow, l_eyebrow, r_eye, l_eye, upper_mouth, lower_mouth])
    return np.round(new_landmarks)

def filter_hand_landmarks(l):
    return l[1:] # palm is discarded because of noise

def filter_body_landmarks(l):
    body_joints = [9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    return l[body_joints] # only upper body joints considered

def parse_face_at(path_to_hdf5, num_frame, all=False):
    """
    :param path_to_hdf5: path to annotations 'hdf5' file
    :param num_frame: frame to extract annotations from
    :param all: if True, all original landmarks are returned. Otherwise, only those used for evaluation are returned.

    :return: confidence, landmarks, valid (only useful for validation and test sets)
    """
    assert os.path.exists(path_to_hdf5) or path_to_hdf5.split(".")[-1].lower() != "hdf5", "HDF5 file could not be opened."
    with h5py.File(path_to_hdf5, "r") as f:
        key_frame = f"{num_frame:05d}"
        if "face" not in f[key_frame]:
            return 0, [], False
        # -------- FACE --------
        confidence = f[key_frame]["face"].attrs["confidence"] if "confidence" in f[key_frame]["face"].attrs.keys() else 0
        valid = f[key_frame]["face"].attrs["valid"] if "valid" in f[key_frame]["face"].attrs.keys() else False
        landmarks = None
        if "landmarks" in f[key_frame]["face"].keys():
            landmarks = f[key_frame]["face"]["landmarks"][()]

    if not all and landmarks is not None:
        landmarks = filter_face_landmarks(landmarks).astype(int)

    return confidence, landmarks, valid

def parse_lhand_at(path_to_hdf5, num_frame, all=False):
    """
    :param path_to_hdf5: path to annotations 'hdf5' file
    :param num_frame: frame to extract annotations from
    :param all: if True, all original landmarks are returned. Otherwise, only those used for evaluation are returned.

    :return: confidence, landmarks, valid (only useful for validation and test sets)
    """
    assert os.path.exists(path_to_hdf5) or path_to_hdf5.split(".")[-1].lower() != "hdf5", "HDF5 file could not be opened."
    with h5py.File(path_to_hdf5, "r") as f:
        key_frame = f"{num_frame:05d}"
        if "hands" not in f[key_frame] or "left" not in f[key_frame]["hands"] or "right" not in f[key_frame]["hands"]:
            return 0, [], False
        # -------- LEFT HAND --------
        confidence = f[key_frame]["hands"]["left"].attrs["confidence"] if "confidence" in f[key_frame]["hands"]["left"].attrs.keys() else 0
        valid = f[key_frame]["hands"]["left"].attrs["valid"] if "valid" in f[key_frame]["hands"]["left"].attrs.keys() else False
        landmarks = None
        if "landmarks" in f[key_frame]["hands"]["left"].keys():
            landmarks = f[key_frame]["hands"]["left"]["landmarks"][()]

    if not all and landmarks is not None:
        landmarks = filter_hand_landmarks(landmarks).astype(int)

    return confidence, landmarks, valid

def parse_rhand_at(path_to_hdf5, num_frame, all=False):
    """
    :param path_to_hdf5: path to annotations 'hdf5' file
    :param num_frame: frame to extract annotations from
    :param all: if True, all original landmarks are returned. Otherwise, only those used for evaluation are returned.

    :return: confidence, landmarks, valid (only useful for validation and test sets)
    """
    assert os.path.exists(path_to_hdf5) or path_to_hdf5.split(".")[-1].lower() != "hdf5", "HDF5 file could not be opened."
    with h5py.File(path_to_hdf5, "r") as f:
        key_frame = f"{num_frame:05d}"
        if "hands" not in f[key_frame] or "left" not in f[key_frame]["hands"] or "right" not in f[key_frame]["hands"]:
            return 0, [], False
        # -------- RIGHT HAND --------
        confidence = f[key_frame]["hands"]["right"].attrs["confidence"] if "confidence" in f[key_frame]["hands"]["right"].attrs.keys() else 0
        valid = f[key_frame]["hands"]["right"].attrs["valid"] if "valid" in f[key_frame]["hands"]["right"].attrs.keys() else False
        landmarks = None
        if "landmarks" in f[key_frame]["hands"]["right"].keys():
            landmarks = f[key_frame]["hands"]["right"]["landmarks"][()]

    if not all and landmarks is not None:
        landmarks = filter_hand_landmarks(landmarks).astype(int)

    return confidence, landmarks, valid

def parse_body_at(path_to_hdf5, num_frame, all=False):
    """
    :param path_to_hdf5: path to annotations 'hdf5' file
    :param num_frame: frame to extract annotations from
    :param all: if True, all original landmarks are returned. Otherwise, only those used for evaluation are returned.

    :return: confidence, landmarks, valid (only useful for validation and test sets)
    """
    assert os.path.exists(path_to_hdf5) or path_to_hdf5.split(".")[-1].lower() != "hdf5", "HDF5 file could not be opened."
    with h5py.File(path_to_hdf5, "r") as f:
        key_frame = f"{num_frame:05d}"
        if "body" not in f[key_frame]:
            return 0, [], False
        # -------- BODY --------
        confidence = f[key_frame]["body"].attrs["confidence"] if "confidence" in f[key_frame]["body"].attrs.keys() else 0
        valid = f[key_frame]["body"].attrs["valid"] if "valid" in f[key_frame]["body"].attrs.keys() else False
        landmarks = None
        if "landmarks" in f[key_frame]["body"].keys():
            landmarks = f[key_frame]["body"]["landmarks"][()]

    if not all and landmarks is not None:
        landmarks = filter_body_landmarks(landmarks).astype(int)

    return confidence, landmarks, valid

def is_valid(path_to_hdf5, num_frame):
    """
    :param path_to_hdf5: path to annotations 'hdf5' file
    :param num_frame: frame to extract annotations from

    :return: boolean indicating if the frame is valid
    """
    assert os.path.exists(path_to_hdf5) or path_to_hdf5.split(".")[-1].lower() != "hdf5", "HDF5 file could not be opened."
    with h5py.File(path_to_hdf5, "r") as f:
        return f[f"{num_frame:05d}"].attrs["valid"]

def is_to_predict(path_to_hdf5, num_frame):
    """
    :param path_to_hdf5: path to annotations 'hdf5' file
    :param num_frame: frame to extract annotations from

    :return: boolean indicating if the frame has to be predicted
    """
    assert os.path.exists(path_to_hdf5) or path_to_hdf5.split(".")[-1].lower() != "hdf5", "HDF5 file could not be opened."
    with h5py.File(path_to_hdf5, "r") as f:
        key_frame = f"{num_frame:05d}"
        return "body" not in f[key_frame] and "face" not in f[key_frame] and "hands" not in f[key_frame]

def is_lhand_visible(path_to_hdf5, num_frame):
    """
    :param path_to_hdf5: path to annotations 'hdf5' file
    :param num_frame: frame to extract annotations from

    :return: boolean indicating if the left hand is visible in that frame
    """
    assert os.path.exists(path_to_hdf5) or path_to_hdf5.split(".")[-1].lower() != "hdf5", "HDF5 file could not be opened."
    with h5py.File(path_to_hdf5, "r") as f:
        return f[f"{num_frame:05d}"]["hands"]["left"].attrs["visible"]

def is_rhand_visible(path_to_hdf5, num_frame):
    """
    :param path_to_hdf5: path to annotations 'hdf5' file
    :param num_frame: frame to extract annotations from

    :return: boolean indicating if the right hand is visible in that frame
    """
    assert os.path.exists(path_to_hdf5) or path_to_hdf5.split(".")[-1].lower() != "hdf5", "HDF5 file could not be opened."
    with h5py.File(path_to_hdf5, "r") as f:
        return f[f"{num_frame:05d}"]["hands"]["right"].attrs["visible"]

def parse_gaze_at(path_to_hdf5, num_frame):
    """
    :param path_to_hdf5: path to annotations 'hdf5' file
    :param num_frame: frame to extract annotations from

    :return: gaze
    """
    assert os.path.exists(path_to_hdf5) or path_to_hdf5.split(".")[-1].lower() != "hdf5", "HDF5 file could not be opened."
    with h5py.File(path_to_hdf5, "r") as f:
        key_frame = f"{num_frame:05d}"
        if "face" not in f[key_frame]:
            return None
        # -------- FACE --------
        return f[key_frame]["face"].attrs["gaze"] if "gaze" in f[key_frame]["face"].attrs.keys() else None
