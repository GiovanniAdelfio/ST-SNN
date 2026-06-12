import cv2
import numpy as np
import torch
import mediapipe as mp
import argparse
import os
import traceback

from pyskl.apis import init_recognizer, inference_recognizer

def parse_args():
    parser = argparse.ArgumentParser(description="Demo for STSNN++ with pose estimation")
    parser.add_argument("--video",      type=str, required=True)
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--labels",     type=str, required=True)
    parser.add_argument("--device",     type=str, default="cuda:0")
    return parser.parse_args()


def convert_33_to_25_joints(landmarks_33):
    # MediaPipe 33 joints -> NTU-RGB+D 60 25 joints mapping
    mediapipe_to_ntu60 = {
        0: 3,  11: 4,  12: 8,  13: 5,  14: 9,  15: 6,  16: 10,
        19: 7, 20: 11, 21: 22, 22: 24, 23: 12, 24: 16,
        25: 13, 26: 17, 27: 14, 28: 18, 29: 15, 30: 19,
    }
    joints_25 = [[0.0, 0.0, 0.0] for _ in range(25)]
    for mp_idx, ntu_idx in mediapipe_to_ntu60.items():
        if mp_idx < len(landmarks_33):
            joints_25[ntu_idx] = landmarks_33[mp_idx].copy()

    # Compute extra body centers
    ls, rs = joints_25[4], joints_25[8]
    lh, rh = joints_25[12], joints_25[16]
    if ls != [0,0,0] and rs != [0,0,0]:
        joints_25[2] = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2, (ls[2]+rs[2])/2]
    if lh != [0,0,0] and rh != [0,0,0]:
        hc = [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2, (lh[2]+rh[2])/2]
        joints_25[0] = hc
        if joints_25[2] != [0,0,0]:
            joints_25[1] = [(joints_25[2][0]+hc[0])/2,
                            (joints_25[2][1]+hc[1])/2,
                            (joints_25[2][2]+hc[2])/2]
    if joints_25[1] != [0,0,0] and joints_25[2] != [0,0,0]:
        joints_25[20] = [(joints_25[1][0]+joints_25[2][0])/2,
                         (joints_25[1][1]+joints_25[2][1])/2,
                         (joints_25[1][2]+joints_25[2][2])/2]
    return joints_25


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    # 1. Load model
    print("[1/3] Loading model...")
    model = init_recognizer(args.config, args.checkpoint, device=args.device)
    model.eval()
    print("-> Model loaded")

    # 2. Skeleton extraction
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                        smooth_landmarks=True, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(args.video)
    rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[2/3] Video {width}x{height} @ {fps:.1f} FPS (press Q to stop)")

    window_name = 'Skeleton Demo'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 600)

    all_joints = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        """
        # Fix orientation if needed
        if rotation in [90, 270] and frame.shape[1] == width:
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == 180 and frame.shape[1] == width:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
                """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = [[l.x, l.y, l.z] for l in results.pose_landmarks.landmark]
            joints = convert_33_to_25_joints(lm)
            h, w = frame.shape[:2]
            for j in joints:
                if j[0] != 0 and j[1] != 0:
                    cv2.circle(frame, (int(j[0]*w), int(j[1]*h)), 4, (0,255,0), -1)
        else:
            joints = [[0.0, 0.0, 0.0] for _ in range(25)]

        # Compute hand extension points (optional)
        lhj, lwj = joints[7], joints[6]
        if lhj[0] != 0:
            joints[21] = [lhj[0]+(lhj[0]-lwj[0])*.5,
                          lhj[1]+(lhj[1]-lwj[1])*.5,
                          lhj[2]+(lhj[2]-lwj[2])*.5]
        rhj, rwj = joints[11], joints[10]
        if rhj[0] != 0:
            joints[23] = [rhj[0]+(rhj[0]-rwj[0])*.5,
                          rhj[1]+(rhj[1]-rwj[1])*.5,
                          rhj[2]+(rhj[2]-rwj[2])*.5]

        all_joints.append(joints)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    # 3. Pipeline and inference
    print("\n[3/3] Running pipeline and inference...")
    T = len(all_joints)
    print(f"-> Extracted frames: {T}")
    if T < 10:
        print("Too few frames.")
        return

    raw_skeleton        = np.zeros((2, T, 25, 3), dtype=np.float32)
    raw_skeleton[0]     = np.array(all_joints, dtype=np.float32)
    scores_arr          = np.zeros((2, T, 25), dtype=np.float32)
    scores_arr[0]       = 1.0

    data = dict(
        frame_dir      = '',
        label          = -1,
        img_shape      = (height, width),
        original_shape = (height, width),
        start_index    = 0,
        modality       = 'Pose',
        total_frames   = T,
        keypoint       = raw_skeleton,
        keypoint_score = scores_arr,
    )

    from pyskl.datasets.pipelines import (
        PreNormalize3D, UniformSample, PoseDecode,
        FormatGCNInput, Collect, ToTensor
    )

    # Pipeline without GenSkeFeat; we will drop the score channel later
    pipeline = [
        PreNormalize3D(),
        UniformSample(clip_len=100, num_clips=10),
        PoseDecode(),
        FormatGCNInput(num_person=2),
        Collect(keys=['keypoint', 'label'], meta_keys=[]),
        ToTensor(keys=['keypoint']),
    ]

    print("-> Applying pipeline...")
    try:
        for transform in pipeline:
            data = transform(data)
    except Exception:
        print("Pipeline error:")
        traceback.print_exc()
        return

    kp = data['keypoint']                     # (10, 2, 100, 25, 4)
    kp = kp[..., :3]                          # drop score, keep xyz
    print(f"-> Shape after dropping score: {kp.shape}")

    if isinstance(kp, np.ndarray):
        kp = torch.from_numpy(kp)
    kp = kp.float().unsqueeze(0).to(args.device)   # (1, 10, 2, 100, 25, 3)
    print(f"-> Tensor for forward_test: {kp.shape}")

    print("-> Forward pass...")
    try:
        with torch.no_grad():
            result = model.forward(kp, return_loss=False)
        print("-> Inference successful")
    except Exception:
        print("Forward error:")
        traceback.print_exc()
        return

    # 4. Results
    if isinstance(result, torch.Tensor):
        result = result.cpu().numpy()
    result = np.array(result).flatten()

    with open(args.labels) as f:
        labels = [l.strip() for l in f]

    top5 = np.argsort(result)[::-1][:5]
    print("\n" + "="*55)
    print(f"Predicted action : {labels[top5[0]]}")
    print(f"Confidence       : {result[top5[0]]*100:.2f}%")
    print("\nTop-5:")
    for i, idx in enumerate(top5, 1):
        bar = "█" * int(result[idx] * 30)
        print(f"    {i}. {labels[idx]:<35} {result[idx]*100:5.1f}%  {bar}")
    print("="*55)


if __name__ == "__main__":
    main()