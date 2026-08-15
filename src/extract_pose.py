"""
관절 좌표 추출 (프로젝트1의 extract_pose.py를 두 영상 비교용으로 확장)

프로젝트1과 하는 일은 동일합니다 (MediaPipe Pose로 33개 랜드마크 추출).
차이점은 영상 경로 / 결과 저장 위치를 하드코딩하지 않고 인자로 받는다는
것뿐입니다 — 이번 프로젝트는 "정답 영상"과 "내 영상" 두 번을 돌려야 하므로,
같은 스크립트를 --video/--out만 바꿔서 두 번 실행하면 됩니다.

사용법:
    python src/extract_pose.py --video data/videos/reference.mp4 --out output/reference/pose_landmarks.csv --frames-dir data/frames/reference
    python src/extract_pose.py --video data/videos/mine.mp4      --out output/mine/pose_landmarks.csv      --frames-dir data/frames/mine
"""
import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "pose_landmarker_full.task"

# 33개 랜드마크 이름 (인덱스 순서 = 모델 출력 순서)
LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

# 몸통/팔/다리 뼈대 연결 (확인용 이미지에 선을 그릴 때 사용)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),   # 어깨-팔
    (11, 23), (12, 24), (23, 24),                        # 몸통
    (23, 25), (25, 27), (24, 26), (26, 28),              # 다리
    (27, 29), (29, 31), (28, 30), (30, 32),              # 발
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, help="입력 영상 경로 (예: data/videos/reference.mp4)")
    p.add_argument("--out", required=True, help="결과 CSV 경로 (예: output/reference/pose_landmarks.csv)")
    p.add_argument("--frames-dir", default=None, help="확인용 이미지 저장 폴더 (기본: out과 같은 폴더 아래 frames/)")
    p.add_argument("--start", type=float, default=0, help="시작 지점 (초). 기본 0 = 처음부터")
    p.add_argument("--duration", type=float, default=1e9, help="처리할 길이 (초). 기본은 영상 끝까지")
    p.add_argument("--step", type=int, default=2, help="N프레임마다 1장 처리 (기본 2)")
    p.add_argument("--save-every", type=int, default=60, help="관절 그린 이미지를 몇 장마다 저장할지")
    return p.parse_args()


def draw_landmarks(frame, landmarks):
    """검출된 관절 위치에 점과 뼈대 선을 그린다."""
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    csv_path = Path(args.out)
    frames_dir = Path(args.frames_dir) if args.frames_dir else csv_path.parent / "frames"

    frames_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # VIDEO 모드: 프레임 간 추적(tracking)을 활용해 더 안정적이고 빠름
    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(args.start * fps)
    end_frame = int((args.start + args.duration) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 각 관절마다 두 종류의 좌표를 함께 저장한다:
    #   _x/_y/_z/_vis  : 화면 기준 정규화 좌표 (0~1, 그림 그리기/디버깅용)
    #   _wx/_wy/_wz    : MediaPipe가 추정한 실제 3D 좌표 (미터 단위, 엉덩이 중심)
    #                    카메라 각도가 달라도 상대적으로 안정적이라 각도 계산에 사용
    header = ["frame", "time_sec"]
    for name in LANDMARK_NAMES:
        header += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_vis",
                   f"{name}_wx", f"{name}_wy", f"{name}_wz"]

    processed = 0
    detected = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        frame_idx = start_frame
        while frame_idx < end_frame:
            ok, frame = cap.read()
            if not ok:
                break  # 영상 끝

            if (frame_idx - start_frame) % args.step != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(frame_idx / fps * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            processed += 1
            if result.pose_landmarks and result.pose_world_landmarks:
                detected += 1
                landmarks = result.pose_landmarks[0]
                world_landmarks = result.pose_world_landmarks[0]
                row = [frame_idx, round(frame_idx / fps, 2)]
                for lm, wlm in zip(landmarks, world_landmarks):
                    row += [round(lm.x, 4), round(lm.y, 4),
                            round(lm.z, 4), round(lm.visibility, 3),
                            round(wlm.x, 4), round(wlm.y, 4), round(wlm.z, 4)]
                writer.writerow(row)

                if processed % args.save_every == 0:
                    draw_landmarks(frame, landmarks)
                    cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.jpg"), frame)

            frame_idx += 1

    cap.release()
    landmarker.close()
    print(f"처리한 프레임: {processed}장")
    print(f"사람 검출 성공: {detected}장 ({detected / max(processed, 1) * 100:.0f}%)")
    print(f"좌표 저장 위치: {csv_path}")
    print(f"확인용 이미지: {frames_dir}")


if __name__ == "__main__":
    main()
