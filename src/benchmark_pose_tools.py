"""
포즈 추정 도구 비교 벤치마크 (MediaPipe vs RTMPose vs YOLO-pose)

[왜 필요한가]
지금까지 MediaPipe로 겪은 문제(검출 실패 39~49%, 무릎 오배치, 깊이 z 불안정)가
도구를 바꾸면 나아지는지 확인하려면, 같은 구간에 같은 잣대를 대봐야 합니다.
이 스크립트는 세 도구를 동일한 클립에 돌려서 아래 지표를 뽑습니다:

  1) 검출률        - 처리한 프레임 중 사람을 찾은 비율 (높을수록 좋음)
  2) 추적 안정성    - 몸 중심이 프레임 사이 크게 순간이동한 횟수 (낮을수록 좋음)
  3) 뼈 길이 일관성 - 같은 사람의 허벅지/정강이 길이가 프레임마다 얼마나 흔들리는지.
                     실제 뼈 길이는 변하지 않으므로, 변동이 크면 관절 위치가
                     부정확하다는 뜻입니다 (낮을수록 좋음). MediaPipe의 무릎
                     오배치를 잡아내는 핵심 지표.
  4) 킥 신호 선명도 - 발끝 상하 움직임에서 "돌핀킥 주기 성분"이 얼마나 뚜렷한지.
                     자기상관(autocorrelation)의 최대값으로 재며, 신호가 주기적일수록
                     1에 가깝습니다 (높을수록 좋음).

세 도구가 주는 관절 이름이 서로 달라서(MediaPipe 33점 / COCO 17점 / Halpe 26점),
비교에 쓰는 관절만 공통 이름으로 맞춰서 사용합니다.

사용법:
    python src/benchmark_pose_tools.py
"""
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent

# 비교할 구간 - 지금 분석에 쓰고 있는 돌핀킥 구간과 동일하게 맞춘다.
CLIPS = [
    ("reference", ROOT / "data" / "videos" / "reference.mp4", 15.0, 18.6),
    ("mine", ROOT / "data" / "videos" / "mine.mp4", 13.0, 17.3),
]

# 도구마다 관절 순서가 다르므로, 비교에 쓰는 관절의 인덱스만 매핑해둔다.
COCO17 = {"L_SHOULDER": 5, "R_SHOULDER": 6, "L_HIP": 11, "R_HIP": 12,
          "L_KNEE": 13, "R_KNEE": 14, "L_ANKLE": 15, "R_ANKLE": 16}
MEDIAPIPE = {"L_SHOULDER": 11, "R_SHOULDER": 12, "L_HIP": 23, "R_HIP": 24,
             "L_KNEE": 25, "R_KNEE": 26, "L_ANKLE": 27, "R_ANKLE": 28}


def read_clip(path, start, end):
    """구간의 프레임을 전부 메모리로 읽어, 모든 도구가 똑같은 입력을 보게 한다."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    frames = []
    for _ in range(int((end - start) * fps)):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def autocorr_peak(sig):
    """신호의 주기성 세기. 자기상관에서 (0 지연을 뺀) 최대값을 돌려준다."""
    s = np.asarray(sig, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 20:
        return 0.0
    s = s - s.mean()
    if s.std() < 1e-9:
        return 0.0
    ac = np.correlate(s, s, mode="full")[len(s) - 1:]
    ac /= ac[0]
    # 최소 0.25초(=7프레임@30fps)부터 봐야 "바로 옆 프레임과 비슷함"을 주기로 오인하지 않는다
    lo = 7
    return float(ac[lo:].max()) if len(ac) > lo else 0.0


def evaluate(name, per_frame_kps, idx_map, n_frames):
    """도구가 뽑은 프레임별 키포인트에서 공통 지표를 계산한다.

    per_frame_kps: 프레임마다 (K,2) 배열 또는 None
    """
    got = [k for k in per_frame_kps if k is not None]
    det_rate = len(got) / n_frames if n_frames else 0.0

    centers, thigh, shin, foot_y, torso_len = [], [], [], [], []
    for k in per_frame_kps:
        if k is None:
            continue
        try:
            lh, rh = k[idx_map["L_HIP"]], k[idx_map["R_HIP"]]
            ls, rs = k[idx_map["L_SHOULDER"]], k[idx_map["R_SHOULDER"]]
            lk, ak = k[idx_map["L_KNEE"]], k[idx_map["L_ANKLE"]]
        except (IndexError, KeyError):
            continue
        hip_c = (lh + rh) / 2
        sh_c = (ls + rs) / 2
        t = np.linalg.norm(sh_c - hip_c)
        if t < 1e-6:
            continue
        centers.append(hip_c)
        torso_len.append(t)
        thigh.append(np.linalg.norm(lk - lh) / t)
        shin.append(np.linalg.norm(ak - lk) / t)
        foot_y.append(-ak[1] / t)   # 화면 y는 아래가 +라 부호 반전

    if len(centers) < 10:
        return dict(tool=name, det=det_rate, jumps=np.nan, thigh_cv=np.nan,
                    shin_cv=np.nan, kick=np.nan)

    centers = np.array(centers)
    scale = np.mean(torso_len)
    step = np.linalg.norm(np.diff(centers, axis=0), axis=1) / scale
    jumps = int((step > 0.5).sum())   # 몸통 길이의 절반 이상 순간이동 = 추적 튐

    return dict(
        tool=name, det=det_rate, jumps=jumps,
        thigh_cv=float(np.std(thigh) / np.mean(thigh) * 100),
        shin_cv=float(np.std(shin) / np.mean(shin) * 100),
        kick=autocorr_peak(foot_y),
    )


def run_mediapipe(frames, fps):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(ROOT / "models" / "pose_landmarker_full.task")),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    lmk = vision.PoseLandmarker.create_from_options(opts)
    out = []
    for i, f in enumerate(frames):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        r = lmk.detect_for_video(img, int(i / fps * 1000))
        if r.pose_landmarks:
            h, w = f.shape[:2]
            out.append(np.array([[p.x * w, p.y * h] for p in r.pose_landmarks[0]]))
        else:
            out.append(None)
    lmk.close()
    return out, MEDIAPIPE


class SwimmerPicker:
    """다인 검출기가 찾은 여러 사람 중 "우리가 보려는 수영선수"를 고른다.

    [왜 필요한가]
    RTMPose/YOLO는 화면의 모든 사람을 찾아줍니다. 이 영상에는 풀사이드에 서 있는
    사람들이 함께 잡히기 때문에, 그냥 첫 번째 결과를 쓰면 프레임마다 다른 사람을
    재게 됩니다(그러면 "뼈 길이"가 사람마다 달라 변동이 폭발합니다 - 실제로 첫
    벤치마크에서 52~215%가 나왔고, 그건 도구 탓이 아니라 이 선택 실수 탓이었습니다).

    고르는 방법: 직전 프레임에서 고른 사람의 엉덩이 중심과 가장 가까운 사람을
    잇습니다(연속성 추적). 첫 프레임은 "가장 크게 잡힌 사람"(카메라가 따라다니는
    주인공이므로 화면에서 가장 큼)으로 시작합니다.
    """

    def __init__(self, idx_map, max_jump_px=120):
        self.idx = idx_map
        self.prev = None
        self.max_jump = max_jump_px

    @staticmethod
    def _span(k):
        return float(np.linalg.norm(k.max(axis=0) - k.min(axis=0)))

    def pick(self, people):
        people = [np.asarray(p, dtype=float) for p in people if p is not None and len(p) >= 17]
        if not people:
            return None
        if self.prev is None:
            best = max(people, key=self._span)
        else:
            def hip(k):
                return (k[self.idx["L_HIP"]] + k[self.idx["R_HIP"]]) / 2
            prev_hip = hip(self.prev)
            best = min(people, key=lambda k: np.linalg.norm(hip(k) - prev_hip))
            # 너무 멀리 떨어진 후보만 남았다면 추적이 끊긴 것으로 보고 크기로 다시 잡는다
            if np.linalg.norm(hip(best) - prev_hip) > self.max_jump:
                best = max(people, key=self._span)
        self.prev = best
        return best


def run_rtmpose(frames, mode="balanced"):
    from rtmlib import Body
    body = Body(mode=mode, backend="onnxruntime", device="cpu")
    picker = SwimmerPicker(COCO17)
    out = []
    for f in frames:
        kps, _scores = body(f)
        out.append(picker.pick([k[:, :2] for k in kps] if len(kps) else []))
    return out, COCO17


def run_yolo(frames, weights="yolo11n-pose.pt"):
    from ultralytics import YOLO
    model = YOLO(weights)
    picker = SwimmerPicker(COCO17)
    out = []
    for f in frames:
        res = model.predict(f, verbose=False, device="cpu")[0]
        kp = res.keypoints
        people = []
        if kp is not None and kp.xy is not None:
            people = [x.cpu().numpy() for x in kp.xy if x.shape[0] >= 17]
        out.append(picker.pick(people))
    return out, COCO17


def main() -> None:
    tools = [
        ("MediaPipe", run_mediapipe, True),
        ("RTMPose(balanced)", lambda fr, fps: run_rtmpose(fr), False),
        ("YOLO11n-pose", lambda fr, fps: run_yolo(fr, "yolo11n-pose.pt"), False),
        ("YOLO11x-pose", lambda fr, fps: run_yolo(fr, "yolo11x-pose.pt"), False),
    ]

    for clip_name, path, start, end in CLIPS:
        frames, fps = read_clip(path, start, end)
        print(f"\n{'=' * 78}\n{clip_name}  {start}~{end}s  ({len(frames)}프레임 @{fps:.0f}fps)\n{'=' * 78}")
        print(f"{'도구':<20}{'검출률':>8}{'추적튐':>8}{'허벅지변동':>11}{'정강이변동':>11}{'킥주기성':>9}{'소요':>8}")
        print("-" * 78)
        for name, fn, needs_fps in tools:
            t0 = time.time()
            try:
                kps, idx_map = fn(frames, fps) if needs_fps else fn(frames, fps)
            except Exception as exc:  # 설치/실행 실패한 도구는 표에 사유를 남기고 넘어간다
                print(f"{name:<20}  실패: {type(exc).__name__}: {str(exc)[:44]}")
                continue
            m = evaluate(name, kps, idx_map, len(frames))
            dt = time.time() - t0
            print(f"{name:<20}{m['det'] * 100:7.0f}%{m['jumps']:8}{m['thigh_cv']:10.1f}%"
                  f"{m['shin_cv']:10.1f}%{m['kick']:9.2f}{dt:7.0f}s")
        print("\n  검출률·킥주기성은 높을수록, 추적튐·뼈길이 변동은 낮을수록 좋음")


if __name__ == "__main__":
    main()
