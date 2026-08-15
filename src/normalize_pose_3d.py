"""
4단계 확장: 대표 스트로크 1회의 3D 관절 좌표를 진행률 0~100%로 리샘플링하기

[왜 평균이 아니라 스트로크 1개인가]
처음엔 8개 관절 각도(normalize_reps.py)처럼 여러 스트로크를 평균 내서 3D
"평균 자세"를 만들려고 했습니다. 그런데 각도 하나(스칼라 값)를 평균하는 것과
전신 3D 좌표(팔다리가 원을 그리며 움직이는 벡터)를 평균하는 것은 전혀 다른
문제였습니다. 스트로크마다 팔이 정확히 같은 타이밍에 같은 위치를 지나가지
않기 때문에, 좌표나 방향을 평균 내면
  - 팔이 실제보다 짧아지거나(원 둘레의 여러 점을 평균 내면 중심에 가까워짐),
  - 이웃한 뼈들이 서로 다른 순간의 몸 방향을 기준으로 섞여 뒤틀린 자세가
    나오거나("짜깁기" 포즈),
  - 스트로크마다 최고조 시점이 살짝 어긋나서 진폭이 뭉개져 제자리 운동처럼
    보이는
문제가 반복해서 나타났습니다. 이런 문제를 하나씩 보정하려고 여러 겹(방향
정규화, 인체비율 보정, 각도 재보정)을 쌓아봤지만, 근본 원인은 "애초에 여러
스트로크의 3D 좌표를 평균 내려 한 것" 자체였습니다.

그래서 접근을 바꿨습니다: 3D 애니메이션은 평균을 아예 내지 않고, 길이가
중앙값에 가장 가까운 "대표 스트로크 1개"의 실제 좌표를 그대로 씁니다. 실제로
촬영된 동작이므로 관절 각도, 뼈 길이, 스윙 궤적이 전부 자동으로 물리적으로
말이 됩니다 - 뭔가를 다시 계산하거나 보정할 필요가 없습니다.

(숫자 기반 피드백/그래프는 지금처럼 계속 여러 스트로크를 평균한 값을 씁니다 -
그건 "평균적으로 얼마나 다른지"를 신뢰성 있게 보여주는 게 목적이라 평균이
맞는 선택입니다. 3D 애니메이션은 반대로 "동작이 실제로 어떻게 생겼는지"를
직관적으로 보여주는 게 목적이라, 통계적 평균보다 실제로 있었던 동작 하나를
보여주는 쪽이 낫습니다. 두 결과물의 숫자가 100% 똑같지 않을 수 있는 이유이기도
합니다 - app.py에도 "여러 스트로크 평균 기준" 숫자와 "이 장면(대표 스트로크)"
숫자를 구분해서 표시합니다.)

[그래도 팔이 짧아 보이는 이유 - MediaPipe의 깊이(z) 추정 자체가 짧게 잡힘]
평균 문제는 없앴지만, MediaPipe가 카메라 1대로 깊이(z)까지 추정하는 한계는
단일 프레임에도 그대로 있습니다(확인 결과 어깨-팔꿈치 길이가 몸통의 약 1/3 -
표준 인체비율표 기준으로는 몸통의 약 2/3이어야 정상). 몸에 붙어 카메라 각도가
크게 안 바뀌는 몸통(어깨-엉덩이)은 비교적 정확하지만, 빠르게 움직이며 카메라
쪽으로/반대쪽으로 향하는 팔다리는 깊이 추정 오차가 커서 짧게 나옵니다. 그래서
팔다리(위팔/아래팔/허벅지/정강이)는 "길이"만 표준 인체비율(Winter/de Leva
인체측정표 기준)로 다시 스케일하고, "방향"은 그 프레임에서 실제로 측정된 값을
그대로 씁니다. 관절 각도는 두 뼈의 방향에만 좌우되므로(길이와 무관), 길이만
바꾸는 이 보정은 각도에 전혀 영향을 주지 않습니다 - 그래서 이전 버전에 있던
"각도 재보정" 단계가 이제는 필요 없습니다.

[약간의 흔들림 보정]
그래도 프레임 단위 관절 검출에는 자잘한 노이즈가 섞여 있어서, 이동평균으로
살짝만 다듬습니다(segment_reps.py 등 프로젝트 다른 곳에서 쓰는 것과 같은
방식) - 창이 작아서(5프레임) 실제 동작의 궤적을 지우지는 않습니다.

사용법:
    python src/normalize_pose_3d.py --landmarks output/reference/pose_landmarks.csv --reps output/reference/reps.csv --output output/reference/normalized_pose_mean.csv
    python src/normalize_pose_3d.py --landmarks output/mine/pose_landmarks.csv      --reps output/mine/reps.csv      --output output/mine/normalized_pose_mean.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from pose_projection import LANDMARK_NAMES

AXES = ["wx", "wy", "wz"]
COLUMNS = [f"{name}_{axis}" for name in LANDMARK_NAMES for axis in AXES]

# 화면 좌표(x, y)도 함께 실어 보낸다.
#
# [왜 3D 좌표만으로는 부족한가 - 돌핀킥의 "몸의 파동"이 3D에 없다]
# MediaPipe의 world 좌표(wx/wy/wz)는 정의상 "엉덩이 중점"을 원점으로 삼습니다
# (실측: 엉덩이 중심의 크기가 0.0007 ≈ 0). 즉 엉덩이가 위아래로 출렁이는 움직임이
# 3D 좌표에는 애초에 담겨 있지 않습니다. 게다가 그림을 그릴 때 몸통(엉덩이->어깨)을
# 기준축으로 삼기 때문에 몸통 기울기 변화(실측 12~23도)까지 0으로 지워집니다.
# 돌핀킥은 가슴->허리->엉덩이->무릎->발로 파동이 전달되는 동작인데, 그 파동을 만드는
# 두 성분이 바로 이 둘이라서, 3D 기반 그림은 "뻣뻣한 판자에 다리만 펄럭이는" 모습이
# 됩니다.
#
# 다행히 이 프로젝트의 영상은 카메라가 고정된 정측면이고 선수가 화면 평면 안에서
# 움직이므로, 화면 좌표 자체가 이미 충실한 측면 뷰입니다. 그래서 화면 좌표를 같이
# 저장해 두고, 측면 스틱맨은 이걸로 그립니다 (pose_projection.screen_frame 참고).
# 좌우 정보는 화면 좌표에 없으므로 정면 뷰는 계속 3D를 씁니다.
SCREEN_AXES = ["x", "y"]
SCREEN_COLUMNS = [f"{name}_{axis}" for name in LANDMARK_NAMES for axis in SCREEN_AXES]

# 뼈대 사슬 (자식 -> 부모). 부모가 먼저 확정돼야 자식 위치를 이어 붙일 수 있으므로
# 반드시 이 순서(부모가 자식보다 앞)로 처리한다. ROOT(왼쪽 엉덩이)는 실측값 그대로.
ROOT = "LEFT_HIP"
BONE_HIERARCHY = [
    ("RIGHT_HIP", "LEFT_HIP"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_ELBOW", "LEFT_SHOULDER"),
    ("LEFT_WRIST", "LEFT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_SHOULDER"),
    ("RIGHT_WRIST", "RIGHT_ELBOW"),
    ("LEFT_KNEE", "LEFT_HIP"),
    ("LEFT_ANKLE", "LEFT_KNEE"),
    ("LEFT_HEEL", "LEFT_ANKLE"),
    ("LEFT_FOOT_INDEX", "LEFT_ANKLE"),
    ("RIGHT_KNEE", "RIGHT_HIP"),
    ("RIGHT_ANKLE", "RIGHT_KNEE"),
    ("RIGHT_HEEL", "RIGHT_ANKLE"),
    ("RIGHT_FOOT_INDEX", "RIGHT_ANKLE"),
]

# 머리 쪽 landmark(코/귀)의 부모는 실제 landmark가 아니라 양쪽 어깨의 중점("목").
NECK = "NECK"
HEAD_HIERARCHY = [
    ("NOSE", NECK),
    ("LEFT_EAR", NECK),
    ("RIGHT_EAR", NECK),
]

# 표준 인체비율표(Winter/de Leva 계열) 기준, 몸통(어깨-엉덩이) 길이 대비 팔다리 비율.
#
# [기본적으로 쓰지 않는다 - --force-limb-ratios 로만 켜진다]
# 이 보정은 "여러 스트로크를 평균하던" 시절의 유산입니다. 그때는 평균 때문에 팔이
# 실제의 1/4로 줄어들어서(위팔이 몸통의 0.09배) 억지로 늘려줄 필요가 있었습니다.
# 지금은 대표 사이클 1개의 실제 좌표를 쓰기 때문에 실측 뼈 길이가 이미 안정적입니다
# (측정 결과 프레임간 변동 1.6~9.3%). 그런데도 이 비율을 강제하면 실측값을 1.2~1.7배
# 늘리게 되고(위팔은 1.5~1.67배), 길이를 늘린 만큼 방향 추정 오차도 함께 증폭돼서
# 손발 끝이 실제보다 크게 튀어 동작이 부자연스러워집니다. 그래서 기본은 실측 길이를
# 그대로 쓰고, 이 표는 필요할 때만 켜는 선택지로 남겨둡니다.
BONE_LENGTH_RATIO_TO_TORSO = {
    "LEFT_ELBOW": 0.65, "RIGHT_ELBOW": 0.65,   # 위팔(어깨-팔꿈치) ≈ 몸통의 65%
    "LEFT_WRIST": 0.50, "RIGHT_WRIST": 0.50,   # 아래팔(팔꿈치-손목) ≈ 몸통의 50%
    "LEFT_KNEE": 0.85, "RIGHT_KNEE": 0.85,     # 허벅지(엉덩이-무릎) ≈ 몸통의 85%
    "LEFT_ANKLE": 0.85, "RIGHT_ANKLE": 0.85,   # 정강이(무릎-발목) ≈ 몸통의 85%
}

SMOOTH_WINDOW = 5  # 프레임 단위 검출 노이즈만 살짝 줄이는 이동평균 창 크기


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--landmarks", required=True, help="pose_landmarks.csv 경로")
    p.add_argument("--reps", required=True, help="reps.csv 경로 (3단계 결과)")
    p.add_argument("--output", required=True, help="정규화된 3D 좌표 저장 경로")
    p.add_argument("--points", type=int, default=101, help="진행률을 몇 개 지점으로 나눌지 (기본 101 = 0~100% 1%간격)")
    p.add_argument("--force-limb-ratios", action="store_true",
                   help="팔다리 길이를 표준 인체비율(BONE_LENGTH_RATIO_TO_TORSO)로 강제한다. "
                        "기본은 실측 길이를 그대로 쓴다 - 자세한 이유는 그 표의 주석 참고")
    return p.parse_args()


def load_landmarks(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    times = np.array([float(r["time_sec"]) for r in rows])
    wanted = COLUMNS + SCREEN_COLUMNS
    data = {col: np.array([float(r[col]) if r[col] else np.nan for r in rows]) for col in wanted}
    return times, data


def load_reps(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_representative_rep(reps):
    """구간(스트로크) 중 길이가 중앙값에 가장 가까운 것 하나를 대표로 고른다
    (app.py의 같은 이름 함수와 동일한 개념)."""
    durations = [float(r["end_time_sec"]) - float(r["start_time_sec"]) for r in reps]
    median_dur = float(np.median(durations))
    idx = int(np.argmin([abs(d - median_dur) for d in durations]))
    return reps[idx]


def smooth(values, window=SMOOTH_WINDOW):
    """단순 이동평균으로 프레임 단위 검출 노이즈만 살짝 줄인다 (실제 동작 궤적은 유지)."""
    half = window // 2
    out = np.full_like(values, np.nan)
    for i in range(len(values)):
        chunk = values[max(0, i - half): i + half + 1]
        valid = chunk[~np.isnan(chunk)]
        if len(valid):
            out[i] = valid.mean()
    return out


def resample_rep(times, values, start_t, end_t, target_pct):
    """한 스트로크 구간 [start_t, end_t]의 값을 진행률(target_pct, 0~100) 지점들로 리샘플링."""
    mask = (times >= start_t) & (times <= end_t) & ~np.isnan(values)
    if mask.sum() < 2:
        return np.full(len(target_pct), np.nan)

    valid_t = times[mask]
    valid_v = values[mask]
    progress = (valid_t - start_t) / (end_t - start_t) * 100

    return np.interp(target_pct, progress, valid_v)


def build_positions(data, force_limb_ratios=False):
    """대표 스트로크가 포함된 원본 프레임 전체에 대해, 뼈대를 부모->자식 순으로 이어
    붙인 전신 3D 좌표를 프레임별로 계산한다.

    기본값은 실측 뼈 길이를 그대로 쓴다. force_limb_ratios=True면 팔다리 길이만
    표준 인체비율로 바꾼다 (방향은 어느 쪽이든 실측값이라 관절 각도는 그대로 유지됨 -
    길이는 각도에 영향을 주지 않기 때문). 왜 기본이 꺼져 있는지는
    BONE_LENGTH_RATIO_TO_TORSO 주석 참고."""
    def raw(name):
        return np.stack([data[f"{name}_wx"], data[f"{name}_wy"], data[f"{name}_wz"]], axis=1)

    resolved = {ROOT: raw(ROOT)}
    torso_len_by_side = {}

    for child, parent in BONE_HIERARCHY:
        bone_vec = raw(child) - resolved.get(parent, raw(parent))
        length = np.linalg.norm(bone_vec, axis=1, keepdims=True)
        length_safe = np.where(length < 1e-9, 1.0, length)
        unit = bone_vec / length_safe

        side = "LEFT" if child.startswith("LEFT") else "RIGHT"
        if child in ("LEFT_SHOULDER", "RIGHT_SHOULDER"):
            torso_len_by_side[side] = length[:, 0]
            final_len = length[:, 0]
        elif force_limb_ratios and child in BONE_LENGTH_RATIO_TO_TORSO:
            final_len = torso_len_by_side[side] * BONE_LENGTH_RATIO_TO_TORSO[child]
        else:
            final_len = length[:, 0]

        resolved[child] = resolved[parent] + unit * final_len[:, None]

    neck = (resolved["LEFT_SHOULDER"] + resolved["RIGHT_SHOULDER"]) / 2
    resolved[NECK] = neck
    for child, parent in HEAD_HIERARCHY:
        resolved[child] = resolved[parent] + (raw(child) - neck)  # 머리는 길이 보정 없이 실측 방향+거리 그대로

    return resolved


def main() -> None:
    args = parse_args()
    times, data = load_landmarks(args.landmarks)
    reps = load_reps(args.reps)
    target_pct = np.linspace(0, 100, args.points)
    rep = pick_representative_rep(reps)
    start_t, end_t = float(rep["start_time_sec"]), float(rep["end_time_sec"])

    resolved = build_positions(data, args.force_limb_ratios)

    mean_curves = {}
    for name in LANDMARK_NAMES:
        if name in resolved:
            pos = resolved[name]
            cols = (smooth(pos[:, 0]), smooth(pos[:, 1]), smooth(pos[:, 2]))
        else:
            cols = (data[f"{name}_wx"], data[f"{name}_wy"], data[f"{name}_wz"])
        for axis, values in zip(AXES, cols):
            mean_curves[f"{name}_{axis}"] = resample_rep(times, values, start_t, end_t, target_pct)

        # 화면 좌표는 뼈대 재구성 없이 그대로 리샘플링한다 - 카메라가 실제로 본 그림이라
        # 엉덩이 출렁임과 몸통 굴곡(= 돌핀킥의 파동)이 이미 들어 있기 때문.
        for axis in SCREEN_AXES:
            col = f"{name}_{axis}"
            mean_curves[col] = resample_rep(times, smooth(data[col]), start_t, end_t, target_pct)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_columns = COLUMNS + SCREEN_COLUMNS
    header = ["progress_pct"] + out_columns
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, pct in enumerate(target_pct):
            row = [round(pct, 1)]
            for col in out_columns:
                v = mean_curves[col][i]
                row.append(round(v, 5) if not np.isnan(v) else "")
            writer.writerow(row)

    duration = end_t - start_t
    print(f"대표 스트로크 #{rep['rep_index']} ({duration:.2f}초, 전체 {len(reps)}개 중 길이가 중앙값에 가장 가까움)를")
    print(f"진행률 {args.points}개 지점으로 리샘플링 (여러 스트로크 평균 아님 - 모듈 설명 참고)")
    print(f"저장 위치: {output_path}")


if __name__ == "__main__":
    main()
