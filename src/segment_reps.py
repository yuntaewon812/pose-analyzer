"""
3단계: 관절 각도 곡선을 "스트로크 1회" 단위 구간으로 나누기

[왜 필요한가]
지금까지 만든 joint_angles.csv는 영상 전체를 쭉 이어놓은 표라서, "정답 영상의
3번째 스트로크"와 "내 영상의 3번째 스트로크"를 짚어낼 방법이 없습니다. 이 단계는
그 경계(어느 프레임에서 스트로크가 시작해서 어느 프레임에서 끝나는지)를 찾아
output/*/reps.csv 에 저장합니다. 다음 단계(시간축 정규화)는 이 경계를 기준으로
각 스트로크를 0~100%로 다시 늘리고 줄여서, 두 영상의 스트로크끼리 비교합니다.

[기준 신호를 어떻게 고르나]
반복의 리듬이 가장 잘 드러나는 관절 하나를 골라, 그 각도가 오르내리는 것을 세어
구간을 나눕니다. 돌핀킥이라면 무릎(--joint left_knee)이 그 신호입니다 - 다리가
접혔다 펴지는 주기가 곧 킥 1회이기 때문입니다.

--joint를 주지 않으면 어깨/팔꿈치 중에서 자동으로 고르는데, 이건 팔이 원을 그리는
자유형 스트로크를 나누려던 초기 설계가 남은 것입니다. 돌핀킥 분석에서는 무릎을
직접 지정하세요 (run_pipeline.py의 DOLPHIN_KICK_ARGS가 그렇게 넘깁니다).

[구간을 나누는 방법 - 문턱값 교차(threshold crossing)]
프로젝트1의 반복 횟수 세기와 같은 원리입니다: 신호의 최댓값과 최솟값의
중간선(threshold)을 정해두고, "중간선을 아래로 뚫고 내려갔다가 다시 위로
뚫고 올라오는" 한 번을 스트로크 1회로 봅니다. 이번엔 횟수만 세는 게 아니라
그 시작 프레임과 끝 프레임을 기록합니다.

[품질 게이트 - 믿을 수 없는 구간은 아예 버린다 (--landmarks)]
수영 영상은 물거품, 물에 잠긴 팔다리, 풀사이드에 서 있는 사람들 때문에 관절
검출이 자주 실패하거나 엉뚱한 사람에게 옮겨갑니다. 이런 구간이 섞이면 이후
단계(정규화/비교/피드백/스틱맨 영상)가 전부 오염되는데, 숫자만 보면 그럴듯해
보여서 알아채기 어렵습니다. 그래서 --landmarks로 pose_landmarks.csv를 주면
각 구간을 아래 3가지로 검사해서 통과한 것만 reps.csv에 남깁니다:

  1) 추적 튐 - 몸 중심(엉덩이)이 프레임 사이에 화면폭의 몇 %를 순간이동했는지.
     사람이 한 프레임에 그렇게 많이 움직일 수는 없으므로, 큰 점프는 "검출이
     다른 사람에게 옮겨갔다"는 신호입니다.
  2) 팔다리 가시성 - MediaPipe의 visibility가 낮으면 "실제로 보고 검출한 게
     아니라 모델이 추측해서 채운 값"이라는 뜻입니다. 팔꿈치/손목/무릎/발목의
     visibility가 낮은 구간은 자세를 논할 근거가 없습니다.
  3) 표본 밀도 - 검출 실패로 구멍이 뚫린 구간은 팔 회전처럼 빠른 동작을
     복원할 표본 자체가 부족합니다 (초당 최소 몇 개는 있어야 함).

사용법:
    python src/segment_reps.py --input output/reference/joint_angles.csv --landmarks output/reference/pose_landmarks.csv --output output/reference/reps.csv
    python src/segment_reps.py --input output/mine/joint_angles.csv      --landmarks output/mine/pose_landmarks.csv      --output output/mine/reps.csv
"""
import argparse
import csv
import math
from pathlib import Path

SMOOTH_WINDOW = 5
CANDIDATE_JOINTS = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"]

# 품질 게이트에서 가시성을 확인할 팔다리 landmark (몸통/머리는 거의 항상 잘 잡히므로 제외)
LIMB_LANDMARKS = [
    "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="joint_angles.csv 경로")
    p.add_argument("--output", required=True, help="reps.csv 저장 경로")
    p.add_argument("--joint", default=None,
                   help="구간 나누기 기준 관절 (기본: 어깨/팔꿈치 중 ROM이 가장 큰 것 자동 선택)")
    p.add_argument("--min-frames", type=int, default=5,
                   help="이보다 짧은 구간은 노이즈로 보고 버림 (기본 5프레임)")
    p.add_argument("--outlier-range", type=float, nargs=2, default=[0.4, 2.5],
                   help="중앙값 대비 이 배율 범위를 벗어나는 구간은 이상치로 제외 (기본 0.4~2.5배)")
    p.add_argument("--landmarks", default=None,
                   help="pose_landmarks.csv 경로. 주면 품질 게이트(추적 끊김/튐/가시성/표본밀도)를 적용한다")
    p.add_argument("--max-jump", type=float, default=0.12,
                   help="몸 중심이 프레임 사이 이만큼(화면폭 비율) 넘게 튀면 추적이 다른 사람으로 옮겨간 것으로 보고 구간을 끊음 (기본 0.12)")
    p.add_argument("--max-frame-gap", type=int, default=3,
                   help="연속한 두 행의 frame 번호 차이가 이보다 크면 검출 실패 공백으로 보고 구간을 끊음 (기본 3)")
    p.add_argument("--min-span-sec", type=float, default=1.5,
                   help="연속 구간이 이보다 짧으면 스트로크를 담기 어려워 버림 (기본 1.5초)")
    p.add_argument("--min-limb-vis", type=float, default=0.6,
                   help="구간 내 팔다리 visibility 중앙값이 이보다 낮으면 제외 (기본 0.6)")
    p.add_argument("--min-samples-per-sec", type=float, default=12.0,
                   help="구간의 초당 표본 수가 이보다 적으면 빠른 동작을 복원할 수 없어 제외 (기본 12)")
    p.add_argument("--max-roll", type=float, default=None,
                   help="몸통 롤(두 어깨의 화면 세로간격/몸통길이)이 이보다 크면 구간을 끊는다. "
                        "턴 직후 '몸이 아직 옆으로 틀어진' 국면을 제외하고 펴진 뒤의 킥만 쓰려면 "
                        "0.25 정도를 준다 (기본: 끄기)")
    p.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW,
                   help=f"기준 신호 이동평균 창 크기 (기본 {SMOOTH_WINDOW}). 신호에 고주파 잡음이 "
                        "많아 문턱값을 여러 번 넘나들며 가짜 구간이 생기면 키운다")
    p.add_argument("--start", type=float, default=None,
                   help="분석할 시간 구간의 시작(초). 영상에 여러 국면(다이빙 직후 돌핀킥 / "
                        "자유형 / 턴 / 글라이드)이 섞여 있을 때 원하는 국면만 지정한다")
    p.add_argument("--end", type=float, default=None, help="분석할 시간 구간의 끝(초)")
    p.add_argument("--streamline-min-shoulder", type=float, default=None,
                   help="지정하면 '양팔을 앞으로 뻗은 스트림라인' 구간만 남긴다 "
                        "(어깨각이 이 값 이상인 부분). 수중 돌핀킥 분석용. 예: 140")
    return p.parse_args()


def restrict_to_streamline(spans, data, times, min_shoulder, min_span_sec):
    """구간에서 "양팔을 앞으로 뻗은 채 유지하는(스트림라인)" 부분만 남긴다.

    [왜 필요한가]
    수중 돌핀킥을 분석하려면 "팔은 뻗어 고정하고 다리만 차는" 국면만 봐야 합니다.
    같은 영상 안에 수면 자유형(팔을 휘두르는) 구간이 섞여 있으면, 발차기 사이클을
    나누는 기준 신호(무릎)에 전혀 다른 리듬이 끼어들어 구간이 엉킵니다.
    어깨 각도(팔꿈치-어깨-엉덩이)가 크면 팔이 몸에서 멀리, 즉 앞/위로 뻗어 있다는
    뜻이므로, 그 값이 계속 높게 유지되는 부분만 골라냅니다.

    [양쪽 팔을 모두 봐야 하는 이유]
    처음엔 두 어깨 중 "큰 값"만 봤는데, 그러면 자유형 구간이 그대로 통과합니다.
    자유형은 한쪽 팔이 앞으로 뻗어 있는 동안 반대쪽 팔이 젓기 때문에, 큰 값 하나는
    거의 항상 높게 유지되기 때문입니다(실제로 자유형 구간이 돌핀킥으로 잘못 선택돼
    발끝 진폭이 1/3인 엉뚱한 구간이 분석됐습니다). 스트림라인은 "양팔을 함께 앞으로
    모아 뻗은" 자세이므로, 두 어깨가 모두 기준을 넘을 때만 인정합니다.
    """
    kept = []
    for s, e in spans:
        run_start = None
        for i in range(s, e + 1):
            shoulders = [data["left_shoulder"][i], data["right_shoulder"][i]]
            valid = [v for v in shoulders if not math.isnan(v)]
            # 양쪽 다 보이면 둘 다, 한쪽만 보이면 그 한쪽이 기준을 넘어야 한다.
            streamlined = bool(valid) and min(valid) >= min_shoulder
            if streamlined:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None and times[i - 1] - times[run_start] >= min_span_sec:
                    kept.append((run_start, i - 1))
                run_start = None
        if run_start is not None and times[e] - times[run_start] >= min_span_sec:
            kept.append((run_start, e))
    return kept


def load_quality_signals(path):
    """pose_landmarks.csv에서 품질 판정에 필요한 신호만 뽑는다: 몸 중심(화면 좌표),
    팔다리 visibility 중앙값, 몸통 롤 지표. calculate_angles.py가 입력 행마다 정확히
    한 행씩 출력하므로, 이 배열들은 joint_angles.csv의 행과 1:1로 같은 순서로 대응한다.

    [롤 지표]
    정측면 카메라에서 엎드린(prone) 자세면 두 어깨가 화면에서 거의 겹쳐 세로 분리가
    0에 가깝고, 옆으로 돌아갈수록 세로로 벌어집니다. 그래서 "두 어깨의 화면 세로
    간격 / 몸통 길이"를 롤 지표로 씁니다.
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    centers, limb_vis, roll = [], [], []
    for r in rows:
        lhx, lhy = float(r["LEFT_HIP_x"]), float(r["LEFT_HIP_y"])
        rhx, rhy = float(r["RIGHT_HIP_x"]), float(r["RIGHT_HIP_y"])
        lsx, lsy = float(r["LEFT_SHOULDER_x"]), float(r["LEFT_SHOULDER_y"])
        rsx, rsy = float(r["RIGHT_SHOULDER_x"]), float(r["RIGHT_SHOULDER_y"])
        cx, cy = (lhx + rhx) / 2, (lhy + rhy) / 2
        sx, sy = (lsx + rsx) / 2, (lsy + rsy) / 2
        centers.append((cx, cy))
        limb_vis.append(median([float(r[f"{n}_vis"]) for n in LIMB_LANDMARKS]))
        torso = math.hypot(sx - cx, sy - cy)
        roll.append(abs(lsy - rsy) / torso if torso > 1e-9 else 0.0)

    # 롤은 프레임 단위로 크게 튄다(같은 국면에서 raw 0.00~0.70). 원값으로 구간을
    # 끊으면 멀쩡한 구간이 잘게 부서지므로, 다른 신호들과 마찬가지로 평활한 뒤 쓴다.
    return centers, limb_vis, moving_average(roll, SMOOTH_WINDOW)


def find_clean_spans(frames, times, centers, args, roll=None):
    """추적이 끊기지 않은 "연속 구간"들을 먼저 찾는다.

    [왜 이걸 먼저 해야 하나]
    검출이 실패한 구간을 그냥 건너뛰고 문턱값 교차로 스트로크를 나누면, 공백을
    가로질러 18초/24초짜리 "스트로크"가 만들어집니다(실제로 그런 일이 생겼습니다).
    그래서 먼저 타임라인을 "믿을 수 있는 연속 구간"으로 쪼개고, 스트로크 나누기는
    각 구간 안에서만 따로 수행합니다.

    구간이 끊기는(믿을 수 없는) 조건은 세 가지입니다:
      - 프레임 공백: 연속한 두 행의 frame 번호 차이가 크면 그 사이는 검출 실패
      - 추적 튐: 몸 중심이 한 프레임에 사람이 움직일 수 없는 거리를 순간이동
      - 몸통 롤 과다(--max-roll): 턴 직후에는 벽을 차고 나오며 몸이 옆으로 크게
        틀어졌다가 서서히 엎드린 자세로 펴집니다. 이 "회전 중" 국면은 다이빙 직후
        돌핀킥과 몸통 방향 자체가 달라서, 같이 묶어 비교하면 자세 차이가 아니라
        회전 차이를 보게 됩니다. 그래서 롤이 큰 프레임에서 구간을 끊어, 몸이
        펴진 뒤의 킥만 남깁니다. (실측: 벽 차고 나온 직후 0.44 -> 1.5초 뒤 0.10)
    (팔다리 가시성은 프레임마다 들쭉날쭉해서 구간을 쪼개는 기준으로 쓰면 잘게
    부서지므로, 구간 전체의 중앙값으로 나중에 한 번만 판정합니다.)
    """
    spans = []
    start = 0
    for i in range(1, len(frames)):
        (x1, y1), (x2, y2) = centers[i - 1], centers[i]
        broken = (frames[i] - frames[i - 1] > args.max_frame_gap
                  or math.hypot(x2 - x1, y2 - y1) > args.max_jump)
        if roll is not None and args.max_roll is not None and roll[i] > args.max_roll:
            broken = True
        if broken:
            spans.append((start, i - 1))
            start = i
    spans.append((start, len(frames) - 1))

    kept, rejected = [], []
    for s, e in spans:
        duration = times[e] - times[s]
        if duration < args.min_span_sec:
            rejected.append(((s, e), f"너무 짧음 ({duration:.2f}초 < {args.min_span_sec:g})"))
            continue
        per_sec = (e - s + 1) / duration
        if per_sec < args.min_samples_per_sec:
            rejected.append(((s, e), f"표본 부족 (초당 {per_sec:.1f}개)"))
            continue
        kept.append((s, e))
    return kept, rejected


def span_limb_vis_reason(s, e, limb_vis, args):
    """구간 전체의 팔다리 가시성이 너무 낮으면 이유를, 괜찮으면 None을 반환."""
    vis_med = median(limb_vis[s:e + 1])
    if vis_med < args.min_limb_vis:
        return f"팔다리 가시성 낮음 (중앙값 {vis_med:.2f} < {args.min_limb_vis:g})"
    return None


def median(values):
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def drop_duration_outliers(segments, times, lo_ratio, hi_ratio):
    """한 스트로크 길이는 대체로 비슷하다고 가정하고, 중앙값에서 너무 벗어난
    구간(감지 공백으로 여러 스트로크가 합쳐졌거나, 떨림으로 잘못 쪼개진 경우)을 제외한다.
    """
    durations = [times[e] - times[s] for s, e in segments]
    if not durations:
        return segments, []
    med = median(durations)
    kept, dropped = [], []
    for seg, dur in zip(segments, durations):
        if lo_ratio * med <= dur <= hi_ratio * med:
            kept.append(seg)
        else:
            dropped.append((seg, dur))
    return kept, dropped


def moving_average(values, window):
    half = window // 2
    smoothed = []
    for i in range(len(values)):
        chunk = [v for v in values[max(0, i - half): i + half + 1]
                 if not math.isnan(v)]
        smoothed.append(sum(chunk) / len(chunk) if chunk else math.nan)
    return smoothed


def pick_best_joint(data, spans, min_coverage=0.7):
    """CANDIDATE_JOINTS 중 "구멍이 적고(coverage) 가동범위(ROM)가 큰" 관절을 고른다.

    [왜 ROM만 보면 안 되나]
    측면에서 찍으면 카메라 반대쪽 팔은 몸에 가려져서 대부분 검출되지 않습니다.
    그런데 어쩌다 검출된 몇 프레임만 보면 ROM이 크게 나와서, ROM만 기준으로 뽑으면
    "데이터가 거의 없는 반대쪽 팔"이 기준 관절로 뽑혀버립니다(실제로 right_shoulder가
    97% NaN인데 뽑혔습니다). 구멍이 많은 신호로 문턱값 교차를 하면 스트로크를
    제대로 못 나눕니다. 그래서 먼저 coverage(값이 있는 비율)가 충분한 관절만
    후보로 남기고, 그 중에서 ROM이 가장 큰 것을 고릅니다.

    판정은 "믿을 수 있는 연속 구간(spans)" 안의 데이터만으로 합니다 - 어차피
    스트로크를 나누는 것도 그 안에서만 하기 때문입니다.
    """
    idx = [i for s, e in spans for i in range(s, e + 1)]
    if not idx:
        return None

    scored = []
    for name in CANDIDATE_JOINTS:
        vals = [data[name][i] for i in idx]
        valid = [v for v in vals if not math.isnan(v)]
        coverage = len(valid) / len(vals)
        if len(valid) < 10:
            continue
        rom = max(valid) - min(valid)
        scored.append((name, coverage, rom))

    if not scored:
        return None
    eligible = [s for s in scored if s[1] >= min_coverage]
    pool = eligible if eligible else scored  # 아무도 기준을 못 넘으면 그중 최선이라도 쓴다
    best = max(pool, key=lambda s: s[2])
    print("  기준 관절 후보 (구간 내 coverage / ROM):")
    for name, cov, rom in scored:
        mark = " <- 선택" if name == best[0] else ("" if cov >= min_coverage else "  (coverage 부족)")
        print(f"    {name:<16} coverage={cov*100:3.0f}%  ROM={rom:3.0f}도{mark}")
    return best[0]


def global_threshold(values):
    """신호 전체의 최댓값/최솟값 중간선. 구간마다 따로 계산하면 구간별로 기준이
    달라져서 스트로크 길이를 비교할 수 없으므로, 문턱값은 영상 전체에서 한 번만 정한다."""
    valid = [v for v in values if not math.isnan(v)]
    if len(valid) < 10:
        return None
    return (min(valid) + max(valid)) / 2


def find_segments(values, threshold, min_frames, lo=0, hi=None):
    """threshold crossing으로 [lo, hi] 범위 안에서만 구간을 나눈다.

    "아래로 내려감(굽히기 시작)"부터 "다시 위로 올라옴(완료)"까지를 한 구간으로 보되,
    한 구간의 시작은 "그 전 구간이 끝난 지점"으로 잡아서 범위가 빈틈없이 구간으로
    나뉘도록 한다 (첫 구간 이전 꼬리부분은 버림). 범위를 넘어가며 이어붙이지 않는
    것이 핵심 - 검출 공백을 가로지르는 가짜 스트로크를 막는다.
    """
    hi = len(values) - 1 if hi is None else hi

    segments = []
    bent = False
    seg_start_idx = None
    for i in range(lo, hi + 1):
        v = values[i]
        if math.isnan(v):
            continue
        if not bent and v < threshold:
            bent = True
            if seg_start_idx is None:
                seg_start_idx = i  # 이 범위 첫 구간의 시작점
        elif bent and v > threshold:
            bent = False
            end_idx = i
            if seg_start_idx is not None and end_idx - seg_start_idx >= min_frames:
                segments.append((seg_start_idx, end_idx))
            seg_start_idx = end_idx  # 다음 구간은 이 지점부터 이어서 시작

    return segments


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input)
    output_csv = Path(args.output)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(input_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    frames = [int(r["frame"]) for r in rows]
    times = [float(r["time_sec"]) for r in rows]

    # 8개 관절 전부 읽어둔다 - 자동 선택은 CANDIDATE_JOINTS(어깨/팔꿈치)만 보지만,
    # --joint로 무릎/엉덩이를 직접 지정할 수도 있어야 한다 (예: 발차기 기준으로 나누기).
    all_joints = list(dict.fromkeys(CANDIDATE_JOINTS + [
        "left_knee", "right_knee", "left_hip", "right_hip",
    ]))
    data = {}
    for name in all_joints:
        data[name] = [float(r[name]) if r[name] else math.nan for r in rows]

    centers = limb_vis = roll = None
    if args.landmarks:
        centers, limb_vis, roll = load_quality_signals(args.landmarks)
        if len(centers) != len(rows):
            raise ValueError(
                f"--landmarks 행 수({len(centers)})와 --input 행 수({len(rows)})가 다릅니다. "
                "같은 pose_landmarks.csv로 calculate_angles.py를 다시 실행하세요."
            )

    # 0) --start/--end로 분석할 국면을 먼저 잘라낸다. 한 영상 안에 다이빙 직후
    #    돌핀킥 / 자유형 / 턴 / 글라이드가 섞여 있어서, 국면을 안 나누면 엉뚱한
    #    구간이 뽑힐 수 있다 (실제로 자유형 구간이 돌핀킥으로 선택된 적이 있다).
    #    품질 신호도 행과 1:1로 대응하므로 같은 인덱스로 함께 잘라야 한다.
    if args.start is not None or args.end is not None:
        lo = args.start if args.start is not None else -math.inf
        hi = args.end if args.end is not None else math.inf
        keep = [i for i, t in enumerate(times) if lo <= t <= hi]
        if len(keep) < 10:
            raise ValueError(f"--start/--end 구간({lo}~{hi}초)에 데이터가 {len(keep)}행뿐입니다")
        rows = [rows[i] for i in keep]
        frames = [frames[i] for i in keep]
        times = [times[i] for i in keep]
        data = {k: [v[i] for i in keep] for k, v in data.items()}
        if centers is not None:
            centers = [centers[i] for i in keep]
            limb_vis = [limb_vis[i] for i in keep]
            roll = [roll[i] for i in keep]
        print(f"  분석 구간 제한: {lo}~{hi}초 -> {len(rows)}행")

    # 1) 먼저 "믿을 수 있는 연속 구간"을 찾는다. 기준 관절 선택도, 스트로크 나누기도
    #    모두 이 구간 안의 데이터만 보고 해야 한다 (품질 게이트가 없으면 전체를 하나로 본다).
    span_rejected, vis_rejected = [], []
    if args.landmarks:
        spans, span_rejected = find_clean_spans(frames, times, centers, args, roll)
        good_spans = []
        for s, e in spans:
            reason = span_limb_vis_reason(s, e, limb_vis, args)
            if reason is None:
                good_spans.append((s, e))
            else:
                vis_rejected.append(((s, e), reason))
        spans = good_spans
    else:
        spans = [(0, len(rows) - 1)]

    if args.streamline_min_shoulder is not None:
        before = len(spans)
        spans = restrict_to_streamline(spans, data, times,
                                        args.streamline_min_shoulder, args.min_span_sec)
        print(f"  스트림라인 필터(어깨각>={args.streamline_min_shoulder:g}): "
              f"연속 구간 {before}개 -> 수중 스트림라인 구간 {len(spans)}개")

    if not spans:
        raise ValueError("믿을 수 있는 연속 구간이 없습니다 (검출 실패/추적 튐이 너무 많음). "
                          "--max-jump/--min-limb-vis/--min-span-sec을 완화하거나 영상을 바꿔야 합니다.")

    # 2) 그 구간 안의 데이터로 기준 관절을 고른다 (coverage + ROM).
    joint = args.joint or pick_best_joint(data, spans)
    if joint is None:
        raise ValueError("구간을 나눌 만한 관절 데이터가 부족합니다 (가려짐이 너무 많음)")

    # 3) 문턱값은 구간 안의 값들로 한 번만 정한다 (구간마다 다르면 비교 불가).
    smoothed = moving_average(data[joint], args.smooth_window)
    in_span = [smoothed[i] for s, e in spans for i in range(s, e + 1)]
    threshold = global_threshold(in_span)
    if threshold is None:
        raise ValueError(f"{joint} 데이터가 너무 적어 문턱값을 정할 수 없습니다")

    segments = []
    for s, e in spans:
        segments.extend(find_segments(smoothed, threshold, args.min_frames, s, e))

    lo_ratio, hi_ratio = args.outlier_range
    segments, dropped = drop_duration_outliers(segments, times, lo_ratio, hi_ratio)

    header = ["rep_index", "start_frame", "start_time_sec", "end_frame", "end_time_sec", "duration_sec"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, (s, e) in enumerate(segments):
            writer.writerow([i, frames[s], times[s], frames[e], times[e],
                              round(times[e] - times[s], 2)])

    durations = [times[e] - times[s] for s, e in segments]
    print(f"기준 관절: {joint} (문턱값 {threshold:.0f}도)")
    if args.landmarks:
        total_clean = sum(times[e] - times[s] for s, e in spans)
        print(f"믿을 수 있는 연속 구간: {len(spans)}개, 합계 {total_clean:.1f}초 "
              f"(끊김/짧음 제외 {len(span_rejected)}개, 가시성 제외 {len(vis_rejected)}개)")
        for (s, e) in spans:
            print(f"    사용: {times[s]:.2f}s ~ {times[e]:.2f}s ({times[e]-times[s]:.2f}초)")
        for (s, e), reason in vis_rejected:
            print(f"    제외: {times[s]:.2f}s ~ {times[e]:.2f}s -> {reason}")
    else:
        print("  [주의] --landmarks 없이 실행돼 품질 게이트를 적용하지 않았습니다 "
              "(검출 실패/추적 튐 구간이 섞여 있을 수 있음)")
    print(f"최종 스트로크 구간 수: {len(segments)} (길이 이상치 제외 {len(dropped)}개)")
    if dropped:
        print("  제외된 구간(길이가 중앙값과 너무 달라 노이즈로 판단):")
        for (s, e), dur in dropped:
            print(f"    {times[s]:.2f}s ~ {times[e]:.2f}s ({dur:.2f}초)")
    if durations:
        print(f"구간 길이 평균: {sum(durations)/len(durations):.2f}초 "
              f"(최소 {min(durations):.2f}초 / 최대 {max(durations):.2f}초)")
    print(f"저장 위치: {output_csv}")


if __name__ == "__main__":
    main()
