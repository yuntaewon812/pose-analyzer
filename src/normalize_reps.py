"""
4단계: 스트로크 구간을 진행률 0~100%로 다시 늘려서 "대표 곡선" 만들기

[왜 필요한가]
3단계에서 나눈 스트로크는 저마다 길이(초)가 다릅니다 (1.2초짜리도 있고 2.5초짜리도
있음). "정답 영상 3번째 스트로크"와 "내 영상 5번째 스트로크"를 프레임/초 단위로
그대로 겹치면 애초에 시간축이 안 맞습니다. 그래서 스트로크마다 실제 길이는
잊어버리고, "이 스트로크가 몇 % 진행됐는지"라는 공통 잣대로 바꿉니다.
예: 1.2초짜리 스트로크의 0.6초 지점 = 2.5초짜리 스트로크의 1.25초 지점 = 둘 다 "50%"

[핵심 개념 - 리샘플링(resampling)과 보간(interpolation)]
어떤 스트로크는 원본 데이터가 0%, 7%, 15%, 23% ... 이런 식으로 불규칙한 간격의
점들만 가지고 있습니다 (프레임 간격이 시간 기준이라, 스트로크를 %로 바꾸면
점 간격이 스트로크마다 달라짐). 그래서 "0%, 1%, 2%, ... 100%"라는 깔끔한 101개
지점으로 다시 계산해야 하는데, 그 사이 값은 원본 점들을 직선으로 이어서
추정합니다 (선형 보간, linear interpolation). numpy.interp가 이 계산을 해줍니다.

[핵심 개념 - 여러 스트로크를 평균 내는 이유]
스트로크 하나만 보면 그 순간의 우연한 흔들림(노이즈)이 그대로 들어있습니다.
같은 영상 안의 모든 스트로크를 진행률별로 평균 내면(예: 모든 스트로크의 "50%
지점" 각도를 평균), 그 사람의 "평소 자세"에 가까운 대표 곡선이 나옵니다.
5단계(각도 차이 계산)는 이 대표 곡선끼리 비교합니다.

사용법:
    python src/normalize_reps.py --angles output/reference/joint_angles.csv --reps output/reference/reps.csv --output output/reference/normalized_mean.csv
    python src/normalize_reps.py --angles output/mine/joint_angles.csv      --reps output/mine/reps.csv      --output output/mine/normalized_mean.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np

JOINT_COLUMNS = [
    "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
    "left_hip", "right_hip", "left_knee", "right_knee",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--angles", required=True, help="joint_angles.csv 경로")
    p.add_argument("--reps", required=True, help="reps.csv 경로 (3단계 결과)")
    p.add_argument("--output", required=True, help="대표 곡선 저장 경로 (normalized_mean.csv)")
    p.add_argument("--points", type=int, default=101, help="진행률을 몇 개 지점으로 나눌지 (기본 101 = 0~100% 1%간격)")
    return p.parse_args()


def load_angles(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    times = np.array([float(r["time_sec"]) for r in rows])
    data = {}
    for col in JOINT_COLUMNS:
        data[col] = np.array([float(r[col]) if r[col] else np.nan for r in rows])
    return times, data


def load_reps(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resample_rep(times, values, start_t, end_t, target_pct):
    """한 스트로크 구간 [start_t, end_t]의 값을 진행률(target_pct, 0~100) 지점들로 리샘플링.

    가려짐(NaN)으로 빈 값은 보간에서 제외한다. 유효한 점이 2개 미만이면
    이 구간은 보간이 불가능하므로 전부 NaN을 반환한다.
    """
    mask = (times >= start_t) & (times <= end_t) & ~np.isnan(values)
    if mask.sum() < 2:
        return np.full(len(target_pct), np.nan)

    valid_t = times[mask]
    valid_v = values[mask]
    progress = (valid_t - start_t) / (end_t - start_t) * 100  # 0~100 진행률로 변환

    return np.interp(target_pct, progress, valid_v)


def main() -> None:
    args = parse_args()
    times, data = load_angles(args.angles)
    reps = load_reps(args.reps)
    target_pct = np.linspace(0, 100, args.points)

    mean_curves = {}
    rep_counts = {}
    for col in JOINT_COLUMNS:
        per_rep = []
        for rep in reps:
            start_t = float(rep["start_time_sec"])
            end_t = float(rep["end_time_sec"])
            per_rep.append(resample_rep(times, data[col], start_t, end_t, target_pct))
        stacked = np.vstack(per_rep)  # (스트로크 개수, 101) 크기 표
        # 스트로크마다 값이 있는(NaN이 아닌) 지점끼리만 평균
        mean_curves[col] = np.nanmean(stacked, axis=0)
        rep_counts[col] = np.sum(~np.isnan(stacked), axis=0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["progress_pct"] + JOINT_COLUMNS
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, pct in enumerate(target_pct):
            row = [round(pct, 1)]
            for col in JOINT_COLUMNS:
                v = mean_curves[col][i]
                row.append(round(v, 1) if not np.isnan(v) else "")
            writer.writerow(row)

    print(f"스트로크 {len(reps)}개를 진행률 {args.points}개 지점으로 정규화 후 평균")
    print(f"저장 위치: {output_path}")
    print()
    for col in JOINT_COLUMNS:
        avg_count = rep_counts[col].mean()
        print(f"  {col:<16} 평균적으로 {avg_count:.1f}/{len(reps)}개 스트로크가 각 지점에 기여")


if __name__ == "__main__":
    main()
