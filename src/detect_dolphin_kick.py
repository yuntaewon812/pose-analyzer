"""
영상에서 "수중 돌핀킥" 구간이 언제 시작해서 언제 끝나는지 자동으로 찾기

[왜 필요한가]
수영 영상은 대부분 영법이 섞여 있습니다 (다이빙 -> 수중 돌핀킥 -> 자유형 -> 턴 ->
다시 돌핀킥 -> 자유형 ...). 분석 구간을 손으로 지정하면 영상이 바뀔 때마다 다시
찾아야 하고, 실수로 자유형 구간을 돌핀킥으로 잘못 잡으면 엉뚱한 비교가 됩니다
(실제로 그런 일이 있었습니다). 이 스크립트는 그 구간을 자동으로 찾아줍니다.

[무엇을 보고 돌핀킥이라고 판단하나 - 측면 촬영에서도 믿을 수 있는 신호만]
처음엔 "돌핀은 두 다리가 함께, 플러터(자유형)는 번갈아 움직인다"로 구분하려 했는데,
측면에서 찍으면 두 발목이 화면에서 거의 겹쳐(몸통 길이의 0.07~0.14배 간격) 좌우
구분이 안 됩니다. 그래서 겹침에 영향을 받지 않는 네 가지를 씁니다:

  1) 팔 움직임 (arm_move) - 손목이 어깨 기준으로 얼마나 움직이는지.
     돌핀킥은 팔을 앞으로 모아 고정하고, 자유형은 팔을 크게 휘두릅니다.
     실측: 돌핀킥 0.05 / 자유형 0.31 로 6배 차이 - 가장 확실한 구분자.
  2) 어깨 비대칭 (shoulder_asym) - 좌우 어깨 각도 차이.
     스트림라인은 양팔이 같은 자세라 차이가 작고(2~14도), 자유형은 한쪽만 뻗어
     크게 벌어집니다(81도).
  3) 팔 뻗음 (arm_ext) - 양쪽 어깨 각도의 작은 쪽. 팔을 앞으로 모아 뻗었는지.
  4) 발 진폭 + 주기성 - 실제로 "차고 있는지". 이게 없으면 그냥 글라이드입니다
     (글라이드 발끝 진폭 0.86 vs 돌핀킥 1.71~2.04).

여기에 데이터 품질(무릎이 실제로 검출됐는지)도 함께 봅니다. 물거품에 다리가
가려지면 이후 분석을 할 수 없기 때문입니다.

[자유형 영상으로 검증하다 발견한 것 - 위 네 가지로는 부족했다]
자유형 영상 2개로 시험했더니 7곳이 돌핀킥으로 잘못 잡혔습니다. 원인은 자유형의
"글라이드 순간"입니다. 팔을 뻗고 미끄러지는 그 1초 동안은 팔 움직임(0.006~0.126)도
어깨 비대칭(0도)도 돌핀킥과 똑같이 보이고, 플러터킥의 발 진폭(1.04~1.67)도 진짜
돌핀킥(1.72)과 겹칩니다. 그래서 두 가지를 더 봅니다:

  5) 엉덩이 상하 진폭 (hip_amp) - 돌핀킥은 몸 전체가 물결쳐 엉덩이가 오르내리지만,
     플러터킥은 엉덩이를 고정한 채 다리만 젓습니다.
  6) 구간 길이 (MIN_SEGMENT_SEC) - 진짜 돌핀킥은 여러 번 연속으로 차서 2초 이상
     이어지지만, 자유형의 글라이드는 1~1.75초로 짧게 끊깁니다.

주의: 이 두 기준값은 영상 4개(돌핀킥 2 + 자유형 2)로 정한 것이라 표본이 적습니다.
새 영상에서 놓치거나 잘못 잡으면 상단 상수를 조정하고, 반드시 기존 4개 영상으로
다시 확인하세요(하나를 맞추려다 다른 것이 깨지기 쉽습니다).

사용법:
    python src/detect_dolphin_kick.py --landmarks output/mine/pose_landmarks.csv --angles output/mine/joint_angles.csv
    python src/detect_dolphin_kick.py --landmarks ... --angles ... --output output/mine/dolphin_segments.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

# 판정 기준 - 위 docstring의 실측값 사이에서 여유를 두고 잡았다.
MAX_ARM_MOVE = 0.15       # 이보다 크면 팔을 휘두르는 중 = 자유형
MAX_SHOULDER_ASYM = 35.0  # 이보다 크면 한쪽 팔만 뻗은 상태 = 자유형
MIN_ARM_EXT = 135.0       # 양팔이 이만큼은 뻗어 있어야 스트림라인
MIN_FOOT_AMP = 1.0        # 이보다 작으면 차지 않고 흘러가는 중 = 글라이드
MIN_PERIODICITY = 0.15    # 발 신호가 주기적이어야 "킥"
MIN_HIP_AMP = 0.30        # 엉덩이가 이만큼 오르내려야 돌핀킥 (플러터킥과 가르는 기준)
MIN_KNEE_COVERAGE = 0.6   # 무릎이 이만큼은 검출돼야 이후 분석 가능

WINDOW_SEC = 1.0          # 판정 창 크기 (킥 1~2회가 들어갈 정도)
HOP_SEC = 0.25            # 창을 옮기는 간격
MIN_SEGMENT_SEC = 2.0     # 이보다 짧은 구간은 버림. 진짜 돌핀킥은 여러 번 연속으로
                          # 차므로 2초 이상 이어지지만, 자유형 글라이드가 잠깐 돌핀킥처럼
                          # 보이는 구간은 1~1.75초로 짧게 끊긴다.
BRIDGE_SEC = 0.5          # 이만큼 이내로 끊긴 구간은 하나로 이어붙임


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--landmarks", required=True, help="pose_landmarks.csv 경로")
    p.add_argument("--angles", required=True, help="joint_angles.csv 경로")
    p.add_argument("--output", default=None, help="구간 저장 경로 (기본: landmarks 옆 dolphin_segments.csv)")
    p.add_argument("--window", type=float, default=WINDOW_SEC)
    p.add_argument("--min-foot-amp", type=float, default=MIN_FOOT_AMP)
    return p.parse_args()


def periodicity(sig):
    """신호가 얼마나 주기적인지 (자기상관 최대값, 0~1). 실제로 '차고 있는지' 판정용."""
    s = np.asarray(sig, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 15 or s.std() < 1e-9:
        return 0.0
    s = s - s.mean()
    ac = np.correlate(s, s, mode="full")[len(s) - 1:]
    ac /= ac[0]
    lo = 5   # 바로 옆 프레임과 비슷한 것을 주기로 오인하지 않도록
    return float(ac[lo:].max()) if len(ac) > lo else 0.0


def window_features(lm_win, ang_win):
    """창 하나의 특징을 계산한다. 데이터가 모자라면 None."""
    if len(lm_win) < 12:
        return None

    hx = (lm_win["LEFT_HIP_x"] + lm_win["RIGHT_HIP_x"]).values / 2
    hy = (lm_win["LEFT_HIP_y"] + lm_win["RIGHT_HIP_y"]).values / 2
    sx = (lm_win["LEFT_SHOULDER_x"] + lm_win["RIGHT_SHOULDER_x"]).values / 2
    sy = (lm_win["LEFT_SHOULDER_y"] + lm_win["RIGHT_SHOULDER_y"]).values / 2
    torso = float(np.nanmean(np.hypot(sx - hx, sy - hy)))
    if not np.isfinite(torso) or torso < 1e-6:
        return None

    # 손목이 어깨에서 얼마나 떨어져 있는지의 변동 = 팔을 휘두르는 정도
    wrist_dist = np.hypot(lm_win["LEFT_WRIST_x"].values - sx,
                          lm_win["LEFT_WRIST_y"].values - sy) / torso
    arm_move = float(np.nanstd(wrist_dist))

    ls, rs = ang_win["left_shoulder"].values, ang_win["right_shoulder"].values
    stacked = np.vstack([ls, rs])
    # 두 어깨가 모두 안 잡힌 프레임은 nanmin이 경고를 내므로 미리 걸러낸다.
    usable = np.isfinite(stacked).any(axis=0)
    both = np.nanmin(stacked[:, usable], axis=0) if usable.any() else np.array([])
    arm_ext = float(np.median(both)) if both.size else np.nan
    diff = np.abs(ls - rs)
    # 한쪽 어깨가 안 잡히면 비대칭을 알 수 없다 -> 0으로 두고 다른 지표에 맡긴다
    shoulder_asym = float(np.nanmedian(diff)) if np.isfinite(diff).any() else 0.0

    fy = (lm_win["LEFT_FOOT_INDEX_y"] + lm_win["RIGHT_FOOT_INDEX_y"]).values / 2
    foot_amp = float((np.nanmax(fy) - np.nanmin(fy)) / torso) if np.isfinite(fy).any() else 0.0
    period = periodicity(-fy)

    # 엉덩이가 위아래로 얼마나 오르내리는지. 돌핀킥은 몸 전체가 물결치므로 엉덩이가
    # 크게 움직이지만, 자유형의 플러터킥은 엉덩이를 고정한 채 다리만 젓는다.
    # 발 진폭만으로는 두 킥이 구분되지 않아서(자유형 영상에서 1.04~1.67로 진짜
    # 돌핀킥 1.72와 겹쳤다) 이 신호를 함께 본다.
    hip_amp = float((np.nanmax(hy) - np.nanmin(hy)) / torso) if np.isfinite(hy).any() else 0.0

    knee_cov = float(ang_win["left_knee"].notna().mean())

    return dict(arm_move=arm_move, arm_ext=arm_ext, shoulder_asym=shoulder_asym,
                foot_amp=foot_amp, period=period, hip_amp=hip_amp, knee_cov=knee_cov)


def failure_reasons(f, min_foot_amp):
    """돌핀킥 조건 중 어긋난 항목을 사람이 읽을 수 있게 돌려준다 (빈 리스트면 통과).

    새 영상에서 구간을 못 찾았을 때 "왜"를 알아야 손을 쓸 수 있으므로, 통과/불통과만
    돌려주지 않고 사유를 남긴다. (예: 다이빙 직후는 물거품 때문에 무릎 검출이 0%라
    탈락하는데, 이건 기준을 낮춰서 될 일이 아니라 데이터가 없는 것이다.)
    """
    if f is None:
        return ["표본 부족"]
    out = []
    if not np.isfinite(f["arm_ext"]):
        out.append("어깨 미검출")
    elif f["arm_ext"] < MIN_ARM_EXT:
        out.append(f"팔 덜 뻗음 {f['arm_ext']:.0f}도")
    if f["arm_move"] > MAX_ARM_MOVE:
        out.append(f"팔 휘두름 {f['arm_move']:.2f} (자유형 의심)")
    if f["shoulder_asym"] > MAX_SHOULDER_ASYM:
        out.append(f"좌우 어깨 비대칭 {f['shoulder_asym']:.0f}도 (자유형 의심)")
    if f["foot_amp"] < min_foot_amp:
        out.append(f"발 진폭 작음 {f['foot_amp']:.2f} (글라이드 의심)")
    if f["hip_amp"] < MIN_HIP_AMP:
        out.append(f"엉덩이 움직임 작음 {f['hip_amp']:.2f} (플러터킥 의심)")
    if f["period"] < MIN_PERIODICITY:
        out.append(f"주기성 낮음 {f['period']:.2f}")
    if f["knee_cov"] < MIN_KNEE_COVERAGE:
        out.append(f"무릎 검출 {f['knee_cov'] * 100:.0f}% (물거품 등으로 가려짐)")
    return out


def is_dolphin(f, min_foot_amp):
    """특징이 돌핀킥 조건을 모두 만족하는지."""
    return not failure_reasons(f, min_foot_amp)


def merge_windows(hits, window, bridge=BRIDGE_SEC, min_len=MIN_SEGMENT_SEC):
    """통과한 창들을 이어 붙여 구간으로 만든다."""
    if not hits:
        return []
    spans = []
    start = prev = hits[0]
    for t in hits[1:]:
        if t - prev <= bridge:
            prev = t
        else:
            spans.append((start, prev + window))
            start = prev = t
    spans.append((start, prev + window))
    return [(a, b) for a, b in spans if b - a >= min_len]


def detect(landmarks_path, angles_path, window=WINDOW_SEC, min_foot_amp=MIN_FOOT_AMP):
    lm = pd.read_csv(landmarks_path)
    ang = pd.read_csv(angles_path)
    if len(lm) != len(ang):
        raise ValueError(f"landmarks({len(lm)})와 angles({len(ang)}) 행 수가 다릅니다")

    t = lm["time_sec"].values
    hits, detail = [], []
    for w in np.arange(t.min(), t.max() - window, HOP_SEC):
        m = (t >= w) & (t < w + window)
        if m.sum() < 12:
            continue
        f = window_features(lm[m], ang[m])
        if f is None:
            continue
        ok = is_dolphin(f, min_foot_amp)
        detail.append((w, f, ok))
        if ok:
            hits.append(float(w))

    return merge_windows(hits, window), detail


def main() -> None:
    args = parse_args()
    out = Path(args.output) if args.output else Path(args.landmarks).parent / "dolphin_segments.csv"

    segments, detail = detect(args.landmarks, args.angles, args.window, args.min_foot_amp)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["segment_index", "start_sec", "end_sec", "duration_sec"])
        for i, (a, b) in enumerate(segments):
            w.writerow([i, round(a, 2), round(b, 2), round(b - a, 2)])

    print(f"검사한 창: {len(detail)}개 (창 {args.window}초, {HOP_SEC}초 간격)")
    print(f"찾은 돌핀킥 구간: {len(segments)}개")
    for i, (a, b) in enumerate(segments):
        inside = [d for d in detail if a <= d[0] < b and d[2]]
        amp = np.mean([d[1]["foot_amp"] for d in inside]) if inside else float("nan")
        print(f"  #{i}  {a:6.2f} ~ {b:6.2f}초  ({b - a:4.2f}초)  발끝진폭 평균={amp:.2f}")
    # 탈락 사유 집계 - 새 영상에서 구간을 못 찾았을 때 무엇이 문제인지 바로 보이게.
    reasons = {}
    for _w, f, ok in detail:
        if ok:
            continue
        for r in failure_reasons(f, args.min_foot_amp):
            # 수치를 뺀 앞부분만 묶어서 집계한다 ("발 진폭 작음 1.23" -> "발 진폭 작음")
            key = r.split(" (")[0].rsplit(" ", 1)[0] if any(c.isdigit() for c in r) else r
            reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        print("\n제외된 창의 사유 (많은 순):")
        for key, cnt in sorted(reasons.items(), key=lambda kv: -kv[1])[:6]:
            print(f"  {key:<22} {cnt:4d}개 창")

    if not segments:
        print("\n조건을 만족하는 구간이 없습니다. 발 진폭이 컸던 창들과 탈락 사유:")
        near = sorted(detail, key=lambda d: -d[1]["foot_amp"])[:5]
        for w, f, _ in near:
            print(f"  {w:6.2f}초  " + " / ".join(failure_reasons(f, args.min_foot_amp)))
    print(f"\n저장 위치: {out}")


if __name__ == "__main__":
    main()
