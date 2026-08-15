"""
정답 영상 vs 내 영상 관절 각도를 겹쳐서 그래프로 그리는 육안 검증 스크립트

3단계(반복 구간 나누기)로 넘어가기 전에, 지금까지 만든 파이프라인이
제대로 작동하는지 - 두 영상 모두 스트로크 패턴처럼 오르내리는 곡선이
나오는지 - 눈으로 먼저 확인하기 위한 스크립트입니다.

아직 시간축을 정규화(진행률 0~100%)하지 않았기 때문에, 두 영상의
길이가 달라서 x축(시간)이 그대로 겹쳐지지는 않습니다. 그건 4단계에서
다룰 문제이고, 지금은 "패턴이 그럴듯하게 나오는가"만 봅니다.

색상은 영상(정답=파랑, 나=주황)을 구분하고, 선 스타일(실선=왼쪽,
점선=오른쪽)로 좌우를 구분합니다 (색상표는 dataviz 스킬의 검증된
2색 조합 사용).

사용법:
    python src/visualize_comparison.py
결과물:
    output/comparison/angle_overlay.png
"""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
REF_CSV = ROOT / "output" / "reference" / "joint_angles.csv"
MINE_CSV = ROOT / "output" / "mine" / "joint_angles.csv"
OUT_PATH = ROOT / "output" / "comparison" / "angle_overlay.png"

SMOOTH_WINDOW = 5

# dataviz 스킬 검증된 카테고리 팔레트 슬롯 1(파랑)/2(주황)
COLOR_REF = "#2a78d6"
COLOR_MINE = "#eb6834"

PANELS = [
    ("팔꿈치 (Elbow)", "left_elbow", "right_elbow"),
    ("어깨 (Shoulder)", "left_shoulder", "right_shoulder"),
    ("엉덩이 (Hip)", "left_hip", "right_hip"),
    ("무릎 (Knee)", "left_knee", "right_knee"),
]


def load(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    data = {"time_sec": [float(r["time_sec"]) for r in rows]}
    for _, left, right in PANELS:
        for col in (left, right):
            data[col] = [float(r[col]) if r[col] else math.nan for r in rows]
    return data


def moving_average(values, window):
    half = window // 2
    smoothed = []
    for i in range(len(values)):
        chunk = [v for v in values[max(0, i - half): i + half + 1]
                 if not math.isnan(v)]
        smoothed.append(sum(chunk) / len(chunk) if chunk else math.nan)
    return smoothed


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ref = load(REF_CSV)
    mine = load(MINE_CSV)

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=False)
    fig.patch.set_facecolor("#fcfcfb")

    for ax, (title, left, right) in zip(axes, PANELS):
        ax.set_facecolor("#fcfcfb")
        ax.plot(ref["time_sec"], moving_average(ref[left], SMOOTH_WINDOW),
                color=COLOR_REF, linewidth=1.8, linestyle="-", label="정답 - 왼쪽")
        ax.plot(ref["time_sec"], moving_average(ref[right], SMOOTH_WINDOW),
                color=COLOR_REF, linewidth=1.8, linestyle="--", label="정답 - 오른쪽")
        ax.plot(mine["time_sec"], moving_average(mine[left], SMOOTH_WINDOW),
                color=COLOR_MINE, linewidth=1.8, linestyle="-", label="나 - 왼쪽")
        ax.plot(mine["time_sec"], moving_average(mine[right], SMOOTH_WINDOW),
                color=COLOR_MINE, linewidth=1.8, linestyle="--", label="나 - 오른쪽")

        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel("각도 (°)")
        ax.set_ylim(0, 190)
        ax.grid(True, color="#e8e8e4", linewidth=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(loc="lower right", frameon=False, fontsize=8, ncol=2)

    axes[-1].set_xlabel("시간 (초) - 두 영상 길이가 달라 아직 정규화 전")
    fig.suptitle("관절 각도 비교: 정답 영상 vs 내 영상 (시간 정규화 전, 3D 좌표 기반)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_PATH, dpi=120)
    plt.close(fig)

    print(f"그래프 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
