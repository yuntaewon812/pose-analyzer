"""
정규화된 대표 스트로크 곡선(정답 vs 나)을 진행률(%) 기준으로 겹쳐 그리기

4단계에서 만든 normalized_mean.csv는 두 영상 모두 "진행률 0~100%"라는 같은
축을 쓰기 때문에, 이번엔 x축을 그대로 겹쳐도 의미가 있습니다 (raw 시간 그래프와
다른 점). 5단계(각도 차이 계산)로 넘어가기 전 육안 검증용입니다.

사용법:
    python src/visualize_normalized.py
결과물:
    output/comparison/normalized_overlay.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
REF_CSV = ROOT / "output" / "reference" / "normalized_mean.csv"
MINE_CSV = ROOT / "output" / "mine" / "normalized_mean.csv"
OUT_PATH = ROOT / "output" / "comparison" / "normalized_overlay.png"

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
    data = {"progress_pct": [float(r["progress_pct"]) for r in rows]}
    for _, left, right in PANELS:
        for col in (left, right):
            data[col] = [float(r[col]) if r[col] else float("nan") for r in rows]
    return data


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ref = load(REF_CSV)
    mine = load(MINE_CSV)

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
    fig.patch.set_facecolor("#fcfcfb")

    for ax, (title, left, right) in zip(axes, PANELS):
        ax.set_facecolor("#fcfcfb")
        ax.plot(ref["progress_pct"], ref[left], color=COLOR_REF, linewidth=1.8,
                 linestyle="-", label="정답 - 왼쪽")
        ax.plot(ref["progress_pct"], ref[right], color=COLOR_REF, linewidth=1.8,
                 linestyle="--", label="정답 - 오른쪽")
        ax.plot(mine["progress_pct"], mine[left], color=COLOR_MINE, linewidth=1.8,
                 linestyle="-", label="나 - 왼쪽")
        ax.plot(mine["progress_pct"], mine[right], color=COLOR_MINE, linewidth=1.8,
                 linestyle="--", label="나 - 오른쪽")

        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel("각도 (°)")
        ax.set_ylim(0, 190)
        ax.grid(True, color="#e8e8e4", linewidth=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(loc="lower right", frameon=False, fontsize=8, ncol=2)

    axes[-1].set_xlabel("스트로크 진행률 (%)")
    fig.suptitle("대표 스트로크 곡선 비교: 정답 영상 vs 내 영상 (진행률 정규화 후)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_PATH, dpi=120)
    plt.close(fig)

    print(f"그래프 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
