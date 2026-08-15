"""
7단계: 정답 대비 각도 차이를 한눈에 보는 최종 리포트 그래프

지금까지 그래프는 "두 곡선을 겹쳐서" 보여줬는데, 이번엔 아예 "차이(diff)"
자체를 그립니다. 0을 기준으로 위/아래로 얼마나 벌어지는지 색으로 구분해서
(빨강=내가 덜 굽힘/더 폄, 파랑=내가 더 굽힘) 문제 구간을 바로 짚어낼 수
있게 합니다. 좌우 관절은 평균 내서 관절 종류당 선 하나로 단순화했습니다
(자세한 좌우 비교는 4단계의 normalized_overlay.png 참고).

사용법:
    python src/visualize_diff.py
결과물:
    output/comparison/diff_chart.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
DIFF_CSV = ROOT / "output" / "comparison" / "angle_diff.csv"
OUT_PATH = ROOT / "output" / "comparison" / "diff_chart.png"

# dataviz 스킬의 diverging(양극) 색 지침: blue <-> red, 중립은 회색
COLOR_NEG = "#2a78d6"   # diff < 0 (내가 더 굽힘)
COLOR_POS = "#e34948"   # diff > 0 (내가 덜 굽힘)

PANELS = [
    ("팔꿈치 (Elbow)", "left_elbow", "right_elbow"),
    ("어깨 (Shoulder)", "left_shoulder", "right_shoulder"),
    ("엉덩이 (Hip)", "left_hip", "right_hip"),
    ("무릎 (Knee)", "left_knee", "right_knee"),
]


def load():
    with open(DIFF_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pct = [float(r["progress_pct"]) for r in rows]
    combined = {}
    for _, left, right in PANELS:
        vals = []
        for r in rows:
            l, rt = r[f"diff_{left}"], r[f"diff_{right}"]
            if l == "" or rt == "":
                vals.append(float("nan"))
            else:
                vals.append((float(l) + float(rt)) / 2)
        combined[left[5:] if left.startswith("left_") else left] = vals
    return pct, combined


def main() -> None:
    pct, combined = load()

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    fig.patch.set_facecolor("#fcfcfb")

    for ax, (title, left, _) in zip(axes, PANELS):
        key = left[5:]
        vals = combined[key]
        ax.set_facecolor("#fcfcfb")
        ax.axhline(0, color="#8a8a86", linewidth=1, linestyle="-")
        ax.plot(pct, vals, color="#52514e", linewidth=1.2)
        ax.fill_between(pct, vals, 0, where=[v >= 0 for v in vals],
                         color=COLOR_POS, alpha=0.55, interpolate=True)
        ax.fill_between(pct, vals, 0, where=[v < 0 for v in vals],
                         color=COLOR_NEG, alpha=0.55, interpolate=True)

        # 가장 크게 벌어진 지점 하나만 직접 라벨링 (모든 점에 라벨 달지 않음)
        valid = [(p, v) for p, v in zip(pct, vals) if v == v]  # NaN 제외
        if valid:
            worst_p, worst_v = max(valid, key=lambda pv: abs(pv[1]))
            ax.annotate(f"{worst_v:+.0f}°", xy=(worst_p, worst_v),
                        xytext=(0, 12 if worst_v >= 0 else -16), textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold", color="#0b0b0b")
            ax.plot(worst_p, worst_v, "o", color="#0b0b0b", markersize=4)

        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel("차이 (°)")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(True, axis="y", color="#e8e8e4", linewidth=0.7)

    axes[-1].set_xlabel("스트로크 진행률 (%)")
    fig.suptitle("정답 대비 각도 차이 (파랑=내가 더 굽힘 / 빨강=내가 덜 굽힘, 좌우 평균)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PATH, dpi=120)
    plt.close(fig)

    print(f"그래프 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
