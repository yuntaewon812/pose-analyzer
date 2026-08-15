"""
5단계: 정답 영상과 내 영상의 대표 곡선을 진행률 지점별로 빼서 차이 계산하기

[왜 이제는 그냥 뺄셈이 되나]
4단계에서 두 영상 모두 "진행률 0%, 1%, 2%, ..., 100%"라는 완전히 같은 101개
지점으로 맞춰놨습니다. 그래서 이제는 두 표에서 같은 줄(같은 진행률)끼리
그냥 값을 빼기만 하면 "그 지점에서 내가 정답보다 몇 도 더/덜 굽혔는지"가 나옵니다.
시간 정규화(4단계)가 끝났기 때문에 가능해진, 이 단계에서 제일 쉬운 계산입니다.

[차이(diff)의 부호를 어떻게 읽나]
diff = 내 각도 - 정답 각도
  - 양수(+): 내가 정답보다 각도가 크다 = 그 관절을 정답보다 덜 굽혔다(더 폈다)
  - 음수(-): 내가 정답보다 각도가 작다 = 그 관절을 정답보다 더 굽혔다

[요약 통계 - 어디를 고쳐야 하는지 우선순위 매기기]
관절 8개 x 진행률 101개 지점 = 808개나 되는 숫자를 사람이 한 번에 보기는
어렵습니다. 그래서 관절별로 "평균적으로 얼마나 다른지"(평균 절대 차이)와
"가장 크게 다른 지점이 어디인지"(최대 차이 발생 진행률)를 요약해서, 6단계
피드백 문장을 만들 때 "이 관절, 이 시점을 가장 먼저 고치라"고 짚을 수 있게 합니다.

사용법:
    python src/compare_angles.py
결과물:
    output/comparison/angle_diff.csv     (지점별 상세 차이)
    output/comparison/diff_summary.txt   (관절별 요약)
"""
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
REF_CSV = ROOT / "output" / "reference" / "normalized_mean.csv"
MINE_CSV = ROOT / "output" / "mine" / "normalized_mean.csv"
DIFF_CSV = ROOT / "output" / "comparison" / "angle_diff.csv"
SUMMARY_TXT = ROOT / "output" / "comparison" / "diff_summary.txt"

JOINT_COLUMNS = [
    "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
    "left_hip", "right_hip", "left_knee", "right_knee",
]


def load(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ref_rows = load(REF_CSV)
    mine_rows = load(MINE_CSV)

    if len(ref_rows) != len(mine_rows):
        raise ValueError("두 파일의 진행률 지점 개수가 다릅니다 (4단계를 같은 --points로 돌렸는지 확인)")

    DIFF_CSV.parent.mkdir(parents=True, exist_ok=True)

    header = ["progress_pct"]
    for col in JOINT_COLUMNS:
        header += [f"ref_{col}", f"mine_{col}", f"diff_{col}"]

    # 관절별로 모든 진행률 지점의 차이를 모아뒀다가, 끝에서 요약 통계를 낸다.
    diffs_by_joint = {col: [] for col in JOINT_COLUMNS}

    with open(DIFF_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ref_row, mine_row in zip(ref_rows, mine_rows):
            pct = ref_row["progress_pct"]
            out_row = [pct]
            for col in JOINT_COLUMNS:
                ref_v = ref_row[col]
                mine_v = mine_row[col]
                if ref_v == "" or mine_v == "":
                    out_row += [ref_v, mine_v, ""]
                    continue
                ref_v, mine_v = float(ref_v), float(mine_v)
                diff = round(mine_v - ref_v, 1)
                out_row += [ref_v, mine_v, diff]
                diffs_by_joint[col].append((float(pct), diff))
            writer.writerow(out_row)

    lines = []
    lines.append("=" * 60)
    lines.append("관절 각도 차이 요약 (diff = 내 각도 - 정답 각도)")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"{'관절':<16}{'평균|차이|':>10}{'최대 차이':>10}{'그 지점':>8}")
    lines.append("-" * 60)

    for col in JOINT_COLUMNS:
        pairs = diffs_by_joint[col]
        if not pairs:
            continue
        mean_abs = sum(abs(d) for _, d in pairs) / len(pairs)
        worst_pct, worst_diff = max(pairs, key=lambda pd: abs(pd[1]))
        lines.append(f"{col:<16}{mean_abs:>10.1f}{worst_diff:>+10.1f}{worst_pct:>7.0f}%")

    lines.append("-" * 60)
    lines.append("")
    lines.append("[읽는 법] 평균|차이|가 클수록 스트로크 내내 정답과 다르게 움직이는 관절.")
    lines.append("최대 차이가 특정 구간(그 지점)에 몰려 있으면, 스트로크 전체가 아니라")
    lines.append("그 순간(예: 캐치 시작 시점)의 자세만 고치면 되는 경우일 수 있음.")
    lines.append("diff가 +면 정답보다 덜 굽힘(더 폄), -면 정답보다 더 굽힘.")

    summary = "\n".join(lines)
    SUMMARY_TXT.write_text(summary, encoding="utf-8")

    print(summary)
    print()
    print(f"상세 차이 저장: {DIFF_CSV}")
    print(f"요약 저장: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
