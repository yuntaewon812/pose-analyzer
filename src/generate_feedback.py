"""
6단계: 각도 차이 숫자를 사람이 읽는 피드백 문장으로 바꾸기

[규칙 기반으로 먼저 만드는 이유]
숫자(diff)를 바로 LLM에 던져서 코칭 문장을 받을 수도 있지만, 그러면 "LLM이
만든 조언이 맞는 소리인지" 확인할 기준이 없습니다. 그래서 먼저 if-then 규칙으로
간단한 피드백을 만들어두고, 나중에 LLM으로 문장을 더 자연스럽게 다듬을 때
이 규칙 기반 결과와 비교해서 검증할 수 있게 합니다.

[정직하게 다루는 부분 - 과신하지 않기]
"엉덩이 각도가 작다"는 사실은 데이터로 확실하지만, 그 원인(코어 힘 부족?
부력 문제? 킥 타이밍?)까지 영상 각도만으로 단정할 수는 없습니다. 그래서 이
스크립트는 "무엇이 다른지"는 자신 있게 말하고, "왜 다른지/어떻게 고치는지"는
일반적으로 알려진 원인 후보를 참고용으로만 제시합니다.

관절을 두 그룹으로 나눠서 설명합니다:
  - 발차기 그룹 (엉덩이, 무릎): 몸통-다리 라인, 킥 동작
  - 상체 그룹 (어깨, 팔꿈치): 스트림라인 자세 유지

사용법:
    python src/generate_feedback.py
결과물:
    output/comparison/feedback.txt
"""
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
DIFF_CSV = ROOT / "output" / "comparison" / "angle_diff.csv"
FEEDBACK_TXT = ROOT / "output" / "comparison" / "feedback.txt"

KICK_JOINTS = ["left_hip", "right_hip", "left_knee", "right_knee"]
PULL_JOINTS = ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"]

# 지금 분석 대상 국면의 이름. 현재 파이프라인은 "다이빙 직후 수중 스트림라인
# 돌핀킥" 구간을 잘라서 비교하므로(segment_reps.py --streamline-min-shoulder),
# 1회 반복 단위는 "스트로크"가 아니라 "킥 사이클"이다. 수면 자유형 스트로크를
# 분석하도록 되돌린다면 이 두 값만 바꾸면 문구 전체가 따라간다.
PHASE_TITLE = "수중 돌핀킥 (스트림라인 구간)"
CYCLE_LABEL = "킥 사이클"

JOINT_LABELS = {
    "left_elbow": "왼쪽 팔꿈치", "right_elbow": "오른쪽 팔꿈치",
    "left_shoulder": "왼쪽 어깨", "right_shoulder": "오른쪽 어깨",
    "left_hip": "왼쪽 엉덩이", "right_hip": "오른쪽 엉덩이",
    "left_knee": "왼쪽 무릎", "right_knee": "오른쪽 무릎",
}

# 관절별로 "각도가 작다(더 굽음)"/"각도가 크다(더 폄)"일 때 참고할 만한
# 일반적인 원인 후보. 확정 진단이 아니라 참고용 힌트임을 강조한다.
JOINT_HINTS = {
    "hip": {
        "smaller": "몸통-허벅지가 더 접혀 있음 → 흔히 '엉덩이가 가라앉는(처지는)' 자세와 관련됨 (코어 힘, 머리 위치, 부력 등 여러 원인이 있을 수 있음)",
        "larger": "몸통-허벅지가 더 펴져 있음 → 몸이 수평에 더 가깝게 유지되는 경향",
    },
    "knee": {
        "smaller": "무릎을 더 굽힘 → 흔히 '바이시클 킥(무릎을 과하게 굽히는 킥)'과 관련되며, 추진력 손실의 흔한 원인으로 꼽힘",
        "larger": "무릎을 덜 굽힘(더 곧은 다리 킥) → 엉덩이 중심의 채찍킥에 가까움",
    },
    "shoulder": {
        "smaller": "팔이 몸통에 더 가깝게 붙음 → 스트림라인에서 팔을 앞으로 충분히 뻗지 못했을 가능성",
        "larger": "팔이 몸통에서 더 멀리 벌어져 움직임 → 회전 폭이 큰 편",
    },
    "elbow": {
        "smaller": "팔꿈치를 더 많이 접음",
        "larger": "팔을 더 곧게 폄",
    },
}

# "29도 차이" 같은 각도 숫자는 그 자체로는 무엇을 어떻게 고쳐야 할지 와닿지 않는다.
# 그래서 관절마다 "각도를 키워야 할 때/줄여야 할 때" 취할 동작을 짧은 지시문으로
# 미리 준비해두고, 실제 diff의 부호(내가 정답보다 작은지/큰지)에 따라 골라 쓴다.
# diff = 내 각도 - 정답 각도. diff<0(=smaller, 내가 더 굽힘)이면 각도를 "키우는"
# 동작을, diff>0(=larger, 내가 덜 굽힘)이면 각도를 "줄이는" 동작을 안내한다.
ACTION_VERBS = {
    "hip": {"smaller": "더 펴서 몸을 수평에 가깝게 유지하세요", "larger": "힘을 빼고 자연스럽게 살짝 굽히세요"},
    "knee": {"smaller": "더 펴세요 (덜 접고 차보세요)", "larger": "더 굽히세요"},
    "shoulder": {"smaller": "몸통에서 더 멀리 들어 올리세요", "larger": "회전 폭을 살짝 줄이세요"},
    "elbow": {"smaller": "더 펴세요", "larger": "더 굽히세요"},
}
JOINT_PARTICLE = {"hip": "를", "knee": "을", "shoulder": "를", "elbow": "를"}  # 을/를 조사


def joint_group(name):
    return "hip" if "hip" in name else "knee" if "knee" in name else "shoulder" if "shoulder" in name else "elbow"


def action_phrase(name, s):
    """각도 차이를, "무엇을 어떻게 하라"는 짧은 지시문으로 바꾼다."""
    group = joint_group(name)
    direction = "smaller" if s["worst_diff"] < 0 else "larger"
    return f"{JOINT_LABELS[name]}{JOINT_PARTICLE[group]} {ACTION_VERBS[group][direction]}"


def load_joint_stats():
    with open(DIFF_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    joints = ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
              "left_hip", "right_hip", "left_knee", "right_knee"]
    stats = {}
    for name in joints:
        pairs = [(float(r["progress_pct"]), float(r[f"diff_{name}"]))
                 for r in rows if r[f"diff_{name}"] != ""]
        if not pairs:
            continue
        mean_abs = sum(abs(d) for _, d in pairs) / len(pairs)
        worst_pct, worst_diff = max(pairs, key=lambda pd: abs(pd[1]))
        stats[name] = {"mean_abs": mean_abs, "worst_pct": worst_pct, "worst_diff": worst_diff}
    return stats


def joint_feedback_line(name, s):
    group = joint_group(name)
    direction = "smaller" if s["worst_diff"] < 0 else "larger"
    hint = JOINT_HINTS[group][direction]
    return (
        f"- {action_phrase(name, s)}\n"
        f"  → {hint}\n"
        f"  (참고 수치: {CYCLE_LABEL} 내내 평균 {s['mean_abs']:.0f}° 차이, "
        f"진행률 {s['worst_pct']:.0f}% 지점에서 최대 {s['worst_diff']:+.0f}° 차이)"
    )


def positive_phrase(name, s, rank=None):
    """차이가 작은 관절은 "여기를 고치라"가 아니라 "이 자세는 이미 잘 맞고 있다"는
    확인 문장으로 보여준다. 고칠 점만 나열하면 "지금 하고 있는 게 맞는 방향인지"를
    알 수 없어서, 상대적으로 가장 잘 맞는 부분도 짚어준다."""
    label = JOINT_LABELS[name]
    tail = " — 8개 관절 중 가장 정답에 가까움" if rank == 0 else ""
    return f"- {label}: 정답과 비슷한 각도를 잘 유지하고 있어요 (평균 {s['mean_abs']:.0f}° 차이{tail})"


def main() -> None:
    stats = load_joint_stats()
    if not stats:
        raise ValueError("angle_diff.csv에 유효한 데이터가 없습니다. 5단계를 먼저 실행하세요.")

    ranked_all = sorted(stats.items(), key=lambda kv: -kv[1]["mean_abs"])

    lines = []
    lines.append("=" * 60)
    lines.append(f"동작 개선 피드백 - {PHASE_TITLE}")
    lines.append("(규칙 기반, 정답 영상 대비)")
    lines.append("=" * 60)
    lines.append("")
    lines.append("[가장 먼저 고치면 좋을 3가지]")
    for name, s in ranked_all[:3]:
        lines.append(joint_feedback_line(name, s))
    lines.append("")

    lines.append("[이미 잘 유지되고 있는 부분]")
    best = sorted(stats.items(), key=lambda kv: kv[1]["mean_abs"])[:3]
    for rank, (name, s) in enumerate(best):
        lines.append(positive_phrase(name, s, rank))
    lines.append("")

    lines.append("[발차기 (엉덩이·무릎)]")
    kick_ranked = sorted(((n, stats[n]) for n in KICK_JOINTS if n in stats),
                         key=lambda kv: -kv[1]["mean_abs"])
    for name, s in kick_ranked:
        lines.append(joint_feedback_line(name, s))
    lines.append("")

    lines.append("[상체 스트림라인 (어깨·팔꿈치)]")
    pull_ranked = sorted(((n, stats[n]) for n in PULL_JOINTS if n in stats),
                         key=lambda kv: -kv[1]["mean_abs"])
    for name, s in pull_ranked:
        lines.append(joint_feedback_line(name, s))
    lines.append("")

    lines.append("-" * 60)
    lines.append("[주의]")
    lines.append("- 위 원인 후보는 일반적으로 알려진 참고 힌트이며, 확정 진단이 아닙니다.")
    lines.append("- 각도는 카메라 1대로 추정한 3D 좌표 기반이라 어느 정도 오차가 있을 수 있습니다.")
    lines.append(f"- {CYCLE_LABEL} 여러 회를 평균한 값이라 개별 {CYCLE_LABEL}의 특이한 순간은 반영되지 않습니다.")

    feedback = "\n".join(lines)
    FEEDBACK_TXT.write_text(feedback, encoding="utf-8")
    print(feedback)
    print()
    print(f"저장 위치: {FEEDBACK_TXT}")


if __name__ == "__main__":
    main()
