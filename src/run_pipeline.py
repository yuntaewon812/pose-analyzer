"""
업로드한 두 영상에 대해 1~7단계를 순서대로 돌리는 실행기.

[왜 별도 파일인가]
지금까지는 단계마다 CLI를 손으로 실행했습니다(extract_pose -> calculate_angles ->
segment_reps -> normalize_* -> compare -> feedback -> render). 앱에서 영상을 업로드해
바로 분석하려면 이 순서를 코드로 묶어야 해서, 앱과 CLI가 같은 코드를 쓰도록
여기에 모았습니다.

[분석 파라미터가 영상마다 다른 이유]
한 영상 안에 다이빙 / 돌핀킥 / 자유형 / 턴 / 글라이드가 섞여 있어서, "어느 구간의
무슨 동작을 볼지"는 자동으로 정해지지 않습니다. 그래서 구간(start/end)과 기준
관절 같은 값은 호출하는 쪽이 정해서 넘기고, 기본값은 수중 돌핀킥에 맞춰 뒀습니다.

사용법:
    python src/run_pipeline.py --ref-start 14.8 --ref-end 18.6 --mine-start 13.0 --mine-end 17.3
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"

# 수중 돌핀킥 기본값 (segment_reps.py 옵션). 다른 동작을 분석하려면 이 값들을 바꾼다.
DOLPHIN_KICK_ARGS = [
    "--joint", "left_knee",            # 발차기 리듬이 가장 잘 드러나는 신호
    "--streamline-min-shoulder", "130",  # 양팔을 앞으로 뻗은 수중 국면만
    "--max-roll", "0.28",              # 턴 직후 몸이 틀어진 국면 제외
    "--min-span-sec", "0.5",
    "--smooth-window", "9",
    "--min-frames", "11",
]


def run(cmd, label, log=print):
    log(f"  $ {label}")
    res = subprocess.run([sys.executable, *cmd], cwd=str(ROOT),
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise RuntimeError(f"{label} 실패:\n{(res.stderr or res.stdout)[-1500:]}")
    return res.stdout or ""


def analyze_one(role, start, end, step, seg_args, log=print, auto_detect=True):
    """한 영상에 대해 좌표 추출 -> 각도 -> (돌핀킥 구간 자동 검출) -> 구간 나누기 -> 정규화."""
    lm = f"output/{role}/pose_landmarks.csv"
    ang = f"output/{role}/joint_angles.csv"
    reps = f"output/{role}/reps.csv"

    log(f"[{role}] 관절 좌표 추출 (가장 오래 걸리는 단계)")
    run(["src/extract_pose.py", "--video", f"data/videos/{role}.mp4",
         "--out", lm, "--frames-dir", f"data/frames/{role}", "--step", str(step),
         "--save-every", "300"], f"extract_pose({role})", log)

    log(f"[{role}] 관절 각도 계산")
    run(["src/calculate_angles.py", "--input", lm, "--output", ang], f"calculate_angles({role})", log)

    # 구간을 직접 지정하지 않았으면 영상에서 돌핀킥 구간을 자동으로 찾는다.
    # 수영 영상은 영법이 섞여 있어서(다이빙-돌핀킥-자유형-턴-...) 이 단계가 없으면
    # 자유형 구간이 돌핀킥으로 잘못 분석될 수 있다.
    if auto_detect and start is None and end is None:
        log(f"[{role}] 돌핀킥 구간 자동 검출")
        from detect_dolphin_kick import detect
        segments, _detail = detect(ROOT / lm, ROOT / ang)
        if segments:
            start, end = max(segments, key=lambda s: s[1] - s[0])   # 가장 긴 구간
            log(f"    -> {start:.2f}~{end:.2f}초 ({end - start:.2f}초) 사용"
                f"{f' (후보 {len(segments)}개 중 가장 긴 것)' if len(segments) > 1 else ''}")
        else:
            log("    -> 돌핀킥 구간을 찾지 못했습니다. 영상 전체로 진행합니다 "
                "(결과가 이상하면 구간을 직접 지정하세요)")

    log(f"[{role}] 반복 구간 나누기")
    cmd = ["src/segment_reps.py", "--input", ang, "--landmarks", lm, "--output", reps, *seg_args]
    if start is not None:
        cmd += ["--start", str(start)]
    if end is not None:
        cmd += ["--end", str(end)]
    out = run(cmd, f"segment_reps({role})", log)
    log("    " + next((l for l in out.splitlines() if "최종" in l), "").strip())

    log(f"[{role}] 진행률 정규화")
    run(["src/normalize_reps.py", "--angles", ang, "--reps", reps,
         "--output", f"output/{role}/normalized_mean.csv"], f"normalize_reps({role})", log)
    run(["src/normalize_pose_3d.py", "--landmarks", lm, "--reps", reps,
         "--output", f"output/{role}/normalized_pose_mean.csv"], f"normalize_pose_3d({role})", log)


def run_all(ref_start=None, ref_end=None, mine_start=None, mine_end=None,
            step=1, seg_args=None, log=print, auto_detect=True):
    seg_args = list(seg_args if seg_args is not None else DOLPHIN_KICK_ARGS)
    analyze_one("reference", ref_start, ref_end, step, seg_args, log, auto_detect)
    analyze_one("mine", mine_start, mine_end, step, seg_args, log, auto_detect)

    log("[비교] 각도 차이 계산")
    run(["src/compare_angles.py"], "compare_angles", log)
    log("[피드백] 문장 생성 (규칙 기반)")
    run(["src/generate_feedback.py"], "generate_feedback", log)

    # API 키가 있으면 관절별 문장을 Claude로 다듬는다. 키가 없거나 호출이 실패해도
    # 규칙 기반 문장이 이미 있으므로 분석 전체를 멈추지는 않는다.
    log("[피드백] 문장 다듬기 (Claude API)")
    try:
        run(["src/generate_feedback_llm.py"], "generate_feedback_llm", log)
    except RuntimeError as e:
        log(f"    건너뜀 — 규칙 기반 문장을 사용합니다. ({str(e).splitlines()[-1][:80]})")
    log("[렌더링] 비교 영상 / 관절별 집중 영상")
    run(["src/render_pose_compare.py"], "render_pose_compare", log)
    log("완료")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref-start", type=float, default=None)
    p.add_argument("--ref-end", type=float, default=None)
    p.add_argument("--mine-start", type=float, default=None)
    p.add_argument("--mine-end", type=float, default=None)
    p.add_argument("--step", type=int, default=1, help="몇 프레임마다 처리할지 (1=전 프레임)")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run_all(a.ref_start, a.ref_end, a.mine_start, a.mine_end, a.step)
