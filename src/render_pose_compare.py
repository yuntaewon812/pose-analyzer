"""
7단계 확장: 정답과 내 동작을 "정면/측면 졸라맨"으로 나란히 그린 비교 영상/이미지 만들기

[왜 필요한가]
실제 영상 프레임을 나란히 보여주는 방식은 정답 영상과 내 영상의 촬영 카메라
각도가 서로 달라서(한쪽은 옆에서, 한쪽은 정면 가까이에서) 자세 차이인지 카메라
각도 차이인지 헷갈립니다. 이 스크립트는 실제 영상 대신 normalize_pose_3d.py가
만든 "몸통 기준 3D 좌표"로 스틱맨을 그립니다. 정답과 내 동작을 한 화면에 겹치면
(교차) 두 뼈대가 서로 가려서 잘 안 보이므로, 패널을 정답/나로 나눠 나란히
그리고, 같은 진행률(%)에 맞춰 같은 타이밍에 움직이게 합니다.

정면(앞에서 마주보는 시점)만으로는 팔이 앞뒤로 뻗는 스트로크 움직임이 잘 안
보이고, 측면(옆에서 보는 시점)만으로는 롤링/발차기 폭이 잘 안 보여서, 두 시점을
모두 만듭니다 (pose_projection.py 참고). 스틱맨은 몸통(회색)/팔(파랑)/다리(주황)로
색을 나누고 머리를 원으로 그려서, 관절점만 찍었을 때보다 어디가 어디인지 한눈에
들어오게 했습니다.

[영상이 브라우저에서 안 움직이고 정지 화면처럼 보였던 이유 - 코덱 문제]
이 컴퓨터에 설치된 OpenCV의 ffmpeg에는 라이선스 문제로 H.264 인코더(openh264
DLL)가 빠져 있어서, cv2.VideoWriter로 만든 mp4는 실제로는 브라우저가 재생할 수
없는 옛날 코덱(MPEG-4 Part 2)으로 인코딩됐습니다. 파일 확장자는 .mp4라서
"영상"처럼 보이지만, 웹 브라우저는 이 코덱을 디코딩하지 못해 재생 버튼을 눌러도
움직이지 않고 사실상 정지 이미지처럼 보였던 것입니다. 그래서 OpenCV 대신
imageio-ffmpeg 패키지가 받아둔 정식 ffmpeg 실행 파일(H.264 인코더 libx264 포함)로
직접 인코딩하도록 바꿨습니다 - 프레임을 그림으로 저장하는 대신, 원시 픽셀
데이터를 파이프로 ffmpeg에 흘려보내 H.264로 압축합니다.

사용법:
    python src/render_pose_compare.py
결과물:
    output/comparison/pose_compare_front.mp4       (정면 졸라맨 비교 영상, 0~100% 반복)
    output/comparison/pose_compare_side.mp4        (측면 졸라맨 비교 영상, 0~100% 반복)
    output/comparison/pose_compare_key_moments.png (차이가 가장 큰 3개 지점, 정면+측면 정지 이미지)
    output/comparison/pose_focus_<관절>.mp4        (차이가 큰 관절마다 동그라미 1개짜리 영상)
"""
import csv
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import calculate_angles as ca
import generate_feedback as fb
from pose_projection import (
    body_frame, blank_canvas, draw_stick_figure, draw_legend, project_points,
    stabilize_frames, screen_frame, project_screen, draw_focus_circle,
    COLOR_REF_BGR, COLOR_MINE_BGR, CANVAS_SIZE, ORIGIN_PX,
)

ROOT = Path(__file__).parent.parent
REF_POSE = ROOT / "output" / "reference" / "normalized_pose_mean.csv"
MINE_POSE = ROOT / "output" / "mine" / "normalized_pose_mean.csv"
OUT_VIDEO = {
    "front": ROOT / "output" / "comparison" / "pose_compare_front.mp4",
    "side": ROOT / "output" / "comparison" / "pose_compare_side.mp4",
}
OUT_KEY_IMG = ROOT / "output" / "comparison" / "pose_compare_key_moments.png"
VIEW_LABELS = {"front": "정면", "side": "측면"}

# 사이클 1회를 101개 지점으로 그리므로, fps가 낮으면 한 사이클이 지나치게 느려진다
# (15fps면 6.7초 - 실제 킥 0.6초의 11배라 "차는 동작"으로 안 보인다). 50fps면 한
# 사이클이 약 2초로, 기술 분석에 적당한 3배 정도의 슬로모션이 된다.
FPS = 50
DIVIDER_PX = 6

# 비교 영상에서 대표 사이클을 몇 번 이어 붙일지.
#
# [왜 반복이 필요한가]
# 이 파이프라인은 킥 사이클 1회를 진행률 0~100%로 정규화해서 비교합니다(정답과 내
# 동작을 같은 국면끼리 겹쳐 보려면 그래야 함). 그래서 영상도 1회분만 그려지는데,
# 실제 영상에서는 5~7회를 연속으로 차기 때문에 "왜 한 번만 차느냐"처럼 보입니다.
# 같은 사이클을 여러 번 이어 붙이면 연속 동작처럼 읽히면서도, 비교 기준(같은
# 진행률끼리 대응)은 그대로 유지됩니다. 다만 이건 "실제로 5회를 찬 기록"이 아니라
# "대표 1회를 반복 재생한 것"이므로, 화면에도 그렇게 표시합니다.
CYCLE_LOOPS = 5

# 이 각도(도) 이상 차이나는 관절만 "집중 비교 영상"을 만든다. 너무 낮게 잡으면
# 영상이 우수수 늘어나 무엇부터 볼지 알 수 없고, 너무 높게 잡으면 고칠 거리를 놓친다.
FOCUS_MIN_DIFF = 10.0


def load_pose_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def nearest_row(rows, pct):
    return min(rows, key=lambda r: abs(float(r["progress_pct"]) - pct))


def nearest_row_index(rows, pct):
    """nearest_row와 같지만 인덱스를 준다 - 미리 계산해둔 안정화 축 배열에서
    같은 위치를 꺼내 쓰기 위해 필요하다."""
    return min(range(len(rows)), key=lambda i: abs(float(rows[i]["progress_pct"]) - pct))


def frames_for_view(rows, view):
    """뷰에 맞는 '기준값'을 시퀀스 전체에서 한 번 계산해 프레임 개수만큼 돌려준다.

    - side : 화면 좌표 기준값 하나를 모든 프레임이 공유 (파동 보존 - screen_frame 참고)
    - front: 프레임마다 몸통 기준축 (좌우 정보는 3D에만 있으므로 3D 사용)
    """
    if view == "side":
        sf = screen_frame(rows)
        return [sf] * len(rows)
    return stabilize_frames(rows)


# 측면 뷰는 몸이 가로로 누워 있어서 정사각 캔버스의 위아래가 대부분 빈 공간이다.
# 그만큼 스틱맨이 작게 보여 자세를 읽기 어려우므로, 그림을 그린 뒤 세로로 잘라내
# 몸이 화면을 채우게 한다 (킥으로 발이 오르내리는 폭은 넉넉히 남긴다).
SIDE_VIEW_HEIGHT = 340


def render_panel(row, label, label_color, view, highlight_names, bf=None, focus_landmark=None):
    """view="side"면 bf에 screen_frame() 결과를, 그 외에는 body_frame() 결과를 받는다.
    측면은 화면 좌표로 그려야 돌핀킥의 파동(엉덩이 출렁임·몸통 굴곡)이 살아남는다.

    focus_landmark를 주면 그 관절 하나에만 집중 표시 동그라미를 그린다."""
    canvas = blank_canvas(view)
    if bf is None:
        bf = screen_frame([row]) if view == "side" else body_frame(row)
    if bf:
        pts = project_screen(row, bf) if view == "side" else project_points(row, bf, view)
        draw_stick_figure(canvas, pts, view, highlight_names)
        if focus_landmark:
            draw_focus_circle(canvas, pts, focus_landmark, view)

    if view == "side":
        cy = ORIGIN_PX["side"][1]
        half = SIDE_VIEW_HEIGHT // 2
        canvas = canvas[max(0, cy - half): min(CANVAS_SIZE, cy + half)].copy()

    # 라벨/범례는 잘라낸 뒤에 그려야 잘리거나 서로 겹치지 않는다.
    cv2.putText(canvas, label, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2, cv2.LINE_AA)
    draw_legend(canvas)
    return canvas


def render_frame(ref_row, mine_row, pct, view, highlight_names=None, ref_bf=None, mine_bf=None,
                  mine_focus=None):
    """정답/내 스틱맨을 각자 패널에 그려서 나란히 붙인다 (겹치지 않게).

    ref_bf/mine_bf에 stabilize_frames()로 미리 안정화한 몸통 축을 넘기면 그걸 쓴다
    (안 넘기면 그 프레임만 보고 축을 계산 - 옆에서 찍은 영상에서는 축이 떨릴 수 있음).
    """
    ref_panel = render_panel(ref_row, f"REFERENCE  {VIEW_LABELS[view]} {pct:.0f}%", COLOR_REF_BGR,
                              view, highlight_names, ref_bf)
    # 동그라미는 "내 동작"에만 그린다 - 고쳐야 할 곳을 짚는 표시이므로.
    mine_panel = render_panel(mine_row, f"MINE  {VIEW_LABELS[view]} {pct:.0f}%", COLOR_MINE_BGR,
                               view, highlight_names, mine_bf, mine_focus)
    divider = np.full((ref_panel.shape[0], DIVIDER_PX, 3), 210, dtype=np.uint8)
    return cv2.hconcat([ref_panel, divider, mine_panel])


def write_video_h264(frames, out_path, fps, width, height):
    """cv2.VideoWriter 대신 ffmpeg(libx264)로 직접 인코딩해 브라우저에서 실제로
    재생되는 mp4를 만든다 (모듈 docstring의 "코덱 문제" 참고)."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 인코딩 실패 (exit {proc.returncode}): {out_path}")


def write_cycle_video(ref_rows, mine_rows, view, out_path, mine_focus=None, corner_text=None):
    """대표 사이클을 CYCLE_LOOPS번 이어 붙여 mp4로 쓴다.

    mine_focus를 주면 "내 동작" 쪽 그 관절에만 집중 표시 동그라미가 그려진다.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ref_bfs, mine_bfs = frames_for_view(ref_rows, view), frames_for_view(mine_rows, view)
    # 측면은 세로로 잘라내므로 크기를 미리 알 수 없다. 첫 프레임을 그려서 확인한다.
    probe = render_frame(ref_rows[0], mine_rows[0], 0.0, view, None, ref_bfs[0], mine_bfs[0], mine_focus)
    h, w = probe.shape[:2]

    def frames():
        for loop in range(CYCLE_LOOPS):
            for i, (ref_row, mine_row) in enumerate(zip(ref_rows, mine_rows)):
                # 마지막 지점(100%)은 다음 반복의 0%와 같은 자세라, 이어 붙일 때
                # 한 프레임 멈칫하는 것처럼 보인다. 마지막 반복에서만 남긴다.
                if i == len(ref_rows) - 1 and loop < CYCLE_LOOPS - 1:
                    continue
                pct = float(ref_row["progress_pct"])
                frame = render_frame(ref_row, mine_row, pct, view, None,
                                      ref_bfs[i], mine_bfs[i], mine_focus)
                # 범례(좌하단)와 겹치지 않도록 오른쪽 위에 표시한다.
                text = corner_text or f"cycle {loop + 1}/{CYCLE_LOOPS} (same cycle repeated)"
                (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.putText(frame, text, (w - tw - 16, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1, cv2.LINE_AA)
                yield frame

    write_video_h264(frames(), out_path, FPS, w, h)
    return w, h


def render_video(ref_rows, mine_rows, view):
    out_path = OUT_VIDEO[view]
    write_cycle_video(ref_rows, mine_rows, view, out_path)
    total = CYCLE_LOOPS * len(ref_rows) - (CYCLE_LOOPS - 1)
    print(f"{VIEW_LABELS[view]} 비교 영상 저장: {out_path} "
          f"(대표 사이클 1회 x {CYCLE_LOOPS}반복 = {total}프레임, {FPS}fps, H.264)")


def focus_joints(min_diff=FOCUS_MIN_DIFF):
    """차이가 큰 관절을 큰 순서로 돌려준다 (관절 이름, 통계). 영상 1개당 관절 1개."""
    stats = fb.load_joint_stats()
    ranked = sorted(stats.items(), key=lambda kv: -kv[1]["mean_abs"])
    picked = [(n, s) for n, s in ranked if s["mean_abs"] >= min_diff]
    # 기준을 넘는 관절이 하나도 없으면(자세가 전반적으로 잘 맞는 경우) 가장 큰 것 하나는 보여준다.
    return picked or ranked[:1]


def focus_video_path(joint):
    return ROOT / "output" / "comparison" / f"pose_focus_{joint}.mp4"


def render_focus_videos(ref_rows, mine_rows, view="side"):
    """차이가 큰 관절마다 "그 관절 하나에만 동그라미가 있는" 비교 영상을 만든다.

    [왜 관절마다 따로 만드나]
    차이가 큰 관절이 여러 개일 때 한 영상에 동그라미를 다 그리면 시선이 분산돼서
    오히려 어디를 고쳐야 할지 알기 어렵습니다. 관절 1개 = 영상 1개로 나누면
    한 번에 한 곳만 보면 되고, 영상마다 그 관절의 피드백을 붙일 수 있습니다.
    """
    made = []
    for joint, s in focus_joints():
        vertex = ca.ANGLE_DEFINITIONS[joint][1]   # 각도의 꼭짓점 = 그 관절의 위치
        out = focus_video_path(joint)
        write_cycle_video(ref_rows, mine_rows, view, out, mine_focus=vertex,
                           corner_text=f"focus: {joint}")
        made.append((joint, s, out))
        print(f"  집중 비교 영상: {fb.JOINT_LABELS[joint]} (평균 {s['mean_abs']:.0f}° 차이) -> {out.name}")
    return made


def render_key_moments(ref_rows, mine_rows):
    stats = fb.load_joint_stats()
    top3 = sorted(stats.items(), key=lambda kv: -kv[1]["mean_abs"])[:3]

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6))
    fig.patch.set_facecolor("#fcfcfb")

    # 정면/측면이 서로 다른 기준값을 쓰므로 뷰별로 따로 준비한다.
    bfs_by_view = {v: (frames_for_view(ref_rows, v), frames_for_view(mine_rows, v))
                   for v in ("front", "side")}

    for col, (name, s) in enumerate(top3):
        pct = s["worst_pct"]
        ref_i = nearest_row_index(ref_rows, pct)
        mine_i = nearest_row_index(mine_rows, pct)
        ref_row, mine_row = ref_rows[ref_i], mine_rows[mine_i]
        a_name, b_name, c_name = ca.ANGLE_DEFINITIONS[name]
        highlight = {a_name, b_name, c_name}

        for row_i, view in enumerate(("front", "side")):
            ax = axes[row_i, col]
            v_ref_bfs, v_mine_bfs = bfs_by_view[view]
            frame_bgr = render_frame(ref_row, mine_row, pct, view, highlight,
                                      v_ref_bfs[ref_i], v_mine_bfs[mine_i])
            ax.imshow(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            title = f"{VIEW_LABELS[view]}"
            if row_i == 0:
                title = f"{fb.action_phrase(name, s)}\n" + title
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.axis("off")

    fig.suptitle("가장 크게 차이나는 순간 — 정면/측면 (빨간 테두리 = 지금 보는 각도, 왼쪽=정답 / 오른쪽=나)",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.text(0.5, 0.955,
             "머리(살구색 원) · 몸통(회색) · 팔(파랑) · 다리(빨강 계열)  —  색은 몸의 부위를 뜻함 (정답/나는 왼쪽·오른쪽 패널로 구분)",
             ha="center", fontsize=9, color="#6b6b66")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_KEY_IMG, dpi=120)
    plt.close(fig)
    print(f"핵심 차이 지점 이미지 저장: {OUT_KEY_IMG}")


def main() -> None:
    ref_rows = load_pose_rows(REF_POSE)
    mine_rows = load_pose_rows(MINE_POSE)
    render_video(ref_rows, mine_rows, "front")
    render_video(ref_rows, mine_rows, "side")
    render_focus_videos(ref_rows, mine_rows)
    render_key_moments(ref_rows, mine_rows)


if __name__ == "__main__":
    main()
