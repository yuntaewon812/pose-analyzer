"""
Streamlit 앱 - 수중 돌핀킥 동작 분석을 한 페이지에서

영상 두 개(정답/내 영상)를 올리면 돌핀킥 구간을 자동으로 찾아 분석하고,
차이가 큰 관절마다 "그 관절 하나에만 동그라미가 있는" 비교 영상과 피드백을 보여준다.

[왜 필요한가]
5~6단계에서 만든 결과물(angle_diff.csv, feedback.txt, 그래프 PNG들)이
output/comparison/ 여기저기 흩어져 있어서, 확인하려면 파일을 하나씩 열어야
했습니다. 이 스크립트는 그 결과물들을 표/그래프/문장으로 묶어서 브라우저
한 페이지에서 훑어볼 수 있게 합니다. 파이프라인 1~6단계를 다시 계산하지
않고, 이미 저장된 CSV/텍스트 결과만 읽어서 보여줍니다.

사용법:
    streamlit run src/app.py
"""
import json
import sys
import traceback
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
import calculate_angles as ca  # ANGLE_DEFINITIONS, calculate_angle, get_point 재사용 (가벼운 모듈, mediapipe 불필요)
import generate_feedback as fb  # load_joint_stats, JOINT_LABELS 등 6단계 로직 재사용
from render_pose_compare import (
    render_frame as render_pose_frame,
    nearest_row_index as nearest_pose_row_index,
    load_pose_rows,
)
from render_pose_compare import (
    frames_for_view, focus_joints, focus_video_path, CYCLE_LOOPS, FOCUS_MIN_DIFF,
)
from run_pipeline import run_all

REF_NORM = ROOT / "output" / "reference" / "normalized_mean.csv"
MINE_NORM = ROOT / "output" / "mine" / "normalized_mean.csv"
REF_REPS = ROOT / "output" / "reference" / "reps.csv"
MINE_REPS = ROOT / "output" / "mine" / "reps.csv"
REF_LANDMARKS = ROOT / "output" / "reference" / "pose_landmarks.csv"
MINE_LANDMARKS = ROOT / "output" / "mine" / "pose_landmarks.csv"
DIFF_CSV = ROOT / "output" / "comparison" / "angle_diff.csv"
FEEDBACK_TXT = ROOT / "output" / "comparison" / "feedback.txt"
FEEDBACK_LLM_TXT = ROOT / "output" / "comparison" / "feedback_llm.txt"
FEEDBACK_LLM_JSON = ROOT / "output" / "comparison" / "feedback_llm.json"
FEEDBACK_LLM_SAMPLE = ROOT / "output" / "comparison" / "feedback_llm_sample.txt"
REF_VIDEO = ROOT / "data" / "videos" / "reference.mp4"
MINE_VIDEO = ROOT / "data" / "videos" / "mine.mp4"
REF_POSE_NORM = ROOT / "output" / "reference" / "normalized_pose_mean.csv"
MINE_POSE_NORM = ROOT / "output" / "mine" / "normalized_pose_mean.csv"
FRONT_COMPARE_VIDEO = ROOT / "output" / "comparison" / "pose_compare_front.mp4"
SIDE_COMPARE_VIDEO = ROOT / "output" / "comparison" / "pose_compare_side.mp4"

# extract_pose.py의 LANDMARK_NAMES/POSE_CONNECTIONS와 동일 (mediapipe 임포트를 피하려 값만 복사)
LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (28, 30), (30, 32),
]
HIGHLIGHT_COLOR = (255, 64, 64)   # 문제의 관절 각도 강조 (빨강, RGB)
BONE_COLOR = (64, 200, 90)        # 나머지 뼈대 (초록)
JOINT_COLOR = (60, 110, 240)      # 나머지 관절점 (파랑)

COLOR_REF = "#2a78d6"
COLOR_MINE = "#eb6834"
COLOR_NEG = "#2a78d6"   # diff < 0 (내가 더 굽힘)
COLOR_POS = "#e34948"   # diff > 0 (내가 덜 굽힘)

PANELS = [
    ("팔꿈치 (Elbow)", "left_elbow", "right_elbow"),
    ("어깨 (Shoulder)", "left_shoulder", "right_shoulder"),
    ("엉덩이 (Hip)", "left_hip", "right_hip"),
    ("무릎 (Knee)", "left_knee", "right_knee"),
]

st.set_page_config(page_title="수중 돌핀킥 동작 분석", layout="wide")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


@st.cache_data
def load_data():
    return (
        pd.read_csv(REF_NORM),
        pd.read_csv(MINE_NORM),
        pd.read_csv(DIFF_CSV),
        pd.read_csv(REF_REPS),
        pd.read_csv(MINE_REPS),
    )


def missing_required_files():
    required = [REF_NORM, MINE_NORM, DIFF_CSV, REF_REPS, MINE_REPS]
    return [p for p in required if not p.exists()]


@st.cache_data
def load_llm_feedback():
    """generate_feedback_llm.py가 만든 관절별 문장. 없으면 None (규칙 기반으로 대체)."""
    if not FEEDBACK_LLM_JSON.exists():
        return None
    try:
        return json.loads(FEEDBACK_LLM_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None   # 파일이 깨졌으면 없는 것으로 보고 규칙 기반 문장을 쓴다


@st.cache_data
def load_landmarks():
    return pd.read_csv(REF_LANDMARKS), pd.read_csv(MINE_LANDMARKS)


@st.cache_data
def load_pose_rows_cached(path_str):
    return load_pose_rows(path_str)


@st.cache_resource
def frames_for_view_cached(path_str, view):
    """뷰에 맞는 기준값 배열 (측면=화면 좌표 기준값, 정면=몸통 기준축).
    numpy 배열 튜플이라 cache_data로는 직렬화가 번거로워 cache_resource를 쓴다
    (읽기 전용으로만 사용)."""
    return frames_for_view(load_pose_rows(path_str), view)


def pick_representative_rep(reps_df):
    """구간(스트로크) 중 길이가 중앙값에 가장 가까운 것 하나를 대표로 고른다."""
    median_dur = reps_df["duration_sec"].median()
    idx = (reps_df["duration_sec"] - median_dur).abs().idxmin()
    return reps_df.loc[idx]


def frame_for_progress(rep, pct):
    """대표 구간 안에서 진행률 pct(%)에 해당하는 프레임 번호를 계산한다."""
    start, end = rep["start_frame"], rep["end_frame"]
    return int(round(start + pct / 100 * (end - start)))


def nearest_landmark_row(landmarks_df, frame_idx):
    """extract_pose.py는 --step 프레임마다 한 줄씩만 저장하므로, 가장 가까운 줄을 찾는다."""
    idx = (landmarks_df["frame"] - frame_idx).abs().idxmin()
    return landmarks_df.loc[idx]


@st.cache_data(show_spinner=False)
def read_video_frame(video_path_str, frame_idx):
    cap = cv2.VideoCapture(video_path_str)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_skeleton(frame_rgb, landmark_row, highlight_names):
    """관절 뼈대를 그리되, 지금 보고 있는 각도(highlight_names 3개 점)만 빨갛게 강조한다."""
    frame = frame_rgb.copy()
    h, w = frame.shape[:2]

    points = {}
    for name in LANDMARK_NAMES:
        x, y = landmark_row.get(f"{name}_x"), landmark_row.get(f"{name}_y")
        if pd.isna(x) or pd.isna(y):
            continue
        points[name] = (int(float(x) * w), int(float(y) * h))

    for a_idx, b_idx in POSE_CONNECTIONS:
        a_name, b_name = LANDMARK_NAMES[a_idx], LANDMARK_NAMES[b_idx]
        if a_name not in points or b_name not in points:
            continue
        is_hl = a_name in highlight_names and b_name in highlight_names
        color = HIGHLIGHT_COLOR if is_hl else BONE_COLOR
        cv2.line(frame, points[a_name], points[b_name], color, 5 if is_hl else 2)

    for name, (x, y) in points.items():
        is_hl = name in highlight_names
        color = HIGHLIGHT_COLOR if is_hl else JOINT_COLOR
        cv2.circle(frame, (x, y), 9 if is_hl else 3, color, -1)

    return frame


def angle_at_row(row, joint_name):
    """pose_landmarks.csv 행(visibility 열 포함)에서 관절 각도를 계산한다."""
    a_name, b_name, c_name = ca.ANGLE_DEFINITIONS[joint_name]
    a_pt = ca.get_point(row, a_name)[:3]
    b_pt = ca.get_point(row, b_name)[:3]
    c_pt = ca.get_point(row, c_name)[:3]
    return ca.calculate_angle(a_pt, b_pt, c_pt)


def angle_at_pose_row(row, joint_name):
    """normalized_pose_mean.csv 행에서 관절 각도를 계산한다 (이 파일에는 좌표만 있고
    visibility 열이 없으므로 angle_at_row를 쓸 수 없다)."""
    def xyz(name):
        try:
            return (float(row[f"{name}_wx"]), float(row[f"{name}_wy"]), float(row[f"{name}_wz"]))
        except (KeyError, TypeError, ValueError):
            return None

    pts = [xyz(n) for n in ca.ANGLE_DEFINITIONS[joint_name]]
    if any(p is None for p in pts):
        return None
    return ca.calculate_angle(*pts)


st.title("🐬 수중 돌핀킥 동작 분석")
st.caption(f"분석 대상: {fb.PHASE_TITLE} — 정답(기준) 영상 대비 내 영상의 관절 각도 차이를 확인하고 피드백을 받아봅니다.")


def save_upload(uploaded, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(uploaded.getbuffer())


def fetch_link(url, dest, log=lambda m: None):
    """유튜브 등 링크에서 영상을 받아 dest에 저장한다.

    download_video.py의 다운로드 로직을 그대로 쓴다(유튜브가 403으로 막는 경우를
    피하려고 여러 클라이언트를 순서대로 시도하는 설정이 거기에 들어 있다).
    """
    from download_video import download
    log(f"영상 받는 중: {url}")
    download(url, dest.name)          # download()는 data/videos/ 아래에 저장한다
    if not dest.exists():
        raise RuntimeError("다운로드는 끝났는데 파일이 만들어지지 않았습니다.")
    log(f"받기 완료: {dest.name} ({dest.stat().st_size / 1e6:.1f}MB)")


with st.expander("📤 영상 올리고 새로 분석하기", expanded=missing_required_files() != []):
    st.caption(
        "정답 영상과 내 영상을 올린 뒤 **분석 실행**을 누르면 관절 추출부터 비교 영상까지 다시 만듭니다. "
        "영상 길이에 따라 몇 분 걸립니다 (관절 추출이 대부분)."
    )
    src_mode = st.radio(
        "영상을 어떻게 넣을까요?", ["파일 올리기", "링크 붙여넣기"],
        horizontal=True, key="src_mode",
        help="유튜브 링크를 붙여넣으면 앱이 직접 받아옵니다.")

    up_ref = up_mine = None
    link_ref = link_mine = ""
    uc1, uc2 = st.columns(2)
    if src_mode == "파일 올리기":
        with uc1:
            up_ref = st.file_uploader("정답 영상 (mp4)", type=["mp4", "mov", "avi"], key="up_ref")
        with uc2:
            up_mine = st.file_uploader("내 영상 (mp4)", type=["mp4", "mov", "avi"], key="up_mine")
    else:
        with uc1:
            link_ref = st.text_input("정답 영상 링크", key="link_ref",
                                     placeholder="https://www.youtube.com/watch?v=...")
        with uc2:
            link_mine = st.text_input("내 영상 링크", key="link_mine",
                                      placeholder="https://www.youtube.com/watch?v=...")
        st.caption(
            "유튜브를 비롯해 yt-dlp가 지원하는 사이트 링크, 직접 mp4 주소도 됩니다. "
            "비공개·연령 제한 영상은 받을 수 없습니다."
        )

    auto_detect = st.checkbox(
        "돌핀킥 구간 자동 검출 (권장)", value=True,
        help="수영 영상은 다이빙-돌핀킥-자유형-턴이 섞여 있습니다. 켜면 팔을 모아 뻗은 채 "
             "다리만 규칙적으로 차는 구간을 찾아 그 부분만 분석합니다.")
    st.markdown(
        "**분석할 구간** — 자동 검출이 엉뚱한 곳을 잡거나 원하는 구간이 따로 있으면 직접 지정하세요 "
        "(값을 넣으면 자동 검출 대신 이 구간을 씁니다)"
    )
    tc1, tc2, tc3, tc4 = st.columns(4)
    ref_start = tc1.number_input("정답 시작(초)", min_value=0.0, value=0.0, step=0.5)
    ref_end = tc2.number_input("정답 끝(초)", min_value=0.0, value=0.0, step=0.5,
                                help="0이면 끝까지")
    mine_start = tc3.number_input("내 영상 시작(초)", min_value=0.0, value=0.0, step=0.5)
    mine_end = tc4.number_input("내 영상 끝(초)", min_value=0.0, value=0.0, step=0.5,
                                 help="0이면 끝까지")

    if st.button("분석 실행", type="primary"):
        # st.empty()는 호출할 때마다 같은 자리를 덮어쓴다. st.container()에 code()를
        # 반복 호출하면 줄이 쌓일 때마다 새 블록이 계속 추가돼서 화면이 지저분해진다.
        box = st.empty()
        lines = []

        def log(msg):
            lines.append(str(msg))
            box.code("\n".join(lines[-14:]))

        if up_ref is not None:
            save_upload(up_ref, REF_VIDEO)
        if up_mine is not None:
            save_upload(up_mine, MINE_VIDEO)

        # 링크가 들어왔으면 먼저 받아온다. 받기에 실패하면 그 사실만 알리고
        # 분석으로 넘어가지 않는다 (예전 영상으로 엉뚱하게 분석되는 것을 막는다).
        download_failed = False
        for url, dest, label in [(link_ref.strip(), REF_VIDEO, "정답 영상"),
                                 (link_mine.strip(), MINE_VIDEO, "내 영상")]:
            if not url:
                continue
            try:
                with st.spinner(f"{label} 받는 중…"):
                    fetch_link(url, dest, log)
            except Exception as exc:
                st.error(f"{label}을(를) 받지 못했습니다: {exc}")
                download_failed = True

        if download_failed:
            st.info("링크가 올바른지, 비공개·연령 제한 영상은 아닌지 확인해 주세요.")
        elif not REF_VIDEO.exists() or not MINE_VIDEO.exists():
            st.error("정답 영상과 내 영상이 모두 필요합니다.")
        else:
            try:
                with st.spinner("분석 중… 관절 추출이 가장 오래 걸립니다"):
                    run_all(
                        ref_start or None, ref_end or None,
                        mine_start or None, mine_end or None,
                        log=log, auto_detect=auto_detect,
                    )
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("분석 완료! 아래 결과가 새 영상 기준으로 갱신됐습니다.")
                st.rerun()
            except Exception as exc:
                # 예외 메시지만 보여주면 원인을 알 수 없는 경우가 많다(예: KeyError는
                # 키 이름만 나온다). 어디서 났는지까지 펼쳐볼 수 있게 남긴다.
                st.error(f"분석 중단: {type(exc).__name__}: {exc}")
                with st.expander("자세한 오류 내용 (개발자용)"):
                    st.code(traceback.format_exc())

missing = missing_required_files()
if missing:
    st.info(
        "아직 분석 결과가 없습니다. 위 **영상 올리고 새로 분석하기**에서 두 영상을 올려주세요.\n\n"
        "(없는 파일: " + ", ".join(p.name for p in missing) + ")"
    )
    st.stop()

ref_norm, mine_norm, diff_df, ref_reps, mine_reps = load_data()
stats = fb.load_joint_stats()
ranked = sorted(stats.items(), key=lambda kv: -kv[1]["mean_abs"])

col_v1, col_v2 = st.columns(2)
with col_v1:
    st.subheader("정답 영상")
    if REF_VIDEO.exists():
        st.video(str(REF_VIDEO))
with col_v2:
    st.subheader("내 영상")
    if MINE_VIDEO.exists():
        st.video(str(MINE_VIDEO))

st.caption(
    "고칠 점은 아래 **📝 피드백 & 문제 장면** 탭에서 관절마다 영상과 함께 봅니다."
)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 각도 곡선 비교", "📊 차이 그래프", "🔁 반복 구간", "📝 피드백 & 문제 장면"]
)

with tab1:
    st.write(f"정답(파랑)과 내 영상(주황)의 대표 {fb.CYCLE_LABEL} 곡선을 진행률(0~100%) 기준으로 겹쳐서 봅니다.")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (title, left, right) in zip(axes.flat, PANELS):
        ax.plot(ref_norm["progress_pct"], ref_norm[left], color=COLOR_REF, lw=1.8, ls="-", label="정답-왼쪽")
        ax.plot(ref_norm["progress_pct"], ref_norm[right], color=COLOR_REF, lw=1.8, ls="--", label="정답-오른쪽")
        ax.plot(mine_norm["progress_pct"], mine_norm[left], color=COLOR_MINE, lw=1.8, ls="-", label="나-왼쪽")
        ax.plot(mine_norm["progress_pct"], mine_norm[right], color=COLOR_MINE, lw=1.8, ls="--", label="나-오른쪽")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("각도 (°)")
        ax.set_ylim(0, 190)
        ax.grid(True, color="#e8e8e4", lw=0.7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(loc="lower right", frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    st.pyplot(fig)

with tab2:
    st.write("0을 기준으로 빨강=내가 덜 굽힘(더 폄), 파랑=내가 더 굽힘. 좌우 관절은 평균했습니다.")
    fig2, axes2 = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (title, left, right) in zip(axes2.flat, PANELS):
        pct = diff_df["progress_pct"]
        vals = (diff_df[f"diff_{left}"] + diff_df[f"diff_{right}"]) / 2
        ax.axhline(0, color="#8a8a86", lw=1)
        ax.plot(pct, vals, color="#52514e", lw=1.2)
        ax.fill_between(pct, vals, 0, where=vals >= 0, color=COLOR_POS, alpha=0.55, interpolate=True)
        ax.fill_between(pct, vals, 0, where=vals < 0, color=COLOR_NEG, alpha=0.55, interpolate=True)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("차이 (°)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, axis="y", color="#e8e8e4", lw=0.7)
    fig2.tight_layout()
    st.pyplot(fig2)

    st.subheader("관절별 요약")
    summary_rows = [
        {
            "관절": fb.JOINT_LABELS[name],
            "다음 행동": fb.ACTION_VERBS[fb.joint_group(name)]["smaller" if s["worst_diff"] < 0 else "larger"],
            "평균|차이|(°)": round(s["mean_abs"], 1),
            "최대 차이(°)": s["worst_diff"],
            "발생 지점(%)": s["worst_pct"],
        }
        for name, s in ranked
    ]
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

with tab3:
    st.write(f"{fb.CYCLE_LABEL} 1회 단위로 나눈 구간입니다. 구간 길이(duration_sec)가 고르면 리듬이 안정적이라는 뜻입니다.")
    c1, c2 = st.columns(2)
    with c1:
        st.caption(f"정답 영상 — {fb.CYCLE_LABEL} {len(ref_reps)}회")
        st.dataframe(ref_reps, hide_index=True, width="stretch")
    with c2:
        st.caption(f"내 영상 — {fb.CYCLE_LABEL} {len(mine_reps)}회")
        st.dataframe(mine_reps, hide_index=True, width="stretch")

with tab4:
    st.subheader("문제 장면 보기")
    st.write(
        "실제 영상은 정답과 내 영상의 촬영 카메라 각도가 서로 달라서(한쪽은 옆에서, "
        "한쪽은 정면 가까이에서), 자세 차이인지 카메라 각도 차이인지 헷갈립니다. 그래서 "
        "카메라 위치와 무관하게 그 사람의 몸(엉덩이-어깨 라인) 기준으로 다시 그린 졸라맨을 "
        "**정답(왼쪽 패널)·내 동작(오른쪽 패널)** 으로 나눠서 보여줍니다. **정면**은 좌우 롤링과 "
        "몸통 라인을, **측면**은 다리가 위아래로 차는 발차기 움직임을 잘 보여줍니다."
    )
    st.caption("🟤 몸통 · 🔵 팔 · 🔴 다리 · ⚪ 머리 — 색은 정답/나 구분이 아니라 몸의 부위를 뜻합니다.")

    pose_files_missing = [p for p in (REF_POSE_NORM, MINE_POSE_NORM) if not p.exists()]
    if pose_files_missing:
        st.warning(
            "정면/측면 비교에 필요한 파일이 없습니다. 먼저 아래를 실행하세요:\n\n"
            "- python src/normalize_pose_3d.py --landmarks output/reference/pose_landmarks.csv "
            "--reps output/reference/reps.csv --output output/reference/normalized_pose_mean.csv\n"
            "- python src/normalize_pose_3d.py --landmarks output/mine/pose_landmarks.csv "
            "--reps output/mine/reps.csv --output output/mine/normalized_pose_mean.csv\n\n"
            + "\n".join(f"- (없음) {p.relative_to(ROOT)}" for p in pose_files_missing)
        )
    else:
        st.subheader(f"전체 {fb.CYCLE_LABEL} 비교 영상 (자동 재생)")
        st.caption(
            f"대표 {fb.CYCLE_LABEL} **1회**를 {CYCLE_LOOPS}번 이어 붙인 영상입니다 — 실제로 "
            f"{CYCLE_LOOPS}번 찬 기록이 아니라, 연속 동작으로 보이도록 같은 사이클을 반복 재생한 것입니다. "
            "정답과 내 동작을 같은 진행률끼리 비교하려면 사이클 1회로 정규화해야 하기 때문입니다."
        )
        vc1, vc2 = st.columns(2)
        with vc1:
            st.caption("정면")
            if FRONT_COMPARE_VIDEO.exists():
                st.video(str(FRONT_COMPARE_VIDEO), autoplay=True, loop=True, muted=True)
            else:
                st.info("src/render_pose_compare.py를 실행하면 이 영상이 만들어집니다.")
        with vc2:
            st.caption("측면")
            if SIDE_COMPARE_VIDEO.exists():
                st.video(str(SIDE_COMPARE_VIDEO), autoplay=True, loop=True, muted=True)
            else:
                st.info("src/render_pose_compare.py를 실행하면 이 영상이 만들어집니다.")

        st.divider()
        st.subheader("특정 관절의 최대 차이 순간")
        joint_options = [name for name, _ in ranked]
        selected = st.selectbox(
            "확인할 관절",
            joint_options,
            format_func=lambda n: (
                f"{fb.JOINT_LABELS[n]} — 평균 {stats[n]['mean_abs']:.0f}° / "
                f"최대 {stats[n]['worst_diff']:+.0f}° ({stats[n]['worst_pct']:.0f}% 지점)"
            ),
        )
        s = stats[selected]
        pct = s["worst_pct"]

        ref_pose_rows = load_pose_rows_cached(str(REF_POSE_NORM))
        mine_pose_rows = load_pose_rows_cached(str(MINE_POSE_NORM))
        # 기준값은 시퀀스 전체를 보고 한 번에 정해야 한다 (한 프레임만 보면 측면
        # 화면좌표의 중심/배율을 잡을 수 없고, 정면은 축이 떨림 - render_pose_compare
        # .frames_for_view 참고). 정면/측면이 서로 다른 기준을 쓰므로 뷰별로 준비한다.
        ref_i = nearest_pose_row_index(ref_pose_rows, pct)
        mine_i = nearest_pose_row_index(mine_pose_rows, pct)
        ref_pose_row, mine_pose_row = ref_pose_rows[ref_i], mine_pose_rows[mine_i]

        a_name, b_name, c_name = ca.ANGLE_DEFINITIONS[selected]
        highlight = {a_name, b_name, c_name}

        # 숫자는 "여러 사이클 평균"(피드백/요약 표와 같은 값), 그림은 "대표 사이클 1개"의
        # 실제 좌표다. 일부러 다른 것을 쓴다 - 숫자는 안정성이, 그림은 실제 동작다움이
        # 중요하기 때문(normalize_pose_3d.py 설명 참고). 그래서 둘이 최대 20~30도까지
        # 다를 수 있어, 아래에 두 값을 나란히 보여줘서 혼동을 막는다.
        ref_norm_val = float(ref_norm.loc[ref_norm["progress_pct"] == pct, selected].iloc[0])
        mine_norm_val = float(mine_norm.loc[mine_norm["progress_pct"] == pct, selected].iloc[0])
        ref_shot = angle_at_pose_row(ref_pose_row, selected)
        mine_shot = angle_at_pose_row(mine_pose_row, selected)

        st.write(f"**{fb.action_phrase(selected, s)}**")
        st.caption(
            f"**여러 사이클 평균(피드백 기준 숫자)** — 정답 {ref_norm_val:.0f}° vs 나 {mine_norm_val:.0f}° "
            f"(차이 {mine_norm_val - ref_norm_val:+.0f}°) · 진행률 {pct:.0f}% 지점"
        )
        if ref_shot is not None and mine_shot is not None:
            st.caption(
                f"**아래 그림(대표 사이클 1개)** — 정답 {ref_shot:.0f}° vs 나 {mine_shot:.0f}° "
                f"(차이 {mine_shot - ref_shot:+.0f}°) · 그림은 실제로 있었던 동작 하나라 위 평균과 다를 수 있음 · "
                "빨간 굵은 테두리 = 지금 보고 있는 각도"
        )

        pc1, pc2 = st.columns(2)
        for col, view, label in [(pc1, "front", "정면"), (pc2, "side", "측면 (몸의 파동이 보이는 뷰)")]:
            v_ref = frames_for_view_cached(str(REF_POSE_NORM), view)
            v_mine = frames_for_view_cached(str(MINE_POSE_NORM), view)
            with col:
                st.caption(label)
                st.image(
                    render_pose_frame(ref_pose_row, mine_pose_row, pct, view, highlight,
                                       v_ref[ref_i], v_mine[mine_i]),
                    channels="BGR",
                )

        with st.expander("원본 카메라 화면 (참고용 — 촬영 각도가 달라 자세 비교에는 적합하지 않음)"):
            scene_files_missing = [
                p for p in (REF_LANDMARKS, MINE_LANDMARKS, REF_VIDEO, MINE_VIDEO) if not p.exists()
            ]
            if scene_files_missing:
                st.warning(
                    "원본 장면 비교에 필요한 파일이 없습니다:\n\n"
                    + "\n".join(f"- {p.relative_to(ROOT)}" for p in scene_files_missing)
                )
            else:
                ref_landmarks_df, mine_landmarks_df = load_landmarks()
                ref_rep = pick_representative_rep(ref_reps)
                mine_rep = pick_representative_rep(mine_reps)

                ref_row = nearest_landmark_row(ref_landmarks_df, frame_for_progress(ref_rep, pct))
                mine_row = nearest_landmark_row(mine_landmarks_df, frame_for_progress(mine_rep, pct))

                ref_angle = angle_at_row(ref_row, selected)
                mine_angle = angle_at_row(mine_row, selected)

                def angle_caption(label, angle):
                    return f"{label} (이 장면 실측: {angle:.0f}°)" if angle is not None else label

                st.caption(
                    f"정답: {fb.CYCLE_LABEL} #{int(ref_rep['rep_index'])} (총 {len(ref_reps)}회 중) · "
                    f"내 영상: {fb.CYCLE_LABEL} #{int(mine_rep['rep_index'])} (총 {len(mine_reps)}회 중) · "
                    f"진행률 {pct:.0f}% 지점 · 실측값은 프레임 하나만의 값이라 위 평균값과 다소 다를 수 있음"
                )

                ref_scene = read_video_frame(str(REF_VIDEO), int(ref_row["frame"]))
                mine_scene = read_video_frame(str(MINE_VIDEO), int(mine_row["frame"]))

                c3, c4 = st.columns(2)
                with c3:
                    st.caption(angle_caption("정답 영상", ref_angle))
                    if ref_scene is not None:
                        st.image(draw_skeleton(ref_scene, ref_row, highlight))
                with c4:
                    st.caption(angle_caption("내 영상", mine_angle))
                    if mine_scene is not None:
                        st.image(draw_skeleton(mine_scene, mine_row, highlight))

    st.divider()
    # 차이가 큰 관절마다 "그 관절 하나에만 동그라미가 있는" 영상 + 그 관절 피드백.
    # 동그라미를 한 영상에 몰아 그리면 어디를 볼지 알기 어려워서 관절당 1개로 나눈다.
    st.subheader("🎯 관절별 집중 비교")
    picked = focus_joints()
    st.caption(
        f"정답과 평균 {FOCUS_MIN_DIFF:.0f}° 이상 차이나는 관절 **{len(picked)}곳**입니다. "
        "영상마다 고칠 관절 한 곳에만 노란 동그라미를 표시했고, 동그라미는 **내 동작(오른쪽)** 에만 있습니다."
    )

    llm = load_llm_feedback()
    if not llm:
        st.caption(
            "💡 `python src/generate_feedback_llm.py`를 실행하면 아래 문장이 코치가 말하듯 "
            "자연스러운 문장으로 바뀝니다 (Claude API 키 필요)."
        )

    for rank, (joint, s) in enumerate(picked, start=1):
        coaching = llm.get("joints", {}).get(joint) if llm else None
        headline = coaching["headline"] if coaching else fb.action_phrase(joint, s)
        st.markdown(f"### {rank}. {fb.JOINT_LABELS[joint]} — {headline}")
        path = focus_video_path(joint)
        if path.exists():
            st.video(str(path), autoplay=True, loop=True, muted=True)
        else:
            st.info("src/render_pose_compare.py를 실행하면 이 영상이 만들어집니다.")

        if coaching:
            st.markdown(coaching["detail"])
        else:
            group = fb.joint_group(joint)
            direction = "smaller" if s["worst_diff"] < 0 else "larger"
            st.markdown(f"**무엇이 다른가** — {fb.JOINT_HINTS[group][direction]}")
        st.caption(
            f"{fb.CYCLE_LABEL} 내내 평균 {s['mean_abs']:.0f}° 차이 · "
            f"진행률 {s['worst_pct']:.0f}% 지점에서 최대 {s['worst_diff']:+.0f}° 차이 "
            f"({'정답보다 더 굽힘' if s['worst_diff'] < 0 else '정답보다 덜 굽힘'})"
        )
        st.divider()

    # 고칠 점만 나열하면 "지금 하고 있는 게 맞는 방향인지" 알 수 없어서,
    # 정답과 가장 비슷하게 맞고 있는 관절도 함께 보여준다.
    st.subheader("✅ 이미 잘 유지되고 있는 부분")
    best_cols = st.columns(3)
    best_ranked = sorted(stats.items(), key=lambda kv: kv[1]["mean_abs"])[:3]
    for rank, (col, (name, s)) in enumerate(zip(best_cols, best_ranked)):
        with col:
            st.markdown(f"**{fb.JOINT_LABELS[name]}**")
            st.write("정답과 비슷한 각도를 잘 유지하고 있어요")
            st.caption(f"평균 {s['mean_abs']:.0f}° 차이 ({len(stats)}개 관절 중 {rank + 1}번째로 작음)")

    if llm and llm.get("encouragement"):
        st.success(llm["encouragement"])

    st.caption(
        "※ 원인 후보는 일반적으로 알려진 참고 힌트이며 확정 진단이 아닙니다. "
        "각도는 카메라 1대로 추정한 좌표 기반이라 오차가 있을 수 있습니다."
        + (f" 문장 생성: Claude {llm['model']}." if llm else "")
    )
