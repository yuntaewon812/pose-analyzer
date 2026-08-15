"""
3D 관절 좌표(wx, wy, wz)를 몸통 기준의 정면/측면 2D 스틱맨 좌표로 바꾸는 공용 도구.

[왜 필요한가]
정답 영상과 내 영상은 촬영 카메라 각도가 다릅니다. 카메라가 찍은 좌표축을 그대로
쓰면 자세 차이인지 카메라 각도 차이인지 구분할 수 없습니다. 이 모듈은 매 프레임
엉덩이-어깨 라인으로 그 사람만의 "위(up)/옆(right)/앞(forward)" 축을 새로 계산해서
(그람-슈미트 직교화), 그 축 기준으로 모든 관절을 투영합니다. 수영 중 몸이 좌우로
롤링해도 그 롤링은 그림에 반영되면서, 카메라가 어느 각도에서 찍었는지는 결과에서
상당 부분 빠집니다 (calculate_angles.py가 각도 계산에 3D 좌표를 쓰는 것과 같은
이유). 몸통 길이로 나눠서 스케일도 1로 맞추므로, 두 사람의 체격 차이도 그림에서
지워지고 "자세(각도/모양)"만 남습니다.

[정면 하나로는 부족한 이유 - 정면 + 측면 둘 다 그리는 이유]
위/옆(right) 축만 보는 "정면" 투영은 롤링이나 발차기 좌우 폭은 잘 보이지만, 팔이
앞뒤로 뻗는 스트로크의 핵심 움직임은 화면과 수직인 축이라 거의 안 보입니다.
반대로 위/앞(forward) 축만 보는 "측면" 투영은 그 앞뒤 뻗음은 잘 보이지만 좌우
롤링/폭은 잘 안 보입니다. 그래서 이 모듈은 두 투영을 모두 제공하고, 호출하는
쪽(render_pose_compare.py, app.py)이 필요에 따라 골라 쓰거나 둘 다 나란히
보여줍니다. "front"는 사람을 마주 본 모습(머리가 위)으로, "side"는 실제 수영
측면 샷처럼 몸이 눕는 모습(머리가 왼쪽)으로 그립니다.

[정답/내 동작을 한 화면에 겹치지 않는 이유]
처음엔 두 사람을 같은 화면에 겹쳐 그렸는데, 특히 차이가 큰 구간에서는 두 뼈대가
서로 가려서 오히려 안 보기 어려웠습니다. 그래서 이 모듈은 스틱맨 하나만 그리는
일을 맡고, 정답/나를 나란히 배치하는 건 호출하는 쪽(render_pose_compare.py의
render_panel/render_frame)이 담당합니다. 두 패널이 분리되니 색을 "정답이냐
나냐"가 아니라 "몸의 어느 부위냐"(머리/몸통/팔/다리)를 구분하는 데 쓸 수 있게
됐고, 그래서 관절점만 이었을 때보다 졸라맨에 훨씬 가깝게 알아보기 쉬워졌습니다.

render_pose_compare.py(전체 스트로크 비교 영상)와 app.py(문제 장면 보기 탭)가
같은 방식으로 그려야 두 결과물이 서로 어긋나지 않으므로, 계산을 이 모듈 하나로
모아 공유합니다.
"""
import math

import cv2
import numpy as np

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
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),   # 어깨-팔
    (11, 23), (12, 24), (23, 24),                        # 몸통
    (23, 25), (25, 27), (24, 26), (26, 28),              # 다리
    (27, 29), (29, 31), (28, 30), (30, 32),              # 발
]

# 졸라맨처럼 몸 부위별로 색을 다르게 줘서 머리/팔/다리가 한눈에 구분되게 한다
# (BGR 순서). 정답/내 동작은 이제 같은 화면에 겹치지 않고 패널을 나눠서 보여주므로,
# 색을 "누구 것인지"가 아니라 "몸의 어느 부위인지" 표시하는 데 쓸 수 있게 됐다.
TORSO_BGR = (90, 85, 80)          # 몸통 - 짙은 회색
ARM_BGR = (219, 152, 52)          # 팔 - 파랑
LEG_BGR = (60, 76, 231)           # 다리 - 주황빛 빨강
HEAD_FILL_BGR = (222, 234, 244)   # 머리 - 살구색 원
HEAD_OUTLINE_BGR = (90, 85, 80)
OUTLINE_BGR = (30, 30, 235)       # 강조용 - "지금 보고 있는 각도"를 두드러지게 (밝은 빨강)

CONNECTION_KIND = {
    (11, 12): "torso", (11, 23): "torso", (12, 24): "torso", (23, 24): "torso",
    (11, 13): "arm", (13, 15): "arm", (12, 14): "arm", (14, 16): "arm",
    (23, 25): "leg", (25, 27): "leg", (24, 26): "leg", (26, 28): "leg",
    (27, 29): "leg", (29, 31): "leg", (28, 30): "leg", (30, 32): "leg",
}
KIND_COLOR = {"torso": TORSO_BGR, "arm": ARM_BGR, "leg": LEG_BGR}

LANDMARK_KIND = {}
for _n in ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"):
    LANDMARK_KIND[_n] = "torso"
for _n in ("LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST"):
    LANDMARK_KIND[_n] = "arm"
for _n in ("LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
           "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"):
    LANDMARK_KIND[_n] = "leg"

# 몸 부위 범례 - (영문 라벨, cv2용 BGR, 웹/matplotlib용 RGB 헥스). cv2 기본 폰트는
# 한글을 못 그려서 영상에 굽는 범례는 영문을 쓰고, app.py 등 한글이 되는 곳에서는
# 이 헥스값으로 "머리/몸통/팔/다리" 한글 범례를 만든다 (색 정의를 한 곳에서만 관리).
LEGEND_ITEMS = [
    ("head", HEAD_FILL_BGR, "#dee8f2"),
    ("torso", TORSO_BGR, "#50555a"),
    ("arm", ARM_BGR, "#3498db"),
    ("leg", LEG_BGR, "#e74c3c"),
]


def draw_legend(canvas, x=16, y=None):
    """캔버스 좌하단에 머리/몸통/팔/다리 색상 범례를 작게 그린다 (cv2 기본 폰트는
    한글을 지원하지 않아 영문 라벨을 쓴다)."""
    if y is None:
        y = canvas.shape[0] - 14 - (len(LEGEND_ITEMS) - 1) * 22
    for i, (label, color, _hex) in enumerate(LEGEND_ITEMS):
        cy = y + i * 22
        cv2.circle(canvas, (x + 6, cy), 7, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, label, (x + 20, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1, cv2.LINE_AA)

# 정답=파랑 / 나=주황 - 스틱맨 자체는 더 이상 이 색을 쓰지 않지만, 패널 라벨
# 텍스트 색으로는 계속 써서 "이건 정답 패널, 이건 내 패널"을 표시한다.
COLOR_REF_BGR = (214, 120, 42)
COLOR_MINE_BGR = (52, 104, 235)

CANVAS_SIZE = 640
PIXELS_PER_UNIT = 150   # 몸통(엉덩이~어깨) 길이를 1.0으로 맞춘 뒤, 이만큼의 픽셀로 그린다
# "front"는 서 있는 모습(머리 위/다리 아래)이라 원점을 살짝 위쪽에 둬서 다리가 그려질
# 아래쪽 공간을 더 확보한다. "side"는 몸이 눕는 모습(머리 왼쪽/발 오른쪽)이라 위아래
# 공간이 거의 대칭으로 필요하므로 원점을 세로 중앙에 둔다.
ORIGIN_PX = {
    "front": (CANVAS_SIZE // 2, int(CANVAS_SIZE * 0.40)),
    "side": (CANVAS_SIZE // 2, CANVAS_SIZE // 2),
}


def get_xyz(row, name):
    x, y, z = row.get(f"{name}_wx", ""), row.get(f"{name}_wy", ""), row.get(f"{name}_wz", "")
    if x == "" or y == "" or z == "":
        return None
    return np.array([float(x), float(y), float(z)])


def body_frame(row):
    """엉덩이 중심을 원점으로, 몸통(엉덩이->어깨)을 up, 엉덩이 라인을 right,
    나머지 한 축(가슴이 향하는 방향)을 forward로 잡는다 (셋 다 서로 직각)."""
    lh, rh = get_xyz(row, "LEFT_HIP"), get_xyz(row, "RIGHT_HIP")
    ls, rs = get_xyz(row, "LEFT_SHOULDER"), get_xyz(row, "RIGHT_SHOULDER")
    if lh is None or rh is None or ls is None or rs is None:
        return None

    mid_hip = (lh + rh) / 2
    mid_shoulder = (ls + rs) / 2
    up_vec = mid_shoulder - mid_hip
    torso_length = np.linalg.norm(up_vec)
    if torso_length < 1e-6:
        return None
    up = up_vec / torso_length

    hip_right = rh - lh
    hip_right = hip_right - np.dot(hip_right, up) * up  # 그람-슈미트: up 성분 제거
    side_len = np.linalg.norm(hip_right)
    if side_len < 1e-6:
        return None
    right = hip_right / side_len

    forward = np.cross(right, up)

    return mid_hip, up, right, forward, torso_length


def get_screen_xy(row, name):
    x, y = row.get(f"{name}_x", ""), row.get(f"{name}_y", "")
    if x == "" or y == "":
        return None
    return float(x), float(y)


def screen_frame(rows):
    """화면 좌표로 측면 스틱맨을 그리기 위한 기준값(중심/배율/좌우방향)을 시퀀스 전체에서 한 번 정한다.

    [왜 화면 좌표로 그리나 - 3D에는 파동이 없다]
    MediaPipe의 world 좌표는 엉덩이 중점이 원점이라 "엉덩이가 위아래로 출렁이는"
    움직임이 아예 없고, 그림을 몸통 기준축으로 그리면 몸통 기울기까지 0이 됩니다.
    돌핀킥은 그 둘이 만드는 파동이 본질이라, 3D로 그리면 뻣뻣한 판자가 됩니다.
    이 영상들은 카메라가 고정된 정측면이고 선수가 화면 평면 안에서 움직이므로,
    화면 좌표가 곧 충실한 측면 뷰입니다.

    [비교하려면 맞춰야 하는 세 가지]
      - 중심: 사이클 동안의 엉덩이 평균 위치를 원점으로 (선수가 화면 어디에 있든 무관하게)
      - 배율: 사이클 동안의 평균 몸통 길이를 1로 (카메라 거리/체격 차이 제거)
      - 좌우: 머리가 항상 화면 왼쪽을 향하도록 (영상에 왕복 랩이 섞여 있어, 방향이
        다른 사이클을 그대로 겹치면 좌우가 뒤집힌 채 비교된다)
    배율을 매 프레임이 아니라 사이클 평균으로 잡는 것이 중요합니다 - 매 프레임
    정규화하면 몸이 기울며 짧아 보이는 변화까지 지워져서 파동이 다시 사라집니다.
    """
    hips, shoulders, noses, ankles = [], [], [], []
    for r in rows:
        lh, rh = get_screen_xy(r, "LEFT_HIP"), get_screen_xy(r, "RIGHT_HIP")
        ls, rs = get_screen_xy(r, "LEFT_SHOULDER"), get_screen_xy(r, "RIGHT_SHOULDER")
        if not (lh and rh and ls and rs):
            continue
        hips.append(((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2))
        shoulders.append(((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2))
        nose = get_screen_xy(r, "NOSE")
        la, ra = get_screen_xy(r, "LEFT_ANKLE"), get_screen_xy(r, "RIGHT_ANKLE")
        if nose:
            noses.append(nose[0])
        if la and ra:
            ankles.append((la[0] + ra[0]) / 2)
    if not hips:
        return None

    hips = np.array(hips)
    shoulders = np.array(shoulders)
    center = hips.mean(axis=0)
    scale = float(np.linalg.norm(shoulders - hips, axis=1).mean())
    if scale < 1e-9:
        return None
    # 머리가 발보다 오른쪽에 있으면 좌우를 뒤집어, 항상 "머리가 왼쪽"으로 통일한다.
    flip = -1.0 if (noses and ankles and np.mean(noses) > np.mean(ankles)) else 1.0
    return center, scale, flip


def project_screen(row, sframe):
    """화면 좌표를 그리기용 (x, y)로 바꾼다. 몸통 길이를 1로 맞추고, 머리가 왼쪽을
    향하게 하며, 화면 y가 아래로 증가하므로 위아래를 뒤집는다."""
    center, scale, flip = sframe
    points = {}
    for name in LANDMARK_NAMES:
        p = get_screen_xy(row, name)
        if p is None:
            continue
        points[name] = (flip * (p[0] - center[0]) / scale,
                        -(p[1] - center[1]) / scale)
    return points


AXIS_SMOOTH_WINDOW = 9


def stabilize_frames(rows, window=AXIS_SMOOTH_WINDOW):
    """연속된 프레임들의 몸통 기준 축을 시간적으로 안정화해서 돌려준다.

    [왜 필요한가 - 옆에서 찍으면 좌우 축이 노이즈가 된다]
    body_frame()은 좌우 축(right)을 "왼쪽 엉덩이 -> 오른쪽 엉덩이" 방향으로 잡습니다.
    그런데 수영을 정확히 옆에서 찍으면 그 좌우 방향이 카메라의 깊이(z) 방향과 거의
    겹칩니다. MediaPipe에서 z는 가장 부정확한 성분이라, 화면상 거의 겹쳐 보이는 두
    엉덩이의 앞뒤 관계가 프레임마다 뒤바뀝니다(실측: 엉덩이 폭이 48% 요동, right
    축이 프레임 사이 최대 93도 회전). 그러면 up(몸통 축)은 멀쩡한데도 right/forward가
    춤을 춰서, 스틱맨이 제자리에서 비틀리는 것처럼 보입니다.

    실제 사람 몸은 1/30초 만에 90도씩 돌지 않으므로 이런 급회전은 전부 추정 오차입니다.
    그래서 up은 프레임별 값을 그대로 믿고(안정적임), right만 시간축으로 다듬습니다:
      1) 부호 정렬 - right가 갑자기 정반대로 뒤집힌 프레임은 좌우 엉덩이가 뒤바뀌어
         검출된 것이므로, 앞 프레임과 방향이 반대면 뒤집어 맞춰준다. (먼저 정렬하지
         않고 평균 내면 서로 상쇄돼 축이 사라진다.)
      2) 이동평균 - 남은 떨림을 창(window) 크기만큼 평균해서 없앤다. 진짜 몸통 롤링은
         이보다 훨씬 느리므로 살아남는다.
      3) 재직교화 - 다듬은 right에서 up 성분을 빼고 정규화한 뒤 forward를 다시 계산해,
         세 축이 서로 직각인 상태를 유지한다.
    """
    frames = [body_frame(r) for r in rows]
    valid_idx = [i for i, f in enumerate(frames) if f is not None]
    if len(valid_idx) < 2:
        return frames

    # 1) 부호 정렬
    aligned = {}
    prev = None
    for i in valid_idx:
        right = frames[i][2]
        if prev is not None and np.dot(right, prev) < 0:
            right = -right
        aligned[i] = right
        prev = right

    # 2) 이동평균 + 3) 재직교화
    half = window // 2
    out = list(frames)
    for pos, i in enumerate(valid_idx):
        neighbours = [aligned[valid_idx[p]]
                      for p in range(max(0, pos - half), min(len(valid_idx), pos + half + 1))]
        smoothed = np.mean(neighbours, axis=0)

        mid_hip, up, orig_right, _forward, torso_length = frames[i]
        smoothed = smoothed - np.dot(smoothed, up) * up
        norm = np.linalg.norm(smoothed)
        right = orig_right if norm < 1e-9 else smoothed / norm  # 평균이 상쇄되면 원래 값 사용
        out[i] = (mid_hip, up, right, np.cross(right, up), torso_length)
    return out


# 정면(위/옆 축)만 보면 스트로크의 핵심 동작인 "팔이 앞뒤로 뻗는 움직임"이 (그 축은
# 화면과 수직이라) 거의 안 보이고, 반대로 측면(위/앞 축)만 보면 좌우 롤링이나 발차기
# 폭이 잘 안 보인다. 그래서 둘 다 그린다.
# - "front": 사람을 마주 보는 시점이라 위(up)=화면 위(머리가 위), 옆(right)=화면 가로.
# - "side": 실제 수영 측면 샷처럼 몸이 눕는 모습으로 보여준다. 위(up, 머리 방향)를
#   화면 "가로" 축으로 돌리고 부호를 반대로 줘서 머리가 화면 왼쪽을 향하게 하고,
#   앞(forward, 가슴이 향하는 방향=몸의 깊이)을 화면 "세로" 축으로 쓴다.
VIEW_AXES = {
    "front": (("right", 1), ("up", 1)),
    "side": (("up", -1), ("forward", 1)),
}


def project_points(row, frame, view="front"):
    """3D 좌표를, 몸통 길이를 1로 맞춘 (x, y) 2D 좌표로 투영한다 (몸 크기 차이 제거)."""
    mid_hip, up, right, forward, torso_length = frame
    axis_lookup = {"up": up, "right": right, "forward": forward}
    (x_name, x_sign), (y_name, y_sign) = VIEW_AXES[view]
    x_axis, y_axis = axis_lookup[x_name], axis_lookup[y_name]

    points = {}
    for name in LANDMARK_NAMES:
        p = get_xyz(row, name)
        if p is None:
            continue
        rel = p - mid_hip
        points[name] = (
            x_sign * float(np.dot(rel, x_axis)) / torso_length,
            y_sign * float(np.dot(rel, y_axis)) / torso_length,
        )
    return points


def to_pixel(x2d, y2d, view="front"):
    ox, oy = ORIGIN_PX[view]
    return (
        ox + int(round(x2d * PIXELS_PER_UNIT)),
        oy - int(round(y2d * PIXELS_PER_UNIT)),  # 위(head)쪽 성분이 화면 위쪽으로 가도록 y 반전
    )


def blank_canvas(view="front"):
    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), 250, dtype=np.uint8)
    _, oy = ORIGIN_PX[view]
    cv2.line(canvas, (0, oy), (CANVAS_SIZE, oy), (225, 225, 225), 1, cv2.LINE_AA)
    return canvas


# 집중 표시용 동그라미 색 (BGR). 뼈대 색(회색/파랑/빨강)과 겹치지 않는 노란 계열을
# 써서, 어느 관절을 보라는 표시인지 한눈에 들어오게 한다.
FOCUS_CIRCLE_BGR = (0, 200, 255)


def draw_focus_circle(canvas, points, name, view="front", radius_units=0.40):
    """지정한 관절 하나에 눈에 띄는 동그라미를 그린다.

    [왜 한 영상에 하나만 그리나]
    차이가 큰 관절이 여러 개일 때 동그라미를 한꺼번에 그리면 어디를 봐야 할지
    오히려 알기 어려워집니다. 그래서 관절 하나당 영상 하나를 만들고, 각 영상에는
    동그라미를 딱 하나만 그립니다 (render_pose_compare.render_focus_videos 참고).
    """
    if name not in points:
        return
    px = to_pixel(*points[name], view)
    r = max(int(round(radius_units * PIXELS_PER_UNIT)), 16)
    cv2.circle(canvas, px, r + 3, (255, 255, 255), 6, cv2.LINE_AA)  # 흰 테두리로 배경과 분리
    cv2.circle(canvas, px, r, FOCUS_CIRCLE_BGR, 4, cv2.LINE_AA)


def draw_head(canvas, points, view="front"):
    """머리를 원으로 그린다 (졸라맨처럼) - 코 위치를 중심으로, 귀 사이 거리로 크기를 잡는다."""
    if "NOSE" not in points:
        return
    if "LEFT_EAR" in points and "RIGHT_EAR" in points:
        ex1, ey1 = points["LEFT_EAR"]
        ex2, ey2 = points["RIGHT_EAR"]
        radius_units = max(math.hypot(ex1 - ex2, ey1 - ey2) * 0.85, 0.12)
    else:
        radius_units = 0.16
    px = to_pixel(*points["NOSE"], view)
    radius_px = max(int(round(radius_units * PIXELS_PER_UNIT)), 8)
    cv2.circle(canvas, px, radius_px, HEAD_FILL_BGR, -1, cv2.LINE_AA)
    cv2.circle(canvas, px, radius_px, HEAD_OUTLINE_BGR, 2, cv2.LINE_AA)


def draw_stick_figure(canvas, points, view="front", highlight_names=None, thickness=6, radius=8):
    """졸라맨처럼 머리(원)/몸통(회색)/팔(파랑)/다리(주황)를 구분해서 그린다.
    highlight_names에 양끝이 모두 들어있는 선/점은 굵은 빨간 테두리로 "지금 보고
    있는 각도"를 강조한다."""
    highlight_names = highlight_names or set()

    draw_head(canvas, points, view)

    for a_idx, b_idx in POSE_CONNECTIONS:
        a_name, b_name = LANDMARK_NAMES[a_idx], LANDMARK_NAMES[b_idx]
        if a_name not in points or b_name not in points:
            continue
        color = KIND_COLOR[CONNECTION_KIND[(a_idx, b_idx)]]
        is_hl = a_name in highlight_names and b_name in highlight_names
        pa, pb = to_pixel(*points[a_name], view), to_pixel(*points[b_name], view)
        if is_hl:
            cv2.line(canvas, pa, pb, OUTLINE_BGR, thickness + 6, cv2.LINE_AA)
        else:
            cv2.line(canvas, pa, pb, color, thickness, cv2.LINE_AA)

    for name, (x2d, y2d) in points.items():
        kind = LANDMARK_KIND.get(name)
        if kind is None:
            continue  # 손가락/눈/입 등 세부 랜드마크는 점을 찍지 않아 뼈대를 단순하게 유지
        is_hl = name in highlight_names
        px = to_pixel(x2d, y2d, view)
        if is_hl:
            cv2.circle(canvas, px, radius + 6, OUTLINE_BGR, -1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, px, radius, KIND_COLOR[kind], -1, cv2.LINE_AA)
