"""
관절 좌표 -> 관절 각도 계산 (프로젝트1의 calculate_angles.py를 인자 기반 + 3D로 확장)

[프로젝트1과의 차이 - 2D가 아니라 3D 좌표를 쓴다]
화면 기준 (x, y) 좌표로 각도를 재면, 카메라를 어느 각도에서 찍었는지에 따라
같은 자세라도 다른 각도로 계산됩니다 (예: 정면 촬영에서는 발차기가 카메라
쪽/반대쪽으로 움직여서 거의 안 보이지만, 옆에서 찍으면 크게 보임).

MediaPipe는 화면 좌표 외에 pose_world_landmarks라는 3D 좌표도 함께 줍니다.
이건 카메라가 찍은 "그림자"가 아니라 모델이 추정한 실제 3D 공간 속 위치라서
(엉덩이 중심 기준, 미터 단위), 촬영 각도가 달라도 상대적으로 비슷한 각도가
나옵니다. 그래서 이 버전은 x,y 대신 wx,wy,wz(3D)로 벡터 각도를 계산합니다.

visibility 필터, "세 점으로 각도 정의" 개념은 프로젝트1과 동일합니다.
입출력 경로는 인자로 받아서 정답 영상 / 내 영상 각각의 CSV에 재사용합니다.

사용법:
    python src/calculate_angles.py --input output/reference/pose_landmarks.csv --output output/reference/joint_angles.csv
    python src/calculate_angles.py --input output/mine/pose_landmarks.csv      --output output/mine/joint_angles.csv
"""
import argparse
import csv
import math
from pathlib import Path

# 계산할 각도 정의: 이름 -> (점A, 가운데점B, 점C). 각도는 항상 B에서 측정.
ANGLE_DEFINITIONS = {
    "left_elbow":     ("LEFT_SHOULDER",  "LEFT_ELBOW",     "LEFT_WRIST"),
    "right_elbow":    ("RIGHT_SHOULDER", "RIGHT_ELBOW",    "RIGHT_WRIST"),
    "left_shoulder":  ("LEFT_ELBOW",     "LEFT_SHOULDER",  "LEFT_HIP"),
    "right_shoulder": ("RIGHT_ELBOW",    "RIGHT_SHOULDER", "RIGHT_HIP"),
    "left_hip":       ("LEFT_SHOULDER",  "LEFT_HIP",       "LEFT_KNEE"),
    "right_hip":      ("RIGHT_SHOULDER", "RIGHT_HIP",      "RIGHT_KNEE"),
    "left_knee":      ("LEFT_HIP",       "LEFT_KNEE",      "LEFT_ANKLE"),
    "right_knee":     ("RIGHT_HIP",      "RIGHT_KNEE",     "RIGHT_ANKLE"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="pose_landmarks.csv 경로")
    p.add_argument("--output", required=True, help="joint_angles.csv 저장 경로")
    p.add_argument("--min-vis", type=float, default=0.5, help="이 값보다 낮은 visibility가 섞이면 각도를 비움")
    return p.parse_args()


def calculate_angle(a, b, c):
    """세 점 a, b, c(각 (x,y,z) 3D 좌표)가 이루는 각도(도 단위)를 반환. b가 꼭짓점."""
    ba = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])

    dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    len_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2)
    len_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2)

    if len_ba == 0 or len_bc == 0:
        return None

    cos_angle = max(-1.0, min(1.0, dot / (len_ba * len_bc)))
    return math.degrees(math.acos(cos_angle))


def get_point(row, name):
    """CSV의 한 행에서 관절 이름으로 (wx, wy, wz, visibility)를 꺼낸다.

    wx/wy/wz = MediaPipe의 3D 월드 좌표 (미터 단위). visibility 판정은
    화면 좌표와 값이 같아서 기존 _vis 열을 그대로 쓴다.
    """
    return (
        float(row[f"{name}_wx"]),
        float(row[f"{name}_wy"]),
        float(row[f"{name}_wz"]),
        float(row[f"{name}_vis"]),
    )


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input)
    output_csv = Path(args.output)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(input_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    header = ["frame", "time_sec"] + list(ANGLE_DEFINITIONS.keys())
    skipped = {name: 0 for name in ANGLE_DEFINITIONS}

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row in rows:
            out = [row["frame"], row["time_sec"]]
            for name, (a_name, b_name, c_name) in ANGLE_DEFINITIONS.items():
                ax, ay, az, a_vis = get_point(row, a_name)
                bx, by, bz, b_vis = get_point(row, b_name)
                cx, cy, cz, c_vis = get_point(row, c_name)

                if min(a_vis, b_vis, c_vis) < args.min_vis:
                    skipped[name] += 1
                    out.append("")
                    continue

                angle = calculate_angle((ax, ay, az), (bx, by, bz), (cx, cy, cz))
                out.append(round(angle, 1) if angle is not None else "")
            writer.writerow(out)

    total = len(rows)
    print(f"입력 프레임: {total}장")
    print(f"저장 위치: {output_csv}")
    print()
    print(f"{'관절':<16}{'계산 성공':>10}{'제외(가려짐)':>12}")
    for name in ANGLE_DEFINITIONS:
        ok = total - skipped[name]
        print(f"{name:<16}{ok:>10}{skipped[name]:>12}")


if __name__ == "__main__":
    main()
