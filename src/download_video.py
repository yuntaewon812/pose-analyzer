"""
영상 링크 다운로드 스크립트 (유튜브 등 대부분의 동영상 사이트 링크 지원)

이 프로젝트는 "정답 영상"과 "내 영상" 두 개를 data/videos/reference.mp4,
data/videos/mine.mp4 라는 정해진 이름으로 필요로 합니다. yt-dlp 기본 옵션은
영상 제목으로 파일명을 정하기 때문에, --role로 reference/mine 중 하나를
지정하면 그 이름으로 바로 저장되도록 했습니다.

사용법:
    python src/download_video.py "영상 URL" --role reference
    python src/download_video.py "영상 URL" --role mine

역할 이름 대신 파일명을 직접 정하고 싶다면:
    python src/download_video.py "영상 URL" --out my_video.mp4
"""
import argparse
from pathlib import Path

import yt_dlp

SAVE_DIR = Path(__file__).parent.parent / "data" / "videos"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url", help="영상 URL (유튜브 등 yt-dlp가 지원하는 사이트, 또는 직접 mp4 링크)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--role", choices=["reference", "mine"],
                        help="정답/내 영상 슬롯 지정 -> data/videos/reference.mp4 또는 mine.mp4로 저장")
    group.add_argument("--out", help="저장할 파일명 직접 지정 (예: my_video.mp4)")
    return p.parse_args()


def download(url: str, filename: str) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    dest = SAVE_DIR / filename

    base = {
        # 파일명을 고정 -> extract_pose.py --video 인자와 그대로 맞물림
        "outtmpl": str(dest),
        # 이미 있으면 물어보지 않고 덮어쓰기 (스크립트 재실행 편의)
        "overwrites": True,
        # 이어받기 끄기. 앞선 시도가 남긴 조각 파일(.part)을 이어받으려다
        # "HTTP Error 416: Requested range not satisfiable"로 실패한 적이 있다.
        "continuedl": False,
    }

    # 해상도가 낮으면 관절 검출 신뢰도가 크게 떨어진다. 실제로 480x360으로 받은
    # 영상에서 어깨 0.46 / 엉덩이 0.12 / 무릎 0.34로 기준(0.5)에 미달해 동작 판정
    # 자체가 불가능했는데, 같은 영상에 2704x2028 화질이 올라와 있었다.
    #
    # 유튜브에서 "영상+음성이 한 파일"인 포맷은 360p까지뿐이고, 고화질은 영상과
    # 음성이 분리돼 있다. 포즈 분석에 음성은 필요 없으므로 bv*(영상만)를 받으면
    # 고화질을 쓰면서 합치는 과정(ffmpeg)도 생략된다.
    #
    # 2순위는 403 우회용이다. 기본 경로가 "HTTP Error 403: Forbidden"으로 막힐 때가
    # 있어서 클라이언트를 바꿔 시도하는데, 이 클라이언트들은 480x360만 노출하므로
    # 어디까지나 최후의 수단이다 (실제로 이걸 1순위로 뒀다가 저화질만 받은 적이 있다).
    attempts = [
        ("고화질(영상만)", {**base, "format": "bv*[height<=1080][ext=mp4]/bv*[height<=1080]"}),
        ("호환 모드(403 우회)", {**base,
                             "format": "best[ext=mp4]/best",
                             "extractor_args": {"youtube": {"player_client": ["android", "web_safari", "tv"]}}}),
    ]

    last_error = None
    for label, options in attempts:
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([url])
            print(f"\n다운로드 완료 ({label}): {dest}")
            return
        except Exception as exc:      # noqa: BLE001 - 다음 방법으로 넘어가기 위해
            print(f"  {label} 실패: {str(exc).splitlines()[-1][:120]}")
            last_error = exc

    raise RuntimeError(f"영상을 받지 못했습니다: {last_error}")


if __name__ == "__main__":
    args = parse_args()
    filename = f"{args.role}.mp4" if args.role else args.out
    download(args.url, filename)
