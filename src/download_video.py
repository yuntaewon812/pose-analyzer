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

    options = {
        # 720p 이하 mp4로 다운로드 (포즈 추정에는 720p면 충분하고 처리 속도가 빠름)
        "format": "best[height<=720][ext=mp4]/best[ext=mp4]/best",
        # 파일명을 고정 -> extract_pose.py --video 인자와 그대로 맞물림
        "outtmpl": str(dest),
        # 이미 있으면 덮어쓰지 않고 물어보지 않게(스크립트 재실행 편의를 위해 덮어쓰기)
        "overwrites": True,
        # 유튜브 기본 경로가 "HTTP Error 403: Forbidden"으로 막히는 경우가 있어서
        # 앱(클라이언트) 종류를 여러 개 지정해 순서대로 시도하게 한다. 하나가 막혀도
        # 다른 것으로 넘어가므로 성공률이 크게 올라간다.
        "extractor_args": {"youtube": {"player_client": ["android", "web_safari", "tv"]}},
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])

    print(f"\n다운로드 완료! 저장 위치: {dest}")


if __name__ == "__main__":
    args = parse_args()
    filename = f"{args.role}.mp4" if args.role else args.out
    download(args.url, filename)
