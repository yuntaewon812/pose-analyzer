"""
관절별 피드백을 Claude API로 생성하기 (규칙 기반 문장을 대체)

[무엇이 달라지나]
generate_feedback.py는 미리 적어둔 표에서 문장을 꺼내 조립합니다. 정확하고 항상
같은 답이 나오지만 기계적이고, 관절마다 같은 문장이 반복됩니다. 이 스크립트는
같은 숫자(angle_diff.csv)를 Claude에게 주고 관절마다 코치가 말하듯 문장을
만들게 합니다. 숫자 계산은 이미 끝났으므로 LLM은 "해석해서 말로 옮기는" 일만
합니다.

[앱에서 어떻게 쓰이나]
관절별 집중 영상 하나마다 문장 하나가 필요하므로, 결과를 관절 이름으로 찾을 수
있게 JSON으로 저장합니다. 앱은 이 파일이 있으면 LLM 문장을, 없으면 규칙 기반
문장을 보여줍니다 (API 키가 없어도 앱이 멈추지 않게).

[사용 전 준비 - API 키]
https://console.anthropic.com 에서 키를 발급받아 환경변수로 설정하세요.
    (PowerShell)  $env:ANTHROPIC_API_KEY = "sk-ant-..."
    (Git Bash)    export ANTHROPIC_API_KEY="sk-ant-..."
키는 코드에 적지 말고 환경변수로 두세요 (실수로 깃에 올라가면 폐기해야 합니다).

사용법:
    python src/generate_feedback_llm.py
결과물:
    output/comparison/feedback_llm.json   (앱이 읽는 파일)
    output/comparison/feedback_llm.txt    (사람이 읽는 사본)
"""
import json
import sys
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
import generate_feedback as fb  # JOINT_LABELS, load_joint_stats 재사용

ROOT = Path(__file__).parent.parent
OUT_JSON = ROOT / "output" / "comparison" / "feedback_llm.json"
OUT_TXT = ROOT / "output" / "comparison" / "feedback_llm.txt"

MODEL = "claude-opus-5"


class JointCoaching(BaseModel):
    """관절 한 곳에 대한 코칭. 앱의 집중 영상 하나에 그대로 붙는다."""
    joint: str = Field(description="관절 키 (예: left_knee). 입력에 주어진 값을 그대로 쓸 것")
    headline: str = Field(description="무엇을 어떻게 할지 한 문장 지시. 예: '왼쪽 무릎을 더 펴세요'")
    detail: str = Field(description="지금 동작이 정답과 어떻게 다른지, 어떻게 고치면 좋을지 2~3문장")


class Coaching(BaseModel):
    joints: list[JointCoaching]
    encouragement: str = Field(description="가장 잘 유지되고 있는 부분을 짚어주는 격려 1~2문장")


PROMPT = """당신은 수영 코치입니다. 수강생의 수중 돌핀킥 동작을 정답 영상과 비교한 결과를 보고 있습니다.

아래는 두 영상의 관절 각도를 같은 진행률 지점끼리 비교해서 얻은 수치입니다.
diff = 수강생 각도 - 정답 각도이므로, 음수면 수강생이 정답보다 더 굽힌 것이고
양수면 덜 굽힌 것입니다.

[관절별 차이]
{joint_data}

[요청]
위 관절 각각에 대해 코칭 문장을 만들어 주세요. 각 관절의 headline은 수강생이 바로
따라 할 수 있는 동작 지시 한 문장으로, detail은 지금 동작이 정답과 어떻게 다른지와
어떻게 고치면 좋을지를 2~3문장으로 써 주세요.

[지켜야 할 것]
- 위에 주어진 것 말고 새로운 각도 수치를 만들어내지 마세요.
- 각도 차이가 "무엇인지"는 데이터로 확실하지만, 그 원인(코어 힘, 부력, 킥 타이밍 등)은
  영상 각도만으로 확정할 수 없습니다. 원인을 말할 때는 "~일 수 있습니다"처럼 여지를 두세요.
- 수영을 배우는 사람이 읽습니다. 전문 용어를 쓸 때는 짧게 풀어서 설명하세요.
- 관절마다 같은 표현을 반복하지 말고, 그 관절에서 실제로 일어나는 일을 구체적으로 쓰세요.
"""


def build_joint_lines(stats):
    """LLM에 넘길 관절별 수치를 차이가 큰 순서로 정리한다."""
    ranked = sorted(stats.items(), key=lambda kv: -kv[1]["mean_abs"])
    lines = []
    for name, s in ranked:
        direction = "수강생이 더 굽힘" if s["worst_diff"] < 0 else "수강생이 덜 굽힘"
        lines.append(
            f"- {name} ({fb.JOINT_LABELS[name]}): 킥 사이클 평균 {s['mean_abs']:.0f}도 차이, "
            f"진행률 {s['worst_pct']:.0f}% 지점에서 최대 {s['worst_diff']:+.0f}도 ({direction})"
        )
    return "\n".join(lines), [name for name, _ in ranked]


def generate(stats):
    """Claude에게 관절별 코칭 문장을 받아온다."""
    joint_data, order = build_joint_lines(stats)
    client = anthropic.Anthropic()   # ANTHROPIC_API_KEY 환경변수를 자동으로 읽는다

    # max_tokens는 넉넉히 - Claude Opus 5는 기본적으로 생각(thinking)을 하고,
    # max_tokens가 생각과 답변을 합쳐서 제한하므로 빠듯하면 문장이 잘린다.
    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": PROMPT.format(joint_data=joint_data)}],
        output_format=Coaching,
    )
    if response.stop_reason == "refusal":
        raise SystemExit("Claude가 이 요청에 응답하지 않았습니다. 프롬프트를 확인해 주세요.")

    result = response.parsed_output
    # 앱이 관절 이름으로 바로 찾을 수 있게 딕셔너리로 바꾼다.
    by_joint = {j.joint: {"headline": j.headline, "detail": j.detail}
                for j in result.joints if j.joint in stats}
    missing = [n for n in order if n not in by_joint]
    if missing:
        print(f"경고: 문장이 오지 않은 관절 {missing} — 이 관절은 규칙 기반 문장이 쓰입니다.")
    return {"model": MODEL, "joints": by_joint, "encouragement": result.encouragement}


def main() -> None:
    stats = fb.load_joint_stats()
    if not stats:
        raise SystemExit("angle_diff.csv에 유효한 데이터가 없습니다. 비교 단계를 먼저 실행하세요.")

    try:
        data = generate(stats)
    # 키가 아예 없으면 SDK가 TypeError를, 키가 틀렸으면 AuthenticationError를 낸다.
    # 둘 다 사용자 입장에서는 "키 설정 문제"라 같은 안내를 보여준다.
    except (anthropic.AuthenticationError, TypeError):
        raise SystemExit(
            "API 키가 없거나 잘못됐습니다. 키를 환경변수로 설정한 뒤 다시 실행하세요.\n"
            '  (PowerShell)  $env:ANTHROPIC_API_KEY = "sk-ant-..."\n'
            '  (Git Bash)    export ANTHROPIC_API_KEY="sk-ant-..."\n'
            "키 발급: https://console.anthropic.com"
        )
    except anthropic.RateLimitError:
        raise SystemExit("요청이 너무 잦습니다. 잠시 후 다시 실행하세요.")
    # 키는 맞는데 계정에 크레딧이 없는 경우. 인증 실패(401)와 헷갈리기 쉬워서
    # "키 문제가 아니라 잔액 문제"라는 걸 분명히 알려준다.
    except anthropic.BadRequestError as e:
        if "credit balance" in str(e).lower():
            raise SystemExit(
                "API 키는 정상이지만 계정 크레딧이 부족합니다.\n"
                "  https://console.anthropic.com → Plans & Billing 에서 크레딧을 충전하세요.\n"
                "  (Claude 구독(Pro/Max)과 API 크레딧은 별개입니다.)"
            )
        raise
    except anthropic.APIConnectionError:
        raise SystemExit("네트워크 연결에 실패했습니다. 인터넷 연결을 확인하세요.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"동작 개선 피드백 - {fb.PHASE_TITLE} (Claude {MODEL})", "=" * 60, ""]
    for name, s in sorted(stats.items(), key=lambda kv: -kv[1]["mean_abs"]):
        c = data["joints"].get(name)
        if not c:
            continue
        lines += [f"[{fb.JOINT_LABELS[name]}] {c['headline']}", f"  {c['detail']}", ""]
    lines += ["-" * 60, data["encouragement"]]
    text = "\n".join(lines)
    OUT_TXT.write_text(text, encoding="utf-8")

    print(text)
    print(f"\n저장 위치: {OUT_JSON}")


if __name__ == "__main__":
    main()
