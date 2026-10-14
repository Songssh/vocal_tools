# SHS GPU Node VC/MDX Integration Manual

작성일: 2026-06-15

이 문서는 `vocal_tools`를 SHS GPU Node에서 가져다 써서 RVC 음성변환과 MDX 음원분리 기능을 붙이는 방법을 정리한다.

`vocal_tools` 자체는 서버가 아니다. FastAPI, 큐, DB, 인증, 파일 업로드 API, 진행률 WebSocket 같은 서비스 계층은 SHS GPU Node가 담당하고, `vocal_tools`는 GPU Node 안에서 호출되는 headless Python 엔진 라이브러리로 쓰는 구조가 맞다.

## 1. 현재 프로젝트 진행 상태

현재 이 프로젝트는 "GPU Node에 import해서 사용할 수 있는 라이브러리" 단계까지 와 있다.

| 영역 | 현재 상태 |
| --- | --- |
| MDX ONNX | `ModelCatalog + Separator + MDXOnnxEngine` 경로 구현됨 |
| RVC 음성변환 | `RVCConverter` 및 payload helper 구현됨 |
| Cover pipeline | `MDX -> RVC -> Mix` 파이프라인 구현됨 |
| Audio mix | 샘플레이트/채널 정렬, gain, normalize, WAV 출력 구현됨 |
| 모델 카탈로그 | builtin/user/remote manifest 로딩 구현됨 |
| RVC voice catalog | 로컬 `.pth`/`.index` voice model 스캔 구현됨 |
| Task payload helper | GPU Node 같은 host service용 JSON-safe 함수 구현됨 |
| CLI | `vocal-tools` 콘솔 명령 구현됨 |
| 테스트 | import/catalog/options/payload validation smoke test 존재 |
| Demucs | `DemucsEngine` 구현됨 |
| VR Arch | external CLI adapter만 있음. bundled backend는 아직 없음 |
| MDXC/Roformer | external CLI adapter만 있음. bundled backend는 아직 없음 |
| SHS GPU Node 직접 연결 | 일부러 없음. host에서 import해서 연결해야 함 |

현재 작업공간에서 확인한 값:

- `python -m unittest discover -s tests`: 8개 테스트 통과
- `python -m vocal_tools.cli --help`: 정상
- `models list --models-dir test_models`: builtin separation model 7개 확인
- `voices list --models-dir test_models`: `iut` RVC voice model 확인
- Git 상태: `test_new2.py`가 untracked 상태로 남아 있음

즉, 지금 바로 GPU Node에 붙일 수 있는 핵심 기능은 다음 세 가지다.

1. MDX 음원분리
2. RVC 음성변환
3. MDX로 보컬을 뽑고 RVC 변환 후 instrumental과 믹스하는 cover pipeline

반대로 아직 이 프로젝트 안에 없는 것은 다음이다.

- GPU Node API route
- job queue 연결
- cancellation 처리
- job/result DB 저장
- SHS AI Tools UI 연결
- public model download service
- RVC voice model remote install flow

## 2. 권장 아키텍처

SHS GPU Node 쪽에서는 `vocal_tools`를 엔진 라이브러리로 설치하고, task handler에서 payload helper를 호출한다.

```text
SHS AI Tools / client
  -> SHS GPU Node API
    -> queue / worker / job store
      -> vocal_tools task helper
        -> MDX ONNX / RVC / mix runtime
          -> output artifacts
```

`vocal_tools`에 서비스 코드를 넣지 않는 이유는 다음과 같다.

- GPU Node가 이미 job routing, queue, auth, progress transport를 갖고 있거나 가질 예정임
- audio engine은 다른 host에서도 재사용될 수 있어야 함
- 모델 경로, output 경로, artifact 저장 정책은 배포 환경마다 다름
- 긴 inference 작업의 cancellation/timeout 정책은 queue worker가 잡는 편이 안전함

GPU Node에서 추가하면 되는 service boundary는 다음이다.

| GPU Node 책임 | `vocal_tools` 책임 |
| --- | --- |
| API route | Python 함수 제공 |
| job id 생성 | 실제 분리/변환 실행 |
| queue enqueue/dequeue | model catalog 조회 |
| progress WebSocket/SSE relay | progress callback 이벤트 emit |
| cancellation/timeout | 단일 함수 호출 단위 처리 |
| 파일 업로드/다운로드 | output path에 artifact 생성 |
| DB 저장 | JSON-safe result 반환 |

## 3. 설치

GPU Node의 Python 환경에서 이 repo를 editable install로 설치한다.

```powershell
cd H:/shs/modules/python/vocal_tools
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

다른 프로젝트의 가상환경에서 설치할 때는 경로를 직접 지정한다.

```powershell
python -m pip install -e H:/shs/modules/python/vocal_tools
```

GPU에서 MDX ONNX를 실행하려면 `onnxruntime-gpu`가 필요하다.

```powershell
python -m pip install -e "H:/shs/modules/python/vocal_tools[gpu]"
```

RVC를 실행하려면 RVC extra가 필요하다. 현재 검증된 로컬 설치 경로에서는 `fairseq==0.12.2` 때문에 pip 버전을 낮춘 뒤 설치하는 편이 안전했다.

```powershell
python -m pip install "pip<24.1"
python -m pip install -e "H:/shs/modules/python/vocal_tools[rvc]"
```

Cover pipeline에서 mix 관련 optional package까지 쓰려면 audio extra도 설치한다.

```powershell
python -m pip install -e "H:/shs/modules/python/vocal_tools[audio]"
```

실서비스 GPU Node에서는 다음도 별도로 확인한다.

- `ffmpeg` CLI가 PATH에 있어야 함
- CUDA용 PyTorch가 설치되어 있어야 함
- `onnxruntime-gpu`가 실제 CUDA provider를 잡는지 확인해야 함
- GPU Node worker process가 모델 파일과 output directory에 접근할 수 있어야 함

## 4. 모델 디렉터리 구조

가장 단순한 구조는 현재 `test_models`와 같은 형태다.

```text
models/
  hubert_base.pt
  rmvpe.pt
  iut/
    iut.pth
    iut.index
  mdx/
    Kim_Vocal_2.onnx
    UVR_MDXNET_KARA_2.onnx
    Reverb_HQ_By_FoxJoy.onnx
    model_data.json
```

권장 환경변수 예시는 다음이다.

```powershell
$env:VOCAL_TOOLS_MODELS_DIR="H:/shs/models/vocal_tools"
$env:VOCAL_TOOLS_OUTPUT_ROOT="H:/shs/output/audio_jobs"
```

MDX는 보통 `models_dir/mdx` 아래의 `.onnx` 파일을 사용한다.

RVC는 `models_dir` 루트 아래에서 다음을 찾는다.

```text
models/hubert_base.pt
models/rmvpe.pt
models/<voice_model>/*.pth
models/<voice_model>/*.index
```

예를 들어 payload에서 `"voice_model": "iut"`를 넘기면 다음을 찾는다.

```text
models/iut/iut.pth
models/iut/iut.index
```

voice model을 `models/rvc/iut`처럼 더 깊게 두고 싶다면 `"voice_model": "rvc/iut"`처럼 상대 경로를 넘기거나, 절대 경로를 넘긴다.

## 5. GPU Node에 추가할 task 종류

처음에는 아래 네 개 task만 추가하면 충분하다.

| Task name 예시 | 역할 | 호출 함수 |
| --- | --- | --- |
| `audio.models.list` | separation/RVC 모델 목록 조회 | `list_audio_models_payload` |
| `audio.separate` | MDX 또는 Demucs 분리 | `run_audio_separate_payload` |
| `audio.rvc.convert` | RVC 단독 음성변환 | `run_audio_rvc_convert_payload` |
| `audio.cover` | MDX -> RVC -> Mix | `run_audio_cover_pipeline_payload` |

서비스 내부 task name은 GPU Node의 기존 naming convention에 맞춰 바꿔도 된다. 중요한 것은 payload helper의 입력/출력 경계를 유지하는 것이다.

## 6. 공통 handler 골격

GPU Node worker 쪽에 다음 같은 wrapper를 둔다.

```python
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from vocal_tools import (
    list_audio_models_payload,
    list_rvc_voice_models_payload,
    run_audio_cover_pipeline_payload,
    run_audio_rvc_convert_payload,
    run_audio_separate_payload,
)


MODELS_DIR = Path(os.environ["VOCAL_TOOLS_MODELS_DIR"])
OUTPUT_ROOT = Path(os.environ["VOCAL_TOOLS_OUTPUT_ROOT"])


def _job_output_dir(job_id: str, kind: str) -> Path:
    path = OUTPUT_ROOT / kind / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _progress(job_id: str):
    def relay(event: dict[str, Any]) -> None:
        # GPU Node의 기존 progress bus, WebSocket, SSE, DB update 등에 연결한다.
        publish_job_progress(job_id, event)

    return relay


def handle_audio_task(job_id: str, task_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload.setdefault("models_dir", str(MODELS_DIR))

    if task_name == "audio.models.list":
        return list_audio_models_payload(payload)

    if task_name == "audio.voices.list":
        return list_rvc_voice_models_payload({"models_dir": payload["models_dir"]})

    if task_name == "audio.separate":
        payload.setdefault("output_dir", str(_job_output_dir(job_id, "separate")))
        return run_audio_separate_payload(payload, progress_callback=_progress(job_id))

    if task_name == "audio.rvc.convert":
        output_dir = _job_output_dir(job_id, "rvc")
        payload.setdefault("output_audio", str(output_dir / "converted.wav"))
        return run_audio_rvc_convert_payload(payload, progress_callback=_progress(job_id))

    if task_name == "audio.cover":
        payload.setdefault("output_dir", str(_job_output_dir(job_id, "cover")))
        return run_audio_cover_pipeline_payload(payload, progress_callback=_progress(job_id))

    raise ValueError(f"Unsupported audio task: {task_name}")
```

위 코드에서 `publish_job_progress()`는 GPU Node의 기존 progress publish 함수로 바꿔야 한다.

## 7. 모델 목록 API

GPU Node가 UI에 모델 목록을 내려줄 때는 `list_audio_models_payload()`를 쓴다.

```python
from vocal_tools import list_audio_models_payload

result = list_audio_models_payload(
    {
        "models_dir": "H:/shs/models/vocal_tools",
        "include_separation_models": True,
        "include_voice_models": True,
    }
)
```

반환값은 JSON-safe dict다.

```json
{
  "separation_models": [],
  "voice_models": [],
  "sources": []
}
```

UI에서 보통 쓰는 필드는 다음이다.

| 필드 | 의미 |
| --- | --- |
| `id` | task payload에 넣을 model id |
| `name` | 사용자에게 보여줄 이름 |
| `arch` | `mdx_onnx`, `demucs`, `vr_arch`, `mdxc` 등 |
| `tasks` | `vocal_separation`, `dereverb`, `karaoke` 등 |
| `output_stems` | 결과 stem 이름 |
| `downloaded` | 로컬에서 바로 실행 가능한지 |
| `download_status` | `downloaded`, `missing`, `managed_by_package` 등 |
| `ui.category` | UI grouping용 |
| `options` | UI control 생성용 option schema |

RVC voice만 따로 조회하려면 다음을 쓴다.

```python
from vocal_tools import list_rvc_voice_models_payload

result = list_rvc_voice_models_payload({"models_dir": "H:/shs/models/vocal_tools"})
```

RVC voice card 주요 필드는 다음이다.

| 필드 | 의미 |
| --- | --- |
| `id` | `voice_model`에 넣을 기본 id |
| `path` | voice model directory |
| `pth_path` | 선택된 `.pth` |
| `index_path` | 선택된 `.index`, 없으면 null |
| `has_index` | index 파일 존재 여부 |
| `downloaded` | `.pth` 존재 여부 |

## 8. MDX 음원분리 추가

GPU Node에서 MDX 분리 job을 실행할 때는 `run_audio_separate_payload()`를 호출한다.

```python
from vocal_tools import run_audio_separate_payload

payload = {
    "models_dir": "H:/shs/models/vocal_tools/mdx",
    "input_audio": "H:/shs/input/song.wav",
    "output_dir": "H:/shs/output/audio_jobs/separate/job-001",
    "model_id": "kim_vocal_2",
    "options": {
        "denoise": False,
        "processor": 0
    },
}

result = run_audio_separate_payload(payload, progress_callback=relay_progress)
```

중요한 점:

- `models_dir`는 MDX만 실행할 때 `models/mdx`를 넘기는 것이 가장 명확하다.
- `model_id`는 builtin manifest의 id를 사용한다.
- `processor`는 CUDA device index다. `0`이면 첫 번째 GPU를 의미한다.
- `denoise=True`는 모델/환경에 따라 시간이 더 걸린다.
- `model_data.json`은 `.onnx` 파일 옆에 있거나 `model_params_path`로 지정해야 한다.

기본 MDX model id는 다음이다.

| model id | 용도 | output stems |
| --- | --- | --- |
| `kim_vocal_2` | 보컬/반주 분리 | `vocals`, `instrumental` |
| `uvr_mdxnet_kara_2` | karaoke/lead vocal removal | `instrumental`, `vocals` |
| `reverb_hq_by_foxjoy` | dereverb | `dry_vocals`, `reverb` |

성공 결과 예시는 다음 형태다.

```json
{
  "input_audio": ".../song.wav",
  "output_dir": ".../job-001",
  "model_id": "kim_vocal_2",
  "model_name": "Kim Vocal 2",
  "arch": "mdx_onnx",
  "stems": {
    "vocals": ".../job-001/song_(vocals).wav",
    "instrumental": ".../job-001/song_(instrumental).wav"
  },
  "options_used": {},
  "metadata": {}
}
```

## 9. RVC 음성변환 추가

RVC 단독 변환은 이미 분리된 보컬 파일을 입력으로 받는다.

```python
from vocal_tools import run_audio_rvc_convert_payload

payload = {
    "models_dir": "H:/shs/models/vocal_tools",
    "input_audio": "H:/shs/output/audio_jobs/separate/job-001/vocals.wav",
    "output_audio": "H:/shs/output/audio_jobs/rvc/job-002/converted.wav",
    "voice_model": "iut",
    "device": "cuda:0",
    "is_half": "auto",
    "options": {
        "pitch_change": 0,
        "f0_method": "rmvpe",
        "index_rate": 0.5,
        "filter_radius": 3,
        "rms_mix_rate": 0.25,
        "protect": 0.33
    },
}

result = run_audio_rvc_convert_payload(payload, progress_callback=relay_progress)
```

필수 파일:

```text
models/hubert_base.pt
models/rmvpe.pt
models/iut/iut.pth
models/iut/iut.index
```

`index` 파일은 선택사항이지만, 있으면 timbre retrieval 품질에 도움이 된다.

RVC option 의미:

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `pitch_change` | `0` | semitone 단위 pitch shift |
| `f0_method` | `rmvpe` | pitch extraction 방식 |
| `index_rate` | `0.5` | index 검색 반영 비율 |
| `filter_radius` | `3` | f0 smoothing radius |
| `rms_mix_rate` | `0.25` | 입력 RMS와 출력 RMS mix 비율 |
| `protect` | `0.33` | 무성음/자음 보호 |
| `crepe_hop_length` | `128` | crepe 계열 f0 hop length |
| `sid` | `0` | speaker id |
| `resample_sr` | `0` | 출력 resample sr, `0`이면 모델 기본값 |

GPU Node에서는 `device="cuda:0"`처럼 명시하는 편이 운영상 추적하기 쉽다. `auto`도 가능하지만, worker별 GPU pinning을 한다면 task payload에서 device를 명시하는 편이 낫다.

## 10. Cover pipeline 추가

사용자가 "원곡 + voice model"만 주고 커버 결과를 원한다면 cover pipeline을 쓰면 된다.

```python
from vocal_tools import run_audio_cover_pipeline_payload

payload = {
    "models_dir": "H:/shs/models/vocal_tools",
    "input_audio": "H:/shs/input/song.wav",
    "output_dir": "H:/shs/output/audio_jobs/cover/job-003",
    "mdx_model_id": "kim_vocal_2",
    "voice_model": "iut",
    "mdx_options": {
        "denoise": False,
        "processor": 0
    },
    "rvc_options": {
        "pitch_change": 0,
        "f0_method": "rmvpe",
        "index_rate": 0.5,
        "protect": 0.33
    },
    "mix_options": {
        "vocal_gain_db": 0.0,
        "instrumental_gain_db": 0.0,
        "normalize": True,
        "peak": 0.98
    },
    "keep_intermediate": True
}

result = run_audio_cover_pipeline_payload(payload, progress_callback=relay_progress)
```

pipeline 내부 순서는 다음이다.

```text
1. MDX로 vocals / instrumental 분리
2. 선택적으로 dereverb MDX pass 실행
3. 선택적으로 karaoke/main vocal split pass 실행
4. vocals를 RVC voice model로 변환
5. converted vocal과 instrumental을 mix
6. JSON-safe result 반환
```

선택 옵션:

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `vocal_stem` | `vocals` | RVC 입력으로 사용할 stem |
| `instrumental_stem` | `instrumental` | 최종 mix에 사용할 반주 stem |
| `rvc_output_name` | voice 기반 자동 이름 | 변환된 보컬 파일명 |
| `mix_output_name` | `cover_mix.wav` | 최종 mix 파일명 |
| `keep_intermediate` | `True` | 중간 산출물 보존 |
| `use_dereverb` | `False` | dereverb 추가 pass |
| `dereverb_model_id` | `reverb_hq_by_foxjoy` | dereverb model id |
| `use_karaoke_split` | `False` | main/backing split 추가 pass |
| `karaoke_model_id` | `uvr_mdxnet_kara_2` | karaoke model id |

GPU Node 첫 연동에서는 optional preprocessing을 끄고 시작하는 것을 권장한다.

```json
{
  "use_dereverb": false,
  "use_karaoke_split": false
}
```

기본 pipeline이 안정화된 뒤 곡/모델별 품질 테스트를 하면서 켜는 편이 디버깅이 쉽다.

## 11. Progress event 연결

모든 payload helper는 `progress_callback`을 받을 수 있다.

```python
def relay_progress(event: dict) -> None:
    print(event)

result = run_audio_cover_pipeline_payload(payload, progress_callback=relay_progress)
```

대표 stage는 다음과 같다.

| Stage | 의미 |
| --- | --- |
| `task_start` | helper task 시작 |
| `separate_progress` | separation 내부 이벤트 |
| `pipeline_progress` | cover pipeline 내부 이벤트 |
| `mdx_start` | cover pipeline의 MDX 시작 |
| `mdx_progress` | MDX 진행 |
| `mdx_complete` | MDX 완료 |
| `rvc_start` | RVC 시작 |
| `rvc_complete` | RVC 완료 |
| `mix_start` | mix 시작 |
| `mix_complete` | mix 완료 |
| `task_complete` | helper task 완료 |

GPU Node에서는 이 event를 그대로 job progress channel에 흘리면 된다.

```python
def relay_progress(job_id: str):
    def relay(event: dict) -> None:
        job_store.append_event(job_id, event)
        websocket_manager.publish(job_id, event)

    return relay
```

현재 MDX/RVC의 세부 percentage는 제한적이다. 그래서 UI에서는 처음에는 stage 기반 progress를 보여주는 것이 현실적이다.

## 12. Cancellation과 timeout

`vocal_tools` 함수 하나가 긴 inference를 수행할 수 있으므로 cancellation은 GPU Node worker 계층에서 잡는 것이 맞다.

권장 방식:

1. job 하나를 worker process 또는 cancellable subprocess에서 실행
2. GPU Node queue가 timeout을 관리
3. cancel 요청이 들어오면 worker process를 중단
4. output directory를 cleanup하거나 partial artifact 상태로 표시

Python thread만으로 강제 중단하는 방식은 권장하지 않는다. PyTorch/onnxruntime/ffmpeg subprocess가 섞이면 안전하게 멈추기 어렵다.

## 13. 에러 처리 정책

GPU Node에서는 예외를 다음처럼 분류하면 된다.

| 예외/상황 | HTTP/job status 예시 | 사용자 메시지 |
| --- | --- | --- |
| payload validation `ValueError` | failed / bad_request | 요청 필드가 잘못됨 |
| input file 없음 | failed / bad_request | 입력 오디오가 없음 |
| model file 없음 | failed / model_missing | 모델 파일이 설치되지 않음 |
| CUDA unavailable | failed / runtime_unavailable | GPU 런타임 확인 필요 |
| inference 중 예외 | failed / inference_error | 처리 중 오류 |
| timeout | cancelled 또는 failed / timeout | 처리 시간이 제한을 초과함 |
| cancel request | cancelled | 사용자가 작업을 취소함 |

handler 예시:

```python
def run_job(job_id: str, task_name: str, payload: dict) -> dict:
    try:
        result = handle_audio_task(job_id, task_name, payload)
    except ValueError as exc:
        return fail_job(job_id, code="bad_request", message=str(exc))
    except FileNotFoundError as exc:
        return fail_job(job_id, code="file_not_found", message=str(exc))
    except Exception as exc:
        return fail_job(job_id, code="inference_error", message=str(exc))

    return complete_job(job_id, result)
```

## 14. API 설계 예시

GPU Node가 FastAPI를 쓴다면 route는 대략 다음처럼 잡을 수 있다.

```python
from fastapi import APIRouter

router = APIRouter(prefix="/audio")


@router.get("/models")
def list_models():
    return list_audio_models_payload(
        {
            "models_dir": str(MODELS_DIR),
            "include_separation_models": True,
            "include_voice_models": True,
        }
    )


@router.post("/separate")
def create_separation_job(payload: dict):
    job_id = enqueue_job("audio.separate", payload)
    return {"job_id": job_id}


@router.post("/rvc")
def create_rvc_job(payload: dict):
    job_id = enqueue_job("audio.rvc.convert", payload)
    return {"job_id": job_id}


@router.post("/cover")
def create_cover_job(payload: dict):
    job_id = enqueue_job("audio.cover", payload)
    return {"job_id": job_id}
```

worker에서는 section 6의 `handle_audio_task()`로 dispatch한다.

## 15. CLI로 먼저 검증하기

GPU Node에 붙이기 전에 같은 환경에서 CLI smoke check를 먼저 한다.

```powershell
python -m vocal_tools.cli models list --models-dir H:/shs/models/vocal_tools
python -m vocal_tools.cli voices list --models-dir H:/shs/models/vocal_tools
```

MDX 단독 검증:

```powershell
python -m vocal_tools.cli separate `
  --models-dir H:/shs/models/vocal_tools/mdx `
  --model-id kim_vocal_2 `
  --input H:/shs/input/song.wav `
  --output-dir H:/shs/output/check_mdx `
  --option denoise=false `
  --option processor=0
```

RVC 단독 검증:

```powershell
python -m vocal_tools.cli rvc-convert `
  --models-dir H:/shs/models/vocal_tools `
  --voice-model iut `
  --input H:/shs/output/check_mdx/vocals.wav `
  --output H:/shs/output/check_rvc/converted.wav `
  --device cuda:0 `
  --option pitch_change=0 `
  --option f0_method=rmvpe
```

Cover pipeline 검증:

```powershell
python -m vocal_tools.cli cover `
  --models-dir H:/shs/models/vocal_tools `
  --input H:/shs/input/song.wav `
  --output-dir H:/shs/output/check_cover `
  --mdx-model-id kim_vocal_2 `
  --voice-model iut `
  --mdx-option denoise=false `
  --mdx-option processor=0 `
  --rvc-option pitch_change=0 `
  --rvc-option f0_method=rmvpe `
  --mix-option normalize=true `
  --mix-option peak=0.98
```

CLI가 성공하면 같은 payload를 GPU Node worker로 옮기면 된다.

## 16. 운영 체크리스트

GPU Node에 붙이기 전 체크:

- `python -m unittest discover -s tests` 통과
- `python -m vocal_tools.cli models list --models-dir <models_dir>` 성공
- `python -m vocal_tools.cli voices list --models-dir <models_dir>` 성공
- `ffmpeg -version` 성공
- PyTorch CUDA 확인 성공
- onnxruntime CUDA provider 확인 성공
- 짧은 10초 이내 오디오로 MDX 검증
- 이미 분리된 vocals로 RVC 검증
- cover pipeline 검증
- 긴 오디오 max duration 정책 결정
- output retention/cleanup 정책 결정
- cancellation 방식 결정
- GPU별 worker concurrency 결정

PyTorch CUDA 확인:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

onnxruntime provider 확인:

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

`CUDAExecutionProvider`가 보여야 MDX ONNX GPU 실행을 기대할 수 있다.

## 17. 권장 1차 구현 순서

1. GPU Node venv에 `vocal_tools` editable install
2. 모델 디렉터리를 `models/hubert_base.pt`, `models/rmvpe.pt`, `models/<voice>`, `models/mdx` 구조로 정리
3. `audio.models.list`, `audio.voices.list` route 추가
4. CLI로 MDX/RVC/cover 짧은 파일 검증
5. GPU Node worker에 `handle_audio_task()` 같은 dispatch wrapper 추가
6. `audio.separate` job 연결
7. `audio.rvc.convert` job 연결
8. `audio.cover` job 연결
9. progress event를 기존 job progress channel에 relay
10. timeout/cancel/output cleanup 정책 연결
11. SHS AI Tools UI에서 모델 목록과 job 생성 payload 연결

## 18. 지금 기준으로 특히 조심할 점

- MDX와 RVC의 `models_dir` 의미가 다르다.
  - MDX 단독: `models/mdx`
  - RVC/cover: root `models`
- `cover` payload의 `models_dir`는 root `models`여야 한다. 내부에서 `models/mdx`를 사용한다.
- RVC support model 파일명은 기본적으로 `hubert_base.pt`, `rmvpe.pt`다.
- RVC voice directory는 `.pth`가 반드시 있어야 하고 `.index`는 선택이다.
- `processor`는 MDX ONNX CUDA device index이고, RVC의 `device`는 `cuda:0` 같은 torch device 문자열이다.
- MDXC/Roformer는 MDX ONNX와 다르다. 현재는 external CLI adapter만 있고 bundled runtime은 없다.
- 기본 smoke test는 inference를 돌리지 않는다. 실제 GPU inference 검증은 CLI 또는 `examples/verify_*` 스크립트로 별도 수행해야 한다.
