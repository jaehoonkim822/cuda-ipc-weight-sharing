# CUDA IPC GPU Memory Sharing PoC

GPU 메모리에 로드된 모델 가중치를 여러 추론 프로세스가 **zero-copy**로 공유하는 개념 검증(PoC)입니다. CUDA IPC를 통해 하나의 Weight Manager가 GPU 메모리를 소유하고, 다수의 Inference Worker가 복사 없이 같은 메모리를 참조하여 추론합니다.

## 목적

모델 서빙 환경에서 동일한 가중치를 프로세스마다 복제하면 GPU 메모리가 선형으로 증가합니다. 이 PoC는 CUDA IPC(`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`)를 활용해 **N개 Worker가 1벌의 가중치만으로 동시 추론**할 수 있음을 검증합니다.

추가로 **Tensor Parallelism**을 두 가지 모드로 구현합니다:
- **집중형 TP (N:1)**: 1 Worker가 모든 WM에서 shard를 수집해 로컬 루프로 추론 (`TPSimpleMLP`)
- **분산형 TP (1:1)**: N Worker가 각각 1 WM을 전담하고 `torch.distributed` NCCL 집합 통신으로 추론 (`DistributedTPForward`)

**ModelSpec/ModelRegistry** 기반으로 설계되어, 임의의 모델을 등록하면 동일한 파이프라인(로드 → 샤딩 → IPC 서빙 → 추론)을 재사용할 수 있습니다.

## 아키텍처

### 단일 GPU (1:N)

```
                  GPU 0
                ┌────────────────┐
                │  Model Weights │  <- 1벌만 존재
                └───────┬────────┘
                        │ CUDA IPC
        ┌───────────────┼───────────────┐
        v               v               v
   Worker 0        Worker 1        Worker 2
   (추론)          (추론)          (추론)
```

Weight Manager가 모델을 GPU에 로드하고, IPC 핸들을 ZMQ REP 소켓으로 서빙합니다. 각 Worker는 ZMQ REQ로 핸들을 받아 `torch.UntypedStorage._new_shared_cuda()`로 같은 GPU 메모리를 열고, `load_state_dict(assign=True)`로 zero-copy 주입합니다.

### 집중형 Tensor Parallelism (N:1)

```
  WM-0 (cuda:0)               WM-1 (cuda:1)
  ┌──────────────┐            ┌──────────────┐
  │ rank=0 shard  │            │ rank=1 shard  │
  │ ipc:///wm_0   │            │ ipc:///wm_1   │
  └──────┬───────┘            └──────┬───────┘
         │ ZMQ REQ/REP                │
         └──────────┐    ┌───────────┘
                    v    v
              ┌──────────────┐
              │    Worker     │
              │  TPSimpleMLP  │
              │  forward(x)   │
              └──────────────┘
```

1 Worker가 모든 WM에 연결하여 shard를 수집하고, 로컬 루프로 TP-aware forward를 수행합니다.

### 분산형 Tensor Parallelism (1:1)

```
  WM-0 (cuda:0)               WM-1 (cuda:1)
  ┌──────────────┐            ┌──────────────┐
  │ rank=0 shard  │            │ rank=1 shard  │
  │ ipc:///wm_0   │            │ ipc:///wm_1   │
  └──────┬───────┘            └──────┬───────┘
         │ ZMQ                        │ ZMQ
         v                            v
  ┌──────────────┐            ┌──────────────┐
  │  Worker-0     │◄──NCCL───►│  Worker-1     │
  │  rank=0       │ all_gather │  rank=1       │
  │DistributedTP  │ all_reduce │DistributedTP  │
  └──────────────┘            └──────────────┘
```

각 Worker가 1개 WM만 전담합니다. `torch.distributed` NCCL `all_gather`/`all_reduce`로 프로세스 간 통신하여 분산 forward를 수행합니다.

**샤딩 전략**: `BaseTensorParallelHandler`로 선언적 정의. 모델별로 Column/Row-parallel 파라미터 패턴만 지정하면 자동 분할됩니다.

- **SimpleMLP** (784→256→10): `fc1` column-parallel, `fc2.weight` row-parallel, `fc2.bias` rank 0 only
- **TinyTransformer** (LLaMA-style): `q/k/v/gate/up_proj` column-parallel, `o/down_proj` row-parallel

## 코드 구조

```
src/cuda_ipc_poc/
├── config.py              # ZMQ_ENDPOINT, TP_WORLD_SIZE 등 설정
├── ipc_channel.py         # ZMQ REP/REQ 기반 핸들 서버/클라이언트
├── handle_codec.py        # IPC 핸들 직렬화 (TP rank 메타데이터 포함)
├── model_spec.py          # ModelSpec, WeightLoader, ModelRegistry
├── tp_handler.py          # BaseTensorParallelHandler + SimpleMlpTPHandler
├── weight_manager.py      # ModelSpec 기반 로드 -> 샤딩 -> IPC 서빙
├── inference_worker.py    # 핸들 수신 -> 텐서 복원 -> ModelSpec 기반 모델 구성
├── tensor_parallel.py     # TPSimpleMLP, DistributedTPForward (legacy 래퍼)
├── model.py               # SimpleMLP, ResNet18 + Registry 등록
└── models/
    └── tiny_llama.py      # TinyTransformer (LLaMA-style) + TPHandler + 등록

scripts/
├── run_weight_manager.py  # WM 프로세스 진입점 (--model-path 지원)
├── run_inference_worker.py # Worker 프로세스 진입점
└── run_verification.py    # 9개 자동 검증 테스트

tests/
├── test_ipc_channel.py    # ZMQ 전송 계층 테스트 (7개)
├── test_handle_codec.py   # 직렬화 + TP 포맷 테스트 (12개)
├── test_tensor_parallel.py # 샤딩/TP forward/분산 TP 테스트 (10개)
├── test_model_spec.py     # Registry + TPHandler 패턴 매칭 테스트 (13개)
└── test_tiny_transformer.py # TinyTransformer forward + TP 샤딩 테스트 (11개)
```

### 핵심 모듈 요약

| 모듈 | 역할 |
|------|------|
| `model_spec.py` | `ModelSpec` 데이터클래스 (model_cls, factory, sample_input_fn, tp_handler, weight_loader). `ModelRegistry`로 이름 기반 조회. `SafetensorsLoader`로 HF 가중치 로딩 |
| `tp_handler.py` | `BaseTensorParallelHandler`: fnmatch 글로브로 Column/Row-parallel 패턴 매칭 후 자동 분할. 매칭 안 되는 파라미터는 replicate |
| `ipc_channel.py` | ZMQ REP(서버) / REQ(클라이언트). Poller 기반 1초 타임아웃, LINGER=0, 클라이언트는 RCVTIMEO=5s + 소켓 재생성으로 REQ/REP 상태 머신 리셋 |
| `handle_codec.py` | `_share_cuda_()` 8-tuple + size/stride/dtype를 직렬화. TP 모드에서는 `{"tp_rank", "tp_world_size", "handles"}` envelope 포맷. `tp_rank` 키 유무로 자동 감지 |
| `weight_manager.py` | `ModelRegistry.get()` → 모델 생성, `spec.tp_handler.process_state_dict()`로 샤딩. `model_path` 지정 시 `WeightLoader`로 외부 가중치 로딩 |
| `inference_worker.py` | 단일/집중형 TP/분산형 TP 3가지 모드. TP-aware forward 내장 모델은 builtin 경로, 그 외는 legacy `TPSimpleMLP` 래퍼 사용 |
| `tensor_parallel.py` | `TPSimpleMLP`: 로컬 루프 TP forward. `DistributedTPForward`: NCCL 분산 forward. `shard_model()`: deprecated (하위 호환) |
| `models/tiny_llama.py` | LLaMA-style TinyTransformer (RMSNorm, SwiGLU MLP, multi-head attention). forward에 `dist.all_reduce` 내장 — 별도 TP 래퍼 불필요 |

## 실행 방법

### 설치

```bash
pip install -e ".[dev]"
```

### 유닛 테스트 (GPU 불필요)

```bash
python -m pytest tests/ -v
# 53 tests: IPC 7 + Codec 12 + TP 10 + ModelSpec 13 + TinyTransformer 11
```

### 단일 GPU 수동 실행

```bash
# 터미널 1: Weight Manager
python scripts/run_weight_manager.py --model mlp --device cuda:0

# 터미널 2: Inference Worker
python scripts/run_inference_worker.py --model mlp --device cuda:0
```

### Tensor Parallelism 수동 실행 (GPU 2개)

```bash
# 터미널 1: WM rank 0
python scripts/run_weight_manager.py --model mlp --device cuda:0 \
  --endpoint ipc:///tmp/wm_0.zmq --tp-rank 0 --tp-world-size 2 --seed 42

# 터미널 2: WM rank 1
python scripts/run_weight_manager.py --model mlp --device cuda:1 \
  --endpoint ipc:///tmp/wm_1.zmq --tp-rank 1 --tp-world-size 2 --seed 42

# 터미널 3: Worker (TP)
python scripts/run_inference_worker.py --model mlp \
  --endpoints "ipc:///tmp/wm_0.zmq,ipc:///tmp/wm_1.zmq"
```

### 분산 Tensor Parallelism 수동 실행 (GPU 2개)

```bash
# 터미널 1: WM rank 0
python scripts/run_weight_manager.py --model mlp --device cuda:0 \
  --endpoint ipc:///tmp/dtp_wm_0.zmq --tp-rank 0 --tp-world-size 2 --seed 42

# 터미널 2: WM rank 1
python scripts/run_weight_manager.py --model mlp --device cuda:1 \
  --endpoint ipc:///tmp/dtp_wm_1.zmq --tp-rank 1 --tp-world-size 2 --seed 42

# 터미널 3: Worker rank 0
python scripts/run_inference_worker.py --model mlp --device cuda:0 \
  --endpoint ipc:///tmp/dtp_wm_0.zmq \
  --distributed-tp --tp-rank 0 --tp-world-size 2

# 터미널 4: Worker rank 1
python scripts/run_inference_worker.py --model mlp --device cuda:1 \
  --endpoint ipc:///tmp/dtp_wm_1.zmq \
  --distributed-tp --tp-rank 1 --tp-world-size 2
```

### 자동 검증 스위트 (GPU 필요)

```bash
python scripts/run_verification.py --model mlp --device cuda:0 -v
```

## 검증 항목과 결과

최대 9개 테스트로 핵심 속성을 자동 검증합니다. Test 7-8은 GPU 2개 이상, Test 9는 4개 이상일 때 자동 실행됩니다.

| # | 테스트 | 검증 내용 | GPU 요구 | 결과 |
|---|--------|-----------|----------|------|
| 1 | Memory Sharing | 3개 Worker가 붙어도 GPU 메모리가 선형 증가하지 않음 | 1+ | PASS |
| 2 | Zero-Copy | WM이 가중치를 수정하면 Worker에서 즉시 관찰됨 | 1+ | PASS |
| 3 | Worker Lifecycle | 10회 Worker 생성/파괴 후에도 WM이 안정적 | 1+ | PASS |
| 4 | Inference Accuracy | 같은 seed로 2개 Worker 추론 결과 일치 (max diff 0.00) | 1+ | PASS |
| 5 | Memory Leak | 10회 warmup 후 40회 사이클에서 메모리 성장 0.0 MB | 1+ | PASS |
| 6 | Crash Recovery | Worker `kill -9` 후 `ipc_collect()` → 새 Worker 정상 동작 | 1+ | PASS |
| 7 | Tensor Parallelism | 집중형 TP: 1 Worker가 2 WM shard 수집, 참조 모델과 비교 | 2+ | PASS |
| 8 | Distributed TP (2 ranks) | 분산형 TP: 2 WM + 2 Worker, NCCL 통신, 전 rank 출력 일치 | 2+ | PASS |
| 9 | Distributed TP (4 ranks) | 분산형 TP: 4 WM + 4 Worker, NCCL 통신, 전 rank 출력 일치 | 4+ | PASS |

### Test 5 설계 노트

`nvidia-smi`는 CUDA 컨텍스트 메모리(프로세스당 ~100-200MB)를 포함합니다. 첫 수 회 worker spawn에서 일회성 증가 후 안정화되므로, 10회 warmup으로 컨텍스트 비용을 흡수한 뒤 baseline을 측정합니다. 이후 40회 사이클에서 성장률이 0에 가까운지 확인합니다.

### Test 7 설계 노트

두 WM 서브프로세스에 동일한 `--seed`를 전달하여 같은 가중치로 초기화합니다. Worker가 양쪽에서 shard를 수집하고 `TPSimpleMLP.forward()`로 추론한 결과를 단일 GPU 참조 모델과 `torch.allclose(atol=1e-5)`로 비교합니다.

### Test 8-9 설계 노트

분산형 TP에서는 각 Worker가 자신의 WM에서만 shard를 받고, `torch.distributed.init_process_group("nccl")`로 NCCL 프로세스 그룹을 초기화합니다. `DistributedTPForward`의 `all_gather`/`all_reduce`가 모든 rank에서 동일한 출력을 보장하므로, (1) rank 0 출력이 단일 GPU 참조와 일치하는지, (2) 모든 rank의 출력이 동일한지를 검증합니다.

## 커스텀 모델 추가

`ModelSpec`을 등록하면 기존 파이프라인을 그대로 사용할 수 있습니다.

```python
from cuda_ipc_poc.model_spec import ModelRegistry, ModelSpec, SafetensorsLoader
from cuda_ipc_poc.tp_handler import BaseTensorParallelHandler

class MyModelTPHandler(BaseTensorParallelHandler):
    COLUMN_WISE_PARAMS = ["layers.*.q_proj.weight", "layers.*.gate_proj.weight"]
    ROW_WISE_PARAMS = ["layers.*.o_proj.weight", "layers.*.down_proj.weight"]
    HAS_BUILTIN_TP_FORWARD = True  # 모델 자체에 dist.all_reduce 포함 시

ModelRegistry.register("my-model", ModelSpec(
    model_cls=MyModel,
    model_factory=lambda: MyModel(hidden=4096, layers=32),
    sample_input_fn=lambda device, batch_size=1: torch.randint(0, 32000, (batch_size, 128), device=device),
    tp_handler=MyModelTPHandler(),
    weight_loader=SafetensorsLoader(),  # safetensors 파일에서 가중치 로딩
))
```

```bash
# safetensors 가중치로 서빙
python scripts/run_weight_manager.py --model my-model --model-path /path/to/model.safetensors

# Worker 연결
python scripts/run_inference_worker.py --model my-model
```

**설계 원칙**: 프로덕션 모델(LLaMA 등)은 forward에 `dist.all_reduce`가 내장되어 있으므로, Worker는 모델 shell을 만들고 sharded state_dict를 `assign=True`로 주입하면 됩니다. 별도 TP forward 래퍼가 불필요합니다.

## 기술적 주의 사항

1. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`** -- PyTorch 2.2+의 expandable segments는 `cudaMemCreate`/`cudaMemMap`을 사용하여 `cudaIpcGetMemHandle`과 호환되지 않습니다. 반드시 CUDA 연산 전에 설정해야 합니다.

2. **ZMQ REQ/REP 상태 머신** -- `send()` 후 `recv()` 타임아웃 시 소켓이 깨진 상태가 됩니다. 재시도하려면 소켓을 닫고 새로 생성해야 합니다.

3. **TP shard 연속성** -- `torch.chunk()`는 뷰를 반환하므로 `.contiguous()` 호출 필수. CUDA IPC는 독립 할당된 메모리만 공유할 수 있습니다.

4. **다중 GPU `_lazy_init()`** -- Worker가 각 device의 IPC 핸들을 열기 전에 해당 device의 CUDA 컨텍스트를 초기화해야 합니다.

5. **하위 호환성** -- `tp_world_size=1` + 단일 endpoint는 기존 단일 GPU 동작과 동일합니다. handle_codec은 `tp_rank` 키 유무로 TP/legacy 포맷을 자동 감지합니다. 집중형 TP(`TPSimpleMLP`)도 그대로 유지됩니다.

6. **분산 TP 환경변수** -- `DistributedTPForward`를 사용하려면 `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE` 환경변수가 Worker 프로세스 시작 전에 설정되어야 합니다. CLI 스크립트의 `--distributed-tp` 플래그가 이를 자동 처리합니다.

## 의존성

- `torch>=2.1.0` -- CUDA IPC, `_share_cuda_()`, `_new_shared_cuda()`
- `pyzmq>=25.0` -- IPC 핸들 전송
- `psutil` -- 프로세스/메모리 유틸리티
- `numpy` -- 수치 연산
- `pytest>=7.0` (dev) -- 테스트
- `torchvision` (optional) -- ResNet18 모델
