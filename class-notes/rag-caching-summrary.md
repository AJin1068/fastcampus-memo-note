# RAG 운영: Caching 정리

슬라이드 "Caching 1. RAG에서 cache할 수 있는 세 층"과 "Caching 3. KV Cache: Prompt Caching이 실제로 동작하는 원리"에 GPU 하드웨어 관점의 설명을 더해 정리한 문서입니다.
더 자세한 GPU 설명은 [kv-cache-gpu.md](kv-cache-gpu.md)에 있습니다.

> **한 줄 요약**: 세 층의 cache는 모두 **"같은 입력을 두 번 계산하지 않는다"** 는 한 가지 원리이고, 어디서 재사용하느냐만 다릅니다.

---

## 1. 전체 그림: cache할 수 있는 세 층

| 층 | 무엇을 아끼나 | hit 조건 | 도구 |
|---|---|---|---|
| ① **응답 cache** (앱 층) | LLM 호출 자체 (GPU를 아예 안 씀) | 질문이 같거나 의미가 비슷함 | GPTCache, Redis/Valkey + 소형 embedding |
| ② **Prompt cache** (API 층) | 같은 prefix의 입력 token 비용 | token prefix가 완전히 같음 | Anthropic, OpenAI 등 API 기능 |
| ③ **KV cache** (추론 엔진 층) | 같은 prefix의 attention 계산 | 같은 token이 같은 위치에 있음 | vLLM, SGLang |

②는 "API 가격표"에 보이는 기능이고, ③은 그 기능이 GPU 안에서 실제로 구현되는 방식입니다. 바깥 층에서 hit할수록 절약이 큽니다.

---

## 2. ① 응답 cache: LLM을 아예 부르지 않음

이전 질문과 답을 저장해 두었다가 같은 질문이 오면 저장된 답을 바로 돌려줍니다.

- **Exact cache**: 질문 문자열이 완전히 같아야 hit입니다. "환불 어떻게 해요?"와 "환불 방법 알려줘"는 다른 질문이라 hit율이 **10~15%** 에 그칩니다.
- **Semantic cache**: 질문을 embedding으로 바꿔 유사도가 threshold 이상이면 같은 질문으로 봅니다.

**실사례**: exact에서 semantic으로 바꾸자 hit율 18% → 67%, 월 LLM 비용 $47K → $12.7K(-73%), 평균 latency 850ms → 300ms로 줄었습니다.
하지만 전역 threshold 0.85에서는 "구독 취소"와 "주문 취소"가 같은 질문으로 묶이는 사고가 났습니다. 그래서 사람이 라벨을 붙인 5,000쌍으로 카테고리별 threshold(FAQ 0.94, 거래 0.97)를 튜닝했습니다. 돈이 걸린 질문일수록 기준을 엄격하게 잡은 것입니다.

**한계**: 사람마다 다른 답(개인화)이나 시간이 지나면 바뀌는 답(재고, 가격)은 cache하면 안 됩니다. TTL(만료 시간)과 invalidation(무효화) 규칙이 없으면 **틀린 답을 빠르게** 주게 됩니다.

---

## 3. 배경 지식: GPU와 LLM 추론

### GPU의 구조
| 구성 요소 | H100 기준 | 비유 |
|---|---|---|
| SM + Tensor Core (계산 유닛) | BF16 약 989 TFLOPS | 엄청 빠른 요리사 |
| **HBM** (GPU 메인 메모리) | 80GB, 약 3.35 TB/s | 큰 창고 |
| SRAM (칩 내부 캐시) | SM당 수백 KB | 요리사 옆 작은 작업대 |

**HBM(High Bandwidth Memory)** 은 DRAM 칩을 수직으로 8~12층 쌓아 GPU 바로 옆에 붙인 메모리입니다. 통로가 매우 넓어 PC용 DDR5 RAM(약 0.05~0.1 TB/s)보다 수십 배 빠릅니다. 모델 가중치와 KV cache가 모두 여기에 올라갑니다. 하지만 GPU의 계산 속도에 비하면 여전히 느리고 용량도 한정되어 있습니다.

### 추론의 두 단계
| 단계 | 하는 일 | 병목 | 체감 지표 |
|---|---|---|---|
| **Prefill** | 입력 전체의 KV를 한 번에 계산 (행렬 × 행렬) | **compute-bound** (계산이 병목) | 첫 글자까지 시간 (TTFT) |
| **Decode** | token을 하나씩 생성 (행렬 × 벡터) | **memory-bound** (HBM에서 읽기가 병목) | 글자가 나오는 속도 |

- Prefill 계산량 ≈ `2 × 파라미터 수 × 입력 token 수`. 8B 모델에 4,000 token이면 약 64 TFLOP으로, 실제로는 150ms 안팎이 걸립니다.
- Decode는 token 하나마다 가중치 전체(8B 모델 FP16이면 16GB)를 HBM에서 읽어야 해서 token당 약 5ms가 걸립니다.
- **RAG는 검색한 문서를 prompt에 붙이므로 입력이 길어 prefill 비용이 큽니다.** 그래서 prefill을 건너뛰는 cache의 효과가 큽니다.

---

## 4. ③ KV cache: Prompt caching이 실제로 동작하는 원리

### KV cache란
Transformer는 token을 하나 생성할 때마다 **앞의 모든 token**에 대한 attention을 계산합니다. 이를 위해 token마다, layer마다 **Key(K)** 와 **Value(V)** 벡터가 필요합니다.
한 번 계산한 앞 token들의 K, V를 **GPU 메모리(HBM)에 저장해 두고 재사용**하는 것이 KV cache입니다. 없으면 token 하나를 만들 때마다 앞부분 전체를 다시 계산해야 합니다.

### 크기
```
KV cache 크기 = 2(K와 V) × layer 수 × hidden 차원 × bytes × token 수
```
- **Llama-2-7B (FP16)**: 2 × 32 × 4,096 × 2바이트 = token당 약 **0.5MB** → 4,096 token이면 약 **2GB**
- 긴 context가 GPU 메모리를 잡아먹는 이유가 이것입니다. KV cache가 차지하는 공간이 곧 **동시에 처리할 수 있는 요청 수의 상한**입니다.
- 참고: 최신 모델은 GQA(Grouped-Query Attention)로 K, V head 수를 줄입니다. 이때는 공식의 "hidden 차원" 대신 `KV head 수 × head 차원`을 씁니다. Llama-3-8B는 KV head가 8개뿐이라 token당 약 **128KB**로, Llama-2-7B의 1/4입니다.

### Prefix caching
원래 KV cache는 한 요청 안에서만 쓰고 버립니다. **Prefix caching**은 이것을 **요청 사이에서** 재사용합니다.

```
[system prompt + few-shot + tool 정의]  ← 매 요청 동일 → KV를 한 번만 계산해 재사용
[검색된 문서 chunk]                      ← 질문마다 다름 → 새로 계산
[사용자 질문]                            ← 질문마다 다름 → 새로 계산
```

- 여러 요청이 같은 prefix(system prompt, 같은 문서)를 공유하면 그 부분의 KV를 한 번만 계산해 둡니다. **API 제공자의 prompt caching이 바로 이것입니다.**
- **Prefix가 완전히 같아야 하는 이유**: 각 token의 K, V는 그 **앞의 모든 token**과 **자기 위치**(RoPE 위치 인코딩)에 의존합니다. 앞부분이 한 글자라도 다르면 그 뒤의 KV는 전부 달라집니다. 그래서 **고정된 내용은 앞에, 바뀌는 내용은 뒤에** 둬야 합니다. system prompt에 현재 시각을 넣으면 cache가 매번 깨집니다.
- **Prefill만 빨라지고 decode 비용은 그대로입니다.** 답변을 생성하는 단계는 cache와 무관하게 똑같이 돌기 때문입니다.

### 구현: vLLM과 SGLang
- **vLLM**: `enable_prefix_caching=True` 한 줄로 Automatic Prefix Caching이 켜집니다.
  - KV cache를 OS 가상 메모리처럼 **16 token짜리 block(page)** 으로 쪼개 관리합니다(PagedAttention).
  - 각 block을 "이 block의 token + 앞 모든 token"의 hash로 식별하고, 새 요청의 앞 block hash가 일치하면 계산 없이 그대로 연결합니다.
  - 같은 prefix block을 여러 요청이 물리적으로 하나만 공유하므로 HBM도 아낍니다. 공간이 부족하면 오래 안 쓴 block부터 지웁니다(LRU).
- **SGLang RadixAttention**: prompt와 생성 결과의 KV를 **radix tree**(공통 접두사를 공유하는 트리)로 관리해 공유 prefix를 자동으로 찾습니다.
  - system prompt → few-shot → 1턴 → 2턴처럼 가지가 갈라지는 구조에 강합니다.
  - few-shot, multi-turn, RAG 벤치마크에서 **최대 5배 throughput**을 보였습니다.

---

## 5. ② Prompt cache: API 층에서 보이는 모습

LLM API 제공자가 같은 prefix를 다시 처리하지 않고 **싼 읽기 비용만** 받는 기능입니다. RAG는 system prompt + few-shot + tool 정의가 매 요청 반복되므로 효과가 큽니다.

가격 구조를 GPU 관점에서 보면 이렇습니다. (Anthropic 기준 대략치: 쓰기는 기본 입력 단가의 약 1.25배(5분 TTL), 읽기는 약 0.1배)
- **읽기가 싼 이유**: 가장 비싼 prefill 계산을 건너뛰기 때문입니다.
- **쓰기가 더 비싼 이유**: 계산은 똑같이 하면서, 결과를 HBM에 계속 붙잡아 두어 그만큼 다른 요청을 못 받기 때문입니다.
- **TTL이 있는 이유**: HBM이 한정 자원이기 때문입니다. 더 오래 두려면 CPU 메모리나 SSD로 내려 보냅니다(LMCache, Mooncake 등).

**활용 팁**
- **작은 KB(수십만 token 이하)** 는 검색 대신 **문서 전체를 cache에 올리고** 질문만 바꾸는 방식이 더 싸고 정확할 수 있습니다. 검색이 관련 chunk를 놓치는 문제가 사라집니다.
- **Anthropic Contextual Retrieval**도 문서 전체를 cache해서, chunk마다 문맥 설명을 생성하는 비용을 문서 100만 token당 약 **$1.02**로 낮췄습니다.

---

## 6. 다시 한 번 정리

| 층 | GPU 관점에서 건너뛰는 것 | 주의할 점 |
|---|---|---|
| ① 응답 cache | prefill + decode 전부 | threshold 튜닝, 개인화·시간 민감 답변 제외, TTL/invalidation |
| ② Prompt cache | prefill (입력 처리) | prefix를 정확히 같게 유지, 바뀌는 내용은 뒤로 |
| ③ KV cache | prefill의 attention 계산 | HBM 용량이 한정되어 eviction 발생 |

세 층 모두 **"같은 입력을 두 번 계산하지 않는다"** 는 원리이고, 재사용하는 위치가 앱이냐, API냐, GPU 메모리냐의 차이입니다.
