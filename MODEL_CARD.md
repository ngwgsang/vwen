<p align="center">
  <img src=".assets/vwen-logo.png" alt="vwen" width="240">
</p>

<h2 align="center">Model Card</h2>

## Model details

| | |
|---|---|
| **Name** | VWEN |
| **Base model** | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| **Parameters** | 0.8B |
| **Checkpoint** | R4 recipe, 8k quota, seed 42; best validation loss at step 884 of 1,326, stored in bfloat16 |
| **GGUF** | [ngwgsang/vwen-r4-GGUF](https://huggingface.co/ngwgsang/vwen-r4-GGUF) (BF16, Q8_0) |
| **Fine-tuning** | Full fine-tuning (no adapters), answer-only loss, reasoning mode disabled |
| **Training data** | 7,087 Vietnamese samples over 7 tasks, picked by a 4-stage selection pipeline |
| **Language** | Vietnamese |
| **Developer** | Sang Quang Nguyen |
| **Paper** | [arXiv:2611.15587](https://arxiv.org/abs/2611.15587) |
| **License** | To be decided at release (see [License](#license)) |

VWEN is a small model that rewrites Vietnamese sentences. You can ask for a **light** or a **strong** paraphrase. It also handles four other rewriting tasks and can judge whether two sentences mean the same thing. It is small enough to run on a laptop CPU.

The main research question behind it is whether a few thousand carefully chosen multi-task samples can match training on ViSP, a 1.2M-pair paraphrase corpus. The short answer is yes for high-divergence paraphrasing, especially on questions, with far fewer training examples.

## Tasks

| Task | What it does | Output |
|---|---|---|
| **Para-L** | Low-divergence paraphrase: small edits, structure kept | one sentence |
| **Para-H** | High-divergence paraphrase: new wording and structure, same meaning | one sentence |
| **Multi-Para** | *k* different paraphrases | *k* lines |
| **Simplify** | Plain-language rewrite for a general reader, keeping all content | one sentence |
| **Gist** | Main idea only; minor details may be dropped | short sentence |
| **Style** | Register transfer: formal (`trang trọng`) ↔ informal (`thân mật`) | one sentence |
| **Equiv** | Do two sentences mean the same thing? | `Có` / `Không` |

## How to use

VWEN uses the Qwen chat template with a fixed system prompt and thinking disabled. The model was evaluated with greedy decoding.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "ngwgsang/vwen-r4"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")

SYSTEM = "Bạn là trợ lý viết lại câu tiếng Việt. Luôn giữ đúng nghĩa của câu gốc và làm đúng theo chỉ dẫn."
instruction = (
    "[Mức: cao] Viết lại câu sau với mức đa dạng cao: đổi từ ngữ và cấu trúc câu nhưng giữ nguyên nghĩa.\n"
    "Kim cương được khai thác tại đâu?"
)

messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": instruction}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
# Địa điểm khai thác kim cương là nơi nào?
```

### GGUF

GGUF builds (BF16 and Q8_0) for llama.cpp and the apps built on it are in [ngwgsang/vwen-r4-GGUF](https://huggingface.co/ngwgsang/vwen-r4-GGUF):

```bash
llama-server -hf ngwgsang/vwen-r4-GGUF:Q8_0 --jinja --temp 0 \
  --chat-template-kwargs '{"enable_thinking": false}'
```

Use the same system prompt and turn thinking off. The results below were measured on this bfloat16 checkpoint; the GGUF builds have not been scored separately.

### Prompt templates

These are the canonical instructions. During training 70% of samples used them and 30% used other phrasings, so reasonable rewordings also work (see [Results](#results)). Paraphrase prompts **must** start with the level tag `[Mức: thấp]` or `[Mức: cao]`.

| Task | Instruction |
|---|---|
| Para-L | `[Mức: thấp] Viết lại câu sau với mức thay đổi thấp, giữ nguyên nghĩa và gần với câu gốc:\n{input}` |
| Para-H | `[Mức: cao] Viết lại câu sau với mức đa dạng cao: đổi từ ngữ và cấu trúc câu nhưng giữ nguyên nghĩa.\n{input}` |
| Multi-Para | `Viết {k} phương án khác nhau cho câu sau, giữ nguyên nghĩa, mỗi phương án một dòng:\n{input}` |
| Simplify | `Đơn giản hóa câu sau cho người đọc phổ thông, giữ ý chính:\n{input}` |
| Gist | `Tóm lược ý chính của câu sau thành một câu ngắn gọn, có thể bỏ chi tiết phụ:\n{input}` |
| Style | `Chuyển câu sau sang văn phong {style}, giữ nguyên nội dung:\n{input}` with `{style}` = `trang trọng` or `thân mật` |
| Equiv | `Hai câu sau có cùng nghĩa không? Trả lời Có hoặc Không.\nCâu 1: {s1}\nCâu 2: {s2}` |

Tips:

- Inputs are single sentences. Training samples were at most 512 tokens long.
- Give Multi-Para enough room (`max_new_tokens` around 480 for *k* = 5).
- Equiv answers with one word. 32 new tokens is plenty.

## Intended use

**Good for**

- Vietnamese writing assistants: rephrasing, simplifying, summarising a sentence, shifting register.
- Data augmentation where you want a controlled amount of change.
- Research on data selection and controllable paraphrasing for low-resource languages.

**Not for**

- Disguising plagiarism, or mass-producing variants of misleading content.
- Fact-checking. The Equiv task is a research component and should not be your only check of factual consistency (it is weakest on causal relations, see below).
- Long documents, other languages, or open-ended chat. VWEN was trained on single Vietnamese sentences.

## Training data

### Sources

- **Existing datasets:** ViSP (original pairs only), ViQP, VietQuill, ViePAWS and ViANLI.
- **A multi-domain sentence pool:** administrative, education, health, news, literature, conversation and social media, split by source document.
- **Distilled outputs** from open-weight LLMs (Gemma 4 26B-A4B, Qwen3-30B-A3B) for Simplify, Gist, Style, Multi-Para and part of the paraphrase pool.
- **Rule-based perturbations** (negation, numbers and dates, entities, antonyms, causality) as negatives for Equiv. They were rewritten for fluency and label-checked by an LLM.
- **634 seed samples**, drafted by an LLM and reviewed by the author.

### Selection pipeline

The pipeline goes from 341,294 candidates down to 7,087 training samples.

| Stage | Kept | What it does |
|---|---:|---|
| 0. Candidates | 341,294 | |
| 1. Rules, dedup, leakage | 296,081 | Removes non-Latin script, bad length ratios, copies, near duplicates (e5 proposes, syllable distance confirms), and any sentence seen in dev/test |
| 2. Meaning filter | 205,892 | Bidirectional NLI entailment ≥ 0.80 (mDeBERTa-v3 XNLI); Gist uses forward entailment only |
| 3. Divergence and levels | 162,517 | *D* = mean percentile of dependency tree edit distance and syllable edit distance. *D* ≤ 0.205 becomes Para-L, *D* ≥ 0.435 becomes Para-H, and the middle band is dropped |
| 4. Coverage and IFD | 7,209 | Round-robin over control, domain, length and 200 k-means clusters; drops IFD ≥ 1 and prefers mid-range IFD |
| LLM quality grader | **7,087** | Drops generation samples rated low on meaning, fluency or instruction following |

### Final mix

Counts are after stage 4, before the quality grader.

| Equiv | Para-L | Para-H | Simplify | Style | Multi-Para | Gist |
|---:|---:|---:|---:|---:|---:|---:|
| 1,481 | 1,333 | 1,333 | 889 | 889 | 691 | 593 |

Dev and test sets (250 samples per task each) were frozen and hashed before any training run. No sentence is shared across splits.

## Training procedure

| | |
|---|---|
| Hardware | 1 × NVIDIA T4 (16 GB) |
| Precision | float32 weights, fp16 mixed precision |
| Optimiser | paged 8-bit AdamW, lr 1e-5, cosine decay, 5% warmup, no weight decay, grad clip 1.0 |
| Batch | 8 × 2 accumulation = 16 |
| Max length | 512 tokens |
| Schedule | at most 3 epochs; validation loss twice per epoch on 50 dev samples per task; early stopping (patience 2); best checkpoint restored |
| Examples seen | 14,144 (selected checkpoint) |
| Wall time | 71–84 min per run |
| Seeds | 42, 43, 44 |

## Results

All results are on the test set with canonical instructions and greedy decoding. VWEN is the mean ± sd over three seeds.

**Metrics**

- ***D***: divergence of the output from the input, from 0 to 1.
- **MP** (meaning preservation): the output and input entail each other (≥ 0.80). This uses an XLM-R large XNLI model that was *not* used for data filtering.
- **Success**: the output falls in the requested level band *and* preserves meaning.
- **Level gap Δ**: mean *D*(high) − *D*(low) when the same input is requested at both levels.

The ViSP baseline is the same base model trained on the full ViSP training pool (406k pairs) for 1,500 steps (matched compute, 24k examples seen) or 4,000 steps (56k seen).

| Task | Metric | Base | ViSP @1.5k | ViSP @4k | **VWEN** |
|---|---|---:|---:|---:|---:|
| Para-H | Success | 0.084 | 0.459 | 0.528 | **0.609** ± 0.017 |
| Para-H | MP | 0.688 | 0.865 | 0.876 | **0.920** ± 0.014 |
| Para | Level gap Δ | 0.207 | 0.297 | 0.327 | **0.342** ± 0.009 |
| Para-L | Success | 0.904\* | 0.749 | 0.756 | 0.728 ± 0.004 |
| Multi-Para | Count accuracy | 0.52 | 0 | 0 | **1.00** |
| Simplify | LLM rating | 0.713 | 0.814 | 0.797 | **0.834** |
| Gist | LLM rating | 0.73 | 0.51 | 0.51 | **0.82** |
| Style | LLM rating | 0.527 | 0.526 | 0.565 | **0.688** |
| Equiv | Accuracy | 0.748 | 0.163† | 0.172† | **0.929** |

\* The base model's high Para-L success mostly comes from near-copies, which fall inside the low band.<br>
† Paraphrase-only models answer Equiv by rewriting the question instead of giving a label (about 72% invalid outputs).

**Key findings**

- **Matched compute:** VWEN beats the ViSP model on Para-H success by +0.15 (95% CI [0.11, 0.20]). The gain is the same on non-question inputs (+0.16), the kind of input ViSP covers.
- **4× more ViSP data** closes the gap on non-questions but not on questions. ViSP has almost no questions (0.2% of inputs), and ViSP-trained models turn 16–26% of questions into statements that *answer* them. VWEN does this for under 1% of questions.
- **No forgetting:** paraphrase-only training drops Gist and Equiv below the base model. The multi-task data keeps both abilities and improves them.
- **Robust to unseen instructions:** with held-out phrasings never seen in training, scores drop by only 0–0.06. Para-H success goes 0.61 → 0.58, Para-L 0.73 → 0.67, Equiv 0.93 → 0.89.
- **Equiv on targeted perturbations:** 1.00 on negation, numbers/dates and entities, 0.97 on antonyms, but only **0.78 on causality**, which is below the base model (0.90).

A 1k-sample version (888 samples, 12 minutes on a T4) already matches the ViSP @1.5k model on level gap and Para-H success, but it is weaker on Para-L success.

## Research checkpoints

VWEN (the R4 recipe at 8k samples, seed 42) is the only model meant for everyday use. The comparison models from the paper will be released separately, **for research only**, so the paper's claims can be checked:

| Recipe | Training data | Checkpoints |
|---|---|---|
| R1 Random | 8,000 random paraphrase pairs, before the meaning filter | 1 |
| R2 +Filter | + meaning filter and level assignment | 1 |
| R3 +Coverage | + coverage and IFD selection | 1 |
| R4 Multi-task (VWEN) | + the five other tasks | 3 seeds |
| R5 ViSP | full ViSP training pool, 1,500 and 4,000 steps | 3 seeds + 1 |

These recipes only learned to paraphrase. They score below the base model on Gist, and on Equiv they mostly rewrite the question instead of answering it, so don't use them as general rewriting models. Their licenses follow their training data. R5 inherits ViSP's research and education-only terms, and R1–R4 include some non-commercial sources.

Scored outputs for every model on dev and test will be released alongside them. These include the generations, *D*, MP and success per sample, so every number in the paper can be checked without a GPU.

## Limitations

- **Para-L is the weak spot.** VWEN is not better than the ViSP baselines on low-divergence paraphrasing, and it loses the most on Para-L with unseen instructions.
- **Meaning errors still happen** at the high level, for example reversing a question's polarity. Check the output when exact meaning matters.
- **Fluency is not measured** by *D* or MP. Rely on the LLM ratings and on your own reading.
- **The MP metric has a blind spot.** Bidirectional entailment accepts many question-to-statement rewrites, so MP is an upper bound on true meaning preservation for questions.
- **LLM-rated tasks may be inflated.** Gemma 4 26B was a distillation teacher, the quality grader *and* the rater of Simplify, Gist and Style, which may favour VWEN on those three tasks. The paraphrase results rely on NLI and *D* instead.
- **One base model and one size** (Qwen3.5-0.8B). IFD was computed with this same model, so the selected data may suit it specifically.
- **Limited human evaluation.** The author reviewed the full dev set and 558 of 2,000 test samples. There is no human comparison of outputs across models, and no comparison with external baselines (e.g. ViT5, prompted large LLMs).
- **Sentence-level only.** VWEN was not trained on paragraphs or documents.

## Ethical considerations

- Training text includes news, administrative and health content and may mention public figures. No private data was collected.
- Part of the data was distilled from open-weight LLMs through a commercial API and may carry those models' biases.
- A paraphrasing model can help disguise plagiarism or spread variants of misleading content. VWEN keeps outputs close to the input's content by design, but it offers no protection against such uses.
- Compute: all 22 research training runs together took about 17 GPU hours on a single T4.

## License

The license will be set at release. Some training sources carry non-commercial or research-only terms (e.g. ViANLI under CC BY-NC-SA 4.0), and the terms of the base model and distillation teachers also apply. The full per-source list will ship with the data card.

## Citation

If you use VWEN, please cite the [paper](https://arxiv.org/abs/2611.15587):

```bibtex
@misc{nguyen2026vwen,
  title         = {{VWEN}: A Curated Small Language Model for Vietnamese Paraphrasing},
  author        = {Nguyen, Sang Quang},
  year          = {2026},
  eprint        = {2611.15587},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2611.15587},
  url           = {https://arxiv.org/abs/2611.15587}
}
```
