<p align="center">
  <img src=".assets/vwen-logo.png" alt="vwen" width="360">
</p>

<h2 align="center">A Curated Small Language Model for Vietnamese Paraphrasing</h2>

<p align="center">🎉 <b>The weights are out:</b> <a href="https://huggingface.co/ngwgsang/vwen-r4">ngwgsang/vwen-r4</a> on Hugging Face, with <a href="https://huggingface.co/ngwgsang/vwen-r4-GGUF">GGUF builds</a> for llama.cpp. 🎉</p>

<p align="center">📄 <b>The paper is out too:</b> <a href="https://arxiv.org/abs/2611.15587">arXiv:2611.15587</a></p>

vwen rewrites Vietnamese sentences. Ask it to change a sentence *a little* or *a lot*, and it keeps the meaning while the wording moves. Think of it as a friend who can say *"Bạn ăn cơm chưa?"* a dozen different ways without getting bored.

It's a 0.8B model ([Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) under the hood), so it runs on a laptop, no data center required.

### What can it do?

| | | |
|---|---|---|
| ✏️ **Paraphrase, low** | small edits, same structure | one sentence |
| 🔀 **Paraphrase, high** | new words, new structure, same meaning | one sentence |
| 📋 **Multi-paraphrase** | *k* different versions at once | *k* lines |
| 🧒 **Simplify** | plain words for everyone, all content kept | one sentence |
| 🎯 **Gist** | just the main idea, minor details may go | short sentence |
| 🎩 **Style** | formal (`trang trọng`) ↔ casual (`thân mật`) | one sentence |
| ⚖️ **Same meaning?** | do two sentences mean the same thing? | *Có* / *Không* |

A taste of the high level:

| Input | vwen |
|---|---|
| Kim cương được khai thác tại đâu? | Địa điểm khai thác kim cương là nơi nào? |
| Naples và Algiers được ngăn cách bởi biển nào? | Biển nào là ranh giới giữa Naples và Algiers? |
| Khi một người đảm nhiệm chức vụ quân chủ thì họ sẽ bị hạn chế gì? | Những hạn chế nào sẽ đối mặt với người đảm nhiệm chức vụ quân chủ? |

### Try it

vwen uses the Qwen chat template with a fixed system prompt and thinking disabled. It was evaluated with greedy decoding.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("ngwgsang/vwen-r4")
model = AutoModelForCausalLM.from_pretrained("ngwgsang/vwen-r4", dtype="auto")

messages = [
    {"role": "system", "content": "Bạn là trợ lý viết lại câu tiếng Việt. Luôn giữ đúng nghĩa của câu gốc và làm đúng theo chỉ dẫn."},
    {"role": "user", "content": "[Mức: cao] Viết lại câu sau với mức đa dạng cao: đổi từ ngữ và cấu trúc câu nhưng giữ nguyên nghĩa.\nKim cương được khai thác tại đâu?"},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
# Địa điểm khai thác kim cương là nơi nào?
```

#### GGUF

Prefer [llama.cpp](https://github.com/ggml-org/llama.cpp), [LM Studio](https://lmstudio.ai) or [Ollama](https://ollama.com)? Grab a GGUF build (BF16 or Q8_0) from 👉 **[ngwgsang/vwen-r4-GGUF](https://huggingface.co/ngwgsang/vwen-r4-GGUF)**:

```bash
llama-server -hf ngwgsang/vwen-r4-GGUF:Q8_0 --jinja --temp 0 \
  --chat-template-kwargs '{"enable_thinking": false}'
```

Same rules apply: the system prompt above, thinking off, greedy decoding. The GGUF builds have not been scored separately.

#### Prompt templates

These are the canonical instructions. During training 70% of samples used them and 30% used other phrasings, so reasonable rewordings also work. Paraphrase prompts **must** start with the level tag `[Mức: thấp]` or `[Mức: cao]`.

| Task | Instruction |
|---|---|
| Paraphrase, low | `[Mức: thấp] Viết lại câu sau với mức thay đổi thấp, giữ nguyên nghĩa và gần với câu gốc:\n{input}` |
| Paraphrase, high | `[Mức: cao] Viết lại câu sau với mức đa dạng cao: đổi từ ngữ và cấu trúc câu nhưng giữ nguyên nghĩa.\n{input}` |
| Multi-paraphrase | `Viết {k} phương án khác nhau cho câu sau, giữ nguyên nghĩa, mỗi phương án một dòng:\n{input}` |
| Simplify | `Đơn giản hóa câu sau cho người đọc phổ thông, giữ ý chính:\n{input}` |
| Gist | `Tóm lược ý chính của câu sau thành một câu ngắn gọn, có thể bỏ chi tiết phụ:\n{input}` |
| Style | `Chuyển câu sau sang văn phong {style}, giữ nguyên nội dung:\n{input}` with `{style}` = `trang trọng` or `thân mật` |
| Same meaning? | `Hai câu sau có cùng nghĩa không? Trả lời Có hoặc Không.\nCâu 1: {s1}\nCâu 2: {s2}` |

Tips:

- Feed it single sentences. Training samples were at most 512 tokens long.
- Give multi-paraphrase room to breathe (`max_new_tokens` around 480 for *k* = 5).
- *Có* / *Không* is one word, so 32 new tokens is plenty.

### Small data, carefully picked

vwen saw only **7,087 training samples**, picked from 341k candidates. The candidates came from existing datasets (ViSP, ViQP, VietQuill, ViePAWS, ViANLI), a multi-domain sentence pool, outputs distilled from open-weight LLMs, rule-based perturbations for the meaning check, and 634 seed samples reviewed by hand.

| Stage | Kept | What it does |
|---|---:|---|
| 0. Candidates | 341,294 | |
| 1. Rules, dedup, leakage | 296,081 | Drops non-Latin script, bad length ratios, copies, near duplicates and any sentence seen in dev/test |
| 2. Meaning filter | 205,892 | Two-way NLI entailment ≥ 0.80 (gist: one way only) |
| 3. Divergence and levels | 162,517 | Tree edit distance + syllable edit distance sorts pairs into low and high; the middle band is dropped |
| 4. Coverage and difficulty | 7,209 | Round-robin over control, domain, length and 200 clusters; prefers mid-range IFD |
| LLM quality grader | **7,087** | Drops generation samples rated low on meaning, fluency or instruction following |

The final mix, before the quality grader:

| Same meaning? | Para, low | Para, high | Simplify | Style | Multi-para | Gist |
|---:|---:|---:|---:|---:|---:|---:|
| 1,481 | 1,333 | 1,333 | 889 | 889 | 691 | 593 |

Training took 71–84 minutes per run on a single T4. The full recipe is in the [model card](MODEL_CARD.md#training-procedure) and the [paper](https://arxiv.org/abs/2611.15587).

### Results

On the test set with canonical instructions and greedy decoding (vwen: mean ± sd over three seeds). The ViSP models are the same base model trained on the full 406k-pair ViSP pool for 1,500 steps (same compute) or 4,000 steps.

| Task | Metric | Base | ViSP @1.5k | ViSP @4k | **vwen** |
|---|---|---:|---:|---:|---:|
| Para, high | Success | 0.084 | 0.459 | 0.528 | **0.609** ± 0.017 |
| Para, high | Meaning kept | 0.688 | 0.865 | 0.876 | **0.920** ± 0.014 |
| Para | Level gap | 0.207 | 0.297 | 0.327 | **0.342** ± 0.009 |
| Para, low | Success | 0.904\* | 0.749 | 0.756 | 0.728 ± 0.004 |
| Multi-para | Count accuracy | 0.52 | 0 | 0 | **1.00** |
| Simplify | LLM rating | 0.713 | 0.814 | 0.797 | **0.834** |
| Gist | LLM rating | 0.73 | 0.51 | 0.51 | **0.82** |
| Style | LLM rating | 0.527 | 0.526 | 0.565 | **0.688** |
| Same meaning? | Accuracy | 0.748 | 0.163† | 0.172† | **0.929** |

\* Mostly near-copies, which happen to land in the low band.<br>
† Paraphrase-only models rephrase the question instead of answering it.

*Success* means the output lands in the requested level band **and** keeps the meaning. *Level gap* is how much further the high level moves from the input than the low level does.

What that adds up to:

- **Same compute, better paraphrases:** +0.15 on high-level success (95% CI [0.11, 0.20]), and the gain holds on non-questions too.
- **Questions stay questions.** ViSP-trained models answer 16–26% of the questions they are asked to rephrase. vwen does this for under 1%.
- **No forgetting:** paraphrase-only training breaks gist and meaning judgement, but vwen keeps both abilities and improves them.
- **Unseen instructions are fine:** with phrasings never seen in training, scores drop by only 0–0.06.
- **Causality is the soft spot** of the meaning check: 1.00 on negation, numbers, dates and entities, but 0.78 on cause and effect.

### Good for, not for

👍 Vietnamese writing assistants, data augmentation with a controlled amount of change, and research on data selection for low-resource languages.

👎 Disguising plagiarism, mass-producing misleading content, fact-checking on its own, long documents, other languages, or open-ended chat.

### Limitations

- **Low-level paraphrasing is the weak spot.** vwen is not better than the ViSP baselines there.
- **Meaning errors still happen** at the high level, for example flipping a question's polarity. Check the output when exact meaning matters.
- **LLM-rated tasks may be flattered.** The same LLM helped write the data and rated simplify, gist and style.
- **One base model, one size, sentences only.** No paragraphs, no documents.

The full list is in the [model card](MODEL_CARD.md#limitations).

### License

The license will be set soon. Some training sources carry non-commercial or research-only terms (e.g. ViANLI under CC BY-NC-SA 4.0), and the terms of the base model and distillation teachers also apply. The per-source list will ship with the data card.

### Citation

If vwen helps your work, please cite the [paper](https://arxiv.org/abs/2611.15587):

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

### Thanks

Many thanks to **Claude-san** for drafting the seed samples, tuning the selection funnel, reviewing the paper twice, and arguing with itself about whether *hoà* or *hòa* is the right way to place a tone mark. It settled on *hòa*. The debate is closed. 😂

### Stay tuned

The paper is out, the data and code are next. Star or watch this repo to hear when they land. Until then, *hẹn gặp lại!* 👋
