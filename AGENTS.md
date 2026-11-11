# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project

**vwen** is a curated small language model for Vietnamese paraphrasing. It rewrites Vietnamese text so the wording changes but the meaning stays the same.

The model weights are released on Hugging Face as [`ngwgsang/vwen-r4`](https://huggingface.co/ngwgsang/vwen-r4) (GGUF builds in [`ngwgsang/vwen-r4-GGUF`](https://huggingface.co/ngwgsang/vwen-r4-GGUF)) and documented in `MODEL_CARD.md`. The paper, dataset and code are not released yet, and there are no build, lint or test commands. Don't invent them. Update this file when they're added.

## Layout

- `README.md`: the public landing page. It mirrors the main content of `MODEL_CARD.md` (details, usage, prompts, data, results, limitations, license, citation) in a lighthearted tone; keep both in sync.
- `MODEL_CARD.md`: the model card, based on the paper draft. Keep its numbers in sync with the paper.
- `.assets/`: images for docs. `vwen-logo.png` is the wordmark logo, shown centered at the top of the README.

## Conventions

- Write all repo docs (README, AGENTS.md, code comments) in **English**. The maintainer may talk to you in Vietnamese.
- The logo already shows the name "vwen", so don't add a separate `# vwen` heading next to it.
- Commit or push only when asked.
