"""Fine-tune Qwen models with grammar-constrained SFT + optional GRPO.

SFT uses grammar-masked logits (custom training loop) so the model
learns valid output format intrinsically. GRPO refines boundaries
using IoU-based reward with constrained generation.

Usage:
  python -m models.qwen.train --model 0.5B
  python -m models.qwen.train --model 3B --phases sft,grpo
  python -m models.qwen.train --model 1.5B --max-seq-len 8192
  python -m models.qwen.train --model 0.5B --sft-epochs 5 --output-dir /workspace/output
"""

import argparse
import json
import math
import os
import re
import time

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader

MODEL_MAP = {
    "0.5B": "unsloth/Qwen2.5-0.5B-Instruct",
    "1.5B": "unsloth/Qwen2.5-1.5B-Instruct",
    "3B": "unsloth/Qwen2.5-3B-Instruct",
}

SYSTEM_PROMPT = "Return content line ranges as start:end pairs."


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Qwen with grammar-constrained SFT + GRPO")
    p.add_argument("--model", default="0.5B", choices=list(MODEL_MAP.keys()),
                   help="Model size (default: 0.5B)")
    p.add_argument("--phases", default="sft,grpo",
                   help="Comma-separated phases to run: sft, grpo (default: sft,grpo)")
    p.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "data"),
                   help="Directory containing train.jsonl and test.jsonl")
    p.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "output"),
                   help="Directory for adapter outputs")
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--sft-epochs", type=int, default=3)
    p.add_argument("--sft-max-steps", type=int, default=0,
                   help="Stop SFT after N optimizer steps (0 = full run); for smoke tests")
    p.add_argument("--sft-lr", type=float, default=2e-4)
    p.add_argument("--sft-batch", type=int, default=1)
    p.add_argument("--sft-grad-accum", type=int, default=8)
    p.add_argument("--grpo-max-steps", type=int, default=0,
                   help="Stop GRPO after N optimizer steps (0 = full run); for smoke tests")
    p.add_argument("--grpo-epochs", type=int, default=1)
    p.add_argument("--grpo-lr", type=float, default=5e-6)
    p.add_argument("--grpo-num-generations", type=int, default=4)
    p.add_argument("--grpo-max-completion", type=int, default=128)
    p.add_argument("--grpo-beta", type=float, default=0.1)
    p.add_argument("--grpo-temperature", type=float, default=0.7)
    p.add_argument("--reward-beta", type=float, default=2.0)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--adapter", default=None,
                   help="Path to existing adapter to resume from (for grpo-only runs)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tokenize_with_labels(row, tokenizer, max_seq_length, response_tpl):
    """Tokenize a chat row with -100 masking for prompt tokens."""
    text = tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=False,
    )
    enc = tokenizer(text, truncation=True, max_length=max_seq_length)
    ids = enc["input_ids"]
    labels = list(ids)
    tpl_len = len(response_tpl)
    found = -1
    for j in range(len(ids) - tpl_len + 1):
        if ids[j:j + tpl_len] == response_tpl:
            found = j
    if found >= 0:
        for j in range(found + tpl_len):
            labels[j] = -100
        # The chat template appends "\n" after <|im_end|>. Tokens after the
        # first EOS are grammar-invalid targets (-inf logit -> inf loss),
        # so mask everything past the EOS.
        eos_id = tokenizer.eos_token_id
        for j in range(found + tpl_len, len(ids)):
            if ids[j] == eos_id:
                for k in range(j + 1, len(ids)):
                    labels[k] = -100
                break
    else:
        labels = [-100] * len(ids)
    return {"input_ids": ids, "labels": labels, "attention_mask": enc["attention_mask"]}


# ---------------------------------------------------------------------------
# Grammar FSM
# ---------------------------------------------------------------------------

TRANSITIONS = {
    0: {**{d: 1 for d in "0123456789"}},
    1: {**{d: 1 for d in "0123456789"}, ":": 2},
    2: {**{d: 3 for d in "0123456789"}},
    3: {**{d: 3 for d in "0123456789"}, ",": 4},
    4: {**{d: 5 for d in "0123456789"}},
    5: {**{d: 5 for d in "0123456789"}, ":": 6},
    6: {**{d: 7 for d in "0123456789"}},
    7: {**{d: 7 for d in "0123456789"}, ",": 4},
}
ACCEPTING = {3, 7}
STATE_TO_IDX = {s: s for s in range(8)}
STATE_TO_IDX[-1] = 8


def build_grammar(tokenizer, model_vocab_size):
    """Build FSM token masks. Returns (stacked_masks, advance_map)."""
    eos_id = tokenizer.eos_token_id
    all_vocab = tokenizer.get_vocab()

    print("Building grammar FSM token masks...")
    state_valid = {}
    for state in TRANSITIONS:
        valid = []
        for token_str, token_id in all_vocab.items():
            if token_id == eos_id:
                continue
            chars = tokenizer.decode([token_id])
            s = state
            ok = True
            for ch in chars:
                if s not in TRANSITIONS or ch not in TRANSITIONS[s]:
                    ok = False
                    break
                s = TRANSITIONS[s][ch]
            if ok and len(chars) > 0:
                valid.append((token_id, s))
        state_valid[state] = valid

    for state in ACCEPTING:
        state_valid[state].append((eos_id, -1))
    state_valid[-1] = [(eos_id, -1)]

    # Build advance lookup
    advance_map = {}
    for s in list(range(8)) + [-1]:
        advance_map[s] = {tid: ns for tid, ns in state_valid.get(s, [])}

    # Build stacked mask tensor (9, model_vocab_size)
    stacked = torch.full((9, model_vocab_size), -math.inf)
    for state in list(range(8)) + [-1]:
        idx = STATE_TO_IDX[state]
        for token_id, _ in state_valid[state]:
            if token_id < model_vocab_size:
                stacked[idx, token_id] = 0.0

    mask_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    stacked = stacked.to(dtype=mask_dtype, device="cuda")

    for state in range(8):
        n = len(state_valid[state])
        acc = "  [ACCEPT]" if state in ACCEPTING else ""
        print(f"  State {state}: {n} valid tokens{acc}")

    # Quick sanity check
    for test_str in ["15:85", "15:85,90:120", "1:1"]:
        toks = tokenizer.encode(test_str, add_special_tokens=False)
        s = 0
        ok = True
        for tid in toks:
            if tid not in advance_map.get(s, {}):
                ok = False
                break
            s = advance_map[s][tid]
        status = "OK" if ok and s in ACCEPTING else "FAIL"
        print(f"  Walk {test_str!r}: {status}")

    return stacked, advance_map, state_valid


class GrammarLogitsProcessor:
    """Transformers-compatible logits processor for constrained inference.

    Derives the FSM state per sequence from the tokens actually generated,
    so it stays correct under sampling (the sampled token is not always the
    argmax) and under batched generation (each row has its own state).
    """

    def __init__(self, stacked_masks, advance_map):
        self.stacked_masks = stacked_masks
        self.advance_map = advance_map
        self.prompt_len = None

    def _state_for(self, generated):
        state = 0
        for tid in generated:
            state = self.advance_map.get(state, {}).get(tid, state)
        return state

    def __call__(self, input_ids, scores):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if self.prompt_len is None:
            self.prompt_len = input_ids.shape[1]
        vocab_size = scores.shape[-1]
        batched = scores.dim() == 2
        for b in range(input_ids.shape[0]):
            state = self._state_for(input_ids[b, self.prompt_len:].tolist())
            idx = STATE_TO_IDX.get(state, 8)
            mask = self.stacked_masks[idx, :vocab_size].to(scores.dtype)
            if batched:
                scores[b] = scores[b] + mask
            else:
                scores = scores + mask
        return scores


# ---------------------------------------------------------------------------
# SFT: Grammar-masked training loop
# ---------------------------------------------------------------------------

class TokenizedDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def make_collate_fn(pad_id):
    def collate_fn(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, attention_mask, labels = [], [], []
        for b in batch:
            pad_len = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(b["attention_mask"] + [0] * pad_len)
            labels.append(b["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }
    return collate_fn


def run_sft(model, tokenizer, train_tokenized, stacked_masks, advance_map, args):
    """Grammar-masked SFT with custom training loop."""
    from unsloth import FastLanguageModel
    # for_training/for_inference only flip flags on the model spine, not the
    # decoder layers — pair them with train()/eval() to keep both in sync.
    FastLanguageModel.for_training(model)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    train_loader = DataLoader(
        TokenizedDataset(train_tokenized),
        batch_size=args.sft_batch, shuffle=True,
        collate_fn=make_collate_fn(pad_id), drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.sft_lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.sft_epochs // args.sft_grad_accum
    if args.sft_max_steps > 0:
        total_steps = min(total_steps, args.sft_max_steps)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps,
    )
    use_bf16 = torch.cuda.is_bf16_supported()

    print(f"\nSFT: {len(train_tokenized)} samples, {args.sft_epochs} epochs, "
          f"{total_steps} steps, bf16={use_bf16}")

    model.train()
    global_step = 0
    accum_loss = 0.0
    t0 = time.time()

    for epoch in range(args.sft_epochs):
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            with autocast("cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                batch_size, seq_len, vocab_size = logits.shape

                all_logit_rows = []
                all_targets = []
                for b in range(batch_size):
                    target_pos = (labels[b] != -100).nonzero(as_tuple=True)[0]
                    if len(target_pos) == 0:
                        continue
                    state = 0
                    for tp in target_pos:
                        lpos = tp.item() - 1
                        token_id = labels[b, tp].item()
                        if 0 <= lpos < seq_len:
                            mask = stacked_masks[STATE_TO_IDX[state], :vocab_size].to(logits.dtype)
                            all_logit_rows.append(logits[b, lpos] + mask)
                            all_targets.append(token_id)
                        state = advance_map.get(state, {}).get(token_id, state)

                if all_logit_rows:
                    stacked_logits = torch.stack(all_logit_rows)
                    target_tensor = torch.tensor(all_targets, device="cuda")
                    loss = F.cross_entropy(stacked_logits, target_tensor)
                else:
                    loss = torch.tensor(0.0, device="cuda", requires_grad=True)
                loss = loss / args.sft_grad_accum

            loss.backward()
            accum_loss += loss.item()

            if (batch_idx + 1) % args.sft_grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    avg = accum_loss / 10 * args.sft_grad_accum
                    lr = scheduler.get_last_lr()[0]
                    print(f"  step {global_step}/{total_steps}  loss={avg:.4f}  lr={lr:.2e}")
                    accum_loss = 0.0

                if args.sft_max_steps > 0 and global_step >= args.sft_max_steps:
                    print(f"  Reached --sft-max-steps {args.sft_max_steps}, stopping.")
                    print(f"SFT complete in {time.time() - t0:.0f}s")
                    return model

        print(f"  Epoch {epoch + 1}/{args.sft_epochs} done.")

    elapsed = time.time() - t0
    print(f"SFT complete in {elapsed:.0f}s")
    return model


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------

def run_grpo(model, tokenizer, raw_train, stacked_masks, advance_map, state_valid, args):
    """GRPO phase with grammar-constrained generation."""
    from trl import GRPOTrainer, GRPOConfig
    from datasets import Dataset as HFDataset

    max_prompt_tokens = args.max_seq_len - args.grpo_max_completion

    def to_grpo_row(row):
        return {
            "prompt": [
                {"role": "system", "content": row["messages"][0]["content"]},
                {"role": "user", "content": row["messages"][1]["content"]},
            ],
            "ground_truth": row["messages"][2]["content"],
        }

    # Filter by prompt token length
    filtered = []
    skipped = 0
    for row in raw_train:
        grpo_row = to_grpo_row(row)
        prompt_text = tokenizer.apply_chat_template(
            grpo_row["prompt"], tokenize=False, add_generation_prompt=True
        )
        if len(tokenizer.encode(prompt_text)) <= max_prompt_tokens:
            filtered.append(grpo_row)
        else:
            skipped += 1

    grpo_data = HFDataset.from_list(filtered)
    print(f"\nGRPO: {len(grpo_data)} samples ({skipped} skipped as too long)")

    beta_sq = args.reward_beta ** 2

    def _extract_content(item):
        if isinstance(item, str):
            return item
        if isinstance(item, list):
            for msg in reversed(item):
                if isinstance(msg, dict) and "content" in msg:
                    return msg["content"]
        return ""

    def fbeta_reward(completions, **kwargs):
        ground_truth = kwargs.get("ground_truth", [])
        prompts = kwargs.get("prompts", [])
        rewards = []
        for i, completion in enumerate(completions):
            text = _extract_content(completion)
            gold = ground_truth[i] if i < len(ground_truth) else ""

            prompt = prompts[i] if i < len(prompts) else ""
            if isinstance(prompt, list):
                user_content = ""
                for msg in prompt:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        user_content = msg["content"]
                prompt_text = user_content
            else:
                prompt_text = prompt

            max_line = get_max_line(prompt_text)
            if max_line == 0:
                rewards.append(0.0)
                continue

            gold_lines = parse_ranges(gold, max_line) or set()
            pred_lines = parse_ranges(text, max_line)
            if pred_lines is None:
                rewards.append(0.0)
                continue
            if not pred_lines and not gold_lines:
                rewards.append(1.0)
                continue

            intersection = len(pred_lines & gold_lines)
            precision = intersection / len(pred_lines) if pred_lines else 0.0
            recall = intersection / len(gold_lines) if gold_lines else 0.0

            if precision + recall == 0:
                rewards.append(0.0)
            else:
                fbeta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
                rewards.append(fbeta)
        return rewards

    from unsloth import FastLanguageModel
    FastLanguageModel.for_training(model)
    model.train()

    grpo_config = GRPOConfig(
        output_dir=os.path.join(args.output_dir, "grpo_tmp"),
        num_generations=args.grpo_num_generations,
        max_completion_length=args.grpo_max_completion,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=args.grpo_epochs,
        max_steps=args.grpo_max_steps if args.grpo_max_steps > 0 else -1,
        learning_rate=args.grpo_lr,
        beta=args.grpo_beta,
        temperature=args.grpo_temperature,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
        save_strategy="no",
        max_prompt_length=max_prompt_tokens,
        use_vllm=False,
        # default (= grad_accum) generates 16 x 8k-token prompts in one
        # generate call -> ~11 GiB attention prefill OOM on a 24 GB card
        steps_per_generation=1,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=fbeta_reward,
        args=grpo_config,
        train_dataset=grpo_data,
    )

    # unsloth 2025.6.3's compiled _prepare_inputs probes
    # self.llm.llm_engine.vllm_config.model_config even when use_vllm=False;
    # stub the chain so the getattr resolves to False instead of AttributeError
    from types import SimpleNamespace
    trainer.llm = SimpleNamespace(
        llm_engine=SimpleNamespace(vllm_config=SimpleNamespace(model_config=SimpleNamespace()))
    )

    # Inject constrained generation
    orig_gen = trainer.model.generate

    def constrained_generate(*a, **kw):
        if kw.get("temperature", 0) > 0 or kw.get("do_sample", False):
            proc = GrammarLogitsProcessor(stacked_masks, advance_map)
            existing = kw.get("logits_processor", None)
            if existing is None:
                kw["logits_processor"] = [proc]
            else:
                existing.append(proc)
        return orig_gen(*a, **kw)

    trainer.model.generate = constrained_generate

    t0 = time.time()
    stats = trainer.train()
    try:
        print(f"GRPO time: {stats.metrics['train_runtime']:.0f}s")
    except Exception as e:
        print(f"(train stats unavailable: {e})")

    # Restore original generate
    model.generate = orig_gen
    print(f"GRPO complete in {time.time() - t0:.0f}s")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def get_max_line(markdown):
    mx = 0
    for m in re.finditer(r"^\s*(\d+)\s*\|", markdown, re.MULTILINE):
        n = int(m.group(1))
        if n > mx:
            mx = n
    return mx


def parse_ranges(text, max_line):
    text = text.strip()
    if not text:
        return set()
    lines = set()
    for part in text.split(","):
        part = part.strip()
        m = re.match(r"^(\d+):(\d+)$", part)
        if not m:
            return None
        s, e = int(m.group(1)), int(m.group(2))
        if s < 1 or e < s or e > max_line:
            continue
        for n in range(s, e + 1):
            lines.add(n)
    return lines


def compute_metrics(predicted, truth):
    if not predicted and not truth:
        return {"precision": 1.0, "recall": 1.0, "iou": 1.0}
    intersection = len(predicted & truth)
    union = len(predicted | truth)
    return {
        "precision": intersection / len(predicted) if predicted else 0.0,
        "recall": intersection / len(truth) if truth else 0.0,
        "iou": intersection / union if union else 0.0,
    }


def run_eval(model, tokenizer, raw_test, stacked_masks, advance_map, label, max_seq_len,
             use_constrained=False, max_samples=0):
    """Evaluate model on test set. Returns list of per-page results."""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    # for_inference leaves decoder layers with training=True after model.train();
    # with transformers >=4.52 GradientCheckpointingLayer then strips use_cache
    # while the model spine still expects it -> IndexError. eval() clears all.
    model.eval()

    if max_samples > 0:
        raw_test = raw_test[:max_samples]

    test_pages = []
    for row in raw_test:
        md = row["messages"][1]["content"]
        gt_text = row["messages"][2]["content"]
        truth = []
        for part in gt_text.split(","):
            m = re.match(r"(\d+):(\d+)", part.strip())
            if m:
                truth.append({"start": int(m.group(1)), "end": int(m.group(2))})
        test_pages.append({
            "page_id": row.get("page_id"),
            "url": row.get("url", ""),
            "markdown": md,
            "truth_ranges": truth,
        })

    results = []
    for i, page in enumerate(test_pages):
        md = page["markdown"]
        max_line = get_max_line(md)
        truth_lines = set()
        for r in page["truth_ranges"]:
            for n in range(r["start"], r["end"] + 1):
                truth_lines.add(n)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": md},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=max_seq_len).to("cuda")
        gen_kwargs = dict(**inputs, max_new_tokens=128, temperature=0.0, do_sample=False)
        if use_constrained:
            gen_kwargs["logits_processor"] = [
                GrammarLogitsProcessor(stacked_masks, advance_map)
            ]
        output = model.generate(**gen_kwargs)
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)

        pred_lines = parse_ranges(response, max_line)
        if pred_lines is None:
            results.append(None)
            if i < 20:
                print(f"  [{i+1}/{len(test_pages)}] pid={page['page_id']} "
                      f"PARSE_ERROR: {response[:80]}")
        else:
            metrics = compute_metrics(pred_lines, truth_lines)
            results.append(metrics)

        if (i + 1) % 100 == 0:
            ok_so_far = [r for r in results if r is not None]
            if ok_so_far:
                iou = sum(r["iou"] for r in ok_so_far) / len(ok_so_far)
                print(f"  [{i+1}/{len(test_pages)}] running IoU={iou:.3f}")

    ok = [r for r in results if r is not None]
    n_fail = sum(1 for r in results if r is None)
    print(f"\n{label} — {len(ok)}/{len(results)} succeeded ({n_fail} parse failures)")
    if ok:
        mi = sum(r["iou"] for r in ok) / len(ok)
        mp = sum(r["precision"] for r in ok) / len(ok)
        mr = sum(r["recall"] for r in ok) / len(ok)
        print(f"  Mean IoU:  {mi:.3f}")
        print(f"  Mean Prec: {mp:.3f}")
        print(f"  Mean Rec:  {mr:.3f}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    phases = [p.strip() for p in args.phases.split(",")]
    model_name = MODEL_MAP[args.model]
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"Model:       {model_name}")
    print(f"Phases:      {phases}")
    print(f"Max seq len: {args.max_seq_len}")
    print(f"Data dir:    {data_dir}")
    print(f"Output dir:  {output_dir}")
    print("=" * 70)

    # Load data
    train_path = os.path.join(data_dir, "train.jsonl")
    test_path = os.path.join(data_dir, "test.jsonl")
    assert os.path.exists(train_path), f"train.jsonl not found in {data_dir}"
    assert os.path.exists(test_path), f"test.jsonl not found in {data_dir}"

    raw_train = load_jsonl(train_path)
    raw_test = load_jsonl(test_path)
    print(f"Loaded {len(raw_train)} train, {len(raw_test)} test samples")

    # Load model
    from unsloth import FastLanguageModel

    if args.adapter and os.path.exists(args.adapter):
        print(f"Loading adapter from {args.adapter}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.adapter,
            max_seq_length=args.max_seq_len,
            dtype=None,
            load_in_4bit=True,
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=args.max_seq_len,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_r,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    model.print_trainable_parameters()

    # Build grammar
    stacked_masks, advance_map, state_valid = build_grammar(tokenizer, model.config.vocab_size)

    # SFT phase
    if "sft" in phases:
        print("\n" + "=" * 70)
        print("PHASE: SFT (grammar-masked)")
        print("=" * 70)

        response_tpl = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        train_tokenized = [
            tokenize_with_labels(r, tokenizer, args.max_seq_len, response_tpl)
            for r in raw_train
        ]
        n_total = len(train_tokenized)
        # Drop samples whose response was truncated away (all labels -100) —
        # they contribute nothing but still cost forward passes.
        train_tokenized = [t for t in train_tokenized
                           if any(l != -100 for l in t["labels"])]
        print(f"Tokenized: {n_total} samples, {len(train_tokenized)} with targets "
              f"({n_total - len(train_tokenized)} dropped as truncated)")

        model = run_sft(model, tokenizer, train_tokenized, stacked_masks, advance_map, args)

        sft_dir = os.path.join(output_dir, f"sft-{args.model.lower()}")
        model.save_pretrained(sft_dir)
        tokenizer.save_pretrained(sft_dir)
        print(f"Saved SFT adapter to {sft_dir}")

        # Eval after SFT
        eval_n = 5 if args.sft_max_steps > 0 else 0
        print("\n--- SFT Eval (unconstrained) ---")
        sft_free = run_eval(model, tokenizer, raw_test, stacked_masks, advance_map,
                            "SFT (free)", args.max_seq_len, use_constrained=False,
                            max_samples=eval_n)
        print("\n--- SFT Eval (constrained) ---")
        sft_constrained = run_eval(model, tokenizer, raw_test, stacked_masks, advance_map,
                                   "SFT (constrained)", args.max_seq_len, use_constrained=True,
                                   max_samples=eval_n)

    # GRPO phase
    if "grpo" in phases:
        print("\n" + "=" * 70)
        print("PHASE: GRPO (constrained generation)")
        print("=" * 70)

        model = run_grpo(model, tokenizer, raw_train, stacked_masks, advance_map,
                         state_valid, args)

        grpo_dir = os.path.join(output_dir, f"grpo-{args.model.lower()}")
        model.save_pretrained(grpo_dir)
        tokenizer.save_pretrained(grpo_dir)
        print(f"Saved GRPO adapter to {grpo_dir}")

        # Eval after GRPO
        eval_n = 5 if args.grpo_max_steps > 0 else 0
        print("\n--- GRPO Eval (unconstrained) ---")
        grpo_free = run_eval(model, tokenizer, raw_test, stacked_masks, advance_map,
                             "GRPO (free)", args.max_seq_len, use_constrained=False,
                             max_samples=eval_n)
        print("\n--- GRPO Eval (constrained) ---")
        grpo_constrained = run_eval(model, tokenizer, raw_test, stacked_masks, advance_map,
                                    "GRPO (constrained)", args.max_seq_len, use_constrained=True,
                                    max_samples=eval_n)

    # Save eval results
    eval_out = {}
    if "sft" in phases:
        eval_out["sft_free"] = sft_free
        eval_out["sft_constrained"] = sft_constrained
    if "grpo" in phases:
        eval_out["grpo_free"] = grpo_free
        eval_out["grpo_constrained"] = grpo_constrained

    eval_path = os.path.join(output_dir, f"eval_{args.model.lower()}.json")
    with open(eval_path, "w") as f:
        json.dump(eval_out, f, indent=2)
    print(f"\nSaved eval results to {eval_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
