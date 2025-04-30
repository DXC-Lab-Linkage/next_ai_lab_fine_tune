#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fine_tune_llama_masking_load.py  (GGUF 版)

$ python fine_tune_llama_masking_load_iot.py --text "山田太郎さんの電話は 090-1234-5678 です"
"""

from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
from llama_cpp import Llama           # pip install llama-cpp-python

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
#  モデルのロード
# ──────────────────────────────────────────────────────────────
MODEL_PATH = "../3_convert_unsolth_to_gguf/model-q8_0.gguf"   # ← 修正
CTX_LEN    = 2048

print("⏳  loading GGUF model …")
llm = Llama(
    model_path = MODEL_PATH,
    n_ctx      = CTX_LEN,
    n_gpu_layers = -1,   # Metal GPU 全乗せ。CPU 実行なら 0
    verbose    = False,
)
print("✅  model ready")


# ──────────────────────────────────────────────────────────────
#  推論関数
# ──────────────────────────────────────────────────────────────
def generate_masked_text(input_text: str) -> str:
    """渡された文章の個人情報をマスキングして返す（llama.cpp 推論）"""
    prompt = f"""
# 個人情報マスキングタスク

あなたは個人情報のマスキングを行うAIアシスタントです。日本語テキスト内の個人情報のみを正確にマスキングしてください。

## マスキング対象の個人情報と形式
- 人名: <マスキング済みの氏名>
- 会社名: <マスキング済みの会社名>
- 住所: <マスキング済みの住所>
- メールアドレス: <マスキング済みのemailアドレス>
- 電話番号: <マスキング済みの電話番号>
- 生年月日: <マスキング済みの生年月日>
- 郵便番号: <マスキング済みの郵便番号>

## 制約条件
- before_mask の個人情報以外の部分は一切変更しないでください。
- before_mask のテキスト構造と文脈を完全に保持したまま個人情報のみをマスキングし、after_mask に出力してください。

### before_mask:
{input_text}

### after_mask:
"""

    out = llm(
        prompt,
        max_tokens      = 2048,
        temperature     = 0.1,
        top_p           = 0.9,
        repeat_penalty  = 1.1,
        stop            = ["### before_mask:"],   # プロンプトの先頭が来たら停止
    )

    generated = out["choices"][0]["text"]
    if "### after_mask:" in generated:
        return generated.split("### after_mask:")[1].strip()
    return generated.strip()


# ──────────────────────────────────────────────────────────────
#  main
# ──────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="GGUF 個人情報マスキング")
    p.add_argument("--text", help="直接テキストを渡す")
    p.add_argument("--file", type=Path, help="テキストファイルのパス")
    args = p.parse_args()

    if args.text:
        src = args.text
    elif args.file:
        if not args.file.exists():
            sys.exit(f"❌  File not found: {args.file}")
        src = args.file.read_text(encoding="utf-8")
    else:
        if sys.stdin.isatty():
            sys.exit("❌  --text か --file を指定するかパイプで入力してください")
        src = sys.stdin.read()

    masked = generate_masked_text(src)
    print(masked)


if __name__ == "__main__":
    main()
