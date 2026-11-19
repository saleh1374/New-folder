"""Build the self-contained Colab notebook and the code zip.

    python tools/build_notebook.py

Creates:
    darc_nli_code.zip          # package + tests, for manual upload
    DARC_NLI_v1_colab.ipynb    # notebook that embeds the zip as base64
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_NOTEBOOK = ROOT / "DARC_NLI_v1_colab.ipynb"
OUT_ZIP = ROOT / "darc_nli_code.zip"

INCLUDE = ["darc_nli", "tests"]


def build_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in INCLUDE:
            for path in sorted((ROOT / folder).rglob("*")):
                if path.is_dir() or "__pycache__" in path.parts:
                    continue
                if path.suffix in {".pyc", ".pyo"}:
                    continue
                archive.write(path, path.relative_to(ROOT).as_posix())
    return buffer.getvalue()


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


def build_notebook(zip_bytes: bytes) -> dict:
    payload = base64.b64encode(zip_bytes).decode("ascii")
    chunks = [payload[i : i + 100] for i in range(0, len(payload), 100)]
    payload_literal = '"\\n"'.join(chunks)

    cells: list[dict] = []

    cells.append(
        md(
            """
# DARC-NLI v1 — پیاده‌سازی SemEval-2027 Task 2 (DiCo-NLI)

این نوتبوک پیاده‌سازی کامل نسخه اول روش پیشنهادی است:

* `XLM-RoBERTa-base` به صورت **Cross-Encoder** (ورودی مشترک `Premise </s> Hypothesis`)
* سه تابع زیان: `CE` + `SupCon` (class-aware) + `DC` (سازگاری جهت‌دار با JSD)
* Batch Sampler اختصاصی که جفت‌های معکوس واقعی را کنار هم نگه می‌دارد و کلاس‌ها را متعادل می‌کند
* ارزیابی با **Scorer رسمی** (`evaluation_functions/`) → `Weighted F1`, `SoftCons`, `HardCons`
* چهار آزمایش اجباری: `CE` / `CE+SupCon` / `CE+DC` / `CE+SupCon+DC`
* اجرای مستقل برای هر چهار Track و گزارش تفکیکی ترکیب زبان‌ها در Track 4

کد در یک بستهٔ پایتون ماژولار (نه یک سلول غول‌آسا) قرار دارد و همان بسته در سلول بعد از base64 باز می‌شود.
"""
        )
    )

    cells.append(
        md("## ۱) بررسی محیط و GPU")
    )
    cells.append(
        code(
            """
import sys, torch, platform
print("python  :", sys.version.split()[0], "|", platform.platform())
print("torch   :", torch.__version__)
print("cuda    :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu     :", torch.cuda.get_device_name(0))
    print("memory  : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024**3))

!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"""
        )
    )

    cells.append(md("## ۲) نصب کتابخانه‌های لازم"))
    cells.append(
        code(
            """
%pip install -q "transformers>=4.40" "accelerate>=0.30" "scikit-learn>=1.3" "pandas>=2.0" pytest tqdm
print("dependencies ready")
"""
        )
    )

    cells.append(md("## ۳) دریافت مخزن رسمی تسک (داده‌ها + Scorer)"))
    cells.append(
        code(
            """
%cd /content
!rm -rf dico_nli_repo
!git clone --depth 1 https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI.git dico_nli_repo
!ls dico_nli_repo/final_data/train | head -4
"""
        )
    )

    cells.append(
        md(
            """
## ۴) باز کردن کد پروژه

بستهٔ `darc_nli` به صورت zip/base64 داخل همین سلول تعبیه شده است؛ نیازی به آپلود دستی نیست.
اگر در محیط دیگری اجرا می‌کنید، همان `darc_nli_code.zip` را در `/content/darc_nli_project` باز کنید.
"""
        )
    )
    cells.append(
        code(
            f"""
import base64, io, zipfile, pathlib, shutil, sys, os

PROJECT = pathlib.Path("/content/darc_nli_project")
PROJECT.mkdir(parents=True, exist_ok=True)

CODE_ZIP_B64 = (
{payload_literal}
)
zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_ZIP_B64))).extractall(PROJECT)

DATA_REPO = PROJECT / "data" / "repo"
if not DATA_REPO.exists():
    DATA_REPO.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree("/content/dico_nli_repo", DATA_REPO)

os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
print("project files:", sorted(p.name for p in (PROJECT / "darc_nli").glob("*.py")))
print("data repo    :", DATA_REPO)
"""
        )
    )

    cells.append(
        md(
            """
## ۵) دروازهٔ صحت: تست‌های واحد

قبل از هر آموزشی این تست‌ها باید سبز باشند: نگاشت برچسب‌ها، رفتار SupCon در نبود نمونه مثبت (بدون `NaN`)، صفر شدن `DC` روی توزیع دقیقاً معکوس، یکتا بودن جفت‌ها، و اینکه Sampler هیچ جفت معکوسی را از هم جدا نکند.
"""
        )
    )
    cells.append(code("!python -m pytest tests -q"))

    cells.append(
        md(
            """
## فاز صفر: Audit داده‌ها

ساختار فایل‌ها، ستون‌ها، توزیع کلاس‌ها، طول متن‌ها (برای انتخاب `max_length`)، صحت `reverse_pair_id`، و نشتی Train/Dev بررسی و در `outputs/audit_report.json` ذخیره می‌شود.
"""
        )
    )
    cells.append(
        code(
            """
!python -m darc_nli.run audit --data-root data/repo --out outputs/audit_report.json 2>&1 | tail -60
"""
        )
    )

    cells.append(
        md(
            """
## ۶) Smoke test

یک اجرای ۳۰ مرحله‌ای روی ۱۶۰ نمونه، برای اطمینان از سالم بودن کل مسیر (DataLoader → سه Loss → ارزیابی → Scorer).
"""
        )
    )
    cells.append(
        code(
            """
!python -m darc_nli.run train --data-root data/repo --track 1 --experiment ce_supcon_dc \\
    --epochs 1 --max-train-steps 30 --max-train-samples 160 --batch-size 8 \\
    --output-dir outputs_smoke 2>&1 | tail -20
"""
        )
    )

    cells.append(
        md(
            """
## ۷) ماتریس آزمایش‌های اصلی

چهار آزمایش × چهار Track با شرایط یکسان (مدل، داده، Seed، تنظیمات، Scorer).

* `baseline`  → فقط CE
* `ce_supcon`  → CE + 0.1·SupCon
* `ce_dc`      → CE + 0.1·DC
* `ce_supcon_dc` → CE + 0.1·SupCon + 0.1·DC

`max_length` به‌صورت خودکار از Audit تعیین می‌شود. اگر حافظه کم آمد، `--grad-accum 2` را اضافه کنید.
"""
        )
    )
    for track, label in [(1, "English"), (2, "Spanish"), (3, "Basque"), (4, "Mixed")]:
        cells.append(md(f"### Track {track} — {label}"))
        cells.append(
            code(
                f"""
!python -m darc_nli.run train --data-root data/repo --track {track} --experiment all \\
    --epochs 3 --batch-size 16 --max-length 0 --summarize 2>&1 | tee -a outputs/train_track{track}.log | tail -40
"""
            )
        )

    cells.append(md("## ۸) جدول نتایج (با Scorer رسمی)"))
    cells.append(code("!python -m darc_nli.run summarize --output-dir outputs"))
    cells.append(
        code(
            """
import pandas as pd
summary = pd.read_csv("outputs/summary.csv")
display(summary)

mono = summary[summary["track"].isin(["english", "spanish", "basque"])]
print("Monolingual macro-average (tracks 1-3)")
display(mono.groupby("experiment")[["weighted_f1", "soft_cons", "hard_cons"]].mean().round(4))
print("All-track macro-average (tracks 1-4)")
display(summary.groupby("experiment")[["weighted_f1", "soft_cons", "hard_cons"]].mean().round(4))
"""
        )
    )

    cells.append(
        md(
            """
## ۹) تحلیل Track 4 بر اساس ترکیب زبان‌ها

پیش‌بینی‌های Dev هر Run با همان Scorer رسمی، روی زیرمجموعهٔ هر ترکیب زبانی (EN-ES، EU-EN، ...) محاسبه شده و در `metadata.json` چک‌پوینت ذخیره شده است.
"""
        )
    )
    cells.append(
        code(
            """
import json, pandas as pd, pathlib

rows = []
for meta_path in sorted(pathlib.Path("outputs/checkpoints/mixed").glob("*/best/metadata.json")):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for lang_pair, scores in meta["dev_per_language"].items():
        rows.append({"experiment": meta["experiment"], "language_pair": lang_pair, **scores})

if rows:
    table = pd.DataFrame(rows)
    display(table.sort_values(["language_pair", "experiment"]).reset_index(drop=True))
    table.to_csv("outputs/track4_per_language.csv", index=False)
else:
    print("no track4 checkpoints yet")
"""
        )
    )

    cells.append(
        md(
            """
## ۱۰) ساخت فایل‌های پیش‌بینی و بستهٔ نتایج

برای هر Track، بهترین چک‌پوینت روی Dev پیش‌بینی می‌دهد و فایل `instance_id,label` تولید می‌شود (همان قالب رسمی CodaBench).
"""
        )
    )
    cells.append(
        code(
            """
import pathlib, json, shutil

for track_dir in sorted(pathlib.Path("outputs/checkpoints").glob("*")):
    best = track_dir / "baseline" / "best"
    if not best.exists():
        continue
    meta = json.loads((best / "metadata.json").read_text(encoding="utf-8"))
    dev_file = pathlib.Path("data/repo/final_data/dev") / f"dico_nli_dev_track{meta['track']}_participant_labeled.csv"
    out = pathlib.Path("submissions") / f"dev_track{meta['track']}_baseline.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    !python -m darc_nli.run predict --checkpoint "{best}" --input "{dev_file}" --output "{out}"
    print(out)

!cd /content/darc_nli_project && zip -qr /content/darc_nli_outputs.zip outputs submissions
print("ready: /content/darc_nli_outputs.zip")
"""
        )
    )

    cells.append(
        md(
            """
## گام‌های بعدی (خارج از نسخه اول)

این نوتبوک عمداً فقط نسخه اول را پیاده می‌کند. برای مقاله/بهبود:

1. جاروب وزن‌ها: `lambda_supcon ∈ {0.05, 0.1, 0.2, 0.5}` و `lambda_dc` به‌صورت مستقل.
2. حداقل سه Seed برای هر تنظیم و گزارش میانگین ± انحراف معیار.
3. Cross-lingual consistency، Factorized Representation، Ensembling — هر کدام به‌عنوان آزمایش مستقل.
4. تحلیل خطا روی `pair_errors` موجود در `scores.json` (جفت‌های ناسازگار).
"""
        )
    )

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    zip_bytes = build_zip()
    OUT_ZIP.write_bytes(zip_bytes)
    notebook = build_notebook(zip_bytes)
    OUT_NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"zip      : {OUT_ZIP} ({len(zip_bytes)/1024:.1f} KiB)")
    print(f"notebook : {OUT_NOTEBOOK} ({OUT_NOTEBOOK.stat().st_size/1024:.1f} KiB)")


if __name__ == "__main__":
    main()
