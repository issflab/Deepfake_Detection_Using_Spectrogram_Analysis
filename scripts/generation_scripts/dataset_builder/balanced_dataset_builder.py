"""
Author: Atharva Pore

Creates a balanced 30 000-image dataset:

  • 500 ORIGINAL × 30 (clean + 29 attacks) = 15 000
  • 500 DEEPFAKE × 30                     = 15 000

Folder layout
-------------
out_root/
 ├─ original/
 │   ├─ clean/
 │   └─ attacked/
 └─ deepfake/
     ├─ clean/
     └─ attacked/
"""
from __future__ import annotations
import os, re, random, shutil
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------
# 1 CONFIGURATION
# ---------------------------------------------------------------------
ORIG_DIR = Path(r"path\to\original_images")       # ← set your original-image folder
DF_DIR   = Path(r"path\to\deepfake_images")       # ← set your deep-fake image folder
OUT_DIR  = Path(r"path\to\balanced_dataset_out")  # ← set desired output folder

DEEPFAKE_ENGINES = [
    "ElevenLabs", "Maskgct", "Ssr", "F5tts",
    "Styletts", "Xtts", "E2tts", "Fish",
]

ATTACK_NAMES = [
    "RT03", "RT06", "RT09",
    "resample22050", "resample44100", "resample8000", "resample11025",
    "recompression128k", "recompression64k", "recompression196k",
    "recompression16k", "recompression256k", "recompression320k",
    "babble0", "babble10", "babble20",
    "volvo0", "volvo10", "volvo20",
    "white0", "white10", "white20",
    "street0", "street10", "street20",
    "cafe0", "cafe10", "cafe20",
    "lpf7000",
]

RANDOM_SEED = 42
# ---------------------------------------------------------------------
# 2️ HELPER FUNCTIONS
# ---------------------------------------------------------------------
def id_from_name(p: Path) -> str:
    """Return digits that follow 'Donald_Trump_' (keeps leading zeros)."""
    m = re.search(r"Donald_Trump_(\d+)", p.stem)
    if m:
        return m.group(1)
    # fallback: first all-digit token
    for tok in p.stem.split('_'):
        if tok.isdigit():
            return tok
    raise ValueError(f"No ID in {p}")

def generate_id_variants(base_id: str) -> list[str]:
    """'00001'→['00001','0001','001','01','1']; '165'→['165','0165',…]."""
    num = str(int(base_id)) if base_id.lstrip('0') else '0'
    variants = [base_id]
    for w in range(1, 6):
        pad = num.zfill(w)
        if pad not in variants:
            variants.append(pad)
    return variants

def attack_variants(atk: str) -> list[str]:
    """Return all plausible spellings for one attack token."""
    variants = {atk}

    # resample / recompression: underscore flip
    if atk.startswith("resample"):
        variants.add(atk.replace("_", "") if "_" in atk else atk.replace("resample", "resample_"))
    if atk.startswith("recompression"):
        variants.add(atk.replace("_", "") if "_" in atk else atk.replace("recompression", "recompression_"))

    # RTxx ⇄ rt_0.x
    rt_map = {"RT03": "rt_0.3", "RT06": "rt_0.6", "RT09": "rt_0.9"}
    if atk in rt_map:
        variants.add(rt_map[atk])
    if atk in rt_map.values():
        inv = {v: k for k, v in rt_map.items()}
        variants.add(inv[atk])

    return list(variants)

def engine_from_name(p: Path) -> str | None:
    if "_Original" in p.stem:
        return None
    for eng in DEEPFAKE_ENGINES:
        if f"_{eng}_" in p.stem or p.stem.endswith(f"_{eng}"):
            return eng
    return None

def is_clean_original(p: Path) -> bool:
    return p.name.endswith("_Original.png")

def is_clean_deepfake(p: Path) -> bool:
    eng = engine_from_name(p)
    return eng is not None and all(a not in p.name for a in ATTACK_NAMES)

def build_attack_filename(base_id: str, is_orig: bool, eng: str | None, attack: str) -> str:
    """
    Compose the exact filename on disk according to your conventions:
      • recompression → prefix + '_recompression{bitrate}_…_laundering.png'
      • others       → prefix + '_{attack}_…_laundering.png'
    """
    # recompression (with or w/out underscore in 'attack')
    if attack.lower().startswith("recompression"):
        bit_rate = re.search(r"(\d+k)", attack, re.IGNORECASE).group(1)
        if is_orig:
            return (f"original_launderingDonald_Trump_{base_id}"
                    f"_Original_recompression{bit_rate}.png")
        return (f"deepfake_launderingDonald_Trump_{base_id}_{eng}"
                f"_recompression{bit_rate}.png")

    # every other attack
    if is_orig:
        return (f"Donald_Trump_{base_id}_Original_{attack}"
                f".png")
    return (f"Donald_Trump_{base_id}_{eng}_{attack}"
            f".png")

def copy_file(src: Path, dst: Path):
    """Hard-link where possible; otherwise copy. Skip if dst exists."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)          # fast on NTFS
    except (FileExistsError, PermissionError):
        return
    except OSError:                # e.g. cross-drive
        try:
            shutil.copy2(src, dst)
        except FileExistsError:
            pass

# ---------------------------------------------------------------------
# 3️ SCAN SOURCE FOLDERS
# ---------------------------------------------------------------------
print("Scanning source folders …")
orig_clean = [p for p in ORIG_DIR.rglob("*.png") if is_clean_original(p)]
df_clean   = [p for p in DF_DIR.rglob("*.png")   if is_clean_deepfake(p)]
print(f"Found {len(orig_clean):5} clean originals")
print(f"Found {len(df_clean):5} clean deepfakes")

df_by_eng: dict[str, list[Path]] = defaultdict(list)
for p in df_clean:
    df_by_eng[engine_from_name(p)].append(p)
for e in DEEPFAKE_ENGINES:
    print(f"  {e:<10}: {len(df_by_eng[e])}")

# ---------------------------------------------------------------------
# 4️ SAMPLE 500+500 BASE IMAGES
# ---------------------------------------------------------------------
random.seed(RANDOM_SEED)
orig_sample = random.sample(orig_clean, 500)

eng_63 = set(random.sample(DEEPFAKE_ENGINES, 4))
eng_62 = set(DEEPFAKE_ENGINES) - eng_63
df_sample = []
for e in eng_63:
    df_sample += random.sample(df_by_eng[e], 63)
for e in eng_62:
    df_sample += random.sample(df_by_eng[e], 62)
assert len(df_sample) == 500

print("\nSampled deep-fake distribution:")
for e in DEEPFAKE_ENGINES:
    print(f"  {e:<10}: {sum(engine_from_name(p)==e for p in df_sample)}")

# ---------------------------------------------------------------------
# 5️ COPY/LINK WITH FULL PERMUTATION CHECK
# ---------------------------------------------------------------------
def process_group(bases: list[Path], is_orig: bool) -> int:
    group = "original" if is_orig else "deepfake"
    attack_copied = 0
    miss_global = Counter()

    for base in bases:
        base_id = id_from_name(base)
        eng = None if is_orig else engine_from_name(base)

        # clean
        copy_file(base, OUT_DIR / group / "clean" / base.name)

        # attacks
        missing_here = []
        for atk in ATTACK_NAMES:
            found = False
            for atk_spell in attack_variants(atk):
                for bid in generate_id_variants(base_id):
                    fname = build_attack_filename(bid, is_orig, eng, atk_spell)
                    src = base.parent / fname
                    if src.exists():
                        copy_file(src, OUT_DIR / group / "attacked" / fname)
                        if not is_orig:
                            attack_copied += 1
                        found = True
                        break
                if found:
                    break
            if not found:
                missing_here.append(atk)
                miss_global[atk] += 1

        if missing_here:
            print(f"[WARN] {group} ID={base_id} missing {len(missing_here)}/29 → {missing_here}")

    if miss_global:
        print(f"\n[{group.upper()}] GLOBAL missing attacks:")
        for atk, cnt in miss_global.most_common():
            print(f"   {atk:<22} {cnt}")

    return attack_copied

print("\nCopying ORIGINAL files …")
process_group(orig_sample, is_orig=True)

print("\nCopying DEEPFAKE files …")
df_attack_copied = process_group(df_sample, is_orig=False)

if df_attack_copied < 14_500:
    print(f"\n⚠️  Only {df_attack_copied} deep-fake attack files copied "
          f"(expected 14 500). Check warnings above.")

# ---------------------------------------------------------------------
# 6️ FINAL SUMMARY
# ---------------------------------------------------------------------
total = sum(1 for _ in OUT_DIR.rglob("*.png"))
print(f"\n✅ Dataset build complete — {total:,} files in {OUT_DIR}")
print("Structure:")
print(f"  {OUT_DIR / 'original' / 'clean'}")
print(f"  {OUT_DIR / 'original' / 'attacked'}")
print(f"  {OUT_DIR / 'deepfake' / 'clean'}")
print(f"  {OUT_DIR / 'deepfake' / 'attacked'}")
