#!/usr/bin/env python3
import argparse
import os
import subprocess
import time
import sys
import logging
import re
import shutil
from pathlib import Path

# --- CONFIGURATION ---
ENV_CONFIG = {
    "IN_STREAM_WORKERS": "1",
    "OCR_PARALLEL_WORKERS": "1",
    # Vital for 8GB cards to prevent "fragmentation" crashes
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
    "OCR_DEVICE": "cuda", 
    "TORCH_DEVICE": "cuda"
}

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MarkerConvert")

# --- HELPER FUNCTIONS ---

import sysconfig # Add this import at the top if missing

def find_marker_binary():
    """
    Robustly locates 'marker_single' on Windows (User & System paths) and Linux.
    """
    candidates = []
    exe_name = "marker_single.exe" if os.name == 'nt' else "marker_single"

    # 1. Check Active Environment (Virtualenv or System)
    if os.name == 'nt':
        candidates.append(Path(sys.prefix) / "Scripts" / exe_name)
    else:
        candidates.append(Path(sys.prefix) / "bin" / exe_name)

    # 2. Check Windows User Install Directory (AppData/Roaming/Python/...)
    # This is where 'pip install' puts things for the Windows Store Python
    if os.name == 'nt':
        try:
            user_scripts = Path(sysconfig.get_path('scripts', f'{os.name}_user'))
            candidates.append(user_scripts / exe_name)
        except Exception:
            pass

    # 3. Check Global PATH
    if shutil.which("marker_single"):
        candidates.append(Path(shutil.which("marker_single")))

    # Test all candidates
    for path in candidates:
        if path.exists():
            return str(path)
            
    return None

def create_frontmatter(title, filename):
    date_str = time.strftime("%Y-%m-%d")
    return (
        "---\n"
        f'title: "{title}"\n'
        f'source_file: "{filename}"\n'
        f'date_converted: {date_str}\n'
        "tags: [scientific-paper, marker-conversion, latex]\n"
        "status: unread\n"
        "---\n\n"
    )

def post_process_markdown(md_path, pdf_path):
    try:
        if not md_path.exists():
            return
        
        # Force UTF-8 to prevent Windows encoding errors
        original_content = md_path.read_text(encoding='utf-8')
        
        # Clean up title
        paper_title = pdf_path.stem.replace("-", " ").replace("_", " ").title()
        frontmatter = create_frontmatter(paper_title, pdf_path.name)
        
        md_path.write_text(frontmatter + original_content, encoding='utf-8')
        logger.info(f"✨ Metadata injected: {md_path.name}")
        
    except Exception as e:
        logger.error(f"Metadata injection failed: {e}")

def run_marker_cmd(marker_exe, input_file, output_dir, device="cuda"):
    """
    Runs the binary in a subprocess with specific environment variables.
    """
    cmd = [
        marker_exe,
        str(input_file),
        "--output_dir", str(output_dir)
    ]
    
    # Merge current environment with our custom config
    current_env = os.environ.copy()
    current_env.update(ENV_CONFIG)
    
    # If falling back to CPU, strictly hide the GPU
    if device == "cpu":
        current_env["TORCH_DEVICE"] = "cpu"
        current_env["OCR_DEVICE"] = "cpu"
        current_env["CUDA_VISIBLE_DEVICES"] = "" 
    
    # capture_output=True handles stdout/stderr redirection automatically
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=current_env,
        encoding='utf-8' # Explicit encoding for Windows subprocess reading
    )

# --- CORE LOGIC ---

def process_single_pdf(marker_exe, pdf_path, output_root):
    logger.info(f"⚙️  Processing: {pdf_path.name}")
    
    # Check output existence to allow resuming interrupted batch jobs
    expected_md = output_root / pdf_path.stem / f"{pdf_path.stem}.md"
    if expected_md.exists():
         logger.info(f"⏩ Skipping (Output exists)")
         return

    # ATTEMPT 1: GPU
    start_time = time.time()
    result = run_marker_cmd(marker_exe, pdf_path, output_root, device="cuda")

    # Detect Failure / OOM
    # Check for "OutOfMemory" or standard PyTorch CUDA errors
    is_oom = "OutOfMemory" in result.stderr or "CUDA out of memory" in result.stderr
    
    if result.returncode != 0 and is_oom:
        logger.warning("⚠️  GPU OOM detected.")
        
        # Parse error for memory stats
        mem_match = re.search(r"(Tried to allocate.*?)\n", result.stderr)
        if mem_match:
            logger.warning(f"📉 Memory Detail: {mem_match.group(1)}")
        else:
            # Fallback debug
            for line in result.stderr.split('\n'):
                if "allocate" in line or "capacity" in line:
                    logger.warning(f"📉 Debug: {line.strip()}")
        
        logger.info("   Falling back to CPU mode (Slow but Safe)...")
        result = run_marker_cmd(marker_exe, pdf_path, output_root, device="cpu")
    
    duration = time.time() - start_time

    if result.returncode == 0:
        logger.info(f"✅ Done in {duration:.1f}s")
        
        # Handle case where marker output filename differs from expectation
        if expected_md.exists():
            post_process_markdown(expected_md, pdf_path)
        else:
            # Look for ANY .md file in the folder
            found_mds = list((output_root / pdf_path.stem).glob("*.md"))
            if found_mds:
                post_process_markdown(found_mds[0], pdf_path)
            else:
                logger.warning(f"Output folder created but MD file missing: {expected_md}")
    else:
        logger.error(f"❌ Failed to convert {pdf_path.name}")
        # Show the tail of the error log
        logger.error(f"STDERR Snippet:\n" + "\n".join(result.stderr.splitlines()[-15:]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path, help="PDF file OR Folder")
    parser.add_argument("--out", type=Path, default=Path("papers"))
    args = parser.parse_args()

    # 1. Validate Binary
    marker_exe = find_marker_binary()
    if not marker_exe:
        logger.critical("❌ Could not find 'marker_single' executable.")
        logger.critical(f"   Looking in: {Path(sys.prefix) / 'Scripts' if os.name == 'nt' else Path(sys.prefix) / 'bin'}")
        logger.critical("   Try running: pip install marker-pdf")
        sys.exit(1)
    
    logger.info(f"🔧 Using Marker Binary: {marker_exe}")

    # 2. Check Hardware
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"⚡ GPU System Active: {torch.cuda.get_device_name(0)} ({vram:.1f} GB)")
        else:
            logger.warning("🐢 No GPU detected by PyTorch.")
    except ImportError:
        logger.warning("PyTorch not installed in this environment.")

    # 3. Path resolution
    args.input_path = args.input_path.resolve()
    args.out = args.out.resolve()

    # 4. Run
    if args.input_path.is_file():
        process_single_pdf(marker_exe, args.input_path, args.out)
    elif args.input_path.is_dir():
        pdfs = list(args.input_path.glob("*.pdf"))
        logger.info(f"📂 Batch Mode: Found {len(pdfs)} files")
        for pdf in pdfs:
            process_single_pdf(marker_exe, pdf, args.out)

if __name__ == "__main__":
    main()