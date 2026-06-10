"""
Auto-download fine-tuned model weights from Kaggle when not present locally.

Requires: KAGGLE_USERNAME + KAGGLE_KEY in environment (or ~/.kaggle/kaggle.json).
If credentials are absent, logs a warning and skips download — fine-tuned
components will gracefully degrade to the base LLM.
"""

import logging
import os
import shutil
import subprocess
import sys
import zipfile
from typing import Optional

logger = logging.getLogger(__name__)

# Kaggle model handles and the local directories they should be extracted into.
# Format: (kaggle_handle, local_dir_name, weight_filename_to_check)
MODELS_REGISTRY = [
    (
        "ziangtran123/reviewer-lora",
        "reviewer_lora",
        "adapter_model.safetensors",
    ),
    (
        "ziangtran123/reranker-ft",
        "reranker_ft",
        "model.safetensors",
    ),
]


def _has_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials are available."""
    # Check env vars
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    # Check ~/.kaggle/kaggle.json
    kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    return os.path.exists(kaggle_json)


def _weight_file_exists(project_root: str, local_dir: str, weight_filename: str) -> bool:
    """Return True if the weight file already exists on disk."""
    weight_path = os.path.join(project_root, local_dir, weight_filename)
    return os.path.isfile(weight_path)


def _download_kaggle_model(kaggle_handle: str, target_dir: str, tmp_dir: str) -> bool:
    """
    Download a Kaggle model using the kaggle CLI.

    The Kaggle model download command tries the most common instance path
    (transformers/default/1) first, then falls back to listing available instances.

    Returns True on success, False on failure.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    owner, model_slug = kaggle_handle.split("/")

    # --- Attempt 1: Use kaggle Python API to discover and download ---
    try:
        from kaggle.api.kaggle_api_extended import KaggleApiClient  # type: ignore
        api = KaggleApiClient()
        api.authenticate()

        # List model instances to find the right framework/variation
        logger.info(f"[Downloader] Listing instances for: {kaggle_handle}")
        instances = api.model_instances_list(owner_slug=owner, model_slug=model_slug, page_size=5)

        if not instances:
            logger.warning(f"[Downloader] No instances found for {kaggle_handle}")
            return False

        instance = instances[0]
        framework = instance.framework
        overview = instance.overview

        # List versions to get latest
        versions = api.model_instances_versions_list(
            owner_slug=owner, model_slug=model_slug,
            framework=framework, instance_slug=overview, page_size=1
        )
        version_number = versions[0].version_number if versions else 1

        logger.info(
            f"[Downloader] Downloading {kaggle_handle}/{framework}/{overview}/v{version_number} → {target_dir}"
        )
        api.model_instances_versions_download(
            owner_slug=owner,
            model_slug=model_slug,
            framework=framework,
            instance_slug=overview,
            version_number=version_number,
            path=tmp_dir,
            untar=True,
            force=False,
            quiet=False,
        )

        # Move downloaded files into target_dir
        _merge_downloaded_files(tmp_dir, target_dir)
        return True

    except ImportError:
        logger.warning("[Downloader] kaggle package not installed. Trying CLI fallback...")
    except Exception as e:
        logger.warning(f"[Downloader] Kaggle Python API failed: {e}. Trying CLI fallback...")

    # --- Attempt 2: Fallback to kaggle CLI subprocess ---
    try:
        # Try common instance path pattern first
        for instance_path in [
            f"{owner}/{model_slug}/transformers/default/1",
            f"{owner}/{model_slug}/pytorch/default/1",
            f"{owner}/{model_slug}/other/default/1",
        ]:
            cmd = [
                sys.executable, "-m", "kaggle",
                "models", "instances", "versions", "download",
                instance_path,
                "--unzip", "--path", tmp_dir
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"[Downloader] CLI download succeeded for path: {instance_path}")
                _merge_downloaded_files(tmp_dir, target_dir)
                return True
            logger.debug(f"[Downloader] CLI attempt failed for {instance_path}: {result.stderr.strip()}")

        logger.error(f"[Downloader] All CLI download attempts failed for {kaggle_handle}")
        return False

    except Exception as e:
        logger.error(f"[Downloader] CLI fallback error: {e}")
        return False


def _merge_downloaded_files(src_dir: str, dst_dir: str):
    """
    Move all files from src_dir (and its first-level subdirectory if any)
    into dst_dir, then clean up src_dir.
    """
    os.makedirs(dst_dir, exist_ok=True)

    entries = os.listdir(src_dir)

    # If there's a single subdirectory, descend into it
    if len(entries) == 1 and os.path.isdir(os.path.join(src_dir, entries[0])):
        actual_src = os.path.join(src_dir, entries[0])
    else:
        actual_src = src_dir

    for fname in os.listdir(actual_src):
        src_path = os.path.join(actual_src, fname)
        dst_path = os.path.join(dst_dir, fname)
        if not os.path.exists(dst_path):
            shutil.move(src_path, dst_path)
            logger.debug(f"[Downloader] Moved: {fname} → {dst_dir}/")
        else:
            logger.debug(f"[Downloader] Skipped (already exists): {fname}")

    # Clean up temp dir
    shutil.rmtree(src_dir, ignore_errors=True)


def ensure_fine_tuned_models(project_root: Optional[str] = None) -> dict:
    """
    Check if fine-tuned model weight files exist locally.
    If any are missing and Kaggle credentials are available, download them.

    Args:
        project_root: Root directory of the project. Defaults to the
                      parent directory of this file's package.

    Returns:
        dict mapping local_dir → True (available) / False (unavailable)
    """
    if project_root is None:
        # src/utils/model_downloader.py → go up 3 levels to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    results = {}
    any_missing = False

    for kaggle_handle, local_dir, weight_filename in MODELS_REGISTRY:
        if _weight_file_exists(project_root, local_dir, weight_filename):
            logger.info(f"[Downloader] ✓ {local_dir}/{weight_filename} already present.")
            results[local_dir] = True
        else:
            logger.warning(
                f"[Downloader] ✗ {local_dir}/{weight_filename} not found. "
                f"Will attempt download from Kaggle: {kaggle_handle}"
            )
            results[local_dir] = False
            any_missing = True

    if not any_missing:
        return results

    if not _has_kaggle_credentials():
        logger.warning(
            "[Downloader] Kaggle credentials not found. "
            "Fine-tuned models will be disabled. "
            "To enable auto-download, add KAGGLE_USERNAME and KAGGLE_KEY to your .env file. "
            "Get your credentials at: https://www.kaggle.com/settings → API → Create New Token"
        )
        return results

    logger.info("[Downloader] Kaggle credentials found. Starting download of missing model weights...")

    for kaggle_handle, local_dir, weight_filename in MODELS_REGISTRY:
        if results[local_dir]:
            continue  # already present

        target_dir = os.path.join(project_root, local_dir)
        tmp_dir = os.path.join(project_root, ".cache", "_kaggle_tmp", local_dir)

        logger.info(f"[Downloader] Downloading: {kaggle_handle} → {local_dir}/")
        success = _download_kaggle_model(kaggle_handle, target_dir, tmp_dir)

        if success and _weight_file_exists(project_root, local_dir, weight_filename):
            logger.info(f"[Downloader] ✓ {local_dir} downloaded successfully.")
            results[local_dir] = True
        else:
            logger.error(
                f"[Downloader] ✗ Failed to download {kaggle_handle}. "
                f"{local_dir} will run without fine-tuned weights."
            )

    return results
