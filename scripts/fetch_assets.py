"""Download the base model and datasets, pinned by commit, into assets/.

Only safetensors, JSON, text and CSV files are fetched: no pickle checkpoints and
no custom code. The commit of every repository goes to assets/manifest.json so a
later run can tell exactly what it was built on.
"""
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent.parent / "assets"
ALLOW = ["*.safetensors", "*.json", "*.txt", "*.csv", "*.md", "LICENSE*"]
REPOS = [
    ("redmadrobot-rnd/rubert-base-pii-ner", "model", "base_model"),
    ("redmadrobot-rnd/pii_train", "dataset", "pii_train"),
    ("redmadrobot-rnd/pii_benchmark", "dataset", "pii_benchmark"),
]


def main() -> None:
    api = HfApi()
    manifest = {}
    for repo_id, repo_type, folder in REPOS:
        info = api.repo_info(repo_id, repo_type=repo_type)
        target = ROOT / folder
        snapshot_download(
            repo_id,
            repo_type=repo_type,
            revision=info.sha,
            local_dir=target,
            allow_patterns=ALLOW,
        )
        files = sorted(p for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts)
        manifest[folder] = {
            "repo": repo_id,
            "type": repo_type,
            "commit": info.sha,
            "license": (info.card_data or {}).get("license") if info.card_data else None,
            "files": {str(p.relative_to(target)).replace("\\", "/"): p.stat().st_size for p in files},
        }
        print(repo_id, info.sha, f"{len(files)} files")
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
