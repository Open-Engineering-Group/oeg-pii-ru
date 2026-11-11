"""Upload the release folder to a private Hugging Face repo and tag the version.

Uses the token saved by `hf auth login`; no token is read from the project.
"""
import hashlib
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "release" / "hf" / "oeg-pii-ru-lite"
REPO = "OEGtech/oeg-pii-ru-lite"
TAG = "v0.1.0"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
    info = api.upload_folder(repo_id=REPO, folder_path=str(FOLDER), commit_message="OEG PII RU Lite 0.1.0 (preview)")
    print("commit:", info.oid if hasattr(info, "oid") else info, flush=True)
    tags = [t.name for t in api.list_repo_refs(REPO).tags]
    if TAG not in tags:
        api.create_tag(REPO, tag=TAG, tag_message="OEG PII RU Lite 0.1.0 (preview)")
    repo = api.model_info(REPO, files_metadata=True)
    print("private:", repo.private, "| tags:", [t.name for t in api.list_repo_refs(REPO).tags], flush=True)
    remote = {s.rfilename: s for s in repo.siblings}
    ok = True
    for path in sorted(FOLDER.iterdir()):
        r = remote.get(path.name)
        if r is None:
            print("MISSING", path.name)
            ok = False
            continue
        if r.lfs:
            same = r.lfs.sha256 == sha256(path)
        else:
            same = r.size == path.stat().st_size
        ok &= same
        print(f"{path.name}: {'ok' if same else 'MISMATCH'} ({path.stat().st_size} bytes)")
    print("all files match:", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
