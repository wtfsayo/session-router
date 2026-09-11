"""Prepare SWE-bench docker images for the dynamic track on Apple Silicon.

The trb harness computes arm64/x86_64 image keys from platform.machine();
Docker Hub only hosts x86_64 instance images under `swebench/sweb.eval.x86_64.
<repo>_<ghid>_<repo>-<num>` (with a numeric owner-id infix we can't derive).
This script, for each held-out instance:

  1. computes the test_spec keys with machine patched to x86_64
  2. finds the matching Hub repo name via the Hub search API
  3. pulls --platform linux/amd64 and retags as the local env+instance keys
     so build_instance_image sees both present and skips building.

Needs a venv with TwinRouterBench installed (`pip install -e trb[dynamic]`);
point TRB_REPO at the checkout. Run as: TRB_REPO=/path/to/trb python
bench/prep_images.py [N]
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import urllib.request

import os

platform.machine = lambda: "x86_64"
TRB = os.environ.get("TRB_REPO", "/tmp/trb")  # TwinRouterBench checkout
sys.path.insert(0, TRB)

from swerouter.harness.container_runner import (  # noqa: E402
    load_dataset_instance, make_test_spec_for_instance)

IDS = [l.strip() for l in
       open(f"{TRB}/data/dynamic/dynamic_heldout100_ids.txt") if l.strip()]


def hub_name(repo: str, suffix: str) -> str | None:
    url = ("https://hub.docker.com/v2/repositories/swebench/"
           f"?page_size=100&name=sweb.eval.x86_64.{repo}")
    with urllib.request.urlopen(url, timeout=30) as r:
        for it in json.load(r)["results"]:
            if it["name"].endswith(suffix):
                return it["name"]
    return None


def have_image(name: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", name],
                          capture_output=True).returncode == 0


def main(ids):
    done, missing = [], []
    for iid in ids:
        repo = iid.split("__")[0]
        suffix = iid.split("__", 1)[1]  # e.g. astropy-13236
        inst = load_dataset_instance(iid)
        spec = make_test_spec_for_instance(inst, image_namespace=None,
                                           windows_compat=True)
        env_key, inst_key = spec.env_image_key, spec.instance_image_key
        if have_image(inst_key) and have_image(env_key):
            done.append(iid); continue
        name = hub_name(repo, suffix)
        if not name:
            missing.append(iid); continue
        src = f"swebench/{name}:latest"
        print(f"{iid}: pull {src}")
        r = subprocess.run(["docker", "pull", "--platform", "linux/amd64", src])
        if r.returncode != 0:
            missing.append(iid); continue
        for dst in (env_key, inst_key):
            subprocess.run(["docker", "tag", src, dst], check=True)
        done.append(iid)
    print(f"\nprepared {len(done)}, missing {len(missing)}")
    for m in missing:
        print("  missing:", m)


if __name__ == "__main__":
    main(IDS[: int(sys.argv[1])] if len(sys.argv) > 1 else IDS)
