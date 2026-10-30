#!/usr/bin/env bash
# Publish the reviewed source and the selected model release to the user's account.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
REPO="${1:-5ovfivex5-bit/craftbot}"
if [[ ! "$REPO" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
    echo "Expected OWNER/REPO" >&2
    exit 2
fi

# Use the same hi environment as training; never copy authentication into the repo.
if [[ "${CRAFTBOT_IN_HI:-0}" != 1 ]]; then
    exec distrobox enter hi -- env CRAFTBOT_IN_HI=1 "$ROOT/scripts/publish.sh" "$REPO"
fi
gh auth status
./scripts/craftbot.sh export-models
./scripts/craftbot.sh test
if [[ ! -d .git ]]; then
    echo "Initialize and commit the reviewed source before publishing." >&2
    exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Working tree has uncommitted changes. Review and commit them before publishing." >&2
    exit 1
fi
if gh repo view "$REPO" --json name >/dev/null 2>&1; then
    visibility="$(gh repo view "$REPO" --json visibility --jq .visibility)"
    if [[ "$visibility" != PUBLIC ]]; then
        echo "Existing $REPO is $visibility; refusing to change an existing repository's visibility." >&2
        exit 1
    fi
    if ! git remote get-url origin >/dev/null 2>&1; then
        git remote add origin "https://github.com/$REPO.git"
    fi
else
    gh repo create "$REPO" --public --source=. --remote=origin \
        --description 'A recurrent learning agent, voxel simulator and training studio. CPU-first PPO, expert demonstrations and pretrained checkpoints.'
fi
origin="$(git remote get-url origin)"
case "$origin" in
    "https://github.com/$REPO.git"|"https://github.com/$REPO"|"git@github.com:$REPO.git") ;;
    *) echo "origin ($origin) does not match $REPO; refusing to push." >&2; exit 1 ;;
esac
git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push -u origin main
if gh release view models-v1 --repo "$REPO" >/dev/null 2>&1; then
    # A model release is immutable; don't silently replace someone else's assets.
    echo "models-v1 already exists; source pushed, model assets left unchanged."
else
    gh release create models-v1 --repo "$REPO" --title 'CraftBot selected policies · v6 + v7' \
        --target main --notes-file docs/release-notes.md \
        models/best-v6.pt models/best-v7.pt models/manifest.json models/SHA256SUMS.sha256
fi
echo "Published: https://github.com/$REPO"
