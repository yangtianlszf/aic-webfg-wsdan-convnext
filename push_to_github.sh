#!/usr/bin/env bash
set -e

# 使用前请先安装并登录 GitHub CLI：
#   gh auth login
#
# 建议仓库名：aic-webfg-wsdan-convnext

REPO_NAME="aic-webfg-wsdan-convnext"
VISIBILITY="private"  # 可改为 public

git init
git add .
git commit -m "Initial commit: AIC webly supervised FGVC project"

gh repo create "$REPO_NAME" --source=. --remote=origin --push --$VISIBILITY
