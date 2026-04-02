#!/usr/bin/env bash
# ============================================================================
# docker/build_images.sh
#
# 构建并（可选）推送 pdebench 的 deal.II / Firedrake Docker 镜像。
#
# 用法：
#   ./docker/build_images.sh               # 仅构建
#   ./docker/build_images.sh --push        # 构建 + 推送到 Docker Hub
#   ./docker/build_images.sh dealii        # 只构建 deal.II 镜像
#   ./docker/build_images.sh firedrake     # 只构建 Firedrake 镜像
#   ./docker/build_images.sh dealii --push # 构建 deal.II 并推送
# ============================================================================

set -euo pipefail

# 改成你的 Docker Hub 用户名，例如 REPO="yusan"
REPO="pdebench"
TAG="latest"
PUSH=false
TARGETS=("dealii" "firedrake")

# ── 解析参数 ──────────────────────────────────────────────────────────────────
POSITIONAL=()
for arg in "$@"; do
  case $arg in
    --push)  PUSH=true ;;
    dealii)  TARGETS=("dealii") ;;
    firedrake) TARGETS=("firedrake") ;;
    *) POSITIONAL+=("$arg") ;;
  esac
done

# ── 切换到项目根目录（Dockerfile 里 COPY 需要以此为 context）────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "========================================================"
echo " PDEBench Docker Image Builder"
echo " Project root : $PROJECT_ROOT"
echo " Targets      : ${TARGETS[*]}"
echo " Push         : $PUSH"
echo "========================================================"

for target in "${TARGETS[@]}"; do
  IMAGE="${REPO}/${target}:${TAG}"
  DOCKERFILE="docker/Dockerfile.${target}"

  echo ""
  echo ">>> Building $IMAGE from $DOCKERFILE ..."
  docker build \
    --file "$DOCKERFILE" \
    --tag  "$IMAGE" \
    .
  echo ">>> Built $IMAGE successfully."

  if [ "$PUSH" = true ]; then
    echo ">>> Pushing $IMAGE ..."
    docker push "$IMAGE"
    echo ">>> Pushed $IMAGE."
  fi
done

echo ""
echo "========================================================"
echo " Done!"
if [ "$PUSH" = false ]; then
  echo " To push images, run:  ./docker/build_images.sh --push"
fi
echo ""
echo " Colleagues can pull with:"
for target in "${TARGETS[@]}"; do
  echo "   docker pull ${REPO}/${target}:${TAG}"
done
echo "========================================================"
