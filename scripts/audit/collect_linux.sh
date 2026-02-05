#!/usr/bin/env bash
set -euo pipefail

TS="$(date +%Y%m%d_%H%M%S)"
OUT="${1:-$HOME/vpnless_rag_audit_${TS}}"
mkdir -p "$OUT"

log() { echo "==> $*"; }

# 0) Host / OS
log "host/os"
{
  hostname
  date -Is
  uname -a
  lsb_release -a 2>/dev/null || true
  cat /etc/os-release 2>/dev/null || true
  free -h
  df -h /
  lsblk
} > "$OUT/00_host_os.txt" 2>&1

# 1) Network (no secrets)
log "network"
{
  ip -br a
  ip r
  resolvectl status 2>/dev/null || true
  getent hosts bedrock-runtime.ap-northeast-1.amazonaws.com || true
  curl -sS -I https://bedrock-runtime.ap-northeast-1.amazonaws.com/ | head -n 20 || true
  curl -sS -I https://github.com/ | head -n 5 || true
} > "$OUT/01_network.txt" 2>&1

# 2) AWS CLI sanity (NO credentials dump)
log "aws cli"
{
  which aws || true
  aws --version || true
  AWS_PROFILE=rag aws sts get-caller-identity --output json || true
  AWS_PROFILE=rag aws bedrock list-foundation-models --region ap-northeast-1 --max-results 20 --output json || true
} > "$OUT/02_aws_cli.txt" 2>&1

# 3) NFS/VAST
log "nfs/vast"
{
  mount | grep -E "(/mnt/nfs2|kawa_nfs)" || true
  df -hT /mnt/nfs2 2>/dev/null || true
  ls -ld /mnt/nfs2 /mnt/nfs2/rag_inbox 2>/dev/null || true
  find /mnt/nfs2/rag_inbox -maxdepth 1 -type f -iname "*.pdf" 2>/dev/null | wc -l || true
} > "$OUT/03_nfs_vast.txt" 2>&1

# 4) RAG artifacts (no content dump)
log "rag artifacts"
{
  ls -lah /home/kawamura/rag_data/index 2>/dev/null || true
  wc -l /home/kawamura/rag_data/index/chunks.jsonl 2>/dev/null || true
  sha256sum /home/kawamura/rag_data/index/chunks.jsonl 2>/dev/null || true
} > "$OUT/04_rag_artifacts.txt" 2>&1

# 5) Python env (no tokens)
log "python env"
{
  python3 --version || true
  if [ -d "$HOME/vpnless-rag-venv" ]; then
    source "$HOME/vpnless-rag-venv/bin/activate"
    python -V
    python -m pip --version
    python -m pip freeze
  else
    echo "venv not found: $HOME/vpnless-rag-venv"
  fi
} > "$OUT/05_python_env.txt" 2>&1

echo "OK: collected into $OUT"