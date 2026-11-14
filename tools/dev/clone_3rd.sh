#!/usr/bin/env bash
# =============================================================================
# kernel-ops | tools/dev/clone_3rd.sh
#
# Multiclone the pinned third-party reference checkouts declared in
# tools/dev/3rd_manifest.tsv. Dev-only: 3rd/ is gitignored and is never linked
# into the shipped library.
#
#   bash tools/dev/clone_3rd.sh --all --jobs 4
#   bash tools/dev/clone_3rd.sh --only sglang --force
#   bash tools/dev/clone_3rd.sh --check
#   bash tools/dev/clone_3rd.sh --list
#   bash tools/dev/clone_3rd.sh --dry-run
#   bash tools/dev/clone_3rd.sh --root 3rd-full --all
#   bash tools/dev/clone_3rd.sh --all --quiet          # logs only, no stream
#
# Output:
#   every git command is logged to <root>/.clone-logs/<name>.log and, unless
#   --quiet, also streamed to stdout as "[<name>] <line>" so parallel jobs
#   stay readable.
#
# Safety:
#   * existing git checkouts are updated, never deleted;
#   * existing non-git directories are moved to <root>/.backup/<name>-<stamp>
#     before a fresh clone (unless --no-backup);
#   * a dirty git checkout is left untouched unless --force, which discards
#     local changes in that tree (this also repairs a partially failed
#     checkout).
#
# Windows: both sglang and TensorRT-LLM contain paths longer than MAX_PATH
# (260). Git for Windows needs core.longpaths for those; this script sets it
# for its own git calls and writes it into every checkout's local config.
# =============================================================================
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SELF_DIR}/../.." && pwd)"
MANIFEST="${REPO_ROOT}/tools/dev/3rd_manifest.tsv"
ROOT=""
JOBS=""
DEPTH="shallow"
MODE="clone"
DRY_RUN=0
FORCE=0
NO_BACKUP=0
ASSUME_YES=0
WRITE_LOCK=1
STREAM=1
ONLY=""
SKIP=""

# Git for Windows goes through the Win32 API and fails to create paths beyond
# MAX_PATH (260) unless core.longpaths is on. GIT_CONFIG_* (git >= 2.31)
# applies the setting to every git child process without touching the user's
# global config; append to anything the caller already exported.
case "$(uname -s 2>/dev/null || echo unknown)" in
  MINGW*|MSYS*|CYGWIN*) IS_WINDOWS=1 ;;
  *)                    IS_WINDOWS=0 ;;
esac
if [[ "$IS_WINDOWS" -eq 1 ]]; then
  _gc="${GIT_CONFIG_COUNT:-0}"
  export "GIT_CONFIG_KEY_${_gc}=core.longpaths"
  export "GIT_CONFIG_VALUE_${_gc}=true"
  export GIT_CONFIG_COUNT=$(( _gc + 1 ))
  unset _gc
fi

usage() {
  cat <<'EOF'
usage: bash tools/dev/clone_3rd.sh [options]

  --all                 clone/update every core entry (default)
  --only NAME[,NAME]    only these manifest names
  --skip NAME[,NAME]    skip these manifest names
  --root DIR            destination root (default: <repo>/3rd)
  --manifest FILE       manifest path (default: tools/dev/3rd_manifest.tsv)
  --jobs N              parallel clones (default: 4, capped by nproc)
  --depth 1|full        1 = shallow tag clone; full = blobless history
  --update              fetch + checkout the pinned ref for existing git trees
  --check               verify HEAD against the manifest; exit 1 on drift
  --list                print the manifest
  --lock                write <root>/MANIFEST.lock (default on)
  --no-lock             do not write the lock file
  --force               checkout a dirty tree, discarding local changes there
                        (repairs a checkout that failed halfway)
  --no-backup           fail if a non-git destination already exists
  --yes                 do not ask before backing up/replacing
  --quiet               log to files only; do not stream output to the terminal
  --dry-run             print git commands without running them
  -h, --help            this help
EOF
}

die() { echo "[clone_3rd][error] $*" >&2; exit 2; }
info() { echo "[clone_3rd] $*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) MODE="clone" ;;
    --only) [[ $# -ge 2 ]] || die "--only needs a value"; ONLY="$2"; shift ;;
    --skip) [[ $# -ge 2 ]] || die "--skip needs a value"; SKIP="$2"; shift ;;
    --root) [[ $# -ge 2 ]] || die "--root needs a value"; ROOT="$2"; shift ;;
    --manifest) [[ $# -ge 2 ]] || die "--manifest needs a value"; MANIFEST="$2"; shift ;;
    --jobs) [[ $# -ge 2 ]] || die "--jobs needs a value"; JOBS="$2"; shift ;;
    --depth) [[ $# -ge 2 ]] || die "--depth needs a value"; DEPTH="$2"; shift ;;
    --update) MODE="update" ;;
    --check) MODE="check" ;;
    --list) MODE="list" ;;
    --lock) WRITE_LOCK=1 ;;
    --no-lock) WRITE_LOCK=0 ;;
    --force) FORCE=1 ;;
    --no-backup) NO_BACKUP=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    --quiet|-q) STREAM=0 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option '$1' (try --help)" ;;
  esac
  shift
done

[[ -f "$MANIFEST" ]] || die "manifest not found: $MANIFEST"
[[ -n "$ROOT" ]] || ROOT="${REPO_ROOT}/3rd"
ROOT="$(cd "$(dirname "$ROOT")" && pwd)/$(basename "$ROOT")" 2>/dev/null || true
mkdir -p "$ROOT"
ROOT="$(cd "$ROOT" && pwd)"

if [[ -z "$JOBS" ]]; then
  if command -v nproc >/dev/null 2>&1; then JOBS="$(nproc)"; else JOBS="4"; fi
fi
[[ "$JOBS" =~ ^[0-9]+$ ]] || die "--jobs must be a number"
(( JOBS > 8 )) && JOBS=8
(( JOBS < 1 )) && JOBS=1

case "$DEPTH" in
  1|shallow) DEPTH="shallow" ;;
  full) DEPTH="full" ;;
  *) die "--depth must be 1 or full" ;;
esac

LOG_DIR="${ROOT}/.clone-logs"
STATUS_FILE="${ROOT}/.clone-status.tsv"
BACKUP_DIR="${ROOT}/.backup"
mkdir -p "$LOG_DIR"

# -----------------------------------------------------------------------------
# manifest reader
# -----------------------------------------------------------------------------
manifest_rows() {
  # Emit: name<TAB>url<TAB>ref<TAB>dest<TAB>group<TAB>license<TAB>extra_refs
  awk -F'\t' '
    /^[[:space:]]*#/ { next }
    NF < 4 { next }
    { print }
  ' "$MANIFEST"
}

matches_csv() {
  local name="$1" csv="$2" item
  [[ -z "$csv" ]] && return 1
  IFS=',' read -r -a _items <<< "$csv"
  for item in "${_items[@]}"; do
    item="$(echo "$item" | xargs)"
    [[ "$item" == "$name" ]] && return 0
  done
  return 1
}

want_repo() {
  local name="$1"
  if [[ -n "$ONLY" ]]; then
    matches_csv "$name" "$ONLY"
    return $?
  fi
  if matches_csv "$name" "$SKIP"; then return 1; fi
  return 0
}

dest_for() {
  local manifest_dest="$1"
  local rel="${manifest_dest#3rd/}"
  echo "${ROOT}/${rel}"
}

# -----------------------------------------------------------------------------
# commands
# -----------------------------------------------------------------------------
# Stream one line with the repo name in front so that --jobs N stays readable,
# and keep the same line in the per-repo log.
stream_prefix() {
  local log="$1" line
  while IFS= read -r line || [[ -n "$line" ]]; do
    printf '[%s] %s\n' "${CTX:-clone_3rd}" "$line"
    printf '%s\n' "$line" >> "$log"
  done
}

run_cmd() {
  local log="$1"; shift
  local rc=0
  printf '+ %s\n' "$*" >> "$log"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "+ $*"
    return 0
  fi
  if [[ "$STREAM" -eq 0 ]]; then
    "$@" >> "$log" 2>&1
    return
  fi
  printf '[%s] + %s\n' "${CTX:-clone_3rd}" "$*"
  # pipefail is active, so a failing git command is what sets rc here.
  "$@" 2>&1 | stream_prefix "$log" || rc=$?
  return "$rc"
}

# Git commands run by hand inside 3rd/ need the same core.longpaths setting as
# the clone itself, otherwise `git status` reports every long file as modified.
persist_local_config() {
  local dest="$1" log="$2"
  [[ "$IS_WINDOWS" -eq 1 ]] || return 0
  run_cmd "$log" git -C "$dest" config core.longpaths true
}

fetch_extra_refs() {
  local dest="$1" extra="$2" log="$3" sha
  [[ -z "$extra" ]] && return 0
  IFS=',' read -r -a _shas <<< "$extra"
  for sha in "${_shas[@]}"; do
    sha="$(echo "$sha" | xargs)"
    [[ -z "$sha" ]] && continue
    if ! git -C "$dest" cat-file -e "${sha}^{commit}" 2>/dev/null; then
      if [[ "$DEPTH" == "shallow" ]]; then
        run_cmd "$log" git -C "$dest" fetch --depth 1 origin "$sha" || return 1
      else
        run_cmd "$log" git -C "$dest" fetch origin "$sha" || return 1
      fi
    fi
  done
}

clone_one() {
  local name="$1" url="$2" ref="$3" manifest_dest="$4" extra="$5"
  local dest log status head
  local -a co_force=()
  local CTX="$name"
  dest="$(dest_for "$manifest_dest")"
  log="${LOG_DIR}/${name}.log"
  : > "$log"

  # --force also repairs a checkout that died halfway (MAX_PATH before
  # core.longpaths, interrupted run): it leaves missing or untracked files and
  # a re-run would otherwise refuse the dirty tree.
  [[ "$FORCE" -eq 1 ]] && co_force=(--force)

  if [[ -e "$dest" && ! -d "$dest/.git" ]]; then
    if [[ "$NO_BACKUP" -eq 1 ]]; then
      echo "[clone_3rd][error] ${dest} exists without .git; --no-backup was given" >> "$log"
      printf '%s\t%s\t%s\t%s\n' "$name" "$ref" "-" "fail" >> "$STATUS_FILE"
      return 1
    fi
    if [[ "$ASSUME_YES" -eq 0 && -t 0 ]]; then
      read -r -p "[clone_3rd] ${dest} exists without .git. Move it to ${BACKUP_DIR}? [y/N] " ans
      case "$ans" in y|Y|yes|YES) ;; *) echo "[clone_3rd] skipped ${name}" >> "$log"; printf '%s\t%s\t%s\t%s\n' "$name" "$ref" "-" "skip" >> "$STATUS_FILE"; return 0 ;; esac
    fi
    local stamp backup
    stamp="$(date +%Y%m%d-%H%M%S)"
    backup="${BACKUP_DIR}/${name}-${stamp}"
    mkdir -p "$BACKUP_DIR"
    run_cmd "$log" mv "$dest" "$backup" || {
      printf '%s\t%s\t%s\t%s\n' "$name" "$ref" "-" "fail" >> "$STATUS_FILE"
      return 1
    }
    echo "[clone_3rd] backed up ${dest} -> ${backup}" >> "$log"
  fi

  if [[ -d "$dest/.git" ]]; then
    local current
    current="$(git -C "$dest" remote get-url origin 2>/dev/null || true)"
    if [[ -n "$current" && "$current" != "$url" ]]; then
      echo "[clone_3rd][error] ${dest} origin is ${current}, expected ${url}; use --force or --root" >> "$log"
      printf '%s\t%s\t%s\t%s\n' "$name" "$ref" "-" "fail" >> "$STATUS_FILE"
      return 1
    fi
    if [[ -n "$(git -C "$dest" status --porcelain 2>/dev/null || true)" && "$FORCE" -eq 0 ]]; then
      echo "[clone_3rd][error] ${dest} is dirty; use --force to checkout anyway" >> "$log"
      printf '%s\t%s\t%s\t%s\n' "$name" "$ref" "-" "dirty" >> "$STATUS_FILE"
      return 1
    fi
    if [[ "$DRY_RUN" -eq 0 ]]; then
      # Refresh tags only when the pinned ref is not resolved locally yet.
      # `git fetch --tags` on a shallow tree pulls a full history per tag
      # (sglang/TensorRT-LLM have hundreds), so never run it unconditionally.
      if [[ ! "$ref" =~ ^[0-9a-fA-F]{40}$ ]] \
        && ! git -C "$dest" rev-parse -q --verify "${ref}^{commit}" >/dev/null 2>&1; then
        run_cmd "$log" git -C "$dest" fetch --depth 1 --tags --force origin || true
      fi
      run_cmd "$log" git -C "$dest" checkout ${co_force[@]+"${co_force[@]}"} --detach "$ref" || {
        run_cmd "$log" git -C "$dest" fetch --depth 1 origin "$ref" || true
        run_cmd "$log" git -C "$dest" checkout ${co_force[@]+"${co_force[@]}"} --detach FETCH_HEAD || {
          printf '%s\t%s\t%s\t%s\n' "$name" "$ref" "-" "fail" >> "$STATUS_FILE"
          return 1
        }
      }
    fi
    fetch_extra_refs "$dest" "$extra" "$log" || true
    persist_local_config "$dest" "$log"
    head="$(git -C "$dest" rev-parse --verify HEAD 2>/dev/null || echo "-")"
    printf '%s\t%s\t%s\t%s\n' "$name" "$ref" "$head" "ok" >> "$STATUS_FILE"
    return 0
  fi

  # Fresh clone.
  if [[ "$ref" =~ ^[0-9a-fA-F]{40}$ && "$DEPTH" == "shallow" ]]; then
    run_cmd "$log" git init -q "$dest" || { printf '%s\t%s\t%s\t%s\n' "$name" "$ref" "-" "fail" >> "$STATUS_FILE"; return 1; }
    run_cmd "$log" git -C "$dest" remote add origin "$url" || true
    run_cmd "$log" git -C "$dest" fetch --depth 1 origin "$ref" || {
      printf '%s\t%s\t%s\t%s\n' "$name" "$ref" "-" "fail" >> "$STATUS_FILE"; return 1; }
    run_cmd "$log" git -C "$dest" checkout -q --detach FETCH_HEAD || {
      printf '%s\t%s\t%s\t%s\n' "$name" "$ref" "-" "fail" >> "$STATUS_FILE"; return 1; }
  elif [[ "$DEPTH" == "full" ]]; then
    run_cmd "$log" git clone --filter=blob:none --no-checkout "$url" "$dest" || {
      printf '%s\t%s\t%s\t%s\n' "$name" "$ref" "-" "fail" >> "$STATUS_FILE"; return 1; }
    run_cmd "$log" git -C "$dest" fetch --tags origin || true
    run_cmd "$log" git -C "$dest" checkout -q --detach "$ref" || {
      printf '%s\t%s\t%s\t%s\n' "$name" "$ref" "-" "fail" >> "$STATUS_FILE"; return 1; }
  else
    run_cmd "$log" git clone --depth 1 --branch "$ref" --single-branch "$url" "$dest" || {
      printf '%s\t%s\t%s\t%s\n' "$name" "$ref" "-" "fail" >> "$STATUS_FILE"; return 1; }
  fi

  fetch_extra_refs "$dest" "$extra" "$log" || true
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%s\t%s\t%s\t%s\n' "$name" "$ref" "-" "dry" >> "$STATUS_FILE"
    return 0
  fi
  persist_local_config "$dest" "$log"
  head="$(git -C "$dest" rev-parse --verify HEAD 2>/dev/null || echo "-")"
  printf '%s\t%s\t%s\t%s\n' "$name" "$ref" "$head" "ok" >> "$STATUS_FILE"
}

# -----------------------------------------------------------------------------
# modes
# -----------------------------------------------------------------------------
list_manifest() {
  printf '%-24s %-10s %-52s %s\n' "NAME" "GROUP" "REF" "DEST"
  while IFS=$'\t' read -r name url ref dest group license extra; do
    printf '%-24s %-10s %-52s %s\n' "$name" "${group:-core}" "$ref" "$dest"
  done < <(manifest_rows)
}

check_manifest() {
  local failed=0
  printf '%-24s %-12s %-42s %-8s %s\n' "NAME" "REF" "HEAD" "STATUS" "DEST"
  while IFS=$'\t' read -r name url ref dest group license extra; do
    want_repo "$name" || continue
    local path head want status
    path="$(dest_for "$dest")"
    if [[ ! -d "$path/.git" ]]; then
      status="missing"; head="-"; failed=1
    else
      head="$(git -C "$path" rev-parse --verify HEAD 2>/dev/null || echo "-")"
      if [[ "$ref" =~ ^[0-9a-fA-F]{40}$ ]]; then
        want="$ref"
      else
        want="$(git -C "$path" rev-parse -q --verify "${ref}^{commit}" 2>/dev/null || echo "-")"
      fi
      if [[ "$head" == "$want" ]]; then status="ok"; else status="drift"; failed=1; fi
    fi
    printf '%-24s %-12s %-42s %-8s %s\n' "$name" "$ref" "$head" "$status" "$path"
  done < <(manifest_rows)
  return "$failed"
}

clone_all() {
  : > "$STATUS_FILE"
  local started=0
  while IFS=$'\t' read -r name url ref dest group license extra; do
    want_repo "$name" || continue
    started=1
    clone_one "$name" "$url" "$ref" "$dest" "$extra" &
    while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n || true; done
  done < <(manifest_rows)
  wait || true

  if [[ "$started" -eq 0 ]]; then
    info "no manifest entry matched --only/--skip"
    return 0
  fi

  # Summary table.
  printf '\n%-24s %-12s %-42s %-8s %s\n' "NAME" "REF" "HEAD" "STATUS" "DEST"
  local failed=0
  while IFS=$'\t' read -r name url ref dest group license extra; do
    want_repo "$name" || continue
    local line status head
    line="$(awk -F'\t' -v n="$name" '$1==n {print; exit}' "$STATUS_FILE" 2>/dev/null || true)"
    if [[ -z "$line" ]]; then
      status="skip"; head="-"
    else
      status="$(echo "$line" | cut -f4)"
      head="$(echo "$line" | cut -f3)"
    fi
    local reason=""
    if [[ "$status" == "fail" || "$status" == "dirty" ]]; then
      failed=1
      reason="$(grep -m1 -E '(^fatal:|^error:|\[clone_3rd\]\[error\])' "${LOG_DIR}/${name}.log" 2>/dev/null || true)"
      [[ -n "$reason" ]] || reason="see ${LOG_DIR}/${name}.log"
    fi
    printf '%-24s %-12s %-42s %-8s %s\n' "$name" "$ref" "$head" "$status" "$(dest_for "$dest")"
    if [[ -n "$reason" ]]; then
      printf '    %s: %s\n' "$name" "$reason"
    fi
  done < <(manifest_rows)

  info "logs: ${LOG_DIR}"

  if [[ "$WRITE_LOCK" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
    local lock="${ROOT}/MANIFEST.lock"
    {
      printf '# generated by tools/dev/clone_3rd.sh %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf '# name\turl\tref\tresolved_sha\tstatus\n'
      while IFS=$'\t' read -r name url ref dest group license extra; do
        want_repo "$name" || continue
        local line status head
        line="$(awk -F'\t' -v n="$name" '$1==n {print; exit}' "$STATUS_FILE" 2>/dev/null || true)"
        status="$(echo "$line" | cut -f4)"; head="$(echo "$line" | cut -f3)"
        printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$url" "$ref" "${head:--}" "${status:-skip}"
      done < <(manifest_rows)
    } > "$lock"
    info "wrote ${lock}"
  fi

  return "$failed"
}

case "$MODE" in
  list) list_manifest ;;
  check) check_manifest ;;
  clone|update) clone_all ;;
  *) die "internal: unknown mode ${MODE}" ;;
esac
