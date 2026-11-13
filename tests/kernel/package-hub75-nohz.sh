#!/usr/bin/env bash
# Package the existing clean 6.18.54 full-dynticks cross-build. No sudo, kernel
# compilation, Pi access, system module installation, boot edits or reboot.
set -Eeuo pipefail

usage() {
    printf 'Usage: bash %s [WORKSPACE]\n' "$0"
    printf 'Default: $HOME/kernel-builds/hub75-nohz-clean-20261023-182812\n'
    printf 'Run on the desktop containing the completed cross-build.\n'
}
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
trap 'printf "ERROR: packaging stopped at line %s (exit %s). Existing build files were not removed.\n" "$LINENO" "$?" >&2' ERR

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
[[ $# -le 1 ]] || { usage >&2; exit 2; }
[[ $EUID -ne 0 ]] || fail 'Run as your normal desktop user, not with sudo.'

WORK=${1:-"$HOME/kernel-builds/hub75-nohz-clean-20261023-182812"}
[[ -d "$WORK" ]] || fail "Workspace not found: $WORK"
WORK=$(cd -- "$WORK" && pwd -P)
SRC="$WORK/src"
OUT="$WORK/out"
CROSS_COMPILE=${CROSS_COMPILE:-aarch64-linux-gnu-}
EXPECTED_COMMIT=703bc4e92a9f98d797f0289bf2adb1e8b04c4bbb

for command in make git find sort xargs modinfo sha256sum tar tee mktemp strings "${CROSS_COMPILE}gcc"; do
    command -v "$command" >/dev/null || fail "Required command not found: $command"
done
for file in "$SRC/Makefile" "$OUT/.config" "$OUT/arch/arm64/boot/Image" \
            "$OUT/Module.symvers" "$OUT/System.map" "$OUT/include/config/kernel.release"; do
    [[ -s "$file" ]] || fail "Missing/empty build file: $file. Complete the kernel build first."
done
[[ $(git -C "$SRC" rev-parse HEAD) == "$EXPECTED_COMMIT" ]] || fail 'Unexpected source commit.'
[[ -z $(git -C "$SRC" status --porcelain) ]] || fail 'Source checkout is not clean; inspect git status before packaging.'
for setting in CONFIG_NO_HZ_FULL=y CONFIG_RCU_NOCB_CPU=y \
               CONFIG_VIRT_CPU_ACCOUNTING_GEN=y CONFIG_CONTEXT_TRACKING_USER=y \
               CONFIG_CPU_ISOLATION=y CONFIG_ARM64_16K_PAGES=y CONFIG_PREEMPT=y \
               CONFIG_HZ=250 CONFIG_RP1_PIO=m CONFIG_MODVERSIONS=y; do
    grep -Fx -- "$setting" "$OUT/.config" >/dev/null || fail "Configuration check failed: $setting"
done

KREL=$(<"$OUT/include/config/kernel.release")
case "$KREL" in
    6.18.54-v8-16k-hub75-nohz|6.18.54-v8-16k-hub75-nohz+) ;;
    *) fail "Unexpected built release: $KREL" ;;
esac
case $("${CROSS_COMPILE}gcc" -dumpmachine) in
    aarch64*) ;;
    *) fail 'Cross compiler does not target AArch64.' ;;
esac

ARCHIVE="$WORK/$KREL.tar.gz"
CHECKSUM="$ARCHIVE.sha256"
[[ ! -e "$ARCHIVE" && ! -e "$CHECKSUM" ]] || fail "Package already exists. Preserve it or choose another workspace; refusing to overwrite: $ARCHIVE"
STAGE=$(mktemp -d "$WORK/package-stage-XXXXXXXX")
mkdir -p "$STAGE/boot" "$STAGE/provenance"
printf 'Workspace: %s\nKernel:    %s\nStaging:   %s\n' "$WORK" "$KREL" "$STAGE"
printf 'No root privileges or Pi-side changes are used.\n\n'

kmake() {
    make --no-print-directory -C "$SRC" O="$OUT" ARCH=arm64 \
        CROSS_COMPILE="$CROSS_COMPILE" "$@"
}
# This is metadata validation, not an Image/modules compilation target.
COMPUTED_KREL=$(kmake -s kernelrelease)
[[ "$COMPUTED_KREL" == "$KREL" ]] || fail "Computed release changed: $COMPUTED_KREL (built: $KREL)"

kmake INSTALL_MOD_PATH="$STAGE" DEPMOD=true modules_install \
    2>&1 | tee "$STAGE/provenance/modules-install.log"

cp "$OUT/arch/arm64/boot/Image" "$STAGE/boot/kernel-hub75-nohz.img"
cp "$OUT/.config" "$STAGE/provenance/kernel.config"
cp "$OUT/Module.symvers" "$STAGE/provenance/Module.symvers"
cp "$OUT/System.map" "$STAGE/provenance/System.map"
for name in build.log config.diff; do
    if [[ -f "$WORK/$name" ]]; then
        cp "$WORK/$name" "$STAGE/provenance/$name"
    fi
done
if [[ -f "$WORK/input/base.config" ]]; then
    cp "$WORK/input/base.config" "$STAGE/provenance/base.config"
fi
printf '%s\n' "$KREL" > "$STAGE/provenance/kernelrelease.txt"
git -C "$SRC" rev-parse HEAD > "$STAGE/provenance/source-commit.txt"
git -C "$SRC" diff HEAD > "$STAGE/provenance/source.diff"
"${CROSS_COMPILE}gcc" --version > "$STAGE/provenance/packaging-compiler.txt"
# Record build-time compiler metadata separately from the current PATH compiler.
grep '^CONFIG_CC_VERSION_TEXT=' "$OUT/.config" > "$STAGE/provenance/build-compiler.txt"
strings "$STAGE/boot/kernel-hub75-nohz.img" \
    | grep -F -- "Linux version $KREL " > "$STAGE/provenance/kernel-banner.txt"

PIO_MODULE=$(find "$STAGE/lib/modules/$KREL" -type f -name 'rp1-pio.ko*' -print -quit)
[[ -n "$PIO_MODULE" ]] || fail 'Staged rp1-pio module not found.'
VERMAGIC=$(modinfo -F vermagic "$PIO_MODULE")
printf '\nPIO module vermagic: %s\n' "$VERMAGIC"
case "$VERMAGIC" in
    "$KREL "*) ;;
    *) fail 'Module release does not match the kernel.' ;;
esac
# Capture first so modinfo failures cannot be mistaken for an absent parameter.
PARAMETERS=$(modinfo -p "$PIO_MODULE")
if [[ "$PARAMETERS" == *heavy_tx_burst4_sm* ]]; then
    fail 'Experimental PIO parameter unexpectedly present.'
fi
printf '%s\n' "$VERMAGIC" > "$STAGE/provenance/pio-vermagic.txt"

# Remove generated host build/source links only inside this NEW staging tree.
for name in build source; do
    link="$STAGE/lib/modules/$KREL/$name"
    if [[ -L "$link" ]]; then
        rm -- "$link"
    fi
done
[[ -z $(git -C "$SRC" status --porcelain) ]] || fail 'Source checkout changed during staging.'
(
    cd "$STAGE"
    find boot lib provenance -type f -print0 \
        | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum -c SHA256SUMS > "$STAGE/verification.log"
)

# Keep interrupted archive output clearly separate from the final package.
PARTIAL=$(mktemp "$WORK/.hub75-nohz-package-XXXXXXXX.tar.gz.partial")
printf '\nCreating archive (compression can take a little while)...\n'
tar -C "$STAGE" -czf "$PARTIAL" boot lib provenance SHA256SUMS
# Recheck before publishing; no existing completed package is intentionally replaced.
[[ ! -e "$ARCHIVE" && ! -e "$CHECKSUM" ]] || fail 'A package appeared while staging; refusing to replace it.'
# Hard-link publication fails rather than overwriting a concurrently created file.
ln -- "$PARTIAL" "$ARCHIVE"
rm -- "$PARTIAL"
(
    cd "$WORK"
    (set -o noclobber; sha256sum "$KREL.tar.gz" > "$KREL.tar.gz.sha256")
    sha256sum -c "$KREL.tar.gz.sha256"
)
printf '\nPackaging completed successfully.\nArchive:  %s\nChecksum: %s\nStaging:  %s\n' "$ARCHIVE" "$CHECKSUM" "$STAGE"
printf '\nNext: transfer and verify on the Pi. Nothing has been installed.\n'
