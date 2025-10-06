#!/usr/bin/env sh
set -eu

arch="$(uname -m 2>/dev/null || echo unknown)"

# default safe flags
fallback_arm64="-march=armv8-a"
fallback_arm32="-mcpu=cortex-a53 -mfpu=neon-vfpv4 -mfloat-abi=hard"

case "$arch" in
  aarch64)
    # try Model name first, which your Pi 5 provides
    model="$(lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^[ \t]+|[ \t]+$/,"",$2); print $2; exit}')"
    case "$model" in
      *Cortex-A76*) echo "-mcpu=cortex-a76 -mtune=cortex-a76"; exit 0 ;;
      *Cortex-A72*) echo "-mcpu=cortex-a72 -mtune=cortex-a72"; exit 0 ;;
      *Cortex-A53*) echo "-mcpu=cortex-a53 -mtune=cortex-a53"; exit 0 ;;
    esac

    # fallback: parse CPU part codes if present
    cpu_part="$(awk -F: '/CPU part/ {gsub(/^[ \t]+|[ \t]+$/,"",$2); print tolower($2); exit}' /proc/cpuinfo 2>/dev/null || true)"
    case "$cpu_part" in
      0xd0b) echo "-mcpu=cortex-a76 -mtune=cortex-a76"; exit 0 ;; # Pi 5
      0xd08) echo "-mcpu=cortex-a72 -mtune=cortex-a72"; exit 0 ;; # Pi 4
      0xd03) echo "-mcpu=cortex-a53 -mtune=cortex-a53"; exit 0 ;; # Pi 3 64-bit userspace
    esac

    # last fallback: let the compiler decide
    if printf "" | ${CC:-cc} -mcpu=native -x c -c - -o /dev/null >/dev/null 2>&1; then
      echo "-mcpu=native"; exit 0
    fi
    echo "$fallback_arm64";;
  armv7l)
    echo "$fallback_arm32";;
  *)
    # dev boxes and everything else
    if printf "" | ${CC:-cc} -march=native -x c -c - -o /dev/null >/dev/null 2>&1; then
      echo "-march=native"; exit 0
    fi
    echo "";;
esac

