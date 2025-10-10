#!/bin/sh


printf "[Manager]\nDefaultCPUAffinity=0 1 2\n" | sudo tee /etc/systemd/system/cpu.conf
systemctl daemon-reexec

systemctl disable --now bluetooth


# Force all generic system work onto CPUs 0-2, keep CPU3 clean for RT
set -eu

ALLOWED_LIST="0-2"   # human-readable list
ALLOWED_HEX="7"      # hex bitmask for cpumasks (CPU0..2 => 0b0111 = 0x7)

log() { printf '%s\n' "$*"; }

# 1) IRQs: move device interrupts off CPU3
log "[*] repinning device IRQs to $ALLOWED_LIST"
grep -E ':[[:space:]]' /proc/interrupts | while read -r line; do
  irq=$(echo "$line" | awk -F: '{print $1}' | xargs) || true
  [ -n "$irq" ] || continue
  f="/proc/irq/$irq/smp_affinity_list"
  [ -w "$f" ] || continue
  # Skip obvious non-device lines by name
  name=$(echo "$line" | awk '{print $NF}')
  case "$name" in
    *IPI*|*timer*|*resched*|*call_function*|*thermal*|*TLB* ) continue ;;
  esac
  echo "$ALLOWED_LIST" > "$f" 2>/dev/null || true
done

# 2) NIC queues: RPS/XPS away from CPU3
log "[*] setting NIC RPS/XPS to $ALLOWED_LIST"
for dev in /sys/class/net/*; do
  [ -d "$dev" ] || continue
  for f in "$dev"/queues/rx-*/rps_cpus; do
    [ -f "$f" ] && echo "$ALLOWED_HEX" > "$f" 2>/dev/null || true
  done
  for f in "$dev"/queues/tx-*/xps_cpus; do
    [ -f "$f" ] && echo "$ALLOWED_HEX" > "$f" 2>/dev/null || true
  done
done

# 3) Unbound workqueues: keep kernel background work off CPU3
#   This is important because many subsystems dispatch via unbound workqueues.
if [ -w /sys/devices/virtual/workqueue/cpumask ]; then
  log "[*] setting unbound workqueue cpumask to 0x$ALLOWED_HEX"
  echo "$ALLOWED_HEX" > /sys/devices/virtual/workqueue/cpumask 2>/dev/null || true
fi

# Writeback (fs) workqueue may have its own mask on some kernels
for f in /sys/bus/workqueue/devices/writeback/cpumask \
         /sys/devices/virtual/workqueue/writeback/cpumask; do
  [ -f "$f" ] && echo "$ALLOWED_HEX" > "$f" 2>/dev/null || true
done

# 4) Misc driver threads that accept userspace affinity changes
#    Heuristic: try to re-affine known offenders if they aren't strictly per-CPU.
log "[*] attempting to move driver/kernel threads off CPU3 (best effort)"
for pid in $(ls -1 /proc | grep -E '^[0-9]+$'); do
  tdir="/proc/$pid/task"
  [ -d "$tdir" ] || continue
  for tid in $(ls -1 "$tdir" | grep -E '^[0-9]+$'); do
    comm=$(cat "$tdir/$tid/comm" 2>/dev/null || true)
    case "$comm" in
      kworker/*|kdevtmpfs|brcmf_wdog*|spi*|cec-*|card*-crtc*|jbd2/*|uas|hwrng|mmc_complete|mop*|moplet* )
        # try sched_setaffinity through taskset
        taskset -pc "$ALLOWED_LIST" "$tid" >/dev/null 2>&1 || true
      ;;
    esac
  done
done

# 5) Keep RT throttling enabled for safety
if [ -f /proc/sys/kernel/sched_rt_runtime_us ]; then
  cur=$(cat /proc/sys/kernel/sched_rt_runtime_us)
  [ "$cur" = "-1" ] && echo 950000 > /proc/sys/kernel/sched_rt_runtime_us || true
fi

log "[*] done"

