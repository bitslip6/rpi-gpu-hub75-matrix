#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dirent.h>
#include <ctype.h>
#include <errno.h>
#include <unistd.h>
#include <sched.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

static const int target_cpu = 3;

static int read_whole_file(const char *path, char **out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    size_t cap = 4096, len = 0;
    char *buf = malloc(cap);
    if (!buf) { close(fd); return -1; }
    for (;;) {
        ssize_t n = read(fd, buf + len, cap - len);
        if (n < 0) { free(buf); close(fd); return -1; }
        if (n == 0) break;
        len += (size_t)n;
        if (len == cap) {
            cap *= 2;
            char *tmp = realloc(buf, cap);
            if (!tmp) { free(buf); close(fd); return -1; }
            buf = tmp;
        }
    }
    buf[len] = '\0';
    close(fd);
    *out = buf;
    return (int)len;
}

static int cpu_in_list_string(const char *s, int cpu) {
    /* supports formats like "0,2-3,7" and "3" */
    const char *p = s;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == ',') p++;
        if (!*p || *p == '\n') break;
        char *end = NULL;
        long a = strtol(p, &end, 10);
        if (end == p) break;
        int has_dash = 0;
        int b = (int)a;
        p = end;
        if (*p == '-') {
            has_dash = 1;
            p++;
            long r = strtol(p, &end, 10);
            if (end == p) break;
            b = (int)r;
            p = end;
        }
        if (!has_dash) {
            if (a == cpu) return 1;
        } else {
            if ((int)a <= cpu && cpu <= b) return 1;
        }
        while (*p && *p != ',') p++;
    }
    return 0;
}

static int hex_mask_includes_cpu(const char *s, int cpu) {
    /* rps_cpus and xps_cpus are hex masks, possibly multiword like "f,0" */
    /* read from right to left, each hex nibble is 4 CPUs */
    int nibble_index = cpu / 4;
    int bit_in_nibble = cpu % 4;
    int nibble_count = 0;

    /* build a compact string without commas and spaces */
    char buf[512]; size_t j = 0;
    for (size_t i = 0; s[i] && j + 1 < sizeof(buf); i++) {
        if (s[i] == ',' || s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t') continue;
        buf[j++] = (char)tolower((unsigned char)s[i]);
    }
    buf[j] = '\0';

    /* walk from end */
    for (ssize_t i = (ssize_t)j - 1; i >= 0; i--) {
        char c = buf[i];
        int val = -1;
        if (c >= '0' && c <= '9') val = c - '0';
        else if (c >= 'a' && c <= 'f') val = 10 + (c - 'a');
        else continue;

        if (nibble_count == nibble_index) {
            return (val & (1 << bit_in_nibble)) != 0;
        }
        nibble_count++;
    }
    /* if mask shorter than needed, bit is zero */
    return 0;
}

static void audit_cmdline(void) {
    char *cmd = NULL;
    if (read_whole_file("/proc/cmdline", &cmd) < 0) return;
    printf("== kernel cmdline ==\n%s\n", cmd);
    free(cmd);
}

static void audit_governor(void) {
    char path[256], *gov = NULL;
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", target_cpu);
    if (read_whole_file(path, &gov) >= 0) {
        printf("== cpu%d governor ==\n%s", target_cpu, gov);
        free(gov);
    }
}

static void audit_irq_affinity(void) {
    DIR *d = opendir("/proc/irq");
    if (!d) return;
    printf("== IRQs with affinity including CPU%d ==\n", target_cpu);
    struct dirent *de;
    int found = 0;
    while ((de = readdir(d))) {
        if (!isdigit((unsigned char)de->d_name[0])) continue;
        char path[512], *list = NULL;
        snprintf(path, sizeof(path), "/proc/irq/%s/smp_affinity_list", de->d_name);
        if (read_whole_file(path, &list) < 0) continue;
        if (cpu_in_list_string(list, target_cpu)) {
            /* fetch a human label from /proc/interrupts */
            char *ints = NULL;
            if (read_whole_file("/proc/interrupts", &ints) >= 0) {
                char *line = ints, *saveptr = NULL;
                while (line) {
                    char *next = strchr(line, '\n');
                    if (next) *next = '\0';
                    if (strncmp(line, de->d_name, strlen(de->d_name)) == 0) {
                        /* print irq and line tail */
                        const char *label = strrchr(line, ' ');
                        printf("irq %-5s  smp_affinity_list=%slabel:%s\n",
                               de->d_name, list, label ? label + 1 : "(unknown)");
                        found = 1;
                        break;
                    }
                    line = next ? next + 1 : NULL;
                }
                free(ints);
            } else {
                printf("irq %-5s  smp_affinity_list=%s", de->d_name, list);
                found = 1;
            }
        }
        free(list);
    }
    if (!found) printf("(none)\n");
    closedir(d);
}

static void audit_rps_xps(void) {
    DIR *net = opendir("/sys/class/net");
    if (!net) return;
    printf("== NIC queue steering that includes CPU%d ==\n", target_cpu);
    struct dirent *nd;
    int found = 0;
    while ((nd = readdir(net))) {
        if (nd->d_name[0] == '.') continue;
        char base[512];
        snprintf(base, sizeof(base), "/sys/class/net/%s/queues", nd->d_name);

        DIR *q = opendir(base);
        if (!q) continue;
        struct dirent *qd;
        while ((qd = readdir(q))) {
            if (qd->d_name[0] == '.') continue;
            char path[768], *mask = NULL;
            /* rps */
            snprintf(path, sizeof(path), "%s/%s/rps_cpus", base, qd->d_name);
            if (read_whole_file(path, &mask) >= 0) {
                if (hex_mask_includes_cpu(mask, target_cpu)) {
                    printf("%s %s rps_cpus=%s", nd->d_name, qd->d_name, mask);
                    found = 1;
                }
                free(mask);
            }
            /* xps */
            snprintf(path, sizeof(path), "%s/%s/xps_cpus", base, qd->d_name);
            if (read_whole_file(path, &mask) >= 0) {
                if (hex_mask_includes_cpu(mask, target_cpu)) {
                    printf("%s %s xps_cpus=%s", nd->d_name, qd->d_name, mask);
                    found = 1;
                }
                free(mask);
            }
        }
        closedir(q);
    }
    if (!found) printf("(none)\n");
    closedir(net);
}

static int parse_pid(const char *name) {
    for (const char *p = name; *p; p++) if (!isdigit((unsigned char)*p)) return -1;
    return atoi(name);
}

static void audit_tasks_affinity(void) {
    DIR *proc = opendir("/proc");
    if (!proc) return;
    printf("== tasks whose affinity includes CPU%d ==\n", target_cpu);
    struct dirent *pe;
    int found = 0;

    while ((pe = readdir(proc))) {
        int pid = parse_pid(pe->d_name);
        if (pid < 1) continue;

        char taskdir[256];
        snprintf(taskdir, sizeof(taskdir), "/proc/%d/task", pid);
        DIR *td = opendir(taskdir);
        if (!td) continue;

        struct dirent *te;
        while ((te = readdir(td))) {
            int tid = parse_pid(te->d_name);
            if (tid < 1) continue;

            cpu_set_t set;
            CPU_ZERO(&set);
            if (sched_getaffinity(tid, sizeof(set), &set) != 0) continue;
            if (CPU_ISSET(target_cpu, &set)) {
                /* read comm */
                char commpath[256], comm[256] = {0};
                snprintf(commpath, sizeof(commpath), "/proc/%d/task/%d/comm", pid, tid);
                int fd = open(commpath, O_RDONLY);
                if (fd >= 0) {
                    ssize_t n = read(fd, comm, sizeof(comm) - 1);
                    if (n > 0) comm[n] = '\0';
                    close(fd);
                }
                /* read policy and prio from /proc/.../stat */
                char statpath[256], *stat = NULL;
                snprintf(statpath, sizeof(statpath), "/proc/%d/task/%d/stat", pid, tid);
                int pol = -1, prio = -1;
                if (read_whole_file(statpath, &stat) >= 0) {
                    /* fields: ... policy at 41, nice at 19, priority at 18, rt_priority at 42 on recent kernels */
                    /* simple parse: split by space and index */
                    int field = 0;
                    char *s = stat, *tok;
                    while ((tok = strsep(&s, " \t")) != NULL) {
                        if (*tok == '\0') continue;
                        field++;
                        if (field == 41) pol = atoi(tok);
                        if (field == 42) { prio = atoi(tok); break; }
                    }
                    free(stat);
                }

                printf("pid=%d tid=%d comm=%s policy=%d rt_prio=%d\n",
                       pid, tid, strlen(comm)?comm:"?", pol, prio);
                found = 1;
            }
        }
        closedir(td);
    }
    if (!found) printf("(none)\n");
    closedir(proc);
}

static void audit_rt_throttle(void) {
    char *v = NULL;
    if (read_whole_file("/proc/sys/kernel/sched_rt_runtime_us", &v) >= 0) {
        printf("== sched_rt_runtime_us ==\n%s", v);
        free(v);
    }
}

int main(void) {
    printf("core3_audit, scanning for anything that can run on CPU%d\n\n", target_cpu);
    audit_cmdline();
    putchar('\n');
    audit_governor();
    putchar('\n');
    audit_irq_affinity();
    putchar('\n');
    audit_rps_xps();
    putchar('\n');
    audit_tasks_affinity();
    putchar('\n');
    audit_rt_throttle();
    return 0;
}