#define _GNU_SOURCE
#include "gpio_lock.h"
#include <errno.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

int h75_gpio_lock_acquire(void) {
    const char *path = "/tmp/rpi-hub75-rio-scope.lock";
    const int flags = O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK;
    int fd = open(path, flags);
    if (fd < 0 && errno == ENOENT) {
        fd = open(path, flags | O_CREAT | O_EXCL, 0644);
        if (fd >= 0)
            fchmod(fd, 0644);
        else if (errno == EEXIST)
            fd = open(path, flags);
    }
    struct stat st;
    if (fd < 0)
        return -1;
    int rc = fstat(fd, &st);
    if (!rc && (!S_ISREG(st.st_mode) || st.st_nlink != 1)) {
        errno = EINVAL;
        rc = -1;
    }
    if (rc || flock(fd, LOCK_EX | LOCK_NB)) {
        int error = errno;
        close(fd);
        errno = error;
        return -1;
    }
    return fd;
}
