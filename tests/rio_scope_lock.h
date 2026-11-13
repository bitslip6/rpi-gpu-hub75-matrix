#ifndef RIO_SCOPE_LOCK_H
#define RIO_SCOPE_LOCK_H

/* Linux local-filesystem advisory lock, standalone benchmark only.
 * Reopen without O_CREAT: protected_regular may otherwise reject sudo opening
 * a user-owned file in sticky /tmp. flock(LOCK_EX) works on a read-only fd on
 * Linux local filesystems. Keep the same path/inode for old/new benchmark users.
 */
#include <errno.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef RIO_SCOPE_LOCK_OPEN
#define RIO_SCOPE_LOCK_OPEN(path, flags, mode) open((path), (flags), (mode))
#endif

static inline int rio_scope_lock_open(const char *path)
{
    const int flags = O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK;
    int fd = -1;
    for (unsigned attempt = 0; attempt < 8; ++attempt) {
        fd = RIO_SCOPE_LOCK_OPEN(path, flags, (mode_t)0);
        if (fd >= 0)
            break;
        if (errno != ENOENT)
            return -1;
        fd = RIO_SCOPE_LOCK_OPEN(path, flags | O_CREAT | O_EXCL, (mode_t)0644);
        if (fd >= 0) {
            /* This NEW empty lock is nonsensitive. Permit root-first then user
             * runs even under umask 077. Never chmod an existing lock file. */
            if (fchmod(fd, 0644)) {
                int error = errno;
                close(fd);
                errno = error;
                return -1;
            }
            break;
        }
        if (errno != EEXIST)
            return -1;
    }
    if (fd < 0) {
        errno = EAGAIN; /* Bounded retry if another creator repeatedly races. */
        return -1;
    }
    struct stat opened, named;
    int error;
    if (fstat(fd, &opened))
        goto fail;
    if (!S_ISREG(opened.st_mode) || opened.st_nlink != 1 ||
        (opened.st_mode & (S_IWGRP | S_IWOTH))) {
        errno = EINVAL;
        goto fail;
    }
    if (flock(fd, LOCK_EX | LOCK_NB))
        goto fail;
    if (lstat(path, &named))
        goto fail;
    if (!S_ISREG(named.st_mode) || opened.st_dev != named.st_dev ||
        opened.st_ino != named.st_ino || named.st_nlink != 1) {
        errno = ESTALE;
        goto fail;
    }
    return fd; /* Caller keeps this descriptor through GPIO restoration. */
fail:
    error = errno;
    close(fd);
    errno = error;
    return -1;
}

#endif
