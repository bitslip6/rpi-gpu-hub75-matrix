/* Temporary regular files only. No GPIO discovery, mapping or system tuning. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

static int traced_open(const char *path, int flags, mode_t mode);
#define RIO_SCOPE_LOCK_OPEN traced_open
#include "rio_scope_lock.h"

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "%s:%d: %s (errno=%d)\n", __FILE__, __LINE__, #condition, errno); \
    abort(); \
} } while (0)

static unsigned opens, creates;
static bool creator_race, endless_race, deny_read;

static int traced_open(const char *path, int flags, mode_t mode)
{
    ++opens;
    if (flags & O_CREAT)
        ++creates;
    struct stat st;
    /* Model the relevant protected_regular rejection without needing root or
     * changing sysctls. An existing-file O_CREAT reopen must never be attempted. */
    if ((flags & O_CREAT) && !(flags & O_EXCL) && lstat(path, &st) == 0) {
        errno = EACCES;
        return -1;
    }
    if (deny_read) {
        errno = EACCES;
        return -1;
    }
    if (endless_race) {
        errno = flags & O_CREAT ? EEXIST : ENOENT;
        return -1;
    }
    if (creator_race && !(flags & O_CREAT)) {
        creator_race = false;
        int fd = open(path, O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
        CHECK(fd >= 0);
        CHECK(close(fd) == 0);
        errno = ENOENT; /* Another process created it after our failed lookup. */
        return -1;
    }
    return open(path, flags, mode);
}

int main(void)
{
    char directory[] = "/tmp/rio-scope-lock-test-XXXXXX";
    CHECK(mkdtemp(directory) != NULL);
    char path[256], alias[256];
    CHECK(snprintf(path, sizeof(path), "%s/lock", directory) > 0);
    CHECK(snprintf(alias, sizeof(alias), "%s/alias", directory) > 0);

    mode_t previous = umask(077);
    int fd = rio_scope_lock_open(path);
    umask(previous);
    CHECK(fd >= 0);
    struct stat original, current;
    CHECK(fstat(fd, &original) == 0);
    CHECK((original.st_mode & 0777) == 0644); /* New root-first locks remain readable. */
    CHECK((fcntl(fd, F_GETFL) & O_ACCMODE) == O_RDONLY);
    CHECK(fcntl(fd, F_GETFD) & FD_CLOEXEC);

    /* Separate open descriptions must conflict even in the same process. */
    CHECK(rio_scope_lock_open(path) < 0 && (errno == EWOULDBLOCK || errno == EAGAIN));
    pid_t child = fork();
    CHECK(child >= 0);
    if (child == 0) {
        close(fd); /* Parent still holds the lock; do not inherit a false test. */
        int contender = rio_scope_lock_open(path);
        bool blocked = contender < 0 && (errno == EWOULDBLOCK || errno == EAGAIN);
        if (contender >= 0) close(contender);
        _exit(blocked ? 0 : 1);
    }
    int status;
    CHECK(waitpid(child, &status, 0) == child);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    CHECK(close(fd) == 0);

    /* Reuse a legacy 0600 file, preserving inode, permissions and contents. */
    CHECK(chmod(path, 0600) == 0);
    int legacy = open(path, O_RDWR | O_CLOEXEC);
    CHECK(legacy >= 0);
    const char content[] = "existing lock data";
    CHECK(write(legacy, content, sizeof(content)) == (ssize_t)sizeof(content));
    CHECK(flock(legacy, LOCK_EX | LOCK_NB) == 0);
    CHECK(rio_scope_lock_open(path) < 0 && (errno == EWOULDBLOCK || errno == EAGAIN));
    CHECK(close(legacy) == 0);
    opens = creates = 0;
    fd = rio_scope_lock_open(path);
    CHECK(fd >= 0 && opens == 1 && creates == 0);
    CHECK(fstat(fd, &current) == 0);
    CHECK(original.st_ino == current.st_ino && original.st_dev == current.st_dev);
    CHECK((current.st_mode & 0777) == 0600);
    char actual[sizeof(content)];
    CHECK(read(fd, actual, sizeof(actual)) == (ssize_t)sizeof(actual));
    CHECK(memcmp(actual, content, sizeof(content)) == 0);
    CHECK(close(fd) == 0);

    CHECK(symlink(path, alias) == 0);
    CHECK(rio_scope_lock_open(alias) < 0 && errno == ELOOP);
    CHECK(unlink(alias) == 0);
    CHECK(link(path, alias) == 0);
    CHECK(rio_scope_lock_open(path) < 0 && errno == EINVAL);
    CHECK(unlink(alias) == 0);
    CHECK(chmod(path, 0666) == 0);
    CHECK(rio_scope_lock_open(path) < 0 && errno == EINVAL);
    CHECK(chmod(path, 0600) == 0);
    CHECK(mkfifo(alias, 0600) == 0);
    CHECK(rio_scope_lock_open(alias) < 0 && errno == EINVAL);
    CHECK(unlink(alias) == 0);
    CHECK(rio_scope_lock_open(directory) < 0 && errno == EINVAL);

    /* Permission errors must not trigger creation, deletion or a fallback path. */
    deny_read = true;
    opens = creates = 0;
    CHECK(rio_scope_lock_open(path) < 0 && errno == EACCES);
    CHECK(opens == 1 && creates == 0);
    deny_read = false;

    creator_race = true;
    opens = creates = 0;
    fd = rio_scope_lock_open(alias);
    CHECK(fd >= 0 && creates == 1 && opens == 3);
    CHECK(fstat(fd, &current) == 0 && (current.st_mode & 0777) == 0600);
    CHECK(close(fd) == 0);
    CHECK(unlink(alias) == 0);
    endless_race = true;
    opens = creates = 0;
    CHECK(rio_scope_lock_open(alias) < 0 && errno == EAGAIN);
    CHECK(opens == 16 && creates == 8);
    endless_race = false;

    CHECK(unlink(path) == 0);
    CHECK(rmdir(directory) == 0);
    puts("RIO scope advisory-lock tests passed (temporary files only; no GPIO).");
    return 0;
}
