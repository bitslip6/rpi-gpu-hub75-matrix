#ifndef H75_GPIO_LOCK_H
#define H75_GPIO_LOCK_H
/* Internal shared GPIO advisory lock. Never unlink the inode on release. */
int h75_gpio_lock_acquire(void); /* fd, or -1 with errno; caller closes fd */
#endif
