 /* debug_log.h */
#ifndef __DEBUG_H__
#define __DEBUG_H__
#include <stddef.h>

#if defined(__GNUC__) || defined(__clang__)
#  define printf_like(fmtpos, argpos) __attribute__((format(printf, fmtpos, argpos)))
#else
#  define printf_like(fmtpos, argpos)
#endif

#ifdef HAVE_DEBUG_FUNC
/* If you provide debug() elsewhere, declare it here for type checking. */
void debug(const char *fmt, ...) printf_like(1, 2);

/* DEBUG(...) calls debug(...) exactly like printf */
#  define DEBUG(...) do { debug(__VA_ARGS__); } while (0)

#elif !defined DEBUG
/* No debug function available, compile to nothing without evaluating args */
#  if defined(__GNUC__) || defined(__clang__)
     /* GNU extension prevents varargs evaluation in disabled branch */
#    define DEBUG(...) do { if (0) (void)printf(__VA_ARGS__); } while (0)
#  else
#    define DEBUG(...) do { (void)0; } while (0)
#  endif
#endif

#endif