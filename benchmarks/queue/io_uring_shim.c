#define _GNU_SOURCE
#include <dlfcn.h>
#include <sys/syscall.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>

long syscall(long number, ...) {
    if (number == SYS_io_uring_setup || number == SYS_io_uring_enter || number == SYS_io_uring_register) {
        // fprintf(stderr, "[SHIM] Blocking io_uring syscall %ld in PID %d\n", number, getpid());
        errno = ENOSYS;
        return -1;
    }
    
    typedef long (*syscall_t)(long, ...);
    static syscall_t real_syscall = NULL;
    if (!real_syscall) {
        real_syscall = (syscall_t)dlsym(RTLD_NEXT, "syscall");
    }
    
    va_list args;
    va_start(args, number);
    long arg1 = va_arg(args, long);
    long arg2 = va_arg(args, long);
    long arg3 = va_arg(args, long);
    long arg4 = va_arg(args, long);
    long arg5 = va_arg(args, long);
    long arg6 = va_arg(args, long);
    va_end(args);
    
    return real_syscall(number, arg1, arg2, arg3, arg4, arg5, arg6);
}
