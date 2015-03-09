.SUFFIXES: .cpp

CXX = clang++ 

HFILES = ceres-ba.h XBuilder.h

SRCS = main.cpp XBuilder.cpp sfm.cpp fmatrix_matching.cpp triangulation.cpp \
       two_view_reconst.cpp \
       pnp.cpp \
       bundle-adjust.cpp ceres-ba.cpp

OBJS = $(SRCS:.cpp=.o)

CFLAGS = -O3 \
         -I/usr/local/include \
         -I/usr/local/include/eigen3

LIBS =  /usr/local/lib/libopencv* \
       -L/usr/local/lib \
       -lglog -lgflags \
       -lceres \
       -framework Accelerate \
       -lamd -lbtf -lcamd -lccolamd -lcholmod -lcolamd -lcxsparse -lklu -lldl -lrbio -lspqr -lumfpack -lsuitesparseconfig 

xBulder: $(OBJS) Makefile
	$(CXX) $(CFLAGS) -o $@ $(SRCS) $(LIBS)

.cpp.o:
	$(CXX) -c $(CFLAGS) $<

clean:
	rm -rf *.o

depend:
	makedepend $(CFLAGS) $(SRCS)

# DO NOT DELETE

main.o: XBuilder.h /usr/include/unistd.h /usr/include/_types.h
main.o: /usr/include/sys/_types.h /usr/include/sys/cdefs.h
main.o: /usr/include/sys/_symbol_aliasing.h
main.o: /usr/include/sys/_posix_availability.h /usr/include/machine/_types.h
main.o: /usr/include/i386/_types.h /usr/include/sys/_pthread/_pthread_types.h
main.o: /usr/include/sys/unistd.h /usr/include/sys/_types/_posix_vdisable.h
main.o: /usr/include/sys/_types/_seek_set.h /usr/include/sys/_types/_size_t.h
main.o: /usr/include/_types/_uint64_t.h /usr/include/Availability.h
main.o: /usr/include/AvailabilityInternal.h
main.o: /usr/include/sys/_types/_ssize_t.h /usr/include/sys/_types/_uid_t.h
main.o: /usr/include/sys/_types/_gid_t.h /usr/include/sys/_types/_intptr_t.h
main.o: /usr/include/sys/_types/_off_t.h /usr/include/sys/_types/_pid_t.h
main.o: /usr/include/sys/_types/_useconds_t.h /usr/include/sys/_types/_null.h
main.o: /usr/include/sys/select.h /usr/include/sys/appleapiopts.h
main.o: /usr/include/sys/_types/_fd_def.h /usr/include/sys/_types/_timespec.h
main.o: /usr/include/sys/_types/_timeval.h /usr/include/sys/_types/_time_t.h
main.o: /usr/include/sys/_types/_suseconds_t.h
main.o: /usr/include/sys/_types/_sigset_t.h
main.o: /usr/include/sys/_types/_fd_setsize.h
main.o: /usr/include/sys/_types/_fd_set.h /usr/include/sys/_types/_fd_clr.h
main.o: /usr/include/sys/_types/_fd_isset.h
main.o: /usr/include/sys/_types/_fd_zero.h /usr/include/sys/_types/_fd_copy.h
main.o: /usr/include/sys/_select.h /usr/include/sys/_types/_dev_t.h
main.o: /usr/include/sys/_types/_mode_t.h /usr/include/sys/_types/_uuid_t.h
main.o: /usr/include/gethostuuid.h /usr/include/sys/param.h
main.o: /usr/include/sys/types.h /usr/include/machine/types.h
main.o: /usr/include/i386/types.h /usr/include/sys/_types/_int8_t.h
main.o: /usr/include/sys/_types/_int16_t.h /usr/include/sys/_types/_int32_t.h
main.o: /usr/include/sys/_types/_int64_t.h
main.o: /usr/include/sys/_types/_uintptr_t.h /usr/include/machine/endian.h
main.o: /usr/include/i386/endian.h /usr/include/sys/_endian.h
main.o: /usr/include/libkern/_OSByteOrder.h
main.o: /usr/include/libkern/i386/_OSByteOrder.h
main.o: /usr/include/sys/_types/_blkcnt_t.h
main.o: /usr/include/sys/_types/_blksize_t.h
main.o: /usr/include/sys/_types/_in_addr_t.h
main.o: /usr/include/sys/_types/_in_port_t.h /usr/include/sys/_types/_ino_t.h
main.o: /usr/include/sys/_types/_ino64_t.h /usr/include/sys/_types/_key_t.h
main.o: /usr/include/sys/_types/_nlink_t.h /usr/include/sys/_types/_id_t.h
main.o: /usr/include/sys/_types/_clock_t.h /usr/include/sys/_types/_rsize_t.h
main.o: /usr/include/sys/_types/_errno_t.h
main.o: /usr/include/sys/_pthread/_pthread_attr_t.h
main.o: /usr/include/sys/_pthread/_pthread_cond_t.h
main.o: /usr/include/sys/_pthread/_pthread_condattr_t.h
main.o: /usr/include/sys/_pthread/_pthread_mutex_t.h
main.o: /usr/include/sys/_pthread/_pthread_mutexattr_t.h
main.o: /usr/include/sys/_pthread/_pthread_once_t.h
main.o: /usr/include/sys/_pthread/_pthread_rwlock_t.h
main.o: /usr/include/sys/_pthread/_pthread_rwlockattr_t.h
main.o: /usr/include/sys/_pthread/_pthread_t.h
main.o: /usr/include/sys/_pthread/_pthread_key_t.h
main.o: /usr/include/sys/_types/_fsblkcnt_t.h
main.o: /usr/include/sys/_types/_fsfilcnt_t.h /usr/include/sys/syslimits.h
main.o: /usr/include/machine/param.h /usr/include/i386/param.h
main.o: /usr/include/i386/_param.h /usr/include/limits.h
main.o: /usr/include/machine/limits.h /usr/include/i386/limits.h
main.o: /usr/include/i386/_limits.h /usr/include/sys/signal.h
main.o: /usr/include/machine/signal.h /usr/include/i386/signal.h
main.o: /usr/include/machine/_mcontext.h /usr/include/i386/_mcontext.h
main.o: /usr/include/mach/i386/_structs.h
main.o: /usr/include/sys/_types/_sigaltstack.h
main.o: /usr/include/sys/_types/_ucontext.h /usr/include/stdio.h
main.o: /usr/include/sys/_types/_va_list.h /usr/include/sys/stdio.h
main.o: /usr/include/secure/_stdio.h /usr/include/secure/_common.h
XBuilder.o: /usr/include/dirent.h /usr/include/_types.h
XBuilder.o: /usr/include/sys/_types.h /usr/include/sys/cdefs.h
XBuilder.o: /usr/include/sys/_symbol_aliasing.h
XBuilder.o: /usr/include/sys/_posix_availability.h
XBuilder.o: /usr/include/machine/_types.h /usr/include/i386/_types.h
XBuilder.o: /usr/include/sys/_pthread/_pthread_types.h
XBuilder.o: /usr/include/sys/dirent.h /usr/include/sys/_types/_ino_t.h
XBuilder.o: /usr/include/Availability.h /usr/include/AvailabilityInternal.h
XBuilder.o: XBuilder.h /usr/include/unistd.h /usr/include/sys/unistd.h
XBuilder.o: /usr/include/sys/_types/_posix_vdisable.h
XBuilder.o: /usr/include/sys/_types/_seek_set.h
XBuilder.o: /usr/include/sys/_types/_size_t.h /usr/include/_types/_uint64_t.h
XBuilder.o: /usr/include/sys/_types/_ssize_t.h
XBuilder.o: /usr/include/sys/_types/_uid_t.h /usr/include/sys/_types/_gid_t.h
XBuilder.o: /usr/include/sys/_types/_intptr_t.h
XBuilder.o: /usr/include/sys/_types/_off_t.h /usr/include/sys/_types/_pid_t.h
XBuilder.o: /usr/include/sys/_types/_useconds_t.h
XBuilder.o: /usr/include/sys/_types/_null.h /usr/include/sys/select.h
XBuilder.o: /usr/include/sys/appleapiopts.h /usr/include/sys/_types/_fd_def.h
XBuilder.o: /usr/include/sys/_types/_timespec.h
XBuilder.o: /usr/include/sys/_types/_timeval.h
XBuilder.o: /usr/include/sys/_types/_time_t.h
XBuilder.o: /usr/include/sys/_types/_suseconds_t.h
XBuilder.o: /usr/include/sys/_types/_sigset_t.h
XBuilder.o: /usr/include/sys/_types/_fd_setsize.h
XBuilder.o: /usr/include/sys/_types/_fd_set.h
XBuilder.o: /usr/include/sys/_types/_fd_clr.h
XBuilder.o: /usr/include/sys/_types/_fd_isset.h
XBuilder.o: /usr/include/sys/_types/_fd_zero.h
XBuilder.o: /usr/include/sys/_types/_fd_copy.h /usr/include/sys/_select.h
XBuilder.o: /usr/include/sys/_types/_dev_t.h
XBuilder.o: /usr/include/sys/_types/_mode_t.h
XBuilder.o: /usr/include/sys/_types/_uuid_t.h /usr/include/gethostuuid.h
XBuilder.o: /usr/include/sys/param.h /usr/include/sys/types.h
XBuilder.o: /usr/include/machine/types.h /usr/include/i386/types.h
XBuilder.o: /usr/include/sys/_types/_int8_t.h
XBuilder.o: /usr/include/sys/_types/_int16_t.h
XBuilder.o: /usr/include/sys/_types/_int32_t.h
XBuilder.o: /usr/include/sys/_types/_int64_t.h
XBuilder.o: /usr/include/sys/_types/_uintptr_t.h
XBuilder.o: /usr/include/machine/endian.h /usr/include/i386/endian.h
XBuilder.o: /usr/include/sys/_endian.h /usr/include/libkern/_OSByteOrder.h
XBuilder.o: /usr/include/libkern/i386/_OSByteOrder.h
XBuilder.o: /usr/include/sys/_types/_blkcnt_t.h
XBuilder.o: /usr/include/sys/_types/_blksize_t.h
XBuilder.o: /usr/include/sys/_types/_in_addr_t.h
XBuilder.o: /usr/include/sys/_types/_in_port_t.h
XBuilder.o: /usr/include/sys/_types/_ino64_t.h
XBuilder.o: /usr/include/sys/_types/_key_t.h
XBuilder.o: /usr/include/sys/_types/_nlink_t.h
XBuilder.o: /usr/include/sys/_types/_id_t.h
XBuilder.o: /usr/include/sys/_types/_clock_t.h
XBuilder.o: /usr/include/sys/_types/_rsize_t.h
XBuilder.o: /usr/include/sys/_types/_errno_t.h
XBuilder.o: /usr/include/sys/_pthread/_pthread_attr_t.h
XBuilder.o: /usr/include/sys/_pthread/_pthread_cond_t.h
XBuilder.o: /usr/include/sys/_pthread/_pthread_condattr_t.h
XBuilder.o: /usr/include/sys/_pthread/_pthread_mutex_t.h
XBuilder.o: /usr/include/sys/_pthread/_pthread_mutexattr_t.h
XBuilder.o: /usr/include/sys/_pthread/_pthread_once_t.h
XBuilder.o: /usr/include/sys/_pthread/_pthread_rwlock_t.h
XBuilder.o: /usr/include/sys/_pthread/_pthread_rwlockattr_t.h
XBuilder.o: /usr/include/sys/_pthread/_pthread_t.h
XBuilder.o: /usr/include/sys/_pthread/_pthread_key_t.h
XBuilder.o: /usr/include/sys/_types/_fsblkcnt_t.h
XBuilder.o: /usr/include/sys/_types/_fsfilcnt_t.h
XBuilder.o: /usr/include/sys/syslimits.h /usr/include/machine/param.h
XBuilder.o: /usr/include/i386/param.h /usr/include/i386/_param.h
XBuilder.o: /usr/include/limits.h /usr/include/machine/limits.h
XBuilder.o: /usr/include/i386/limits.h /usr/include/i386/_limits.h
XBuilder.o: /usr/include/sys/signal.h /usr/include/machine/signal.h
XBuilder.o: /usr/include/i386/signal.h /usr/include/machine/_mcontext.h
XBuilder.o: /usr/include/i386/_mcontext.h /usr/include/mach/i386/_structs.h
XBuilder.o: /usr/include/sys/_types/_sigaltstack.h
XBuilder.o: /usr/include/sys/_types/_ucontext.h /usr/include/stdio.h
XBuilder.o: /usr/include/sys/_types/_va_list.h /usr/include/sys/stdio.h
XBuilder.o: /usr/include/secure/_stdio.h /usr/include/secure/_common.h
sfm.o: XBuilder.h /usr/include/unistd.h /usr/include/_types.h
sfm.o: /usr/include/sys/_types.h /usr/include/sys/cdefs.h
sfm.o: /usr/include/sys/_symbol_aliasing.h
sfm.o: /usr/include/sys/_posix_availability.h /usr/include/machine/_types.h
sfm.o: /usr/include/i386/_types.h /usr/include/sys/_pthread/_pthread_types.h
sfm.o: /usr/include/sys/unistd.h /usr/include/sys/_types/_posix_vdisable.h
sfm.o: /usr/include/sys/_types/_seek_set.h /usr/include/sys/_types/_size_t.h
sfm.o: /usr/include/_types/_uint64_t.h /usr/include/Availability.h
sfm.o: /usr/include/AvailabilityInternal.h /usr/include/sys/_types/_ssize_t.h
sfm.o: /usr/include/sys/_types/_uid_t.h /usr/include/sys/_types/_gid_t.h
sfm.o: /usr/include/sys/_types/_intptr_t.h /usr/include/sys/_types/_off_t.h
sfm.o: /usr/include/sys/_types/_pid_t.h /usr/include/sys/_types/_useconds_t.h
sfm.o: /usr/include/sys/_types/_null.h /usr/include/sys/select.h
sfm.o: /usr/include/sys/appleapiopts.h /usr/include/sys/_types/_fd_def.h
sfm.o: /usr/include/sys/_types/_timespec.h /usr/include/sys/_types/_timeval.h
sfm.o: /usr/include/sys/_types/_time_t.h
sfm.o: /usr/include/sys/_types/_suseconds_t.h
sfm.o: /usr/include/sys/_types/_sigset_t.h
sfm.o: /usr/include/sys/_types/_fd_setsize.h
sfm.o: /usr/include/sys/_types/_fd_set.h /usr/include/sys/_types/_fd_clr.h
sfm.o: /usr/include/sys/_types/_fd_isset.h /usr/include/sys/_types/_fd_zero.h
sfm.o: /usr/include/sys/_types/_fd_copy.h /usr/include/sys/_select.h
sfm.o: /usr/include/sys/_types/_dev_t.h /usr/include/sys/_types/_mode_t.h
sfm.o: /usr/include/sys/_types/_uuid_t.h /usr/include/gethostuuid.h
sfm.o: /usr/include/sys/param.h /usr/include/sys/types.h
sfm.o: /usr/include/machine/types.h /usr/include/i386/types.h
sfm.o: /usr/include/sys/_types/_int8_t.h /usr/include/sys/_types/_int16_t.h
sfm.o: /usr/include/sys/_types/_int32_t.h /usr/include/sys/_types/_int64_t.h
sfm.o: /usr/include/sys/_types/_uintptr_t.h /usr/include/machine/endian.h
sfm.o: /usr/include/i386/endian.h /usr/include/sys/_endian.h
sfm.o: /usr/include/libkern/_OSByteOrder.h
sfm.o: /usr/include/libkern/i386/_OSByteOrder.h
sfm.o: /usr/include/sys/_types/_blkcnt_t.h
sfm.o: /usr/include/sys/_types/_blksize_t.h
sfm.o: /usr/include/sys/_types/_in_addr_t.h
sfm.o: /usr/include/sys/_types/_in_port_t.h /usr/include/sys/_types/_ino_t.h
sfm.o: /usr/include/sys/_types/_ino64_t.h /usr/include/sys/_types/_key_t.h
sfm.o: /usr/include/sys/_types/_nlink_t.h /usr/include/sys/_types/_id_t.h
sfm.o: /usr/include/sys/_types/_clock_t.h /usr/include/sys/_types/_rsize_t.h
sfm.o: /usr/include/sys/_types/_errno_t.h
sfm.o: /usr/include/sys/_pthread/_pthread_attr_t.h
sfm.o: /usr/include/sys/_pthread/_pthread_cond_t.h
sfm.o: /usr/include/sys/_pthread/_pthread_condattr_t.h
sfm.o: /usr/include/sys/_pthread/_pthread_mutex_t.h
sfm.o: /usr/include/sys/_pthread/_pthread_mutexattr_t.h
sfm.o: /usr/include/sys/_pthread/_pthread_once_t.h
sfm.o: /usr/include/sys/_pthread/_pthread_rwlock_t.h
sfm.o: /usr/include/sys/_pthread/_pthread_rwlockattr_t.h
sfm.o: /usr/include/sys/_pthread/_pthread_t.h
sfm.o: /usr/include/sys/_pthread/_pthread_key_t.h
sfm.o: /usr/include/sys/_types/_fsblkcnt_t.h
sfm.o: /usr/include/sys/_types/_fsfilcnt_t.h /usr/include/sys/syslimits.h
sfm.o: /usr/include/machine/param.h /usr/include/i386/param.h
sfm.o: /usr/include/i386/_param.h /usr/include/limits.h
sfm.o: /usr/include/machine/limits.h /usr/include/i386/limits.h
sfm.o: /usr/include/i386/_limits.h /usr/include/sys/signal.h
sfm.o: /usr/include/machine/signal.h /usr/include/i386/signal.h
sfm.o: /usr/include/machine/_mcontext.h /usr/include/i386/_mcontext.h
sfm.o: /usr/include/mach/i386/_structs.h
sfm.o: /usr/include/sys/_types/_sigaltstack.h
sfm.o: /usr/include/sys/_types/_ucontext.h /usr/include/stdio.h
sfm.o: /usr/include/sys/_types/_va_list.h /usr/include/sys/stdio.h
sfm.o: /usr/include/secure/_stdio.h /usr/include/secure/_common.h
fmatrix_matching.o: XBuilder.h /usr/include/unistd.h /usr/include/_types.h
fmatrix_matching.o: /usr/include/sys/_types.h /usr/include/sys/cdefs.h
fmatrix_matching.o: /usr/include/sys/_symbol_aliasing.h
fmatrix_matching.o: /usr/include/sys/_posix_availability.h
fmatrix_matching.o: /usr/include/machine/_types.h /usr/include/i386/_types.h
fmatrix_matching.o: /usr/include/sys/_pthread/_pthread_types.h
fmatrix_matching.o: /usr/include/sys/unistd.h
fmatrix_matching.o: /usr/include/sys/_types/_posix_vdisable.h
fmatrix_matching.o: /usr/include/sys/_types/_seek_set.h
fmatrix_matching.o: /usr/include/sys/_types/_size_t.h
fmatrix_matching.o: /usr/include/_types/_uint64_t.h
fmatrix_matching.o: /usr/include/Availability.h
fmatrix_matching.o: /usr/include/AvailabilityInternal.h
fmatrix_matching.o: /usr/include/sys/_types/_ssize_t.h
fmatrix_matching.o: /usr/include/sys/_types/_uid_t.h
fmatrix_matching.o: /usr/include/sys/_types/_gid_t.h
fmatrix_matching.o: /usr/include/sys/_types/_intptr_t.h
fmatrix_matching.o: /usr/include/sys/_types/_off_t.h
fmatrix_matching.o: /usr/include/sys/_types/_pid_t.h
fmatrix_matching.o: /usr/include/sys/_types/_useconds_t.h
fmatrix_matching.o: /usr/include/sys/_types/_null.h /usr/include/sys/select.h
fmatrix_matching.o: /usr/include/sys/appleapiopts.h
fmatrix_matching.o: /usr/include/sys/_types/_fd_def.h
fmatrix_matching.o: /usr/include/sys/_types/_timespec.h
fmatrix_matching.o: /usr/include/sys/_types/_timeval.h
fmatrix_matching.o: /usr/include/sys/_types/_time_t.h
fmatrix_matching.o: /usr/include/sys/_types/_suseconds_t.h
fmatrix_matching.o: /usr/include/sys/_types/_sigset_t.h
fmatrix_matching.o: /usr/include/sys/_types/_fd_setsize.h
fmatrix_matching.o: /usr/include/sys/_types/_fd_set.h
fmatrix_matching.o: /usr/include/sys/_types/_fd_clr.h
fmatrix_matching.o: /usr/include/sys/_types/_fd_isset.h
fmatrix_matching.o: /usr/include/sys/_types/_fd_zero.h
fmatrix_matching.o: /usr/include/sys/_types/_fd_copy.h
fmatrix_matching.o: /usr/include/sys/_select.h
fmatrix_matching.o: /usr/include/sys/_types/_dev_t.h
fmatrix_matching.o: /usr/include/sys/_types/_mode_t.h
fmatrix_matching.o: /usr/include/sys/_types/_uuid_t.h
fmatrix_matching.o: /usr/include/gethostuuid.h /usr/include/sys/param.h
fmatrix_matching.o: /usr/include/sys/types.h /usr/include/machine/types.h
fmatrix_matching.o: /usr/include/i386/types.h
fmatrix_matching.o: /usr/include/sys/_types/_int8_t.h
fmatrix_matching.o: /usr/include/sys/_types/_int16_t.h
fmatrix_matching.o: /usr/include/sys/_types/_int32_t.h
fmatrix_matching.o: /usr/include/sys/_types/_int64_t.h
fmatrix_matching.o: /usr/include/sys/_types/_uintptr_t.h
fmatrix_matching.o: /usr/include/machine/endian.h /usr/include/i386/endian.h
fmatrix_matching.o: /usr/include/sys/_endian.h
fmatrix_matching.o: /usr/include/libkern/_OSByteOrder.h
fmatrix_matching.o: /usr/include/libkern/i386/_OSByteOrder.h
fmatrix_matching.o: /usr/include/sys/_types/_blkcnt_t.h
fmatrix_matching.o: /usr/include/sys/_types/_blksize_t.h
fmatrix_matching.o: /usr/include/sys/_types/_in_addr_t.h
fmatrix_matching.o: /usr/include/sys/_types/_in_port_t.h
fmatrix_matching.o: /usr/include/sys/_types/_ino_t.h
fmatrix_matching.o: /usr/include/sys/_types/_ino64_t.h
fmatrix_matching.o: /usr/include/sys/_types/_key_t.h
fmatrix_matching.o: /usr/include/sys/_types/_nlink_t.h
fmatrix_matching.o: /usr/include/sys/_types/_id_t.h
fmatrix_matching.o: /usr/include/sys/_types/_clock_t.h
fmatrix_matching.o: /usr/include/sys/_types/_rsize_t.h
fmatrix_matching.o: /usr/include/sys/_types/_errno_t.h
fmatrix_matching.o: /usr/include/sys/_pthread/_pthread_attr_t.h
fmatrix_matching.o: /usr/include/sys/_pthread/_pthread_cond_t.h
fmatrix_matching.o: /usr/include/sys/_pthread/_pthread_condattr_t.h
fmatrix_matching.o: /usr/include/sys/_pthread/_pthread_mutex_t.h
fmatrix_matching.o: /usr/include/sys/_pthread/_pthread_mutexattr_t.h
fmatrix_matching.o: /usr/include/sys/_pthread/_pthread_once_t.h
fmatrix_matching.o: /usr/include/sys/_pthread/_pthread_rwlock_t.h
fmatrix_matching.o: /usr/include/sys/_pthread/_pthread_rwlockattr_t.h
fmatrix_matching.o: /usr/include/sys/_pthread/_pthread_t.h
fmatrix_matching.o: /usr/include/sys/_pthread/_pthread_key_t.h
fmatrix_matching.o: /usr/include/sys/_types/_fsblkcnt_t.h
fmatrix_matching.o: /usr/include/sys/_types/_fsfilcnt_t.h
fmatrix_matching.o: /usr/include/sys/syslimits.h /usr/include/machine/param.h
fmatrix_matching.o: /usr/include/i386/param.h /usr/include/i386/_param.h
fmatrix_matching.o: /usr/include/limits.h /usr/include/machine/limits.h
fmatrix_matching.o: /usr/include/i386/limits.h /usr/include/i386/_limits.h
fmatrix_matching.o: /usr/include/sys/signal.h /usr/include/machine/signal.h
fmatrix_matching.o: /usr/include/i386/signal.h
fmatrix_matching.o: /usr/include/machine/_mcontext.h
fmatrix_matching.o: /usr/include/i386/_mcontext.h
fmatrix_matching.o: /usr/include/mach/i386/_structs.h
fmatrix_matching.o: /usr/include/sys/_types/_sigaltstack.h
fmatrix_matching.o: /usr/include/sys/_types/_ucontext.h /usr/include/stdio.h
fmatrix_matching.o: /usr/include/sys/_types/_va_list.h
fmatrix_matching.o: /usr/include/sys/stdio.h /usr/include/secure/_stdio.h
fmatrix_matching.o: /usr/include/secure/_common.h
triangulation.o: XBuilder.h /usr/include/unistd.h /usr/include/_types.h
triangulation.o: /usr/include/sys/_types.h /usr/include/sys/cdefs.h
triangulation.o: /usr/include/sys/_symbol_aliasing.h
triangulation.o: /usr/include/sys/_posix_availability.h
triangulation.o: /usr/include/machine/_types.h /usr/include/i386/_types.h
triangulation.o: /usr/include/sys/_pthread/_pthread_types.h
triangulation.o: /usr/include/sys/unistd.h
triangulation.o: /usr/include/sys/_types/_posix_vdisable.h
triangulation.o: /usr/include/sys/_types/_seek_set.h
triangulation.o: /usr/include/sys/_types/_size_t.h
triangulation.o: /usr/include/_types/_uint64_t.h /usr/include/Availability.h
triangulation.o: /usr/include/AvailabilityInternal.h
triangulation.o: /usr/include/sys/_types/_ssize_t.h
triangulation.o: /usr/include/sys/_types/_uid_t.h
triangulation.o: /usr/include/sys/_types/_gid_t.h
triangulation.o: /usr/include/sys/_types/_intptr_t.h
triangulation.o: /usr/include/sys/_types/_off_t.h
triangulation.o: /usr/include/sys/_types/_pid_t.h
triangulation.o: /usr/include/sys/_types/_useconds_t.h
triangulation.o: /usr/include/sys/_types/_null.h /usr/include/sys/select.h
triangulation.o: /usr/include/sys/appleapiopts.h
triangulation.o: /usr/include/sys/_types/_fd_def.h
triangulation.o: /usr/include/sys/_types/_timespec.h
triangulation.o: /usr/include/sys/_types/_timeval.h
triangulation.o: /usr/include/sys/_types/_time_t.h
triangulation.o: /usr/include/sys/_types/_suseconds_t.h
triangulation.o: /usr/include/sys/_types/_sigset_t.h
triangulation.o: /usr/include/sys/_types/_fd_setsize.h
triangulation.o: /usr/include/sys/_types/_fd_set.h
triangulation.o: /usr/include/sys/_types/_fd_clr.h
triangulation.o: /usr/include/sys/_types/_fd_isset.h
triangulation.o: /usr/include/sys/_types/_fd_zero.h
triangulation.o: /usr/include/sys/_types/_fd_copy.h
triangulation.o: /usr/include/sys/_select.h /usr/include/sys/_types/_dev_t.h
triangulation.o: /usr/include/sys/_types/_mode_t.h
triangulation.o: /usr/include/sys/_types/_uuid_t.h /usr/include/gethostuuid.h
triangulation.o: /usr/include/sys/param.h /usr/include/sys/types.h
triangulation.o: /usr/include/machine/types.h /usr/include/i386/types.h
triangulation.o: /usr/include/sys/_types/_int8_t.h
triangulation.o: /usr/include/sys/_types/_int16_t.h
triangulation.o: /usr/include/sys/_types/_int32_t.h
triangulation.o: /usr/include/sys/_types/_int64_t.h
triangulation.o: /usr/include/sys/_types/_uintptr_t.h
triangulation.o: /usr/include/machine/endian.h /usr/include/i386/endian.h
triangulation.o: /usr/include/sys/_endian.h
triangulation.o: /usr/include/libkern/_OSByteOrder.h
triangulation.o: /usr/include/libkern/i386/_OSByteOrder.h
triangulation.o: /usr/include/sys/_types/_blkcnt_t.h
triangulation.o: /usr/include/sys/_types/_blksize_t.h
triangulation.o: /usr/include/sys/_types/_in_addr_t.h
triangulation.o: /usr/include/sys/_types/_in_port_t.h
triangulation.o: /usr/include/sys/_types/_ino_t.h
triangulation.o: /usr/include/sys/_types/_ino64_t.h
triangulation.o: /usr/include/sys/_types/_key_t.h
triangulation.o: /usr/include/sys/_types/_nlink_t.h
triangulation.o: /usr/include/sys/_types/_id_t.h
triangulation.o: /usr/include/sys/_types/_clock_t.h
triangulation.o: /usr/include/sys/_types/_rsize_t.h
triangulation.o: /usr/include/sys/_types/_errno_t.h
triangulation.o: /usr/include/sys/_pthread/_pthread_attr_t.h
triangulation.o: /usr/include/sys/_pthread/_pthread_cond_t.h
triangulation.o: /usr/include/sys/_pthread/_pthread_condattr_t.h
triangulation.o: /usr/include/sys/_pthread/_pthread_mutex_t.h
triangulation.o: /usr/include/sys/_pthread/_pthread_mutexattr_t.h
triangulation.o: /usr/include/sys/_pthread/_pthread_once_t.h
triangulation.o: /usr/include/sys/_pthread/_pthread_rwlock_t.h
triangulation.o: /usr/include/sys/_pthread/_pthread_rwlockattr_t.h
triangulation.o: /usr/include/sys/_pthread/_pthread_t.h
triangulation.o: /usr/include/sys/_pthread/_pthread_key_t.h
triangulation.o: /usr/include/sys/_types/_fsblkcnt_t.h
triangulation.o: /usr/include/sys/_types/_fsfilcnt_t.h
triangulation.o: /usr/include/sys/syslimits.h /usr/include/machine/param.h
triangulation.o: /usr/include/i386/param.h /usr/include/i386/_param.h
triangulation.o: /usr/include/limits.h /usr/include/machine/limits.h
triangulation.o: /usr/include/i386/limits.h /usr/include/i386/_limits.h
triangulation.o: /usr/include/sys/signal.h /usr/include/machine/signal.h
triangulation.o: /usr/include/i386/signal.h /usr/include/machine/_mcontext.h
triangulation.o: /usr/include/i386/_mcontext.h
triangulation.o: /usr/include/mach/i386/_structs.h
triangulation.o: /usr/include/sys/_types/_sigaltstack.h
triangulation.o: /usr/include/sys/_types/_ucontext.h /usr/include/stdio.h
triangulation.o: /usr/include/sys/_types/_va_list.h /usr/include/sys/stdio.h
triangulation.o: /usr/include/secure/_stdio.h /usr/include/secure/_common.h
two_view_reconst.o: XBuilder.h /usr/include/unistd.h /usr/include/_types.h
two_view_reconst.o: /usr/include/sys/_types.h /usr/include/sys/cdefs.h
two_view_reconst.o: /usr/include/sys/_symbol_aliasing.h
two_view_reconst.o: /usr/include/sys/_posix_availability.h
two_view_reconst.o: /usr/include/machine/_types.h /usr/include/i386/_types.h
two_view_reconst.o: /usr/include/sys/_pthread/_pthread_types.h
two_view_reconst.o: /usr/include/sys/unistd.h
two_view_reconst.o: /usr/include/sys/_types/_posix_vdisable.h
two_view_reconst.o: /usr/include/sys/_types/_seek_set.h
two_view_reconst.o: /usr/include/sys/_types/_size_t.h
two_view_reconst.o: /usr/include/_types/_uint64_t.h
two_view_reconst.o: /usr/include/Availability.h
two_view_reconst.o: /usr/include/AvailabilityInternal.h
two_view_reconst.o: /usr/include/sys/_types/_ssize_t.h
two_view_reconst.o: /usr/include/sys/_types/_uid_t.h
two_view_reconst.o: /usr/include/sys/_types/_gid_t.h
two_view_reconst.o: /usr/include/sys/_types/_intptr_t.h
two_view_reconst.o: /usr/include/sys/_types/_off_t.h
two_view_reconst.o: /usr/include/sys/_types/_pid_t.h
two_view_reconst.o: /usr/include/sys/_types/_useconds_t.h
two_view_reconst.o: /usr/include/sys/_types/_null.h /usr/include/sys/select.h
two_view_reconst.o: /usr/include/sys/appleapiopts.h
two_view_reconst.o: /usr/include/sys/_types/_fd_def.h
two_view_reconst.o: /usr/include/sys/_types/_timespec.h
two_view_reconst.o: /usr/include/sys/_types/_timeval.h
two_view_reconst.o: /usr/include/sys/_types/_time_t.h
two_view_reconst.o: /usr/include/sys/_types/_suseconds_t.h
two_view_reconst.o: /usr/include/sys/_types/_sigset_t.h
two_view_reconst.o: /usr/include/sys/_types/_fd_setsize.h
two_view_reconst.o: /usr/include/sys/_types/_fd_set.h
two_view_reconst.o: /usr/include/sys/_types/_fd_clr.h
two_view_reconst.o: /usr/include/sys/_types/_fd_isset.h
two_view_reconst.o: /usr/include/sys/_types/_fd_zero.h
two_view_reconst.o: /usr/include/sys/_types/_fd_copy.h
two_view_reconst.o: /usr/include/sys/_select.h
two_view_reconst.o: /usr/include/sys/_types/_dev_t.h
two_view_reconst.o: /usr/include/sys/_types/_mode_t.h
two_view_reconst.o: /usr/include/sys/_types/_uuid_t.h
two_view_reconst.o: /usr/include/gethostuuid.h /usr/include/sys/param.h
two_view_reconst.o: /usr/include/sys/types.h /usr/include/machine/types.h
two_view_reconst.o: /usr/include/i386/types.h
two_view_reconst.o: /usr/include/sys/_types/_int8_t.h
two_view_reconst.o: /usr/include/sys/_types/_int16_t.h
two_view_reconst.o: /usr/include/sys/_types/_int32_t.h
two_view_reconst.o: /usr/include/sys/_types/_int64_t.h
two_view_reconst.o: /usr/include/sys/_types/_uintptr_t.h
two_view_reconst.o: /usr/include/machine/endian.h /usr/include/i386/endian.h
two_view_reconst.o: /usr/include/sys/_endian.h
two_view_reconst.o: /usr/include/libkern/_OSByteOrder.h
two_view_reconst.o: /usr/include/libkern/i386/_OSByteOrder.h
two_view_reconst.o: /usr/include/sys/_types/_blkcnt_t.h
two_view_reconst.o: /usr/include/sys/_types/_blksize_t.h
two_view_reconst.o: /usr/include/sys/_types/_in_addr_t.h
two_view_reconst.o: /usr/include/sys/_types/_in_port_t.h
two_view_reconst.o: /usr/include/sys/_types/_ino_t.h
two_view_reconst.o: /usr/include/sys/_types/_ino64_t.h
two_view_reconst.o: /usr/include/sys/_types/_key_t.h
two_view_reconst.o: /usr/include/sys/_types/_nlink_t.h
two_view_reconst.o: /usr/include/sys/_types/_id_t.h
two_view_reconst.o: /usr/include/sys/_types/_clock_t.h
two_view_reconst.o: /usr/include/sys/_types/_rsize_t.h
two_view_reconst.o: /usr/include/sys/_types/_errno_t.h
two_view_reconst.o: /usr/include/sys/_pthread/_pthread_attr_t.h
two_view_reconst.o: /usr/include/sys/_pthread/_pthread_cond_t.h
two_view_reconst.o: /usr/include/sys/_pthread/_pthread_condattr_t.h
two_view_reconst.o: /usr/include/sys/_pthread/_pthread_mutex_t.h
two_view_reconst.o: /usr/include/sys/_pthread/_pthread_mutexattr_t.h
two_view_reconst.o: /usr/include/sys/_pthread/_pthread_once_t.h
two_view_reconst.o: /usr/include/sys/_pthread/_pthread_rwlock_t.h
two_view_reconst.o: /usr/include/sys/_pthread/_pthread_rwlockattr_t.h
two_view_reconst.o: /usr/include/sys/_pthread/_pthread_t.h
two_view_reconst.o: /usr/include/sys/_pthread/_pthread_key_t.h
two_view_reconst.o: /usr/include/sys/_types/_fsblkcnt_t.h
two_view_reconst.o: /usr/include/sys/_types/_fsfilcnt_t.h
two_view_reconst.o: /usr/include/sys/syslimits.h /usr/include/machine/param.h
two_view_reconst.o: /usr/include/i386/param.h /usr/include/i386/_param.h
two_view_reconst.o: /usr/include/limits.h /usr/include/machine/limits.h
two_view_reconst.o: /usr/include/i386/limits.h /usr/include/i386/_limits.h
two_view_reconst.o: /usr/include/sys/signal.h /usr/include/machine/signal.h
two_view_reconst.o: /usr/include/i386/signal.h
two_view_reconst.o: /usr/include/machine/_mcontext.h
two_view_reconst.o: /usr/include/i386/_mcontext.h
two_view_reconst.o: /usr/include/mach/i386/_structs.h
two_view_reconst.o: /usr/include/sys/_types/_sigaltstack.h
two_view_reconst.o: /usr/include/sys/_types/_ucontext.h /usr/include/stdio.h
two_view_reconst.o: /usr/include/sys/_types/_va_list.h
two_view_reconst.o: /usr/include/sys/stdio.h /usr/include/secure/_stdio.h
two_view_reconst.o: /usr/include/secure/_common.h
bundle-adjust.o: XBuilder.h /usr/include/unistd.h /usr/include/_types.h
bundle-adjust.o: /usr/include/sys/_types.h /usr/include/sys/cdefs.h
bundle-adjust.o: /usr/include/sys/_symbol_aliasing.h
bundle-adjust.o: /usr/include/sys/_posix_availability.h
bundle-adjust.o: /usr/include/machine/_types.h /usr/include/i386/_types.h
bundle-adjust.o: /usr/include/sys/_pthread/_pthread_types.h
bundle-adjust.o: /usr/include/sys/unistd.h
bundle-adjust.o: /usr/include/sys/_types/_posix_vdisable.h
bundle-adjust.o: /usr/include/sys/_types/_seek_set.h
bundle-adjust.o: /usr/include/sys/_types/_size_t.h
bundle-adjust.o: /usr/include/_types/_uint64_t.h /usr/include/Availability.h
bundle-adjust.o: /usr/include/AvailabilityInternal.h
bundle-adjust.o: /usr/include/sys/_types/_ssize_t.h
bundle-adjust.o: /usr/include/sys/_types/_uid_t.h
bundle-adjust.o: /usr/include/sys/_types/_gid_t.h
bundle-adjust.o: /usr/include/sys/_types/_intptr_t.h
bundle-adjust.o: /usr/include/sys/_types/_off_t.h
bundle-adjust.o: /usr/include/sys/_types/_pid_t.h
bundle-adjust.o: /usr/include/sys/_types/_useconds_t.h
bundle-adjust.o: /usr/include/sys/_types/_null.h /usr/include/sys/select.h
bundle-adjust.o: /usr/include/sys/appleapiopts.h
bundle-adjust.o: /usr/include/sys/_types/_fd_def.h
bundle-adjust.o: /usr/include/sys/_types/_timespec.h
bundle-adjust.o: /usr/include/sys/_types/_timeval.h
bundle-adjust.o: /usr/include/sys/_types/_time_t.h
bundle-adjust.o: /usr/include/sys/_types/_suseconds_t.h
bundle-adjust.o: /usr/include/sys/_types/_sigset_t.h
bundle-adjust.o: /usr/include/sys/_types/_fd_setsize.h
bundle-adjust.o: /usr/include/sys/_types/_fd_set.h
bundle-adjust.o: /usr/include/sys/_types/_fd_clr.h
bundle-adjust.o: /usr/include/sys/_types/_fd_isset.h
bundle-adjust.o: /usr/include/sys/_types/_fd_zero.h
bundle-adjust.o: /usr/include/sys/_types/_fd_copy.h
bundle-adjust.o: /usr/include/sys/_select.h /usr/include/sys/_types/_dev_t.h
bundle-adjust.o: /usr/include/sys/_types/_mode_t.h
bundle-adjust.o: /usr/include/sys/_types/_uuid_t.h /usr/include/gethostuuid.h
bundle-adjust.o: /usr/include/sys/param.h /usr/include/sys/types.h
bundle-adjust.o: /usr/include/machine/types.h /usr/include/i386/types.h
bundle-adjust.o: /usr/include/sys/_types/_int8_t.h
bundle-adjust.o: /usr/include/sys/_types/_int16_t.h
bundle-adjust.o: /usr/include/sys/_types/_int32_t.h
bundle-adjust.o: /usr/include/sys/_types/_int64_t.h
bundle-adjust.o: /usr/include/sys/_types/_uintptr_t.h
bundle-adjust.o: /usr/include/machine/endian.h /usr/include/i386/endian.h
bundle-adjust.o: /usr/include/sys/_endian.h
bundle-adjust.o: /usr/include/libkern/_OSByteOrder.h
bundle-adjust.o: /usr/include/libkern/i386/_OSByteOrder.h
bundle-adjust.o: /usr/include/sys/_types/_blkcnt_t.h
bundle-adjust.o: /usr/include/sys/_types/_blksize_t.h
bundle-adjust.o: /usr/include/sys/_types/_in_addr_t.h
bundle-adjust.o: /usr/include/sys/_types/_in_port_t.h
bundle-adjust.o: /usr/include/sys/_types/_ino_t.h
bundle-adjust.o: /usr/include/sys/_types/_ino64_t.h
bundle-adjust.o: /usr/include/sys/_types/_key_t.h
bundle-adjust.o: /usr/include/sys/_types/_nlink_t.h
bundle-adjust.o: /usr/include/sys/_types/_id_t.h
bundle-adjust.o: /usr/include/sys/_types/_clock_t.h
bundle-adjust.o: /usr/include/sys/_types/_rsize_t.h
bundle-adjust.o: /usr/include/sys/_types/_errno_t.h
bundle-adjust.o: /usr/include/sys/_pthread/_pthread_attr_t.h
bundle-adjust.o: /usr/include/sys/_pthread/_pthread_cond_t.h
bundle-adjust.o: /usr/include/sys/_pthread/_pthread_condattr_t.h
bundle-adjust.o: /usr/include/sys/_pthread/_pthread_mutex_t.h
bundle-adjust.o: /usr/include/sys/_pthread/_pthread_mutexattr_t.h
bundle-adjust.o: /usr/include/sys/_pthread/_pthread_once_t.h
bundle-adjust.o: /usr/include/sys/_pthread/_pthread_rwlock_t.h
bundle-adjust.o: /usr/include/sys/_pthread/_pthread_rwlockattr_t.h
bundle-adjust.o: /usr/include/sys/_pthread/_pthread_t.h
bundle-adjust.o: /usr/include/sys/_pthread/_pthread_key_t.h
bundle-adjust.o: /usr/include/sys/_types/_fsblkcnt_t.h
bundle-adjust.o: /usr/include/sys/_types/_fsfilcnt_t.h
bundle-adjust.o: /usr/include/sys/syslimits.h /usr/include/machine/param.h
bundle-adjust.o: /usr/include/i386/param.h /usr/include/i386/_param.h
bundle-adjust.o: /usr/include/limits.h /usr/include/machine/limits.h
bundle-adjust.o: /usr/include/i386/limits.h /usr/include/i386/_limits.h
bundle-adjust.o: /usr/include/sys/signal.h /usr/include/machine/signal.h
bundle-adjust.o: /usr/include/i386/signal.h /usr/include/machine/_mcontext.h
bundle-adjust.o: /usr/include/i386/_mcontext.h
bundle-adjust.o: /usr/include/mach/i386/_structs.h
bundle-adjust.o: /usr/include/sys/_types/_sigaltstack.h
bundle-adjust.o: /usr/include/sys/_types/_ucontext.h /usr/include/stdio.h
bundle-adjust.o: /usr/include/sys/_types/_va_list.h /usr/include/sys/stdio.h
bundle-adjust.o: /usr/include/secure/_stdio.h /usr/include/secure/_common.h
ceres-ba.o: ceres-ba.h /usr/include/stdio.h /usr/include/sys/cdefs.h
ceres-ba.o: /usr/include/sys/_symbol_aliasing.h
ceres-ba.o: /usr/include/sys/_posix_availability.h
ceres-ba.o: /usr/include/Availability.h /usr/include/AvailabilityInternal.h
ceres-ba.o: /usr/include/_types.h /usr/include/sys/_types.h
ceres-ba.o: /usr/include/machine/_types.h /usr/include/i386/_types.h
ceres-ba.o: /usr/include/sys/_pthread/_pthread_types.h
ceres-ba.o: /usr/include/sys/_types/_va_list.h
ceres-ba.o: /usr/include/sys/_types/_size_t.h /usr/include/sys/_types/_null.h
ceres-ba.o: /usr/include/sys/stdio.h /usr/include/sys/_types/_off_t.h
ceres-ba.o: /usr/include/sys/_types/_ssize_t.h /usr/include/secure/_stdio.h
ceres-ba.o: /usr/include/secure/_common.h
