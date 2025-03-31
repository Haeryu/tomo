pub const __builtin_bswap16 = @import("std").zig.c_builtins.__builtin_bswap16;
pub const __builtin_bswap32 = @import("std").zig.c_builtins.__builtin_bswap32;
pub const __builtin_bswap64 = @import("std").zig.c_builtins.__builtin_bswap64;
pub const __builtin_signbit = @import("std").zig.c_builtins.__builtin_signbit;
pub const __builtin_signbitf = @import("std").zig.c_builtins.__builtin_signbitf;
pub const __builtin_popcount = @import("std").zig.c_builtins.__builtin_popcount;
pub const __builtin_ctz = @import("std").zig.c_builtins.__builtin_ctz;
pub const __builtin_clz = @import("std").zig.c_builtins.__builtin_clz;
pub const __builtin_sqrt = @import("std").zig.c_builtins.__builtin_sqrt;
pub const __builtin_sqrtf = @import("std").zig.c_builtins.__builtin_sqrtf;
pub const __builtin_sin = @import("std").zig.c_builtins.__builtin_sin;
pub const __builtin_sinf = @import("std").zig.c_builtins.__builtin_sinf;
pub const __builtin_cos = @import("std").zig.c_builtins.__builtin_cos;
pub const __builtin_cosf = @import("std").zig.c_builtins.__builtin_cosf;
pub const __builtin_exp = @import("std").zig.c_builtins.__builtin_exp;
pub const __builtin_expf = @import("std").zig.c_builtins.__builtin_expf;
pub const __builtin_exp2 = @import("std").zig.c_builtins.__builtin_exp2;
pub const __builtin_exp2f = @import("std").zig.c_builtins.__builtin_exp2f;
pub const __builtin_log = @import("std").zig.c_builtins.__builtin_log;
pub const __builtin_logf = @import("std").zig.c_builtins.__builtin_logf;
pub const __builtin_log2 = @import("std").zig.c_builtins.__builtin_log2;
pub const __builtin_log2f = @import("std").zig.c_builtins.__builtin_log2f;
pub const __builtin_log10 = @import("std").zig.c_builtins.__builtin_log10;
pub const __builtin_log10f = @import("std").zig.c_builtins.__builtin_log10f;
pub const __builtin_abs = @import("std").zig.c_builtins.__builtin_abs;
pub const __builtin_labs = @import("std").zig.c_builtins.__builtin_labs;
pub const __builtin_llabs = @import("std").zig.c_builtins.__builtin_llabs;
pub const __builtin_fabs = @import("std").zig.c_builtins.__builtin_fabs;
pub const __builtin_fabsf = @import("std").zig.c_builtins.__builtin_fabsf;
pub const __builtin_floor = @import("std").zig.c_builtins.__builtin_floor;
pub const __builtin_floorf = @import("std").zig.c_builtins.__builtin_floorf;
pub const __builtin_ceil = @import("std").zig.c_builtins.__builtin_ceil;
pub const __builtin_ceilf = @import("std").zig.c_builtins.__builtin_ceilf;
pub const __builtin_trunc = @import("std").zig.c_builtins.__builtin_trunc;
pub const __builtin_truncf = @import("std").zig.c_builtins.__builtin_truncf;
pub const __builtin_round = @import("std").zig.c_builtins.__builtin_round;
pub const __builtin_roundf = @import("std").zig.c_builtins.__builtin_roundf;
pub const __builtin_strlen = @import("std").zig.c_builtins.__builtin_strlen;
pub const __builtin_strcmp = @import("std").zig.c_builtins.__builtin_strcmp;
pub const __builtin_object_size = @import("std").zig.c_builtins.__builtin_object_size;
pub const __builtin___memset_chk = @import("std").zig.c_builtins.__builtin___memset_chk;
pub const __builtin_memset = @import("std").zig.c_builtins.__builtin_memset;
pub const __builtin___memcpy_chk = @import("std").zig.c_builtins.__builtin___memcpy_chk;
pub const __builtin_memcpy = @import("std").zig.c_builtins.__builtin_memcpy;
pub const __builtin_expect = @import("std").zig.c_builtins.__builtin_expect;
pub const __builtin_nanf = @import("std").zig.c_builtins.__builtin_nanf;
pub const __builtin_huge_valf = @import("std").zig.c_builtins.__builtin_huge_valf;
pub const __builtin_inff = @import("std").zig.c_builtins.__builtin_inff;
pub const __builtin_isnan = @import("std").zig.c_builtins.__builtin_isnan;
pub const __builtin_isinf = @import("std").zig.c_builtins.__builtin_isinf;
pub const __builtin_isinf_sign = @import("std").zig.c_builtins.__builtin_isinf_sign;
pub const __has_builtin = @import("std").zig.c_builtins.__has_builtin;
pub const __builtin_assume = @import("std").zig.c_builtins.__builtin_assume;
pub const __builtin_unreachable = @import("std").zig.c_builtins.__builtin_unreachable;
pub const __builtin_constant_p = @import("std").zig.c_builtins.__builtin_constant_p;
pub const __builtin_mul_overflow = @import("std").zig.c_builtins.__builtin_mul_overflow;
pub const __builtin_va_list = [*c]u8;
pub const __gnuc_va_list = __builtin_va_list;
pub const va_list = __gnuc_va_list;
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:604:3: warning: TODO implement translation of stmt class GCCAsmStmtClass

// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:599:36: warning: unable to translate function, demoted to extern
pub extern fn __debugbreak() void;
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:626:3: warning: TODO implement translation of stmt class GCCAsmStmtClass

// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:620:60: warning: unable to translate function, demoted to extern
pub extern fn __fastfail(arg_code: c_uint) noreturn;
pub extern fn __mingw_get_crt_info() [*c]const u8;
pub const rsize_t = usize;
pub const ptrdiff_t = c_longlong;
pub const wchar_t = c_ushort;
pub const wint_t = c_ushort;
pub const wctype_t = c_ushort;
pub const errno_t = c_int;
pub const __time32_t = c_long;
pub const __time64_t = c_longlong;
pub const time_t = __time64_t;
pub const struct_threadlocaleinfostruct = extern struct {
    _locale_pctype: [*c]const c_ushort = @import("std").mem.zeroes([*c]const c_ushort),
    _locale_mb_cur_max: c_int = @import("std").mem.zeroes(c_int),
    _locale_lc_codepage: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const struct_threadmbcinfostruct = opaque {};
pub const pthreadlocinfo = [*c]struct_threadlocaleinfostruct;
pub const pthreadmbcinfo = ?*struct_threadmbcinfostruct;
pub const struct___lc_time_data = opaque {};
pub const struct_localeinfo_struct = extern struct {
    locinfo: pthreadlocinfo = @import("std").mem.zeroes(pthreadlocinfo),
    mbcinfo: pthreadmbcinfo = @import("std").mem.zeroes(pthreadmbcinfo),
};
pub const _locale_tstruct = struct_localeinfo_struct;
pub const _locale_t = [*c]struct_localeinfo_struct;
pub const struct_tagLC_ID = extern struct {
    wLanguage: c_ushort = @import("std").mem.zeroes(c_ushort),
    wCountry: c_ushort = @import("std").mem.zeroes(c_ushort),
    wCodePage: c_ushort = @import("std").mem.zeroes(c_ushort),
};
pub const LC_ID = struct_tagLC_ID;
pub const LPLC_ID = [*c]struct_tagLC_ID;
pub const threadlocinfo = struct_threadlocaleinfostruct;
pub extern fn _wdupenv_s(_Buffer: [*c][*c]wchar_t, _BufferSizeInWords: [*c]usize, _VarName: [*c]const wchar_t) errno_t;
pub extern fn _itow_s(_Val: c_int, _DstBuf: [*c]wchar_t, _SizeInWords: usize, _Radix: c_int) errno_t;
pub extern fn _ltow_s(_Val: c_long, _DstBuf: [*c]wchar_t, _SizeInWords: usize, _Radix: c_int) errno_t;
pub extern fn _ultow_s(_Val: c_ulong, _DstBuf: [*c]wchar_t, _SizeInWords: usize, _Radix: c_int) errno_t;
pub extern fn _wgetenv_s(_ReturnSize: [*c]usize, _DstBuf: [*c]wchar_t, _DstSizeInWords: usize, _VarName: [*c]const wchar_t) errno_t;
pub extern fn _i64tow_s(_Val: c_longlong, _DstBuf: [*c]wchar_t, _SizeInWords: usize, _Radix: c_int) errno_t;
pub extern fn _ui64tow_s(_Val: c_ulonglong, _DstBuf: [*c]wchar_t, _SizeInWords: usize, _Radix: c_int) errno_t;
pub extern fn _wmakepath_s(_PathResult: [*c]wchar_t, _SizeInWords: usize, _Drive: [*c]const wchar_t, _Dir: [*c]const wchar_t, _Filename: [*c]const wchar_t, _Ext: [*c]const wchar_t) errno_t;
pub extern fn _wputenv_s(_Name: [*c]const wchar_t, _Value: [*c]const wchar_t) errno_t;
pub extern fn _wsearchenv_s(_Filename: [*c]const wchar_t, _EnvVar: [*c]const wchar_t, _ResultPath: [*c]wchar_t, _SizeInWords: usize) errno_t;
pub extern fn _wsplitpath_s(_FullPath: [*c]const wchar_t, _Drive: [*c]wchar_t, _DriveSizeInWords: usize, _Dir: [*c]wchar_t, _DirSizeInWords: usize, _Filename: [*c]wchar_t, _FilenameSizeInWords: usize, _Ext: [*c]wchar_t, _ExtSizeInWords: usize) errno_t;
pub const _onexit_t = ?*const fn () callconv(.c) c_int;
pub const struct__div_t = extern struct {
    quot: c_int = @import("std").mem.zeroes(c_int),
    rem: c_int = @import("std").mem.zeroes(c_int),
};
pub const div_t = struct__div_t;
pub const struct__ldiv_t = extern struct {
    quot: c_long = @import("std").mem.zeroes(c_long),
    rem: c_long = @import("std").mem.zeroes(c_long),
};
pub const ldiv_t = struct__ldiv_t;
pub const _LDOUBLE = extern struct {
    ld: [10]u8 = @import("std").mem.zeroes([10]u8),
};
pub const _CRT_DOUBLE = extern struct {
    x: f64 = @import("std").mem.zeroes(f64),
};
pub const _CRT_FLOAT = extern struct {
    f: f32 = @import("std").mem.zeroes(f32),
};
pub const _LONGDOUBLE = extern struct {
    x: c_longdouble = @import("std").mem.zeroes(c_longdouble),
};
pub const _LDBL12 = extern struct {
    ld12: [12]u8 = @import("std").mem.zeroes([12]u8),
};
pub extern fn ___mb_cur_max_func() c_int;
pub const _purecall_handler = ?*const fn () callconv(.c) void;
pub extern fn _set_purecall_handler(_Handler: _purecall_handler) _purecall_handler;
pub extern fn _get_purecall_handler() _purecall_handler;
pub const _invalid_parameter_handler = ?*const fn ([*c]const wchar_t, [*c]const wchar_t, [*c]const wchar_t, c_uint, usize) callconv(.c) void;
pub extern fn _set_invalid_parameter_handler(_Handler: _invalid_parameter_handler) _invalid_parameter_handler;
pub extern fn _get_invalid_parameter_handler() _invalid_parameter_handler;
pub extern fn _errno() [*c]c_int;
pub extern fn _set_errno(_Value: c_int) errno_t;
pub extern fn _get_errno(_Value: [*c]c_int) errno_t;
pub extern fn __doserrno() [*c]c_ulong;
pub extern fn _set_doserrno(_Value: c_ulong) errno_t;
pub extern fn _get_doserrno(_Value: [*c]c_ulong) errno_t;
pub extern fn __sys_errlist() [*c][*c]u8;
pub extern fn __sys_nerr() [*c]c_int;
pub extern fn __p___argv() [*c][*c][*c]u8;
pub extern fn __p__fmode() [*c]c_int;
pub extern fn __p___argc() [*c]c_int;
pub extern fn __p___wargv() [*c][*c][*c]wchar_t;
pub extern fn __p__pgmptr() [*c][*c]u8;
pub extern fn __p__wpgmptr() [*c][*c]wchar_t;
pub extern fn _get_pgmptr(_Value: [*c][*c]u8) errno_t;
pub extern fn _get_wpgmptr(_Value: [*c][*c]wchar_t) errno_t;
pub extern fn _set_fmode(_Mode: c_int) errno_t;
pub extern fn _get_fmode(_PMode: [*c]c_int) errno_t;
pub extern fn __p__environ() [*c][*c][*c]u8;
pub extern fn __p__wenviron() [*c][*c][*c]wchar_t;
pub extern fn __p__osplatform() [*c]c_uint;
pub extern fn __p__osver() [*c]c_uint;
pub extern fn __p__winver() [*c]c_uint;
pub extern fn __p__winmajor() [*c]c_uint;
pub extern fn __p__winminor() [*c]c_uint;
pub extern fn _get_osplatform(_Value: [*c]c_uint) errno_t;
pub extern fn _get_osver(_Value: [*c]c_uint) errno_t;
pub extern fn _get_winver(_Value: [*c]c_uint) errno_t;
pub extern fn _get_winmajor(_Value: [*c]c_uint) errno_t;
pub extern fn _get_winminor(_Value: [*c]c_uint) errno_t;
pub extern fn exit(_Code: c_int) noreturn;
pub extern fn _exit(_Code: c_int) noreturn;
pub extern fn quick_exit(_Code: c_int) noreturn;
pub extern fn _Exit(c_int) noreturn;
pub extern fn abort() noreturn;
pub extern fn _set_abort_behavior(_Flags: c_uint, _Mask: c_uint) c_uint;
pub extern fn abs(_X: c_int) c_int;
pub extern fn labs(_X: c_long) c_long;
pub inline fn _abs64(arg_x: c_longlong) c_longlong {
    var x = arg_x;
    _ = &x;
    return __builtin_llabs(x);
}
pub extern fn atexit(?*const fn () callconv(.c) void) c_int;
pub extern fn at_quick_exit(?*const fn () callconv(.c) void) c_int;
pub extern fn atof(_String: [*c]const u8) f64;
pub extern fn _atof_l(_String: [*c]const u8, _Locale: _locale_t) f64;
pub extern fn atoi(_Str: [*c]const u8) c_int;
pub extern fn _atoi_l(_Str: [*c]const u8, _Locale: _locale_t) c_int;
pub extern fn atol(_Str: [*c]const u8) c_long;
pub extern fn _atol_l(_Str: [*c]const u8, _Locale: _locale_t) c_long;
pub extern fn bsearch(_Key: ?*const anyopaque, _Base: ?*const anyopaque, _NumOfElements: usize, _SizeOfElements: usize, _PtFuncCompare: ?*const fn (?*const anyopaque, ?*const anyopaque) callconv(.c) c_int) ?*anyopaque;
pub extern fn qsort(_Base: ?*anyopaque, _NumOfElements: usize, _SizeOfElements: usize, _PtFuncCompare: ?*const fn (?*const anyopaque, ?*const anyopaque) callconv(.c) c_int) void;
pub extern fn _byteswap_ushort(_Short: c_ushort) c_ushort;
pub extern fn _byteswap_ulong(_Long: c_ulong) c_ulong;
pub extern fn _byteswap_uint64(_Int64: c_ulonglong) c_ulonglong;
pub extern fn div(_Numerator: c_int, _Denominator: c_int) div_t;
pub extern fn getenv(_VarName: [*c]const u8) [*c]u8;
pub extern fn _itoa(_Value: c_int, _Dest: [*c]u8, _Radix: c_int) [*c]u8;
pub extern fn _i64toa(_Val: c_longlong, _DstBuf: [*c]u8, _Radix: c_int) [*c]u8;
pub extern fn _ui64toa(_Val: c_ulonglong, _DstBuf: [*c]u8, _Radix: c_int) [*c]u8;
pub extern fn _atoi64(_String: [*c]const u8) c_longlong;
pub extern fn _atoi64_l(_String: [*c]const u8, _Locale: _locale_t) c_longlong;
pub extern fn _strtoi64(_String: [*c]const u8, _EndPtr: [*c][*c]u8, _Radix: c_int) c_longlong;
pub extern fn _strtoi64_l(_String: [*c]const u8, _EndPtr: [*c][*c]u8, _Radix: c_int, _Locale: _locale_t) c_longlong;
pub extern fn _strtoui64(_String: [*c]const u8, _EndPtr: [*c][*c]u8, _Radix: c_int) c_ulonglong;
pub extern fn _strtoui64_l(_String: [*c]const u8, _EndPtr: [*c][*c]u8, _Radix: c_int, _Locale: _locale_t) c_ulonglong;
pub extern fn ldiv(_Numerator: c_long, _Denominator: c_long) ldiv_t;
pub extern fn _ltoa(_Value: c_long, _Dest: [*c]u8, _Radix: c_int) [*c]u8;
pub extern fn mblen(_Ch: [*c]const u8, _MaxCount: usize) c_int;
pub extern fn _mblen_l(_Ch: [*c]const u8, _MaxCount: usize, _Locale: _locale_t) c_int;
pub extern fn _mbstrlen(_Str: [*c]const u8) usize;
pub extern fn _mbstrlen_l(_Str: [*c]const u8, _Locale: _locale_t) usize;
pub extern fn _mbstrnlen(_Str: [*c]const u8, _MaxCount: usize) usize;
pub extern fn _mbstrnlen_l(_Str: [*c]const u8, _MaxCount: usize, _Locale: _locale_t) usize;
pub extern fn mbtowc(noalias _DstCh: [*c]wchar_t, noalias _SrcCh: [*c]const u8, _SrcSizeInBytes: usize) c_int;
pub extern fn _mbtowc_l(noalias _DstCh: [*c]wchar_t, noalias _SrcCh: [*c]const u8, _SrcSizeInBytes: usize, _Locale: _locale_t) c_int;
pub extern fn mbstowcs(noalias _Dest: [*c]wchar_t, noalias _Source: [*c]const u8, _MaxCount: usize) usize;
pub extern fn _mbstowcs_l(noalias _Dest: [*c]wchar_t, noalias _Source: [*c]const u8, _MaxCount: usize, _Locale: _locale_t) usize;
pub extern fn mkstemp(template_name: [*c]u8) c_int;
pub extern fn rand() c_int;
pub extern fn _set_error_mode(_Mode: c_int) c_int;
pub extern fn srand(_Seed: c_uint) void;
pub extern fn strtod(_Str: [*c]const u8, _EndPtr: [*c][*c]u8) f64;
pub extern fn strtof(nptr: [*c]const u8, endptr: [*c][*c]u8) f32;
pub extern fn strtold([*c]const u8, [*c][*c]u8) c_longdouble;
pub extern fn __strtod(noalias [*c]const u8, noalias [*c][*c]u8) f64;
pub extern fn __mingw_strtof(noalias [*c]const u8, noalias [*c][*c]u8) f32;
pub extern fn __mingw_strtod(noalias [*c]const u8, noalias [*c][*c]u8) f64;
pub extern fn __mingw_strtold(noalias [*c]const u8, noalias [*c][*c]u8) c_longdouble;
pub extern fn _strtof_l(noalias _Str: [*c]const u8, noalias _EndPtr: [*c][*c]u8, _Locale: _locale_t) f32;
pub extern fn _strtod_l(noalias _Str: [*c]const u8, noalias _EndPtr: [*c][*c]u8, _Locale: _locale_t) f64;
pub extern fn strtol(_Str: [*c]const u8, _EndPtr: [*c][*c]u8, _Radix: c_int) c_long;
pub extern fn _strtol_l(noalias _Str: [*c]const u8, noalias _EndPtr: [*c][*c]u8, _Radix: c_int, _Locale: _locale_t) c_long;
pub extern fn strtoul(_Str: [*c]const u8, _EndPtr: [*c][*c]u8, _Radix: c_int) c_ulong;
pub extern fn _strtoul_l(noalias _Str: [*c]const u8, noalias _EndPtr: [*c][*c]u8, _Radix: c_int, _Locale: _locale_t) c_ulong;
pub extern fn system(_Command: [*c]const u8) c_int;
pub extern fn _ultoa(_Value: c_ulong, _Dest: [*c]u8, _Radix: c_int) [*c]u8;
pub extern fn wctomb(_MbCh: [*c]u8, _WCh: wchar_t) c_int;
pub extern fn _wctomb_l(_MbCh: [*c]u8, _WCh: wchar_t, _Locale: _locale_t) c_int;
pub extern fn wcstombs(noalias _Dest: [*c]u8, noalias _Source: [*c]const wchar_t, _MaxCount: usize) usize;
pub extern fn _wcstombs_l(noalias _Dest: [*c]u8, noalias _Source: [*c]const wchar_t, _MaxCount: usize, _Locale: _locale_t) usize;
pub extern fn calloc(_NumOfElements: c_ulonglong, _SizeOfElements: c_ulonglong) ?*anyopaque;
pub extern fn free(_Memory: ?*anyopaque) void;
pub extern fn malloc(_Size: c_ulonglong) ?*anyopaque;
pub extern fn realloc(_Memory: ?*anyopaque, _NewSize: c_ulonglong) ?*anyopaque;
pub extern fn _aligned_free(_Memory: ?*anyopaque) void;
pub extern fn _aligned_malloc(_Size: usize, _Alignment: usize) ?*anyopaque;
pub extern fn _aligned_offset_malloc(_Size: usize, _Alignment: usize, _Offset: usize) ?*anyopaque;
pub extern fn _aligned_realloc(_Memory: ?*anyopaque, _Size: usize, _Alignment: usize) ?*anyopaque;
pub extern fn _aligned_offset_realloc(_Memory: ?*anyopaque, _Size: usize, _Alignment: usize, _Offset: usize) ?*anyopaque;
pub extern fn _recalloc(_Memory: ?*anyopaque, _Count: usize, _Size: usize) ?*anyopaque;
pub extern fn _aligned_recalloc(_Memory: ?*anyopaque, _Count: usize, _Size: usize, _Alignment: usize) ?*anyopaque;
pub extern fn _aligned_offset_recalloc(_Memory: ?*anyopaque, _Count: usize, _Size: usize, _Alignment: usize, _Offset: usize) ?*anyopaque;
pub extern fn _aligned_msize(_Memory: ?*anyopaque, _Alignment: usize, _Offset: usize) usize;
pub extern fn _itow(_Value: c_int, _Dest: [*c]wchar_t, _Radix: c_int) [*c]wchar_t;
pub extern fn _ltow(_Value: c_long, _Dest: [*c]wchar_t, _Radix: c_int) [*c]wchar_t;
pub extern fn _ultow(_Value: c_ulong, _Dest: [*c]wchar_t, _Radix: c_int) [*c]wchar_t;
pub extern fn __mingw_wcstod(noalias _Str: [*c]const wchar_t, noalias _EndPtr: [*c][*c]wchar_t) f64;
pub extern fn __mingw_wcstof(noalias nptr: [*c]const wchar_t, noalias endptr: [*c][*c]wchar_t) f32;
pub extern fn __mingw_wcstold(noalias [*c]const wchar_t, noalias [*c][*c]wchar_t) c_longdouble;
pub extern fn wcstod(noalias _Str: [*c]const wchar_t, noalias _EndPtr: [*c][*c]wchar_t) f64;
pub extern fn wcstof(noalias nptr: [*c]const wchar_t, noalias endptr: [*c][*c]wchar_t) f32;
pub extern fn wcstold(noalias [*c]const wchar_t, noalias [*c][*c]wchar_t) c_longdouble;
pub extern fn _wcstod_l(noalias _Str: [*c]const wchar_t, noalias _EndPtr: [*c][*c]wchar_t, _Locale: _locale_t) f64;
pub extern fn _wcstof_l(noalias _Str: [*c]const wchar_t, noalias _EndPtr: [*c][*c]wchar_t, _Locale: _locale_t) f32;
pub extern fn wcstol(noalias _Str: [*c]const wchar_t, noalias _EndPtr: [*c][*c]wchar_t, _Radix: c_int) c_long;
pub extern fn _wcstol_l(noalias _Str: [*c]const wchar_t, noalias _EndPtr: [*c][*c]wchar_t, _Radix: c_int, _Locale: _locale_t) c_long;
pub extern fn wcstoul(noalias _Str: [*c]const wchar_t, noalias _EndPtr: [*c][*c]wchar_t, _Radix: c_int) c_ulong;
pub extern fn _wcstoul_l(noalias _Str: [*c]const wchar_t, noalias _EndPtr: [*c][*c]wchar_t, _Radix: c_int, _Locale: _locale_t) c_ulong;
pub extern fn _wgetenv(_VarName: [*c]const wchar_t) [*c]wchar_t;
pub extern fn _wsystem(_Command: [*c]const wchar_t) c_int;
pub extern fn _wtof(_Str: [*c]const wchar_t) f64;
pub extern fn _wtof_l(_Str: [*c]const wchar_t, _Locale: _locale_t) f64;
pub extern fn _wtoi(_Str: [*c]const wchar_t) c_int;
pub extern fn _wtoi_l(_Str: [*c]const wchar_t, _Locale: _locale_t) c_int;
pub extern fn _wtol(_Str: [*c]const wchar_t) c_long;
pub extern fn _wtol_l(_Str: [*c]const wchar_t, _Locale: _locale_t) c_long;
pub extern fn _i64tow(_Val: c_longlong, _DstBuf: [*c]wchar_t, _Radix: c_int) [*c]wchar_t;
pub extern fn _ui64tow(_Val: c_ulonglong, _DstBuf: [*c]wchar_t, _Radix: c_int) [*c]wchar_t;
pub extern fn _wtoi64(_Str: [*c]const wchar_t) c_longlong;
pub extern fn _wtoi64_l(_Str: [*c]const wchar_t, _Locale: _locale_t) c_longlong;
pub extern fn _wcstoi64(_Str: [*c]const wchar_t, _EndPtr: [*c][*c]wchar_t, _Radix: c_int) c_longlong;
pub extern fn _wcstoi64_l(_Str: [*c]const wchar_t, _EndPtr: [*c][*c]wchar_t, _Radix: c_int, _Locale: _locale_t) c_longlong;
pub extern fn _wcstoui64(_Str: [*c]const wchar_t, _EndPtr: [*c][*c]wchar_t, _Radix: c_int) c_ulonglong;
pub extern fn _wcstoui64_l(_Str: [*c]const wchar_t, _EndPtr: [*c][*c]wchar_t, _Radix: c_int, _Locale: _locale_t) c_ulonglong;
pub extern fn _putenv(_EnvString: [*c]const u8) c_int;
pub extern fn _wputenv(_EnvString: [*c]const wchar_t) c_int;
pub extern fn _fullpath(_FullPath: [*c]u8, _Path: [*c]const u8, _SizeInBytes: usize) [*c]u8;
pub extern fn _ecvt(_Val: f64, _NumOfDigits: c_int, _PtDec: [*c]c_int, _PtSign: [*c]c_int) [*c]u8;
pub extern fn _fcvt(_Val: f64, _NumOfDec: c_int, _PtDec: [*c]c_int, _PtSign: [*c]c_int) [*c]u8;
pub extern fn _gcvt(_Val: f64, _NumOfDigits: c_int, _DstBuf: [*c]u8) [*c]u8;
pub extern fn _atodbl(_Result: [*c]_CRT_DOUBLE, _Str: [*c]u8) c_int;
pub extern fn _atoldbl(_Result: [*c]_LDOUBLE, _Str: [*c]u8) c_int;
pub extern fn _atoflt(_Result: [*c]_CRT_FLOAT, _Str: [*c]u8) c_int;
pub extern fn _atodbl_l(_Result: [*c]_CRT_DOUBLE, _Str: [*c]u8, _Locale: _locale_t) c_int;
pub extern fn _atoldbl_l(_Result: [*c]_LDOUBLE, _Str: [*c]u8, _Locale: _locale_t) c_int;
pub extern fn _atoflt_l(_Result: [*c]_CRT_FLOAT, _Str: [*c]u8, _Locale: _locale_t) c_int;
pub extern fn _lrotl(c_ulong, c_int) c_ulong;
pub extern fn _lrotr(c_ulong, c_int) c_ulong;
pub extern fn _makepath(_Path: [*c]u8, _Drive: [*c]const u8, _Dir: [*c]const u8, _Filename: [*c]const u8, _Ext: [*c]const u8) void;
pub extern fn _onexit(_Func: _onexit_t) _onexit_t;
pub extern fn perror(_ErrMsg: [*c]const u8) void;
pub extern fn _rotl64(_Val: c_ulonglong, _Shift: c_int) c_ulonglong;
pub extern fn _rotr64(Value: c_ulonglong, Shift: c_int) c_ulonglong;
pub extern fn _rotr(_Val: c_uint, _Shift: c_int) c_uint;
pub extern fn _rotl(_Val: c_uint, _Shift: c_int) c_uint;
pub extern fn _searchenv(_Filename: [*c]const u8, _EnvVar: [*c]const u8, _ResultPath: [*c]u8) void;
pub extern fn _splitpath(_FullPath: [*c]const u8, _Drive: [*c]u8, _Dir: [*c]u8, _Filename: [*c]u8, _Ext: [*c]u8) void;
pub extern fn _swab(_Buf1: [*c]u8, _Buf2: [*c]u8, _SizeInBytes: c_int) void;
pub extern fn _wfullpath(_FullPath: [*c]wchar_t, _Path: [*c]const wchar_t, _SizeInWords: usize) [*c]wchar_t;
pub extern fn _wmakepath(_ResultPath: [*c]wchar_t, _Drive: [*c]const wchar_t, _Dir: [*c]const wchar_t, _Filename: [*c]const wchar_t, _Ext: [*c]const wchar_t) void;
pub extern fn _wperror(_ErrMsg: [*c]const wchar_t) void;
pub extern fn _wsearchenv(_Filename: [*c]const wchar_t, _EnvVar: [*c]const wchar_t, _ResultPath: [*c]wchar_t) void;
pub extern fn _wsplitpath(_FullPath: [*c]const wchar_t, _Drive: [*c]wchar_t, _Dir: [*c]wchar_t, _Filename: [*c]wchar_t, _Ext: [*c]wchar_t) void;
pub const _beep = @compileError("unable to resolve function type clang.TypeClass.MacroQualified");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/stdlib.h:635:24
pub const _seterrormode = @compileError("unable to resolve function type clang.TypeClass.MacroQualified");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/stdlib.h:637:24
pub const _sleep = @compileError("unable to resolve function type clang.TypeClass.MacroQualified");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/stdlib.h:638:24
pub extern fn ecvt(_Val: f64, _NumOfDigits: c_int, _PtDec: [*c]c_int, _PtSign: [*c]c_int) [*c]u8;
pub extern fn fcvt(_Val: f64, _NumOfDec: c_int, _PtDec: [*c]c_int, _PtSign: [*c]c_int) [*c]u8;
pub extern fn gcvt(_Val: f64, _NumOfDigits: c_int, _DstBuf: [*c]u8) [*c]u8;
pub extern fn itoa(_Val: c_int, _DstBuf: [*c]u8, _Radix: c_int) [*c]u8;
pub extern fn ltoa(_Val: c_long, _DstBuf: [*c]u8, _Radix: c_int) [*c]u8;
pub extern fn putenv(_EnvString: [*c]const u8) c_int;
pub extern fn swab(_Buf1: [*c]u8, _Buf2: [*c]u8, _SizeInBytes: c_int) void;
pub extern fn ultoa(_Val: c_ulong, _Dstbuf: [*c]u8, _Radix: c_int) [*c]u8;
pub extern fn onexit(_Func: _onexit_t) _onexit_t;
pub const lldiv_t = extern struct {
    quot: c_longlong = @import("std").mem.zeroes(c_longlong),
    rem: c_longlong = @import("std").mem.zeroes(c_longlong),
};
pub extern fn lldiv(c_longlong, c_longlong) lldiv_t;
pub extern fn llabs(c_longlong) c_longlong;
pub extern fn strtoll([*c]const u8, [*c][*c]u8, c_int) c_longlong;
pub extern fn strtoull([*c]const u8, [*c][*c]u8, c_int) c_ulonglong;
pub extern fn atoll([*c]const u8) c_longlong;
pub extern fn wtoll([*c]const wchar_t) c_longlong;
pub extern fn lltoa(c_longlong, [*c]u8, c_int) [*c]u8;
pub extern fn ulltoa(c_ulonglong, [*c]u8, c_int) [*c]u8;
pub extern fn lltow(c_longlong, [*c]wchar_t, c_int) [*c]wchar_t;
pub extern fn ulltow(c_ulonglong, [*c]wchar_t, c_int) [*c]wchar_t;
pub extern fn _dupenv_s(_PBuffer: [*c][*c]u8, _PBufferSizeInBytes: [*c]usize, _VarName: [*c]const u8) errno_t;
pub extern fn bsearch_s(_Key: ?*const anyopaque, _Base: ?*const anyopaque, _NumOfElements: rsize_t, _SizeOfElements: rsize_t, _PtFuncCompare: ?*const fn (?*anyopaque, ?*const anyopaque, ?*const anyopaque) callconv(.c) c_int, _Context: ?*anyopaque) ?*anyopaque;
pub extern fn getenv_s(_ReturnSize: [*c]usize, _DstBuf: [*c]u8, _DstSize: rsize_t, _VarName: [*c]const u8) errno_t;
pub extern fn _itoa_s(_Value: c_int, _DstBuf: [*c]u8, _Size: usize, _Radix: c_int) errno_t;
pub extern fn _i64toa_s(_Val: c_longlong, _DstBuf: [*c]u8, _Size: usize, _Radix: c_int) errno_t;
pub extern fn _ui64toa_s(_Val: c_ulonglong, _DstBuf: [*c]u8, _Size: usize, _Radix: c_int) errno_t;
pub extern fn _ltoa_s(_Val: c_long, _DstBuf: [*c]u8, _Size: usize, _Radix: c_int) errno_t;
pub extern fn mbstowcs_s(_PtNumOfCharConverted: [*c]usize, _DstBuf: [*c]wchar_t, _SizeInWords: usize, _SrcBuf: [*c]const u8, _MaxCount: usize) errno_t;
pub extern fn _mbstowcs_s_l(_PtNumOfCharConverted: [*c]usize, _DstBuf: [*c]wchar_t, _SizeInWords: usize, _SrcBuf: [*c]const u8, _MaxCount: usize, _Locale: _locale_t) errno_t;
pub extern fn _ultoa_s(_Val: c_ulong, _DstBuf: [*c]u8, _Size: usize, _Radix: c_int) errno_t;
pub extern fn wctomb_s(_SizeConverted: [*c]c_int, _MbCh: [*c]u8, _SizeInBytes: rsize_t, _WCh: wchar_t) errno_t;
pub extern fn _wctomb_s_l(_SizeConverted: [*c]c_int, _MbCh: [*c]u8, _SizeInBytes: usize, _WCh: wchar_t, _Locale: _locale_t) errno_t;
pub extern fn wcstombs_s(_PtNumOfCharConverted: [*c]usize, _Dst: [*c]u8, _DstSizeInBytes: usize, _Src: [*c]const wchar_t, _MaxCountInBytes: usize) errno_t;
pub extern fn _wcstombs_s_l(_PtNumOfCharConverted: [*c]usize, _Dst: [*c]u8, _DstSizeInBytes: usize, _Src: [*c]const wchar_t, _MaxCountInBytes: usize, _Locale: _locale_t) errno_t;
pub extern fn _ecvt_s(_DstBuf: [*c]u8, _Size: usize, _Val: f64, _NumOfDights: c_int, _PtDec: [*c]c_int, _PtSign: [*c]c_int) errno_t;
pub extern fn _fcvt_s(_DstBuf: [*c]u8, _Size: usize, _Val: f64, _NumOfDec: c_int, _PtDec: [*c]c_int, _PtSign: [*c]c_int) errno_t;
pub extern fn _gcvt_s(_DstBuf: [*c]u8, _Size: usize, _Val: f64, _NumOfDigits: c_int) errno_t;
pub extern fn _makepath_s(_PathResult: [*c]u8, _Size: usize, _Drive: [*c]const u8, _Dir: [*c]const u8, _Filename: [*c]const u8, _Ext: [*c]const u8) errno_t;
pub extern fn _putenv_s(_Name: [*c]const u8, _Value: [*c]const u8) errno_t;
pub extern fn _searchenv_s(_Filename: [*c]const u8, _EnvVar: [*c]const u8, _ResultPath: [*c]u8, _SizeInBytes: usize) errno_t;
pub extern fn _splitpath_s(_FullPath: [*c]const u8, _Drive: [*c]u8, _DriveSize: usize, _Dir: [*c]u8, _DirSize: usize, _Filename: [*c]u8, _FilenameSize: usize, _Ext: [*c]u8, _ExtSize: usize) errno_t;
pub extern fn qsort_s(_Base: ?*anyopaque, _NumOfElements: usize, _SizeOfElements: usize, _PtFuncCompare: ?*const fn (?*anyopaque, ?*const anyopaque, ?*const anyopaque) callconv(.c) c_int, _Context: ?*anyopaque) void;
pub const struct__heapinfo = extern struct {
    _pentry: [*c]c_int = @import("std").mem.zeroes([*c]c_int),
    _size: usize = @import("std").mem.zeroes(usize),
    _useflag: c_int = @import("std").mem.zeroes(c_int),
};
pub const _HEAPINFO = struct__heapinfo;
pub extern fn __p__amblksiz() [*c]c_uint;
pub extern fn __mingw_aligned_malloc(_Size: usize, _Alignment: usize) ?*anyopaque;
pub extern fn __mingw_aligned_free(_Memory: ?*anyopaque) void;
pub extern fn __mingw_aligned_offset_realloc(_Memory: ?*anyopaque, _Size: usize, _Alignment: usize, _Offset: usize) ?*anyopaque;
pub extern fn __mingw_aligned_realloc(_Memory: ?*anyopaque, _Size: usize, _Offset: usize) ?*anyopaque;
pub inline fn _mm_malloc(arg___size: usize, arg___align: usize) ?*anyopaque {
    var __size = arg___size;
    _ = &__size;
    var __align = arg___align;
    _ = &__align;
    if (__align == @as(usize, @bitCast(@as(c_longlong, @as(c_int, 1))))) {
        return malloc(__size);
    }
    if (!((__align & (__align -% @as(usize, @bitCast(@as(c_longlong, @as(c_int, 1)))))) != 0) and (__align < @sizeOf(?*anyopaque))) {
        __align = @sizeOf(?*anyopaque);
    }
    var __mallocedMemory: ?*anyopaque = undefined;
    _ = &__mallocedMemory;
    __mallocedMemory = __mingw_aligned_malloc(__size, __align);
    return __mallocedMemory;
}
pub inline fn _mm_free(arg___p: ?*anyopaque) void {
    var __p = arg___p;
    _ = &__p;
    __mingw_aligned_free(__p);
}
pub extern fn _resetstkoflw() c_int;
pub extern fn _set_malloc_crt_max_wait(_NewValue: c_ulong) c_ulong;
pub extern fn _expand(_Memory: ?*anyopaque, _NewSize: usize) ?*anyopaque;
pub extern fn _msize(_Memory: ?*anyopaque) usize;
pub extern fn _get_sbh_threshold() usize;
pub extern fn _set_sbh_threshold(_NewValue: usize) c_int;
pub extern fn _set_amblksiz(_Value: usize) errno_t;
pub extern fn _get_amblksiz(_Value: [*c]usize) errno_t;
pub extern fn _heapadd(_Memory: ?*anyopaque, _Size: usize) c_int;
pub extern fn _heapchk() c_int;
pub extern fn _heapmin() c_int;
pub extern fn _heapset(_Fill: c_uint) c_int;
pub extern fn _heapwalk(_EntryInfo: [*c]_HEAPINFO) c_int;
pub extern fn _heapused(_Used: [*c]usize, _Commit: [*c]usize) usize;
pub extern fn _get_heap_handle() isize;
pub fn _MarkAllocaS(arg__Ptr: ?*anyopaque, arg__Marker: c_uint) callconv(.c) ?*anyopaque {
    var _Ptr = arg__Ptr;
    _ = &_Ptr;
    var _Marker = arg__Marker;
    _ = &_Marker;
    if (_Ptr != null) {
        @as([*c]c_uint, @ptrCast(@alignCast(_Ptr))).* = _Marker;
        _Ptr = @as(?*anyopaque, @ptrCast(@as([*c]u8, @ptrCast(@alignCast(_Ptr))) + @as(usize, @bitCast(@as(isize, @intCast(@as(c_int, 16)))))));
    }
    return _Ptr;
}
pub fn _freea(arg__Memory: ?*anyopaque) callconv(.c) void {
    var _Memory = arg__Memory;
    _ = &_Memory;
    var _Marker: c_uint = undefined;
    _ = &_Marker;
    if (_Memory != null) {
        _Memory = @as(?*anyopaque, @ptrCast(@as([*c]u8, @ptrCast(@alignCast(_Memory))) - @as(usize, @bitCast(@as(isize, @intCast(@as(c_int, 16)))))));
        _Marker = @as([*c]c_uint, @ptrCast(@alignCast(_Memory))).*;
        if (_Marker == @as(c_uint, @bitCast(@as(c_int, 56797)))) {
            free(_Memory);
        }
    }
}
pub const int_least8_t = i8;
pub const uint_least8_t = u8;
pub const int_least16_t = c_short;
pub const uint_least16_t = c_ushort;
pub const int_least32_t = c_int;
pub const uint_least32_t = c_uint;
pub const int_least64_t = c_longlong;
pub const uint_least64_t = c_ulonglong;
pub const int_fast8_t = i8;
pub const uint_fast8_t = u8;
pub const int_fast16_t = c_short;
pub const uint_fast16_t = c_ushort;
pub const int_fast32_t = c_int;
pub const uint_fast32_t = c_uint;
pub const int_fast64_t = c_longlong;
pub const uint_fast64_t = c_ulonglong;
pub const intmax_t = c_longlong;
pub const uintmax_t = c_ulonglong;
pub const cuuint32_t = u32;
pub const cuuint64_t = u64;
pub const CUdeviceptr_v2 = c_ulonglong;
pub const CUdeviceptr = CUdeviceptr_v2;
pub const CUdevice_v1 = c_int;
pub const CUdevice = CUdevice_v1;
pub const struct_CUctx_st = opaque {};
pub const CUcontext = ?*struct_CUctx_st;
pub const struct_CUmod_st = opaque {};
pub const CUmodule = ?*struct_CUmod_st;
pub const struct_CUfunc_st = opaque {};
pub const CUfunction = ?*struct_CUfunc_st;
pub const struct_CUlib_st = opaque {};
pub const CUlibrary = ?*struct_CUlib_st;
pub const struct_CUkern_st = opaque {};
pub const CUkernel = ?*struct_CUkern_st;
pub const struct_CUarray_st = opaque {};
pub const CUarray = ?*struct_CUarray_st;
pub const struct_CUmipmappedArray_st = opaque {};
pub const CUmipmappedArray = ?*struct_CUmipmappedArray_st;
pub const struct_CUtexref_st = opaque {};
pub const CUtexref = ?*struct_CUtexref_st;
pub const struct_CUsurfref_st = opaque {};
pub const CUsurfref = ?*struct_CUsurfref_st;
pub const struct_CUevent_st = opaque {};
pub const CUevent = ?*struct_CUevent_st;
pub const struct_CUstream_st = opaque {};
pub const CUstream = ?*struct_CUstream_st;
pub const struct_CUgraphicsResource_st = opaque {};
pub const CUgraphicsResource = ?*struct_CUgraphicsResource_st;
pub const CUtexObject_v1 = c_ulonglong;
pub const CUtexObject = CUtexObject_v1;
pub const CUsurfObject_v1 = c_ulonglong;
pub const CUsurfObject = CUsurfObject_v1;
pub const struct_CUextMemory_st = opaque {};
pub const CUexternalMemory = ?*struct_CUextMemory_st;
pub const struct_CUextSemaphore_st = opaque {};
pub const CUexternalSemaphore = ?*struct_CUextSemaphore_st;
pub const struct_CUgraph_st = opaque {};
pub const CUgraph = ?*struct_CUgraph_st;
pub const struct_CUgraphNode_st = opaque {};
pub const CUgraphNode = ?*struct_CUgraphNode_st;
pub const struct_CUgraphExec_st = opaque {};
pub const CUgraphExec = ?*struct_CUgraphExec_st;
pub const struct_CUmemPoolHandle_st = opaque {};
pub const CUmemoryPool = ?*struct_CUmemPoolHandle_st;
pub const struct_CUuserObject_st = opaque {};
pub const CUuserObject = ?*struct_CUuserObject_st;
pub const CUgraphConditionalHandle = cuuint64_t;
pub const struct_CUgraphDeviceUpdatableNode_st = opaque {};
pub const CUgraphDeviceNode = ?*struct_CUgraphDeviceUpdatableNode_st;
pub const struct_CUasyncCallbackEntry_st = opaque {};
pub const CUasyncCallbackHandle = ?*struct_CUasyncCallbackEntry_st;
pub const struct_CUgreenCtx_st = opaque {};
pub const CUgreenCtx = ?*struct_CUgreenCtx_st;
pub const struct_CUuuid_st = extern struct {
    bytes: [16]u8 = @import("std").mem.zeroes([16]u8),
};
pub const CUuuid = struct_CUuuid_st;
pub const struct_CUmemFabricHandle_st = extern struct {
    data: [64]u8 = @import("std").mem.zeroes([64]u8),
};
pub const CUmemFabricHandle_v1 = struct_CUmemFabricHandle_st;
pub const CUmemFabricHandle = CUmemFabricHandle_v1;
pub const struct_CUipcEventHandle_st = extern struct {
    reserved: [64]u8 = @import("std").mem.zeroes([64]u8),
};
pub const CUipcEventHandle_v1 = struct_CUipcEventHandle_st;
pub const CUipcEventHandle = CUipcEventHandle_v1;
pub const struct_CUipcMemHandle_st = extern struct {
    reserved: [64]u8 = @import("std").mem.zeroes([64]u8),
};
pub const CUipcMemHandle_v1 = struct_CUipcMemHandle_st;
pub const CUipcMemHandle = CUipcMemHandle_v1;
pub const CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS: c_int = 1;
pub const enum_CUipcMem_flags_enum = c_uint;
pub const CUipcMem_flags = enum_CUipcMem_flags_enum;
pub const CU_MEM_ATTACH_GLOBAL: c_int = 1;
pub const CU_MEM_ATTACH_HOST: c_int = 2;
pub const CU_MEM_ATTACH_SINGLE: c_int = 4;
pub const enum_CUmemAttach_flags_enum = c_uint;
pub const CUmemAttach_flags = enum_CUmemAttach_flags_enum;
pub const CU_CTX_SCHED_AUTO: c_int = 0;
pub const CU_CTX_SCHED_SPIN: c_int = 1;
pub const CU_CTX_SCHED_YIELD: c_int = 2;
pub const CU_CTX_SCHED_BLOCKING_SYNC: c_int = 4;
pub const CU_CTX_BLOCKING_SYNC: c_int = 4;
pub const CU_CTX_SCHED_MASK: c_int = 7;
pub const CU_CTX_MAP_HOST: c_int = 8;
pub const CU_CTX_LMEM_RESIZE_TO_MAX: c_int = 16;
pub const CU_CTX_COREDUMP_ENABLE: c_int = 32;
pub const CU_CTX_USER_COREDUMP_ENABLE: c_int = 64;
pub const CU_CTX_SYNC_MEMOPS: c_int = 128;
pub const CU_CTX_FLAGS_MASK: c_int = 255;
pub const enum_CUctx_flags_enum = c_uint;
pub const CUctx_flags = enum_CUctx_flags_enum;
pub const CU_EVENT_SCHED_AUTO: c_int = 0;
pub const CU_EVENT_SCHED_SPIN: c_int = 1;
pub const CU_EVENT_SCHED_YIELD: c_int = 2;
pub const CU_EVENT_SCHED_BLOCKING_SYNC: c_int = 4;
pub const enum_CUevent_sched_flags_enum = c_uint;
pub const CUevent_sched_flags = enum_CUevent_sched_flags_enum;
pub const NVCL_EVENT_SCHED_AUTO: c_int = 0;
pub const NVCL_EVENT_SCHED_SPIN: c_int = 1;
pub const NVCL_EVENT_SCHED_YIELD: c_int = 2;
pub const NVCL_EVENT_SCHED_BLOCKING_SYNC: c_int = 4;
pub const enum_cl_event_flags_enum = c_uint;
pub const cl_event_flags = enum_cl_event_flags_enum;
pub const NVCL_CTX_SCHED_AUTO: c_int = 0;
pub const NVCL_CTX_SCHED_SPIN: c_int = 1;
pub const NVCL_CTX_SCHED_YIELD: c_int = 2;
pub const NVCL_CTX_SCHED_BLOCKING_SYNC: c_int = 4;
pub const enum_cl_context_flags_enum = c_uint;
pub const cl_context_flags = enum_cl_context_flags_enum;
pub const CU_STREAM_DEFAULT: c_int = 0;
pub const CU_STREAM_NON_BLOCKING: c_int = 1;
pub const enum_CUstream_flags_enum = c_uint;
pub const CUstream_flags = enum_CUstream_flags_enum;
pub const CU_EVENT_DEFAULT: c_int = 0;
pub const CU_EVENT_BLOCKING_SYNC: c_int = 1;
pub const CU_EVENT_DISABLE_TIMING: c_int = 2;
pub const CU_EVENT_INTERPROCESS: c_int = 4;
pub const enum_CUevent_flags_enum = c_uint;
pub const CUevent_flags = enum_CUevent_flags_enum;
pub const CU_EVENT_RECORD_DEFAULT: c_int = 0;
pub const CU_EVENT_RECORD_EXTERNAL: c_int = 1;
pub const enum_CUevent_record_flags_enum = c_uint;
pub const CUevent_record_flags = enum_CUevent_record_flags_enum;
pub const CU_EVENT_WAIT_DEFAULT: c_int = 0;
pub const CU_EVENT_WAIT_EXTERNAL: c_int = 1;
pub const enum_CUevent_wait_flags_enum = c_uint;
pub const CUevent_wait_flags = enum_CUevent_wait_flags_enum;
pub const CU_STREAM_WAIT_VALUE_GEQ: c_int = 0;
pub const CU_STREAM_WAIT_VALUE_EQ: c_int = 1;
pub const CU_STREAM_WAIT_VALUE_AND: c_int = 2;
pub const CU_STREAM_WAIT_VALUE_NOR: c_int = 3;
pub const CU_STREAM_WAIT_VALUE_FLUSH: c_int = 1073741824;
pub const enum_CUstreamWaitValue_flags_enum = c_uint;
pub const CUstreamWaitValue_flags = enum_CUstreamWaitValue_flags_enum;
pub const CU_STREAM_WRITE_VALUE_DEFAULT: c_int = 0;
pub const CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER: c_int = 1;
pub const enum_CUstreamWriteValue_flags_enum = c_uint;
pub const CUstreamWriteValue_flags = enum_CUstreamWriteValue_flags_enum;
pub const CU_STREAM_MEM_OP_WAIT_VALUE_32: c_int = 1;
pub const CU_STREAM_MEM_OP_WRITE_VALUE_32: c_int = 2;
pub const CU_STREAM_MEM_OP_WAIT_VALUE_64: c_int = 4;
pub const CU_STREAM_MEM_OP_WRITE_VALUE_64: c_int = 5;
pub const CU_STREAM_MEM_OP_BARRIER: c_int = 6;
pub const CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES: c_int = 3;
pub const enum_CUstreamBatchMemOpType_enum = c_uint;
pub const CUstreamBatchMemOpType = enum_CUstreamBatchMemOpType_enum;
pub const CU_STREAM_MEMORY_BARRIER_TYPE_SYS: c_int = 0;
pub const CU_STREAM_MEMORY_BARRIER_TYPE_GPU: c_int = 1;
pub const enum_CUstreamMemoryBarrier_flags_enum = c_uint;
pub const CUstreamMemoryBarrier_flags = enum_CUstreamMemoryBarrier_flags_enum;
const union_unnamed_2 = extern union {
    value: cuuint32_t,
    value64: cuuint64_t,
};
pub const struct_CUstreamMemOpWaitValueParams_st_1 = extern struct {
    operation: CUstreamBatchMemOpType = @import("std").mem.zeroes(CUstreamBatchMemOpType),
    address: CUdeviceptr = @import("std").mem.zeroes(CUdeviceptr),
    unnamed_0: union_unnamed_2 = @import("std").mem.zeroes(union_unnamed_2),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
    alias: CUdeviceptr = @import("std").mem.zeroes(CUdeviceptr),
};
const union_unnamed_4 = extern union {
    value: cuuint32_t,
    value64: cuuint64_t,
};
pub const struct_CUstreamMemOpWriteValueParams_st_3 = extern struct {
    operation: CUstreamBatchMemOpType = @import("std").mem.zeroes(CUstreamBatchMemOpType),
    address: CUdeviceptr = @import("std").mem.zeroes(CUdeviceptr),
    unnamed_0: union_unnamed_4 = @import("std").mem.zeroes(union_unnamed_4),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
    alias: CUdeviceptr = @import("std").mem.zeroes(CUdeviceptr),
};
pub const struct_CUstreamMemOpFlushRemoteWritesParams_st_5 = extern struct {
    operation: CUstreamBatchMemOpType = @import("std").mem.zeroes(CUstreamBatchMemOpType),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const struct_CUstreamMemOpMemoryBarrierParams_st_6 = extern struct {
    operation: CUstreamBatchMemOpType = @import("std").mem.zeroes(CUstreamBatchMemOpType),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const union_CUstreamBatchMemOpParams_union = extern union {
    operation: CUstreamBatchMemOpType,
    waitValue: struct_CUstreamMemOpWaitValueParams_st_1,
    writeValue: struct_CUstreamMemOpWriteValueParams_st_3,
    flushRemoteWrites: struct_CUstreamMemOpFlushRemoteWritesParams_st_5,
    memoryBarrier: struct_CUstreamMemOpMemoryBarrierParams_st_6,
    pad: [6]cuuint64_t,
};
pub const CUstreamBatchMemOpParams_v1 = union_CUstreamBatchMemOpParams_union;
pub const CUstreamBatchMemOpParams = CUstreamBatchMemOpParams_v1;
pub const struct_CUDA_BATCH_MEM_OP_NODE_PARAMS_v1_st = extern struct {
    ctx: CUcontext = @import("std").mem.zeroes(CUcontext),
    count: c_uint = @import("std").mem.zeroes(c_uint),
    paramArray: [*c]CUstreamBatchMemOpParams = @import("std").mem.zeroes([*c]CUstreamBatchMemOpParams),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const CUDA_BATCH_MEM_OP_NODE_PARAMS_v1 = struct_CUDA_BATCH_MEM_OP_NODE_PARAMS_v1_st;
pub const CUDA_BATCH_MEM_OP_NODE_PARAMS = CUDA_BATCH_MEM_OP_NODE_PARAMS_v1;
pub const struct_CUDA_BATCH_MEM_OP_NODE_PARAMS_v2_st = extern struct {
    ctx: CUcontext = @import("std").mem.zeroes(CUcontext),
    count: c_uint = @import("std").mem.zeroes(c_uint),
    paramArray: [*c]CUstreamBatchMemOpParams = @import("std").mem.zeroes([*c]CUstreamBatchMemOpParams),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const CUDA_BATCH_MEM_OP_NODE_PARAMS_v2 = struct_CUDA_BATCH_MEM_OP_NODE_PARAMS_v2_st;
pub const CU_OCCUPANCY_DEFAULT: c_int = 0;
pub const CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE: c_int = 1;
pub const enum_CUoccupancy_flags_enum = c_uint;
pub const CUoccupancy_flags = enum_CUoccupancy_flags_enum;
pub const CU_STREAM_ADD_CAPTURE_DEPENDENCIES: c_int = 0;
pub const CU_STREAM_SET_CAPTURE_DEPENDENCIES: c_int = 1;
pub const enum_CUstreamUpdateCaptureDependencies_flags_enum = c_uint;
pub const CUstreamUpdateCaptureDependencies_flags = enum_CUstreamUpdateCaptureDependencies_flags_enum;
pub const CU_ASYNC_NOTIFICATION_TYPE_OVER_BUDGET: c_int = 1;
pub const enum_CUasyncNotificationType_enum = c_uint;
pub const CUasyncNotificationType = enum_CUasyncNotificationType_enum;
const struct_unnamed_8 = extern struct {
    bytesOverBudget: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
};
const union_unnamed_7 = extern union {
    overBudget: struct_unnamed_8,
};
pub const struct_CUasyncNotificationInfo_st = extern struct {
    type: CUasyncNotificationType = @import("std").mem.zeroes(CUasyncNotificationType),
    info: union_unnamed_7 = @import("std").mem.zeroes(union_unnamed_7),
};
pub const CUasyncNotificationInfo = struct_CUasyncNotificationInfo_st;
pub const CUasyncCallback = ?*const fn ([*c]CUasyncNotificationInfo, ?*anyopaque, CUasyncCallbackHandle) callconv(.c) void;
pub const CU_AD_FORMAT_UNSIGNED_INT8: c_int = 1;
pub const CU_AD_FORMAT_UNSIGNED_INT16: c_int = 2;
pub const CU_AD_FORMAT_UNSIGNED_INT32: c_int = 3;
pub const CU_AD_FORMAT_SIGNED_INT8: c_int = 8;
pub const CU_AD_FORMAT_SIGNED_INT16: c_int = 9;
pub const CU_AD_FORMAT_SIGNED_INT32: c_int = 10;
pub const CU_AD_FORMAT_HALF: c_int = 16;
pub const CU_AD_FORMAT_FLOAT: c_int = 32;
pub const CU_AD_FORMAT_NV12: c_int = 176;
pub const CU_AD_FORMAT_UNORM_INT8X1: c_int = 192;
pub const CU_AD_FORMAT_UNORM_INT8X2: c_int = 193;
pub const CU_AD_FORMAT_UNORM_INT8X4: c_int = 194;
pub const CU_AD_FORMAT_UNORM_INT16X1: c_int = 195;
pub const CU_AD_FORMAT_UNORM_INT16X2: c_int = 196;
pub const CU_AD_FORMAT_UNORM_INT16X4: c_int = 197;
pub const CU_AD_FORMAT_SNORM_INT8X1: c_int = 198;
pub const CU_AD_FORMAT_SNORM_INT8X2: c_int = 199;
pub const CU_AD_FORMAT_SNORM_INT8X4: c_int = 200;
pub const CU_AD_FORMAT_SNORM_INT16X1: c_int = 201;
pub const CU_AD_FORMAT_SNORM_INT16X2: c_int = 202;
pub const CU_AD_FORMAT_SNORM_INT16X4: c_int = 203;
pub const CU_AD_FORMAT_BC1_UNORM: c_int = 145;
pub const CU_AD_FORMAT_BC1_UNORM_SRGB: c_int = 146;
pub const CU_AD_FORMAT_BC2_UNORM: c_int = 147;
pub const CU_AD_FORMAT_BC2_UNORM_SRGB: c_int = 148;
pub const CU_AD_FORMAT_BC3_UNORM: c_int = 149;
pub const CU_AD_FORMAT_BC3_UNORM_SRGB: c_int = 150;
pub const CU_AD_FORMAT_BC4_UNORM: c_int = 151;
pub const CU_AD_FORMAT_BC4_SNORM: c_int = 152;
pub const CU_AD_FORMAT_BC5_UNORM: c_int = 153;
pub const CU_AD_FORMAT_BC5_SNORM: c_int = 154;
pub const CU_AD_FORMAT_BC6H_UF16: c_int = 155;
pub const CU_AD_FORMAT_BC6H_SF16: c_int = 156;
pub const CU_AD_FORMAT_BC7_UNORM: c_int = 157;
pub const CU_AD_FORMAT_BC7_UNORM_SRGB: c_int = 158;
pub const CU_AD_FORMAT_P010: c_int = 159;
pub const CU_AD_FORMAT_P016: c_int = 161;
pub const CU_AD_FORMAT_NV16: c_int = 162;
pub const CU_AD_FORMAT_P210: c_int = 163;
pub const CU_AD_FORMAT_P216: c_int = 164;
pub const CU_AD_FORMAT_YUY2: c_int = 165;
pub const CU_AD_FORMAT_Y210: c_int = 166;
pub const CU_AD_FORMAT_Y216: c_int = 167;
pub const CU_AD_FORMAT_AYUV: c_int = 168;
pub const CU_AD_FORMAT_Y410: c_int = 169;
pub const CU_AD_FORMAT_Y416: c_int = 177;
pub const CU_AD_FORMAT_Y444_PLANAR8: c_int = 178;
pub const CU_AD_FORMAT_Y444_PLANAR10: c_int = 179;
pub const CU_AD_FORMAT_YUV444_8bit_SemiPlanar: c_int = 180;
pub const CU_AD_FORMAT_YUV444_16bit_SemiPlanar: c_int = 181;
pub const CU_AD_FORMAT_UNORM_INT_101010_2: c_int = 80;
pub const CU_AD_FORMAT_MAX: c_int = 2147483647;
pub const enum_CUarray_format_enum = c_uint;
pub const CUarray_format = enum_CUarray_format_enum;
pub const CU_TR_ADDRESS_MODE_WRAP: c_int = 0;
pub const CU_TR_ADDRESS_MODE_CLAMP: c_int = 1;
pub const CU_TR_ADDRESS_MODE_MIRROR: c_int = 2;
pub const CU_TR_ADDRESS_MODE_BORDER: c_int = 3;
pub const enum_CUaddress_mode_enum = c_uint;
pub const CUaddress_mode = enum_CUaddress_mode_enum;
pub const CU_TR_FILTER_MODE_POINT: c_int = 0;
pub const CU_TR_FILTER_MODE_LINEAR: c_int = 1;
pub const enum_CUfilter_mode_enum = c_uint;
pub const CUfilter_mode = enum_CUfilter_mode_enum;
pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 1;
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: c_int = 2;
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: c_int = 3;
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: c_int = 4;
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: c_int = 5;
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: c_int = 6;
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: c_int = 7;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: c_int = 8;
pub const CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK: c_int = 8;
pub const CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: c_int = 9;
pub const CU_DEVICE_ATTRIBUTE_WARP_SIZE: c_int = 10;
pub const CU_DEVICE_ATTRIBUTE_MAX_PITCH: c_int = 11;
pub const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: c_int = 12;
pub const CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK: c_int = 12;
pub const CU_DEVICE_ATTRIBUTE_CLOCK_RATE: c_int = 13;
pub const CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT: c_int = 14;
pub const CU_DEVICE_ATTRIBUTE_GPU_OVERLAP: c_int = 15;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT: c_int = 17;
pub const CU_DEVICE_ATTRIBUTE_INTEGRATED: c_int = 18;
pub const CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY: c_int = 19;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_MODE: c_int = 20;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH: c_int = 21;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH: c_int = 22;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT: c_int = 23;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH: c_int = 24;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT: c_int = 25;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH: c_int = 26;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH: c_int = 27;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT: c_int = 28;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS: c_int = 29;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH: c_int = 27;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT: c_int = 28;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES: c_int = 29;
pub const CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT: c_int = 30;
pub const CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS: c_int = 31;
pub const CU_DEVICE_ATTRIBUTE_ECC_ENABLED: c_int = 32;
pub const CU_DEVICE_ATTRIBUTE_PCI_BUS_ID: c_int = 33;
pub const CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID: c_int = 34;
pub const CU_DEVICE_ATTRIBUTE_TCC_DRIVER: c_int = 35;
pub const CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: c_int = 36;
pub const CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: c_int = 37;
pub const CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: c_int = 38;
pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: c_int = 39;
pub const CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT: c_int = 40;
pub const CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING: c_int = 41;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH: c_int = 42;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS: c_int = 43;
pub const CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER: c_int = 44;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH: c_int = 45;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT: c_int = 46;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE: c_int = 47;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE: c_int = 48;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE: c_int = 49;
pub const CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID: c_int = 50;
pub const CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT: c_int = 51;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH: c_int = 52;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH: c_int = 53;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS: c_int = 54;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH: c_int = 55;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH: c_int = 56;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT: c_int = 57;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH: c_int = 58;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT: c_int = 59;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH: c_int = 60;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH: c_int = 61;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS: c_int = 62;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH: c_int = 63;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT: c_int = 64;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS: c_int = 65;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH: c_int = 66;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH: c_int = 67;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS: c_int = 68;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH: c_int = 69;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH: c_int = 70;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT: c_int = 71;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH: c_int = 72;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH: c_int = 73;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT: c_int = 74;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH: c_int = 77;
pub const CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED: c_int = 78;
pub const CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED: c_int = 79;
pub const CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED: c_int = 80;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: c_int = 81;
pub const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR: c_int = 82;
pub const CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY: c_int = 83;
pub const CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD: c_int = 84;
pub const CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID: c_int = 85;
pub const CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED: c_int = 86;
pub const CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO: c_int = 87;
pub const CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS: c_int = 88;
pub const CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS: c_int = 89;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED: c_int = 90;
pub const CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM: c_int = 91;
pub const CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1: c_int = 92;
pub const CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1: c_int = 93;
pub const CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1: c_int = 94;
pub const CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH: c_int = 95;
pub const CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH: c_int = 96;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: c_int = 97;
pub const CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES: c_int = 98;
pub const CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED: c_int = 99;
pub const CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES: c_int = 100;
pub const CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST: c_int = 101;
pub const CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED: c_int = 102;
pub const CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED: c_int = 102;
pub const CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED: c_int = 103;
pub const CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED: c_int = 104;
pub const CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED: c_int = 105;
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR: c_int = 106;
pub const CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED: c_int = 107;
pub const CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE: c_int = 108;
pub const CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE: c_int = 109;
pub const CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED: c_int = 110;
pub const CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK: c_int = 111;
pub const CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED: c_int = 112;
pub const CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED: c_int = 113;
pub const CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED: c_int = 114;
pub const CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED: c_int = 115;
pub const CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED: c_int = 116;
pub const CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS: c_int = 117;
pub const CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING: c_int = 118;
pub const CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES: c_int = 119;
pub const CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH: c_int = 120;
pub const CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED: c_int = 121;
pub const CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS: c_int = 122;
pub const CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR: c_int = 123;
pub const CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED: c_int = 124;
pub const CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED: c_int = 125;
pub const CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT: c_int = 126;
pub const CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED: c_int = 127;
pub const CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED: c_int = 128;
pub const CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS: c_int = 129;
pub const CU_DEVICE_ATTRIBUTE_NUMA_CONFIG: c_int = 130;
pub const CU_DEVICE_ATTRIBUTE_NUMA_ID: c_int = 131;
pub const CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED: c_int = 132;
pub const CU_DEVICE_ATTRIBUTE_MPS_ENABLED: c_int = 133;
pub const CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID: c_int = 134;
pub const CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED: c_int = 135;
pub const CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK: c_int = 136;
pub const CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH: c_int = 137;
pub const CU_DEVICE_ATTRIBUTE_GPU_PCI_DEVICE_ID: c_int = 139;
pub const CU_DEVICE_ATTRIBUTE_GPU_PCI_SUBSYSTEM_ID: c_int = 140;
pub const CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED: c_int = 143;
pub const CU_DEVICE_ATTRIBUTE_MAX: c_int = 144;
pub const enum_CUdevice_attribute_enum = c_uint;
pub const CUdevice_attribute = enum_CUdevice_attribute_enum;
pub const struct_CUdevprop_st = extern struct {
    maxThreadsPerBlock: c_int = @import("std").mem.zeroes(c_int),
    maxThreadsDim: [3]c_int = @import("std").mem.zeroes([3]c_int),
    maxGridSize: [3]c_int = @import("std").mem.zeroes([3]c_int),
    sharedMemPerBlock: c_int = @import("std").mem.zeroes(c_int),
    totalConstantMemory: c_int = @import("std").mem.zeroes(c_int),
    SIMDWidth: c_int = @import("std").mem.zeroes(c_int),
    memPitch: c_int = @import("std").mem.zeroes(c_int),
    regsPerBlock: c_int = @import("std").mem.zeroes(c_int),
    clockRate: c_int = @import("std").mem.zeroes(c_int),
    textureAlign: c_int = @import("std").mem.zeroes(c_int),
};
pub const CUdevprop_v1 = struct_CUdevprop_st;
pub const CUdevprop = CUdevprop_v1;
pub const CU_POINTER_ATTRIBUTE_CONTEXT: c_int = 1;
pub const CU_POINTER_ATTRIBUTE_MEMORY_TYPE: c_int = 2;
pub const CU_POINTER_ATTRIBUTE_DEVICE_POINTER: c_int = 3;
pub const CU_POINTER_ATTRIBUTE_HOST_POINTER: c_int = 4;
pub const CU_POINTER_ATTRIBUTE_P2P_TOKENS: c_int = 5;
pub const CU_POINTER_ATTRIBUTE_SYNC_MEMOPS: c_int = 6;
pub const CU_POINTER_ATTRIBUTE_BUFFER_ID: c_int = 7;
pub const CU_POINTER_ATTRIBUTE_IS_MANAGED: c_int = 8;
pub const CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL: c_int = 9;
pub const CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE: c_int = 10;
pub const CU_POINTER_ATTRIBUTE_RANGE_START_ADDR: c_int = 11;
pub const CU_POINTER_ATTRIBUTE_RANGE_SIZE: c_int = 12;
pub const CU_POINTER_ATTRIBUTE_MAPPED: c_int = 13;
pub const CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES: c_int = 14;
pub const CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE: c_int = 15;
pub const CU_POINTER_ATTRIBUTE_ACCESS_FLAGS: c_int = 16;
pub const CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE: c_int = 17;
pub const CU_POINTER_ATTRIBUTE_MAPPING_SIZE: c_int = 18;
pub const CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR: c_int = 19;
pub const CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID: c_int = 20;
pub const CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE: c_int = 21;
pub const enum_CUpointer_attribute_enum = c_uint;
pub const CUpointer_attribute = enum_CUpointer_attribute_enum;
pub const CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 0;
pub const CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: c_int = 1;
pub const CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: c_int = 2;
pub const CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: c_int = 3;
pub const CU_FUNC_ATTRIBUTE_NUM_REGS: c_int = 4;
pub const CU_FUNC_ATTRIBUTE_PTX_VERSION: c_int = 5;
pub const CU_FUNC_ATTRIBUTE_BINARY_VERSION: c_int = 6;
pub const CU_FUNC_ATTRIBUTE_CACHE_MODE_CA: c_int = 7;
pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;
pub const CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: c_int = 9;
pub const CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET: c_int = 10;
pub const CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH: c_int = 11;
pub const CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT: c_int = 12;
pub const CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH: c_int = 13;
pub const CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED: c_int = 14;
pub const CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE: c_int = 15;
pub const CU_FUNC_ATTRIBUTE_MAX: c_int = 16;
pub const enum_CUfunction_attribute_enum = c_uint;
pub const CUfunction_attribute = enum_CUfunction_attribute_enum;
pub const CU_FUNC_CACHE_PREFER_NONE: c_int = 0;
pub const CU_FUNC_CACHE_PREFER_SHARED: c_int = 1;
pub const CU_FUNC_CACHE_PREFER_L1: c_int = 2;
pub const CU_FUNC_CACHE_PREFER_EQUAL: c_int = 3;
pub const enum_CUfunc_cache_enum = c_uint;
pub const CUfunc_cache = enum_CUfunc_cache_enum;
pub const CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE: c_int = 0;
pub const CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE: c_int = 1;
pub const CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: c_int = 2;
pub const enum_CUsharedconfig_enum = c_uint;
pub const CUsharedconfig = enum_CUsharedconfig_enum;
pub const CU_SHAREDMEM_CARVEOUT_DEFAULT: c_int = -1;
pub const CU_SHAREDMEM_CARVEOUT_MAX_SHARED: c_int = 100;
pub const CU_SHAREDMEM_CARVEOUT_MAX_L1: c_int = 0;
pub const enum_CUshared_carveout_enum = c_int;
pub const CUshared_carveout = enum_CUshared_carveout_enum;
pub const CU_MEMORYTYPE_HOST: c_int = 1;
pub const CU_MEMORYTYPE_DEVICE: c_int = 2;
pub const CU_MEMORYTYPE_ARRAY: c_int = 3;
pub const CU_MEMORYTYPE_UNIFIED: c_int = 4;
pub const enum_CUmemorytype_enum = c_uint;
pub const CUmemorytype = enum_CUmemorytype_enum;
pub const CU_COMPUTEMODE_DEFAULT: c_int = 0;
pub const CU_COMPUTEMODE_PROHIBITED: c_int = 2;
pub const CU_COMPUTEMODE_EXCLUSIVE_PROCESS: c_int = 3;
pub const enum_CUcomputemode_enum = c_uint;
pub const CUcomputemode = enum_CUcomputemode_enum;
pub const CU_MEM_ADVISE_SET_READ_MOSTLY: c_int = 1;
pub const CU_MEM_ADVISE_UNSET_READ_MOSTLY: c_int = 2;
pub const CU_MEM_ADVISE_SET_PREFERRED_LOCATION: c_int = 3;
pub const CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION: c_int = 4;
pub const CU_MEM_ADVISE_SET_ACCESSED_BY: c_int = 5;
pub const CU_MEM_ADVISE_UNSET_ACCESSED_BY: c_int = 6;
pub const enum_CUmem_advise_enum = c_uint;
pub const CUmem_advise = enum_CUmem_advise_enum;
pub const CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY: c_int = 1;
pub const CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION: c_int = 2;
pub const CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY: c_int = 3;
pub const CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION: c_int = 4;
pub const CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_TYPE: c_int = 5;
pub const CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_ID: c_int = 6;
pub const CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_TYPE: c_int = 7;
pub const CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_ID: c_int = 8;
pub const enum_CUmem_range_attribute_enum = c_uint;
pub const CUmem_range_attribute = enum_CUmem_range_attribute_enum;
pub const CU_JIT_MAX_REGISTERS: c_int = 0;
pub const CU_JIT_THREADS_PER_BLOCK: c_int = 1;
pub const CU_JIT_WALL_TIME: c_int = 2;
pub const CU_JIT_INFO_LOG_BUFFER: c_int = 3;
pub const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: c_int = 4;
pub const CU_JIT_ERROR_LOG_BUFFER: c_int = 5;
pub const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_int = 6;
pub const CU_JIT_OPTIMIZATION_LEVEL: c_int = 7;
pub const CU_JIT_TARGET_FROM_CUCONTEXT: c_int = 8;
pub const CU_JIT_TARGET: c_int = 9;
pub const CU_JIT_FALLBACK_STRATEGY: c_int = 10;
pub const CU_JIT_GENERATE_DEBUG_INFO: c_int = 11;
pub const CU_JIT_LOG_VERBOSE: c_int = 12;
pub const CU_JIT_GENERATE_LINE_INFO: c_int = 13;
pub const CU_JIT_CACHE_MODE: c_int = 14;
pub const CU_JIT_NEW_SM3X_OPT: c_int = 15;
pub const CU_JIT_FAST_COMPILE: c_int = 16;
pub const CU_JIT_GLOBAL_SYMBOL_NAMES: c_int = 17;
pub const CU_JIT_GLOBAL_SYMBOL_ADDRESSES: c_int = 18;
pub const CU_JIT_GLOBAL_SYMBOL_COUNT: c_int = 19;
pub const CU_JIT_LTO: c_int = 20;
pub const CU_JIT_FTZ: c_int = 21;
pub const CU_JIT_PREC_DIV: c_int = 22;
pub const CU_JIT_PREC_SQRT: c_int = 23;
pub const CU_JIT_FMA: c_int = 24;
pub const CU_JIT_REFERENCED_KERNEL_NAMES: c_int = 25;
pub const CU_JIT_REFERENCED_KERNEL_COUNT: c_int = 26;
pub const CU_JIT_REFERENCED_VARIABLE_NAMES: c_int = 27;
pub const CU_JIT_REFERENCED_VARIABLE_COUNT: c_int = 28;
pub const CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES: c_int = 29;
pub const CU_JIT_POSITION_INDEPENDENT_CODE: c_int = 30;
pub const CU_JIT_MIN_CTA_PER_SM: c_int = 31;
pub const CU_JIT_MAX_THREADS_PER_BLOCK: c_int = 32;
pub const CU_JIT_OVERRIDE_DIRECTIVE_VALUES: c_int = 33;
pub const CU_JIT_NUM_OPTIONS: c_int = 34;
pub const enum_CUjit_option_enum = c_uint;
pub const CUjit_option = enum_CUjit_option_enum;
pub const CU_TARGET_COMPUTE_30: c_int = 30;
pub const CU_TARGET_COMPUTE_32: c_int = 32;
pub const CU_TARGET_COMPUTE_35: c_int = 35;
pub const CU_TARGET_COMPUTE_37: c_int = 37;
pub const CU_TARGET_COMPUTE_50: c_int = 50;
pub const CU_TARGET_COMPUTE_52: c_int = 52;
pub const CU_TARGET_COMPUTE_53: c_int = 53;
pub const CU_TARGET_COMPUTE_60: c_int = 60;
pub const CU_TARGET_COMPUTE_61: c_int = 61;
pub const CU_TARGET_COMPUTE_62: c_int = 62;
pub const CU_TARGET_COMPUTE_70: c_int = 70;
pub const CU_TARGET_COMPUTE_72: c_int = 72;
pub const CU_TARGET_COMPUTE_75: c_int = 75;
pub const CU_TARGET_COMPUTE_80: c_int = 80;
pub const CU_TARGET_COMPUTE_86: c_int = 86;
pub const CU_TARGET_COMPUTE_87: c_int = 87;
pub const CU_TARGET_COMPUTE_89: c_int = 89;
pub const CU_TARGET_COMPUTE_90: c_int = 90;
pub const CU_TARGET_COMPUTE_100: c_int = 100;
pub const CU_TARGET_COMPUTE_101: c_int = 101;
pub const CU_TARGET_COMPUTE_120: c_int = 120;
pub const CU_TARGET_COMPUTE_90A: c_int = 65626;
pub const CU_TARGET_COMPUTE_100A: c_int = 65636;
pub const CU_TARGET_COMPUTE_101A: c_int = 65637;
pub const CU_TARGET_COMPUTE_120A: c_int = 65656;
pub const enum_CUjit_target_enum = c_uint;
pub const CUjit_target = enum_CUjit_target_enum;
pub const CU_PREFER_PTX: c_int = 0;
pub const CU_PREFER_BINARY: c_int = 1;
pub const enum_CUjit_fallback_enum = c_uint;
pub const CUjit_fallback = enum_CUjit_fallback_enum;
pub const CU_JIT_CACHE_OPTION_NONE: c_int = 0;
pub const CU_JIT_CACHE_OPTION_CG: c_int = 1;
pub const CU_JIT_CACHE_OPTION_CA: c_int = 2;
pub const enum_CUjit_cacheMode_enum = c_uint;
pub const CUjit_cacheMode = enum_CUjit_cacheMode_enum;
pub const CU_JIT_INPUT_CUBIN: c_int = 0;
pub const CU_JIT_INPUT_PTX: c_int = 1;
pub const CU_JIT_INPUT_FATBINARY: c_int = 2;
pub const CU_JIT_INPUT_OBJECT: c_int = 3;
pub const CU_JIT_INPUT_LIBRARY: c_int = 4;
pub const CU_JIT_INPUT_NVVM: c_int = 5;
pub const CU_JIT_NUM_INPUT_TYPES: c_int = 6;
pub const enum_CUjitInputType_enum = c_uint;
pub const CUjitInputType = enum_CUjitInputType_enum;
pub const struct_CUlinkState_st = opaque {};
pub const CUlinkState = ?*struct_CUlinkState_st;
pub const CU_GRAPHICS_REGISTER_FLAGS_NONE: c_int = 0;
pub const CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY: c_int = 1;
pub const CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD: c_int = 2;
pub const CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST: c_int = 4;
pub const CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER: c_int = 8;
pub const enum_CUgraphicsRegisterFlags_enum = c_uint;
pub const CUgraphicsRegisterFlags = enum_CUgraphicsRegisterFlags_enum;
pub const CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE: c_int = 0;
pub const CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY: c_int = 1;
pub const CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD: c_int = 2;
pub const enum_CUgraphicsMapResourceFlags_enum = c_uint;
pub const CUgraphicsMapResourceFlags = enum_CUgraphicsMapResourceFlags_enum;
pub const CU_CUBEMAP_FACE_POSITIVE_X: c_int = 0;
pub const CU_CUBEMAP_FACE_NEGATIVE_X: c_int = 1;
pub const CU_CUBEMAP_FACE_POSITIVE_Y: c_int = 2;
pub const CU_CUBEMAP_FACE_NEGATIVE_Y: c_int = 3;
pub const CU_CUBEMAP_FACE_POSITIVE_Z: c_int = 4;
pub const CU_CUBEMAP_FACE_NEGATIVE_Z: c_int = 5;
pub const enum_CUarray_cubemap_face_enum = c_uint;
pub const CUarray_cubemap_face = enum_CUarray_cubemap_face_enum;
pub const CU_LIMIT_STACK_SIZE: c_int = 0;
pub const CU_LIMIT_PRINTF_FIFO_SIZE: c_int = 1;
pub const CU_LIMIT_MALLOC_HEAP_SIZE: c_int = 2;
pub const CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH: c_int = 3;
pub const CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT: c_int = 4;
pub const CU_LIMIT_MAX_L2_FETCH_GRANULARITY: c_int = 5;
pub const CU_LIMIT_PERSISTING_L2_CACHE_SIZE: c_int = 6;
pub const CU_LIMIT_SHMEM_SIZE: c_int = 7;
pub const CU_LIMIT_CIG_ENABLED: c_int = 8;
pub const CU_LIMIT_CIG_SHMEM_FALLBACK_ENABLED: c_int = 9;
pub const CU_LIMIT_MAX: c_int = 10;
pub const enum_CUlimit_enum = c_uint;
pub const CUlimit = enum_CUlimit_enum;
pub const CU_RESOURCE_TYPE_ARRAY: c_int = 0;
pub const CU_RESOURCE_TYPE_MIPMAPPED_ARRAY: c_int = 1;
pub const CU_RESOURCE_TYPE_LINEAR: c_int = 2;
pub const CU_RESOURCE_TYPE_PITCH2D: c_int = 3;
pub const enum_CUresourcetype_enum = c_uint;
pub const CUresourcetype = enum_CUresourcetype_enum;
pub const CUhostFn = ?*const fn (?*anyopaque) callconv(.c) void;
pub const CU_ACCESS_PROPERTY_NORMAL: c_int = 0;
pub const CU_ACCESS_PROPERTY_STREAMING: c_int = 1;
pub const CU_ACCESS_PROPERTY_PERSISTING: c_int = 2;
pub const enum_CUaccessProperty_enum = c_uint;
pub const CUaccessProperty = enum_CUaccessProperty_enum;
pub const struct_CUaccessPolicyWindow_st = extern struct {
    base_ptr: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    num_bytes: usize = @import("std").mem.zeroes(usize),
    hitRatio: f32 = @import("std").mem.zeroes(f32),
    hitProp: CUaccessProperty = @import("std").mem.zeroes(CUaccessProperty),
    missProp: CUaccessProperty = @import("std").mem.zeroes(CUaccessProperty),
};
pub const CUaccessPolicyWindow_v1 = struct_CUaccessPolicyWindow_st;
pub const CUaccessPolicyWindow = CUaccessPolicyWindow_v1;
pub const struct_CUDA_KERNEL_NODE_PARAMS_st = extern struct {
    func: CUfunction = @import("std").mem.zeroes(CUfunction),
    gridDimX: c_uint = @import("std").mem.zeroes(c_uint),
    gridDimY: c_uint = @import("std").mem.zeroes(c_uint),
    gridDimZ: c_uint = @import("std").mem.zeroes(c_uint),
    blockDimX: c_uint = @import("std").mem.zeroes(c_uint),
    blockDimY: c_uint = @import("std").mem.zeroes(c_uint),
    blockDimZ: c_uint = @import("std").mem.zeroes(c_uint),
    sharedMemBytes: c_uint = @import("std").mem.zeroes(c_uint),
    kernelParams: [*c]?*anyopaque = @import("std").mem.zeroes([*c]?*anyopaque),
    extra: [*c]?*anyopaque = @import("std").mem.zeroes([*c]?*anyopaque),
};
pub const CUDA_KERNEL_NODE_PARAMS_v1 = struct_CUDA_KERNEL_NODE_PARAMS_st;
pub const struct_CUDA_KERNEL_NODE_PARAMS_v2_st = extern struct {
    func: CUfunction = @import("std").mem.zeroes(CUfunction),
    gridDimX: c_uint = @import("std").mem.zeroes(c_uint),
    gridDimY: c_uint = @import("std").mem.zeroes(c_uint),
    gridDimZ: c_uint = @import("std").mem.zeroes(c_uint),
    blockDimX: c_uint = @import("std").mem.zeroes(c_uint),
    blockDimY: c_uint = @import("std").mem.zeroes(c_uint),
    blockDimZ: c_uint = @import("std").mem.zeroes(c_uint),
    sharedMemBytes: c_uint = @import("std").mem.zeroes(c_uint),
    kernelParams: [*c]?*anyopaque = @import("std").mem.zeroes([*c]?*anyopaque),
    extra: [*c]?*anyopaque = @import("std").mem.zeroes([*c]?*anyopaque),
    kern: CUkernel = @import("std").mem.zeroes(CUkernel),
    ctx: CUcontext = @import("std").mem.zeroes(CUcontext),
};
pub const CUDA_KERNEL_NODE_PARAMS_v2 = struct_CUDA_KERNEL_NODE_PARAMS_v2_st;
pub const CUDA_KERNEL_NODE_PARAMS = CUDA_KERNEL_NODE_PARAMS_v2;
pub const struct_CUDA_KERNEL_NODE_PARAMS_v3_st = extern struct {
    func: CUfunction = @import("std").mem.zeroes(CUfunction),
    gridDimX: c_uint = @import("std").mem.zeroes(c_uint),
    gridDimY: c_uint = @import("std").mem.zeroes(c_uint),
    gridDimZ: c_uint = @import("std").mem.zeroes(c_uint),
    blockDimX: c_uint = @import("std").mem.zeroes(c_uint),
    blockDimY: c_uint = @import("std").mem.zeroes(c_uint),
    blockDimZ: c_uint = @import("std").mem.zeroes(c_uint),
    sharedMemBytes: c_uint = @import("std").mem.zeroes(c_uint),
    kernelParams: [*c]?*anyopaque = @import("std").mem.zeroes([*c]?*anyopaque),
    extra: [*c]?*anyopaque = @import("std").mem.zeroes([*c]?*anyopaque),
    kern: CUkernel = @import("std").mem.zeroes(CUkernel),
    ctx: CUcontext = @import("std").mem.zeroes(CUcontext),
};
pub const CUDA_KERNEL_NODE_PARAMS_v3 = struct_CUDA_KERNEL_NODE_PARAMS_v3_st;
pub const struct_CUDA_MEMSET_NODE_PARAMS_st = extern struct {
    dst: CUdeviceptr = @import("std").mem.zeroes(CUdeviceptr),
    pitch: usize = @import("std").mem.zeroes(usize),
    value: c_uint = @import("std").mem.zeroes(c_uint),
    elementSize: c_uint = @import("std").mem.zeroes(c_uint),
    width: usize = @import("std").mem.zeroes(usize),
    height: usize = @import("std").mem.zeroes(usize),
};
pub const CUDA_MEMSET_NODE_PARAMS_v1 = struct_CUDA_MEMSET_NODE_PARAMS_st;
pub const CUDA_MEMSET_NODE_PARAMS = CUDA_MEMSET_NODE_PARAMS_v1;
pub const struct_CUDA_MEMSET_NODE_PARAMS_v2_st = extern struct {
    dst: CUdeviceptr = @import("std").mem.zeroes(CUdeviceptr),
    pitch: usize = @import("std").mem.zeroes(usize),
    value: c_uint = @import("std").mem.zeroes(c_uint),
    elementSize: c_uint = @import("std").mem.zeroes(c_uint),
    width: usize = @import("std").mem.zeroes(usize),
    height: usize = @import("std").mem.zeroes(usize),
    ctx: CUcontext = @import("std").mem.zeroes(CUcontext),
};
pub const CUDA_MEMSET_NODE_PARAMS_v2 = struct_CUDA_MEMSET_NODE_PARAMS_v2_st;
pub const struct_CUDA_HOST_NODE_PARAMS_st = extern struct {
    @"fn": CUhostFn = @import("std").mem.zeroes(CUhostFn),
    userData: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
};
pub const CUDA_HOST_NODE_PARAMS_v1 = struct_CUDA_HOST_NODE_PARAMS_st;
pub const CUDA_HOST_NODE_PARAMS = CUDA_HOST_NODE_PARAMS_v1;
pub const struct_CUDA_HOST_NODE_PARAMS_v2_st = extern struct {
    @"fn": CUhostFn = @import("std").mem.zeroes(CUhostFn),
    userData: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
};
pub const CUDA_HOST_NODE_PARAMS_v2 = struct_CUDA_HOST_NODE_PARAMS_v2_st;
pub const CU_GRAPH_COND_TYPE_IF: c_int = 0;
pub const CU_GRAPH_COND_TYPE_WHILE: c_int = 1;
pub const CU_GRAPH_COND_TYPE_SWITCH: c_int = 2;
pub const enum_CUgraphConditionalNodeType_enum = c_uint;
pub const CUgraphConditionalNodeType = enum_CUgraphConditionalNodeType_enum;
pub const struct_CUDA_CONDITIONAL_NODE_PARAMS = extern struct {
    handle: CUgraphConditionalHandle = @import("std").mem.zeroes(CUgraphConditionalHandle),
    type: CUgraphConditionalNodeType = @import("std").mem.zeroes(CUgraphConditionalNodeType),
    size: c_uint = @import("std").mem.zeroes(c_uint),
    phGraph_out: [*c]CUgraph = @import("std").mem.zeroes([*c]CUgraph),
    ctx: CUcontext = @import("std").mem.zeroes(CUcontext),
};
pub const CUDA_CONDITIONAL_NODE_PARAMS = struct_CUDA_CONDITIONAL_NODE_PARAMS;
pub const CU_GRAPH_NODE_TYPE_KERNEL: c_int = 0;
pub const CU_GRAPH_NODE_TYPE_MEMCPY: c_int = 1;
pub const CU_GRAPH_NODE_TYPE_MEMSET: c_int = 2;
pub const CU_GRAPH_NODE_TYPE_HOST: c_int = 3;
pub const CU_GRAPH_NODE_TYPE_GRAPH: c_int = 4;
pub const CU_GRAPH_NODE_TYPE_EMPTY: c_int = 5;
pub const CU_GRAPH_NODE_TYPE_WAIT_EVENT: c_int = 6;
pub const CU_GRAPH_NODE_TYPE_EVENT_RECORD: c_int = 7;
pub const CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL: c_int = 8;
pub const CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT: c_int = 9;
pub const CU_GRAPH_NODE_TYPE_MEM_ALLOC: c_int = 10;
pub const CU_GRAPH_NODE_TYPE_MEM_FREE: c_int = 11;
pub const CU_GRAPH_NODE_TYPE_BATCH_MEM_OP: c_int = 12;
pub const CU_GRAPH_NODE_TYPE_CONDITIONAL: c_int = 13;
pub const enum_CUgraphNodeType_enum = c_uint;
pub const CUgraphNodeType = enum_CUgraphNodeType_enum;
pub const CU_GRAPH_DEPENDENCY_TYPE_DEFAULT: c_int = 0;
pub const CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC: c_int = 1;
pub const enum_CUgraphDependencyType_enum = c_uint;
pub const CUgraphDependencyType = enum_CUgraphDependencyType_enum;
pub const struct_CUgraphEdgeData_st = extern struct {
    from_port: u8 = @import("std").mem.zeroes(u8),
    to_port: u8 = @import("std").mem.zeroes(u8),
    type: u8 = @import("std").mem.zeroes(u8),
    reserved: [5]u8 = @import("std").mem.zeroes([5]u8),
};
pub const CUgraphEdgeData = struct_CUgraphEdgeData_st;
pub const CUDA_GRAPH_INSTANTIATE_SUCCESS: c_int = 0;
pub const CUDA_GRAPH_INSTANTIATE_ERROR: c_int = 1;
pub const CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE: c_int = 2;
pub const CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED: c_int = 3;
pub const CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED: c_int = 4;
pub const CUDA_GRAPH_INSTANTIATE_CONDITIONAL_HANDLE_UNUSED: c_int = 5;
pub const enum_CUgraphInstantiateResult_enum = c_uint;
pub const CUgraphInstantiateResult = enum_CUgraphInstantiateResult_enum;
pub const struct_CUDA_GRAPH_INSTANTIATE_PARAMS_st = extern struct {
    flags: cuuint64_t = @import("std").mem.zeroes(cuuint64_t),
    hUploadStream: CUstream = @import("std").mem.zeroes(CUstream),
    hErrNode_out: CUgraphNode = @import("std").mem.zeroes(CUgraphNode),
    result_out: CUgraphInstantiateResult = @import("std").mem.zeroes(CUgraphInstantiateResult),
};
pub const CUDA_GRAPH_INSTANTIATE_PARAMS = struct_CUDA_GRAPH_INSTANTIATE_PARAMS_st;
pub const CU_SYNC_POLICY_AUTO: c_int = 1;
pub const CU_SYNC_POLICY_SPIN: c_int = 2;
pub const CU_SYNC_POLICY_YIELD: c_int = 3;
pub const CU_SYNC_POLICY_BLOCKING_SYNC: c_int = 4;
pub const enum_CUsynchronizationPolicy_enum = c_uint;
pub const CUsynchronizationPolicy = enum_CUsynchronizationPolicy_enum;
pub const CU_CLUSTER_SCHEDULING_POLICY_DEFAULT: c_int = 0;
pub const CU_CLUSTER_SCHEDULING_POLICY_SPREAD: c_int = 1;
pub const CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING: c_int = 2;
pub const enum_CUclusterSchedulingPolicy_enum = c_uint;
pub const CUclusterSchedulingPolicy = enum_CUclusterSchedulingPolicy_enum;
pub const CU_LAUNCH_MEM_SYNC_DOMAIN_DEFAULT: c_int = 0;
pub const CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE: c_int = 1;
pub const enum_CUlaunchMemSyncDomain_enum = c_uint;
pub const CUlaunchMemSyncDomain = enum_CUlaunchMemSyncDomain_enum;
pub const struct_CUlaunchMemSyncDomainMap_st = extern struct {
    default_: u8 = @import("std").mem.zeroes(u8),
    remote: u8 = @import("std").mem.zeroes(u8),
};
pub const CUlaunchMemSyncDomainMap = struct_CUlaunchMemSyncDomainMap_st;
pub const CU_LAUNCH_ATTRIBUTE_IGNORE: c_int = 0;
pub const CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW: c_int = 1;
pub const CU_LAUNCH_ATTRIBUTE_COOPERATIVE: c_int = 2;
pub const CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY: c_int = 3;
pub const CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION: c_int = 4;
pub const CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE: c_int = 5;
pub const CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION: c_int = 6;
pub const CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT: c_int = 7;
pub const CU_LAUNCH_ATTRIBUTE_PRIORITY: c_int = 8;
pub const CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP: c_int = 9;
pub const CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN: c_int = 10;
pub const CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION: c_int = 11;
pub const CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT: c_int = 12;
pub const CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE: c_int = 13;
pub const CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: c_int = 14;
pub const enum_CUlaunchAttributeID_enum = c_uint;
pub const CUlaunchAttributeID = enum_CUlaunchAttributeID_enum;
const struct_unnamed_9 = extern struct {
    x: c_uint = @import("std").mem.zeroes(c_uint),
    y: c_uint = @import("std").mem.zeroes(c_uint),
    z: c_uint = @import("std").mem.zeroes(c_uint),
};
const struct_unnamed_10 = extern struct {
    event: CUevent = @import("std").mem.zeroes(CUevent),
    flags: c_int = @import("std").mem.zeroes(c_int),
    triggerAtBlockStart: c_int = @import("std").mem.zeroes(c_int),
};
const struct_unnamed_11 = extern struct {
    event: CUevent = @import("std").mem.zeroes(CUevent),
    flags: c_int = @import("std").mem.zeroes(c_int),
};
const struct_unnamed_12 = extern struct {
    x: c_uint = @import("std").mem.zeroes(c_uint),
    y: c_uint = @import("std").mem.zeroes(c_uint),
    z: c_uint = @import("std").mem.zeroes(c_uint),
};
const struct_unnamed_13 = extern struct {
    deviceUpdatable: c_int = @import("std").mem.zeroes(c_int),
    devNode: CUgraphDeviceNode = @import("std").mem.zeroes(CUgraphDeviceNode),
};
pub const union_CUlaunchAttributeValue_union = extern union {
    pad: [64]u8,
    accessPolicyWindow: CUaccessPolicyWindow,
    cooperative: c_int,
    syncPolicy: CUsynchronizationPolicy,
    clusterDim: struct_unnamed_9,
    clusterSchedulingPolicyPreference: CUclusterSchedulingPolicy,
    programmaticStreamSerializationAllowed: c_int,
    programmaticEvent: struct_unnamed_10,
    launchCompletionEvent: struct_unnamed_11,
    priority: c_int,
    memSyncDomainMap: CUlaunchMemSyncDomainMap,
    memSyncDomain: CUlaunchMemSyncDomain,
    preferredClusterDim: struct_unnamed_12,
    deviceUpdatableKernelNode: struct_unnamed_13,
    sharedMemCarveout: c_uint,
};
pub const CUlaunchAttributeValue = union_CUlaunchAttributeValue_union;
pub const struct_CUlaunchAttribute_st = extern struct {
    id: CUlaunchAttributeID = @import("std").mem.zeroes(CUlaunchAttributeID),
    pad: [4]u8 = @import("std").mem.zeroes([4]u8),
    value: CUlaunchAttributeValue = @import("std").mem.zeroes(CUlaunchAttributeValue),
};
pub const CUlaunchAttribute = struct_CUlaunchAttribute_st;
pub const struct_CUlaunchConfig_st = extern struct {
    gridDimX: c_uint = @import("std").mem.zeroes(c_uint),
    gridDimY: c_uint = @import("std").mem.zeroes(c_uint),
    gridDimZ: c_uint = @import("std").mem.zeroes(c_uint),
    blockDimX: c_uint = @import("std").mem.zeroes(c_uint),
    blockDimY: c_uint = @import("std").mem.zeroes(c_uint),
    blockDimZ: c_uint = @import("std").mem.zeroes(c_uint),
    sharedMemBytes: c_uint = @import("std").mem.zeroes(c_uint),
    hStream: CUstream = @import("std").mem.zeroes(CUstream),
    attrs: [*c]CUlaunchAttribute = @import("std").mem.zeroes([*c]CUlaunchAttribute),
    numAttrs: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const CUlaunchConfig = struct_CUlaunchConfig_st;
pub const CUkernelNodeAttrID = CUlaunchAttributeID;
pub const CUkernelNodeAttrValue_v1 = CUlaunchAttributeValue;
pub const CUkernelNodeAttrValue = CUkernelNodeAttrValue_v1;
pub const CU_STREAM_CAPTURE_STATUS_NONE: c_int = 0;
pub const CU_STREAM_CAPTURE_STATUS_ACTIVE: c_int = 1;
pub const CU_STREAM_CAPTURE_STATUS_INVALIDATED: c_int = 2;
pub const enum_CUstreamCaptureStatus_enum = c_uint;
pub const CUstreamCaptureStatus = enum_CUstreamCaptureStatus_enum;
pub const CU_STREAM_CAPTURE_MODE_GLOBAL: c_int = 0;
pub const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: c_int = 1;
pub const CU_STREAM_CAPTURE_MODE_RELAXED: c_int = 2;
pub const enum_CUstreamCaptureMode_enum = c_uint;
pub const CUstreamCaptureMode = enum_CUstreamCaptureMode_enum;
pub const CUstreamAttrID = CUlaunchAttributeID;
pub const CUstreamAttrValue_v1 = CUlaunchAttributeValue;
pub const CUstreamAttrValue = CUstreamAttrValue_v1;
pub const CU_GET_PROC_ADDRESS_DEFAULT: c_int = 0;
pub const CU_GET_PROC_ADDRESS_LEGACY_STREAM: c_int = 1;
pub const CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM: c_int = 2;
pub const enum_CUdriverProcAddress_flags_enum = c_uint;
pub const CUdriverProcAddress_flags = enum_CUdriverProcAddress_flags_enum;
pub const CU_GET_PROC_ADDRESS_SUCCESS: c_int = 0;
pub const CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND: c_int = 1;
pub const CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT: c_int = 2;
pub const enum_CUdriverProcAddressQueryResult_enum = c_uint;
pub const CUdriverProcAddressQueryResult = enum_CUdriverProcAddressQueryResult_enum;
pub const CU_EXEC_AFFINITY_TYPE_SM_COUNT: c_int = 0;
pub const CU_EXEC_AFFINITY_TYPE_MAX: c_int = 1;
pub const enum_CUexecAffinityType_enum = c_uint;
pub const CUexecAffinityType = enum_CUexecAffinityType_enum;
pub const struct_CUexecAffinitySmCount_st = extern struct {
    val: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const CUexecAffinitySmCount_v1 = struct_CUexecAffinitySmCount_st;
pub const CUexecAffinitySmCount = CUexecAffinitySmCount_v1;
const union_unnamed_14 = extern union {
    smCount: CUexecAffinitySmCount,
};
pub const struct_CUexecAffinityParam_st = extern struct {
    type: CUexecAffinityType = @import("std").mem.zeroes(CUexecAffinityType),
    param: union_unnamed_14 = @import("std").mem.zeroes(union_unnamed_14),
};
pub const CUexecAffinityParam_v1 = struct_CUexecAffinityParam_st;
pub const CUexecAffinityParam = CUexecAffinityParam_v1;
pub const CIG_DATA_TYPE_D3D12_COMMAND_QUEUE: c_int = 1;
pub const enum_CUcigDataType_enum = c_uint;
pub const CUcigDataType = enum_CUcigDataType_enum;
pub const struct_CUctxCigParam_st = extern struct {
    sharedDataType: CUcigDataType = @import("std").mem.zeroes(CUcigDataType),
    sharedData: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
};
pub const CUctxCigParam = struct_CUctxCigParam_st;
pub const struct_CUctxCreateParams_st = extern struct {
    execAffinityParams: [*c]CUexecAffinityParam = @import("std").mem.zeroes([*c]CUexecAffinityParam),
    numExecAffinityParams: c_int = @import("std").mem.zeroes(c_int),
    cigParams: [*c]CUctxCigParam = @import("std").mem.zeroes([*c]CUctxCigParam),
};
pub const CUctxCreateParams = struct_CUctxCreateParams_st;
pub const CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE: c_int = 0;
pub const CU_LIBRARY_BINARY_IS_PRESERVED: c_int = 1;
pub const CU_LIBRARY_NUM_OPTIONS: c_int = 2;
pub const enum_CUlibraryOption_enum = c_uint;
pub const CUlibraryOption = enum_CUlibraryOption_enum;
pub const struct_CUlibraryHostUniversalFunctionAndDataTable_st = extern struct {
    functionTable: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    functionWindowSize: usize = @import("std").mem.zeroes(usize),
    dataTable: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    dataWindowSize: usize = @import("std").mem.zeroes(usize),
};
pub const CUlibraryHostUniversalFunctionAndDataTable = struct_CUlibraryHostUniversalFunctionAndDataTable_st;
pub const CUDA_SUCCESS: c_int = 0;
pub const CUDA_ERROR_INVALID_VALUE: c_int = 1;
pub const CUDA_ERROR_OUT_OF_MEMORY: c_int = 2;
pub const CUDA_ERROR_NOT_INITIALIZED: c_int = 3;
pub const CUDA_ERROR_DEINITIALIZED: c_int = 4;
pub const CUDA_ERROR_PROFILER_DISABLED: c_int = 5;
pub const CUDA_ERROR_PROFILER_NOT_INITIALIZED: c_int = 6;
pub const CUDA_ERROR_PROFILER_ALREADY_STARTED: c_int = 7;
pub const CUDA_ERROR_PROFILER_ALREADY_STOPPED: c_int = 8;
pub const CUDA_ERROR_STUB_LIBRARY: c_int = 34;
pub const CUDA_ERROR_DEVICE_UNAVAILABLE: c_int = 46;
pub const CUDA_ERROR_NO_DEVICE: c_int = 100;
pub const CUDA_ERROR_INVALID_DEVICE: c_int = 101;
pub const CUDA_ERROR_DEVICE_NOT_LICENSED: c_int = 102;
pub const CUDA_ERROR_INVALID_IMAGE: c_int = 200;
pub const CUDA_ERROR_INVALID_CONTEXT: c_int = 201;
pub const CUDA_ERROR_CONTEXT_ALREADY_CURRENT: c_int = 202;
pub const CUDA_ERROR_MAP_FAILED: c_int = 205;
pub const CUDA_ERROR_UNMAP_FAILED: c_int = 206;
pub const CUDA_ERROR_ARRAY_IS_MAPPED: c_int = 207;
pub const CUDA_ERROR_ALREADY_MAPPED: c_int = 208;
pub const CUDA_ERROR_NO_BINARY_FOR_GPU: c_int = 209;
pub const CUDA_ERROR_ALREADY_ACQUIRED: c_int = 210;
pub const CUDA_ERROR_NOT_MAPPED: c_int = 211;
pub const CUDA_ERROR_NOT_MAPPED_AS_ARRAY: c_int = 212;
pub const CUDA_ERROR_NOT_MAPPED_AS_POINTER: c_int = 213;
pub const CUDA_ERROR_ECC_UNCORRECTABLE: c_int = 214;
pub const CUDA_ERROR_UNSUPPORTED_LIMIT: c_int = 215;
pub const CUDA_ERROR_CONTEXT_ALREADY_IN_USE: c_int = 216;
pub const CUDA_ERROR_PEER_ACCESS_UNSUPPORTED: c_int = 217;
pub const CUDA_ERROR_INVALID_PTX: c_int = 218;
pub const CUDA_ERROR_INVALID_GRAPHICS_CONTEXT: c_int = 219;
pub const CUDA_ERROR_NVLINK_UNCORRECTABLE: c_int = 220;
pub const CUDA_ERROR_JIT_COMPILER_NOT_FOUND: c_int = 221;
pub const CUDA_ERROR_UNSUPPORTED_PTX_VERSION: c_int = 222;
pub const CUDA_ERROR_JIT_COMPILATION_DISABLED: c_int = 223;
pub const CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY: c_int = 224;
pub const CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC: c_int = 225;
pub const CUDA_ERROR_CONTAINED: c_int = 226;
pub const CUDA_ERROR_INVALID_SOURCE: c_int = 300;
pub const CUDA_ERROR_FILE_NOT_FOUND: c_int = 301;
pub const CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: c_int = 302;
pub const CUDA_ERROR_SHARED_OBJECT_INIT_FAILED: c_int = 303;
pub const CUDA_ERROR_OPERATING_SYSTEM: c_int = 304;
pub const CUDA_ERROR_INVALID_HANDLE: c_int = 400;
pub const CUDA_ERROR_ILLEGAL_STATE: c_int = 401;
pub const CUDA_ERROR_LOSSY_QUERY: c_int = 402;
pub const CUDA_ERROR_NOT_FOUND: c_int = 500;
pub const CUDA_ERROR_NOT_READY: c_int = 600;
pub const CUDA_ERROR_ILLEGAL_ADDRESS: c_int = 700;
pub const CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: c_int = 701;
pub const CUDA_ERROR_LAUNCH_TIMEOUT: c_int = 702;
pub const CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: c_int = 703;
pub const CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: c_int = 704;
pub const CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: c_int = 705;
pub const CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: c_int = 708;
pub const CUDA_ERROR_CONTEXT_IS_DESTROYED: c_int = 709;
pub const CUDA_ERROR_ASSERT: c_int = 710;
pub const CUDA_ERROR_TOO_MANY_PEERS: c_int = 711;
pub const CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED: c_int = 712;
pub const CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED: c_int = 713;
pub const CUDA_ERROR_HARDWARE_STACK_ERROR: c_int = 714;
pub const CUDA_ERROR_ILLEGAL_INSTRUCTION: c_int = 715;
pub const CUDA_ERROR_MISALIGNED_ADDRESS: c_int = 716;
pub const CUDA_ERROR_INVALID_ADDRESS_SPACE: c_int = 717;
pub const CUDA_ERROR_INVALID_PC: c_int = 718;
pub const CUDA_ERROR_LAUNCH_FAILED: c_int = 719;
pub const CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE: c_int = 720;
pub const CUDA_ERROR_TENSOR_MEMORY_LEAK: c_int = 721;
pub const CUDA_ERROR_NOT_PERMITTED: c_int = 800;
pub const CUDA_ERROR_NOT_SUPPORTED: c_int = 801;
pub const CUDA_ERROR_SYSTEM_NOT_READY: c_int = 802;
pub const CUDA_ERROR_SYSTEM_DRIVER_MISMATCH: c_int = 803;
pub const CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: c_int = 804;
pub const CUDA_ERROR_MPS_CONNECTION_FAILED: c_int = 805;
pub const CUDA_ERROR_MPS_RPC_FAILURE: c_int = 806;
pub const CUDA_ERROR_MPS_SERVER_NOT_READY: c_int = 807;
pub const CUDA_ERROR_MPS_MAX_CLIENTS_REACHED: c_int = 808;
pub const CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED: c_int = 809;
pub const CUDA_ERROR_MPS_CLIENT_TERMINATED: c_int = 810;
pub const CUDA_ERROR_CDP_NOT_SUPPORTED: c_int = 811;
pub const CUDA_ERROR_CDP_VERSION_MISMATCH: c_int = 812;
pub const CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED: c_int = 900;
pub const CUDA_ERROR_STREAM_CAPTURE_INVALIDATED: c_int = 901;
pub const CUDA_ERROR_STREAM_CAPTURE_MERGE: c_int = 902;
pub const CUDA_ERROR_STREAM_CAPTURE_UNMATCHED: c_int = 903;
pub const CUDA_ERROR_STREAM_CAPTURE_UNJOINED: c_int = 904;
pub const CUDA_ERROR_STREAM_CAPTURE_ISOLATION: c_int = 905;
pub const CUDA_ERROR_STREAM_CAPTURE_IMPLICIT: c_int = 906;
pub const CUDA_ERROR_CAPTURED_EVENT: c_int = 907;
pub const CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD: c_int = 908;
pub const CUDA_ERROR_TIMEOUT: c_int = 909;
pub const CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE: c_int = 910;
pub const CUDA_ERROR_EXTERNAL_DEVICE: c_int = 911;
pub const CUDA_ERROR_INVALID_CLUSTER_SIZE: c_int = 912;
pub const CUDA_ERROR_FUNCTION_NOT_LOADED: c_int = 913;
pub const CUDA_ERROR_INVALID_RESOURCE_TYPE: c_int = 914;
pub const CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION: c_int = 915;
pub const CUDA_ERROR_KEY_ROTATION: c_int = 916;
pub const CUDA_ERROR_UNKNOWN: c_int = 999;
pub const enum_cudaError_enum = c_uint;
pub const CUresult = enum_cudaError_enum;
pub const CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK: c_int = 1;
pub const CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED: c_int = 2;
pub const CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED: c_int = 3;
pub const CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED: c_int = 4;
pub const CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED: c_int = 4;
pub const enum_CUdevice_P2PAttribute_enum = c_uint;
pub const CUdevice_P2PAttribute = enum_CUdevice_P2PAttribute_enum;
pub const CUstreamCallback = ?*const fn (CUstream, CUresult, ?*anyopaque) callconv(.c) void;
pub const CUoccupancyB2DSize = ?*const fn (c_int) callconv(.c) usize;
pub const struct_CUDA_MEMCPY2D_st = extern struct {
    srcXInBytes: usize = @import("std").mem.zeroes(usize),
    srcY: usize = @import("std").mem.zeroes(usize),
    srcMemoryType: CUmemorytype = @import("std").mem.zeroes(CUmemorytype),
    srcHost: ?*const anyopaque = @import("std").mem.zeroes(?*const anyopaque),
    srcDevice: CUdeviceptr = @import("std").mem.zeroes(CUdeviceptr),
    srcArray: CUarray = @import("std").mem.zeroes(CUarray),
    srcPitch: usize = @import("std").mem.zeroes(usize),
    dstXInBytes: usize = @import("std").mem.zeroes(usize),
    dstY: usize = @import("std").mem.zeroes(usize),
    dstMemoryType: CUmemorytype = @import("std").mem.zeroes(CUmemorytype),
    dstHost: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    dstDevice: CUdeviceptr = @import("std").mem.zeroes(CUdeviceptr),
    dstArray: CUarray = @import("std").mem.zeroes(CUarray),
    dstPitch: usize = @import("std").mem.zeroes(usize),
    WidthInBytes: usize = @import("std").mem.zeroes(usize),
    Height: usize = @import("std").mem.zeroes(usize),
};
pub const CUDA_MEMCPY2D_v2 = struct_CUDA_MEMCPY2D_st;
pub const CUDA_MEMCPY2D = CUDA_MEMCPY2D_v2;
pub const struct_CUDA_MEMCPY3D_st = extern struct {
    srcXInBytes: usize = @import("std").mem.zeroes(usize),
    srcY: usize = @import("std").mem.zeroes(usize),
    srcZ: usize = @import("std").mem.zeroes(usize),
    srcLOD: usize = @import("std").mem.zeroes(usize),
    srcMemoryType: CUmemorytype = @import("std").mem.zeroes(CUmemorytype),
    srcHost: ?*const anyopaque = @import("std").mem.zeroes(?*const anyopaque),
    srcDevice: CUdeviceptr = @import("std").mem.zeroes(CUdeviceptr),
    srcArray: CUarray = @import("std").mem.zeroes(CUarray),
    reserved0: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    srcPitch: usize = @import("std").mem.zeroes(usize),
    srcHeight: usize = @import("std").mem.zeroes(usize),
    dstXInBytes: usize = @import("std").mem.zeroes(usize),
    dstY: usize = @import("std").mem.zeroes(usize),
    dstZ: usize = @import("std").mem.zeroes(usize),
    dstLOD: usize = @import("std").mem.zeroes(usize),
    dstMemoryType: CUmemorytype = @import("std").mem.zeroes(CUmemorytype),
    dstHost: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    dstDevice: CUdeviceptr = @import("std").mem.zeroes(CUdeviceptr),
    dstArray: CUarray = @import("std").mem.zeroes(CUarray),
    reserved1: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    dstPitch: usize = @import("std").mem.zeroes(usize),
    dstHeight: usize = @import("std").mem.zeroes(usize),
    WidthInBytes: usize = @import("std").mem.zeroes(usize),
    Height: usize = @import("std").mem.zeroes(usize),
    Depth: usize = @import("std").mem.zeroes(usize),
};
pub const CUDA_MEMCPY3D_v2 = struct_CUDA_MEMCPY3D_st;
pub const CUDA_MEMCPY3D = CUDA_MEMCPY3D_v2;
pub const struct_CUDA_MEMCPY3D_PEER_st = extern struct {
    srcXInBytes: usize = @import("std").mem.zeroes(usize),
    srcY: usize = @import("std").mem.zeroes(usize),
    srcZ: usize = @import("std").mem.zeroes(usize),
    srcLOD: usize = @import("std").mem.zeroes(usize),
    srcMemoryType: CUmemorytype = @import("std").mem.zeroes(CUmemorytype),
    srcHost: ?*const anyopaque = @import("std").mem.zeroes(?*const anyopaque),
    srcDevice: CUdeviceptr = @import("std").mem.zeroes(CUdeviceptr),
    srcArray: CUarray = @import("std").mem.zeroes(CUarray),
    srcContext: CUcontext = @import("std").mem.zeroes(CUcontext),
    srcPitch: usize = @import("std").mem.zeroes(usize),
    srcHeight: usize = @import("std").mem.zeroes(usize),
    dstXInBytes: usize = @import("std").mem.zeroes(usize),
    dstY: usize = @import("std").mem.zeroes(usize),
    dstZ: usize = @import("std").mem.zeroes(usize),
    dstLOD: usize = @import("std").mem.zeroes(usize),
    dstMemoryType: CUmemorytype = @import("std").mem.zeroes(CUmemorytype),
    dstHost: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    dstDevice: CUdeviceptr = @import("std").mem.zeroes(CUdeviceptr),
    dstArray: CUarray = @import("std").mem.zeroes(CUarray),
    dstContext: CUcontext = @import("std").mem.zeroes(CUcontext),
    dstPitch: usize = @import("std").mem.zeroes(usize),
    dstHeight: usize = @import("std").mem.zeroes(usize),
    WidthInBytes: usize = @import("std").mem.zeroes(usize),
    Height: usize = @import("std").mem.zeroes(usize),
    Depth: usize = @import("std").mem.zeroes(usize),
};
pub const CUDA_MEMCPY3D_PEER_v1 = struct_CUDA_MEMCPY3D_PEER_st;
pub const CUDA_MEMCPY3D_PEER = CUDA_MEMCPY3D_PEER_v1;
pub const struct_CUDA_MEMCPY_NODE_PARAMS_st = extern struct {
    flags: c_int = @import("std").mem.zeroes(c_int),
    reserved: c_int = @import("std").mem.zeroes(c_int),
    copyCtx: CUcontext = @import("std").mem.zeroes(CUcontext),
    copyParams: CUDA_MEMCPY3D = @import("std").mem.zeroes(CUDA_MEMCPY3D),
};
pub const CUDA_MEMCPY_NODE_PARAMS = struct_CUDA_MEMCPY_NODE_PARAMS_st;
pub const struct_CUDA_ARRAY_DESCRIPTOR_st = extern struct {
    Width: usize = @import("std").mem.zeroes(usize),
    Height: usize = @import("std").mem.zeroes(usize),
    Format: CUarray_format = @import("std").mem.zeroes(CUarray_format),
    NumChannels: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const CUDA_ARRAY_DESCRIPTOR_v2 = struct_CUDA_ARRAY_DESCRIPTOR_st;
pub const CUDA_ARRAY_DESCRIPTOR = CUDA_ARRAY_DESCRIPTOR_v2;
pub const struct_CUDA_ARRAY3D_DESCRIPTOR_st = extern struct {
    Width: usize = @import("std").mem.zeroes(usize),
    Height: usize = @import("std").mem.zeroes(usize),
    Depth: usize = @import("std").mem.zeroes(usize),
    Format: CUarray_format = @import("std").mem.zeroes(CUarray_format),
    NumChannels: c_uint = @import("std").mem.zeroes(c_uint),
    Flags: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const CUDA_ARRAY3D_DESCRIPTOR_v2 = struct_CUDA_ARRAY3D_DESCRIPTOR_st;
pub const CUDA_ARRAY3D_DESCRIPTOR = CUDA_ARRAY3D_DESCRIPTOR_v2;
const struct_unnamed_15 = extern struct {
    width: c_uint = @import("std").mem.zeroes(c_uint),
    height: c_uint = @import("std").mem.zeroes(c_uint),
    depth: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const struct_CUDA_ARRAY_SPARSE_PROPERTIES_st = extern struct {
    tileExtent: struct_unnamed_15 = @import("std").mem.zeroes(struct_unnamed_15),
    miptailFirstLevel: c_uint = @import("std").mem.zeroes(c_uint),
    miptailSize: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
    reserved: [4]c_uint = @import("std").mem.zeroes([4]c_uint),
};
pub const CUDA_ARRAY_SPARSE_PROPERTIES_v1 = struct_CUDA_ARRAY_SPARSE_PROPERTIES_st;
pub const CUDA_ARRAY_SPARSE_PROPERTIES = CUDA_ARRAY_SPARSE_PROPERTIES_v1;
pub const struct_CUDA_ARRAY_MEMORY_REQUIREMENTS_st = extern struct {
    size: usize = @import("std").mem.zeroes(usize),
    alignment: usize = @import("std").mem.zeroes(usize),
    reserved: [4]c_uint = @import("std").mem.zeroes([4]c_uint),
};
pub const CUDA_ARRAY_MEMORY_REQUIREMENTS_v1 = struct_CUDA_ARRAY_MEMORY_REQUIREMENTS_st;
pub const CUDA_ARRAY_MEMORY_REQUIREMENTS = CUDA_ARRAY_MEMORY_REQUIREMENTS_v1;
const struct_unnamed_17 = extern struct {
    hArray: CUarray = @import("std").mem.zeroes(CUarray),
};
const struct_unnamed_18 = extern struct {
    hMipmappedArray: CUmipmappedArray = @import("std").mem.zeroes(CUmipmappedArray),
};
const struct_unnamed_19 = extern struct {
    devPtr: CUdeviceptr = @import("std").mem.zeroes(CUdeviceptr),
    format: CUarray_format = @import("std").mem.zeroes(CUarray_format),
    numChannels: c_uint = @import("std").mem.zeroes(c_uint),
    sizeInBytes: usize = @import("std").mem.zeroes(usize),
};
const struct_unnamed_20 = extern struct {
    devPtr: CUdeviceptr = @import("std").mem.zeroes(CUdeviceptr),
    format: CUarray_format = @import("std").mem.zeroes(CUarray_format),
    numChannels: c_uint = @import("std").mem.zeroes(c_uint),
    width: usize = @import("std").mem.zeroes(usize),
    height: usize = @import("std").mem.zeroes(usize),
    pitchInBytes: usize = @import("std").mem.zeroes(usize),
};
const struct_unnamed_21 = extern struct {
    reserved: [32]c_int = @import("std").mem.zeroes([32]c_int),
};
const union_unnamed_16 = extern union {
    array: struct_unnamed_17,
    mipmap: struct_unnamed_18,
    linear: struct_unnamed_19,
    pitch2D: struct_unnamed_20,
    reserved: struct_unnamed_21,
};
pub const struct_CUDA_RESOURCE_DESC_st = extern struct {
    resType: CUresourcetype = @import("std").mem.zeroes(CUresourcetype),
    res: union_unnamed_16 = @import("std").mem.zeroes(union_unnamed_16),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const CUDA_RESOURCE_DESC_v1 = struct_CUDA_RESOURCE_DESC_st;
pub const CUDA_RESOURCE_DESC = CUDA_RESOURCE_DESC_v1;
pub const struct_CUDA_TEXTURE_DESC_st = extern struct {
    addressMode: [3]CUaddress_mode = @import("std").mem.zeroes([3]CUaddress_mode),
    filterMode: CUfilter_mode = @import("std").mem.zeroes(CUfilter_mode),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
    maxAnisotropy: c_uint = @import("std").mem.zeroes(c_uint),
    mipmapFilterMode: CUfilter_mode = @import("std").mem.zeroes(CUfilter_mode),
    mipmapLevelBias: f32 = @import("std").mem.zeroes(f32),
    minMipmapLevelClamp: f32 = @import("std").mem.zeroes(f32),
    maxMipmapLevelClamp: f32 = @import("std").mem.zeroes(f32),
    borderColor: [4]f32 = @import("std").mem.zeroes([4]f32),
    reserved: [12]c_int = @import("std").mem.zeroes([12]c_int),
};
pub const CUDA_TEXTURE_DESC_v1 = struct_CUDA_TEXTURE_DESC_st;
pub const CUDA_TEXTURE_DESC = CUDA_TEXTURE_DESC_v1;
pub const CU_RES_VIEW_FORMAT_NONE: c_int = 0;
pub const CU_RES_VIEW_FORMAT_UINT_1X8: c_int = 1;
pub const CU_RES_VIEW_FORMAT_UINT_2X8: c_int = 2;
pub const CU_RES_VIEW_FORMAT_UINT_4X8: c_int = 3;
pub const CU_RES_VIEW_FORMAT_SINT_1X8: c_int = 4;
pub const CU_RES_VIEW_FORMAT_SINT_2X8: c_int = 5;
pub const CU_RES_VIEW_FORMAT_SINT_4X8: c_int = 6;
pub const CU_RES_VIEW_FORMAT_UINT_1X16: c_int = 7;
pub const CU_RES_VIEW_FORMAT_UINT_2X16: c_int = 8;
pub const CU_RES_VIEW_FORMAT_UINT_4X16: c_int = 9;
pub const CU_RES_VIEW_FORMAT_SINT_1X16: c_int = 10;
pub const CU_RES_VIEW_FORMAT_SINT_2X16: c_int = 11;
pub const CU_RES_VIEW_FORMAT_SINT_4X16: c_int = 12;
pub const CU_RES_VIEW_FORMAT_UINT_1X32: c_int = 13;
pub const CU_RES_VIEW_FORMAT_UINT_2X32: c_int = 14;
pub const CU_RES_VIEW_FORMAT_UINT_4X32: c_int = 15;
pub const CU_RES_VIEW_FORMAT_SINT_1X32: c_int = 16;
pub const CU_RES_VIEW_FORMAT_SINT_2X32: c_int = 17;
pub const CU_RES_VIEW_FORMAT_SINT_4X32: c_int = 18;
pub const CU_RES_VIEW_FORMAT_FLOAT_1X16: c_int = 19;
pub const CU_RES_VIEW_FORMAT_FLOAT_2X16: c_int = 20;
pub const CU_RES_VIEW_FORMAT_FLOAT_4X16: c_int = 21;
pub const CU_RES_VIEW_FORMAT_FLOAT_1X32: c_int = 22;
pub const CU_RES_VIEW_FORMAT_FLOAT_2X32: c_int = 23;
pub const CU_RES_VIEW_FORMAT_FLOAT_4X32: c_int = 24;
pub const CU_RES_VIEW_FORMAT_UNSIGNED_BC1: c_int = 25;
pub const CU_RES_VIEW_FORMAT_UNSIGNED_BC2: c_int = 26;
pub const CU_RES_VIEW_FORMAT_UNSIGNED_BC3: c_int = 27;
pub const CU_RES_VIEW_FORMAT_UNSIGNED_BC4: c_int = 28;
pub const CU_RES_VIEW_FORMAT_SIGNED_BC4: c_int = 29;
pub const CU_RES_VIEW_FORMAT_UNSIGNED_BC5: c_int = 30;
pub const CU_RES_VIEW_FORMAT_SIGNED_BC5: c_int = 31;
pub const CU_RES_VIEW_FORMAT_UNSIGNED_BC6H: c_int = 32;
pub const CU_RES_VIEW_FORMAT_SIGNED_BC6H: c_int = 33;
pub const CU_RES_VIEW_FORMAT_UNSIGNED_BC7: c_int = 34;
pub const enum_CUresourceViewFormat_enum = c_uint;
pub const CUresourceViewFormat = enum_CUresourceViewFormat_enum;
pub const struct_CUDA_RESOURCE_VIEW_DESC_st = extern struct {
    format: CUresourceViewFormat = @import("std").mem.zeroes(CUresourceViewFormat),
    width: usize = @import("std").mem.zeroes(usize),
    height: usize = @import("std").mem.zeroes(usize),
    depth: usize = @import("std").mem.zeroes(usize),
    firstMipmapLevel: c_uint = @import("std").mem.zeroes(c_uint),
    lastMipmapLevel: c_uint = @import("std").mem.zeroes(c_uint),
    firstLayer: c_uint = @import("std").mem.zeroes(c_uint),
    lastLayer: c_uint = @import("std").mem.zeroes(c_uint),
    reserved: [16]c_uint = @import("std").mem.zeroes([16]c_uint),
};
pub const CUDA_RESOURCE_VIEW_DESC_v1 = struct_CUDA_RESOURCE_VIEW_DESC_st;
pub const CUDA_RESOURCE_VIEW_DESC = CUDA_RESOURCE_VIEW_DESC_v1;
pub const struct_CUtensorMap_st = extern struct {
    @"opaque": [16]cuuint64_t align(64) = @import("std").mem.zeroes([16]cuuint64_t),
};
pub const CUtensorMap = struct_CUtensorMap_st;
pub const CU_TENSOR_MAP_DATA_TYPE_UINT8: c_int = 0;
pub const CU_TENSOR_MAP_DATA_TYPE_UINT16: c_int = 1;
pub const CU_TENSOR_MAP_DATA_TYPE_UINT32: c_int = 2;
pub const CU_TENSOR_MAP_DATA_TYPE_INT32: c_int = 3;
pub const CU_TENSOR_MAP_DATA_TYPE_UINT64: c_int = 4;
pub const CU_TENSOR_MAP_DATA_TYPE_INT64: c_int = 5;
pub const CU_TENSOR_MAP_DATA_TYPE_FLOAT16: c_int = 6;
pub const CU_TENSOR_MAP_DATA_TYPE_FLOAT32: c_int = 7;
pub const CU_TENSOR_MAP_DATA_TYPE_FLOAT64: c_int = 8;
pub const CU_TENSOR_MAP_DATA_TYPE_BFLOAT16: c_int = 9;
pub const CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ: c_int = 10;
pub const CU_TENSOR_MAP_DATA_TYPE_TFLOAT32: c_int = 11;
pub const CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ: c_int = 12;
pub const CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B: c_int = 13;
pub const CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B: c_int = 14;
pub const CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B: c_int = 15;
pub const enum_CUtensorMapDataType_enum = c_uint;
pub const CUtensorMapDataType = enum_CUtensorMapDataType_enum;
pub const CU_TENSOR_MAP_INTERLEAVE_NONE: c_int = 0;
pub const CU_TENSOR_MAP_INTERLEAVE_16B: c_int = 1;
pub const CU_TENSOR_MAP_INTERLEAVE_32B: c_int = 2;
pub const enum_CUtensorMapInterleave_enum = c_uint;
pub const CUtensorMapInterleave = enum_CUtensorMapInterleave_enum;
pub const CU_TENSOR_MAP_SWIZZLE_NONE: c_int = 0;
pub const CU_TENSOR_MAP_SWIZZLE_32B: c_int = 1;
pub const CU_TENSOR_MAP_SWIZZLE_64B: c_int = 2;
pub const CU_TENSOR_MAP_SWIZZLE_128B: c_int = 3;
pub const CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B: c_int = 4;
pub const CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B: c_int = 5;
pub const CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B: c_int = 6;
pub const enum_CUtensorMapSwizzle_enum = c_uint;
pub const CUtensorMapSwizzle = enum_CUtensorMapSwizzle_enum;
pub const CU_TENSOR_MAP_L2_PROMOTION_NONE: c_int = 0;
pub const CU_TENSOR_MAP_L2_PROMOTION_L2_64B: c_int = 1;
pub const CU_TENSOR_MAP_L2_PROMOTION_L2_128B: c_int = 2;
pub const CU_TENSOR_MAP_L2_PROMOTION_L2_256B: c_int = 3;
pub const enum_CUtensorMapL2promotion_enum = c_uint;
pub const CUtensorMapL2promotion = enum_CUtensorMapL2promotion_enum;
pub const CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE: c_int = 0;
pub const CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA: c_int = 1;
pub const enum_CUtensorMapFloatOOBfill_enum = c_uint;
pub const CUtensorMapFloatOOBfill = enum_CUtensorMapFloatOOBfill_enum;
pub const CU_TENSOR_MAP_IM2COL_WIDE_MODE_W: c_int = 0;
pub const CU_TENSOR_MAP_IM2COL_WIDE_MODE_W128: c_int = 1;
pub const enum_CUtensorMapIm2ColWideMode_enum = c_uint;
pub const CUtensorMapIm2ColWideMode = enum_CUtensorMapIm2ColWideMode_enum;
pub const struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st = extern struct {
    p2pToken: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    vaSpaceToken: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1 = struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st;
pub const CUDA_POINTER_ATTRIBUTE_P2P_TOKENS = CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1;
pub const CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE: c_int = 0;
pub const CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ: c_int = 1;
pub const CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE: c_int = 3;
pub const enum_CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum = c_uint;
pub const CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS = enum_CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum;
pub const struct_CUDA_LAUNCH_PARAMS_st = extern struct {
    function: CUfunction = @import("std").mem.zeroes(CUfunction),
    gridDimX: c_uint = @import("std").mem.zeroes(c_uint),
    gridDimY: c_uint = @import("std").mem.zeroes(c_uint),
    gridDimZ: c_uint = @import("std").mem.zeroes(c_uint),
    blockDimX: c_uint = @import("std").mem.zeroes(c_uint),
    blockDimY: c_uint = @import("std").mem.zeroes(c_uint),
    blockDimZ: c_uint = @import("std").mem.zeroes(c_uint),
    sharedMemBytes: c_uint = @import("std").mem.zeroes(c_uint),
    hStream: CUstream = @import("std").mem.zeroes(CUstream),
    kernelParams: [*c]?*anyopaque = @import("std").mem.zeroes([*c]?*anyopaque),
};
pub const CUDA_LAUNCH_PARAMS_v1 = struct_CUDA_LAUNCH_PARAMS_st;
pub const CUDA_LAUNCH_PARAMS = CUDA_LAUNCH_PARAMS_v1;
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD: c_int = 1;
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32: c_int = 2;
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT: c_int = 3;
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP: c_int = 4;
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE: c_int = 5;
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE: c_int = 6;
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT: c_int = 7;
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF: c_int = 8;
pub const enum_CUexternalMemoryHandleType_enum = c_uint;
pub const CUexternalMemoryHandleType = enum_CUexternalMemoryHandleType_enum;
const struct_unnamed_23 = extern struct {
    handle: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    name: ?*const anyopaque = @import("std").mem.zeroes(?*const anyopaque),
};
const union_unnamed_22 = extern union {
    fd: c_int,
    win32: struct_unnamed_23,
    nvSciBufObject: ?*const anyopaque,
};
pub const struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st = extern struct {
    type: CUexternalMemoryHandleType = @import("std").mem.zeroes(CUexternalMemoryHandleType),
    handle: union_unnamed_22 = @import("std").mem.zeroes(union_unnamed_22),
    size: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
    reserved: [16]c_uint = @import("std").mem.zeroes([16]c_uint),
};
pub const CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 = struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st;
pub const CUDA_EXTERNAL_MEMORY_HANDLE_DESC = CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1;
pub const struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st = extern struct {
    offset: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    size: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
    reserved: [16]c_uint = @import("std").mem.zeroes([16]c_uint),
};
pub const CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1 = struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st;
pub const CUDA_EXTERNAL_MEMORY_BUFFER_DESC = CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1;
pub const struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st = extern struct {
    offset: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    arrayDesc: CUDA_ARRAY3D_DESCRIPTOR = @import("std").mem.zeroes(CUDA_ARRAY3D_DESCRIPTOR),
    numLevels: c_uint = @import("std").mem.zeroes(c_uint),
    reserved: [16]c_uint = @import("std").mem.zeroes([16]c_uint),
};
pub const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1 = struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st;
pub const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC = CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1;
pub const CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD: c_int = 1;
pub const CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32: c_int = 2;
pub const CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT: c_int = 3;
pub const CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE: c_int = 4;
pub const CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE: c_int = 5;
pub const CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC: c_int = 6;
pub const CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX: c_int = 7;
pub const CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT: c_int = 8;
pub const CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD: c_int = 9;
pub const CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32: c_int = 10;
pub const enum_CUexternalSemaphoreHandleType_enum = c_uint;
pub const CUexternalSemaphoreHandleType = enum_CUexternalSemaphoreHandleType_enum;
const struct_unnamed_25 = extern struct {
    handle: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    name: ?*const anyopaque = @import("std").mem.zeroes(?*const anyopaque),
};
const union_unnamed_24 = extern union {
    fd: c_int,
    win32: struct_unnamed_25,
    nvSciSyncObj: ?*const anyopaque,
};
pub const struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st = extern struct {
    type: CUexternalSemaphoreHandleType = @import("std").mem.zeroes(CUexternalSemaphoreHandleType),
    handle: union_unnamed_24 = @import("std").mem.zeroes(union_unnamed_24),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
    reserved: [16]c_uint = @import("std").mem.zeroes([16]c_uint),
};
pub const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1 = struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st;
pub const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC = CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1;
const struct_unnamed_27 = extern struct {
    value: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
};
const union_unnamed_28 = extern union {
    fence: ?*anyopaque,
    reserved: c_ulonglong,
};
const struct_unnamed_29 = extern struct {
    key: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
};
const struct_unnamed_26 = extern struct {
    fence: struct_unnamed_27 = @import("std").mem.zeroes(struct_unnamed_27),
    nvSciSync: union_unnamed_28 = @import("std").mem.zeroes(union_unnamed_28),
    keyedMutex: struct_unnamed_29 = @import("std").mem.zeroes(struct_unnamed_29),
    reserved: [12]c_uint = @import("std").mem.zeroes([12]c_uint),
};
pub const struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st = extern struct {
    params: struct_unnamed_26 = @import("std").mem.zeroes(struct_unnamed_26),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
    reserved: [16]c_uint = @import("std").mem.zeroes([16]c_uint),
};
pub const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1 = struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st;
pub const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS = CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1;
const struct_unnamed_31 = extern struct {
    value: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
};
const union_unnamed_32 = extern union {
    fence: ?*anyopaque,
    reserved: c_ulonglong,
};
const struct_unnamed_33 = extern struct {
    key: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    timeoutMs: c_uint = @import("std").mem.zeroes(c_uint),
};
const struct_unnamed_30 = extern struct {
    fence: struct_unnamed_31 = @import("std").mem.zeroes(struct_unnamed_31),
    nvSciSync: union_unnamed_32 = @import("std").mem.zeroes(union_unnamed_32),
    keyedMutex: struct_unnamed_33 = @import("std").mem.zeroes(struct_unnamed_33),
    reserved: [10]c_uint = @import("std").mem.zeroes([10]c_uint),
};
pub const struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st = extern struct {
    params: struct_unnamed_30 = @import("std").mem.zeroes(struct_unnamed_30),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
    reserved: [16]c_uint = @import("std").mem.zeroes([16]c_uint),
};
pub const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1 = struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st;
pub const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1;
pub const struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st = extern struct {
    extSemArray: [*c]CUexternalSemaphore = @import("std").mem.zeroes([*c]CUexternalSemaphore),
    paramsArray: [*c]const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS = @import("std").mem.zeroes([*c]const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS),
    numExtSems: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1 = struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st;
pub const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS = CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1;
pub const struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2_st = extern struct {
    extSemArray: [*c]CUexternalSemaphore = @import("std").mem.zeroes([*c]CUexternalSemaphore),
    paramsArray: [*c]const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS = @import("std").mem.zeroes([*c]const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS),
    numExtSems: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2 = struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2_st;
pub const struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st = extern struct {
    extSemArray: [*c]CUexternalSemaphore = @import("std").mem.zeroes([*c]CUexternalSemaphore),
    paramsArray: [*c]const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = @import("std").mem.zeroes([*c]const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS),
    numExtSems: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1 = struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st;
pub const CUDA_EXT_SEM_WAIT_NODE_PARAMS = CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1;
pub const struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2_st = extern struct {
    extSemArray: [*c]CUexternalSemaphore = @import("std").mem.zeroes([*c]CUexternalSemaphore),
    paramsArray: [*c]const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = @import("std").mem.zeroes([*c]const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS),
    numExtSems: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2 = struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2_st;
pub const CUmemGenericAllocationHandle_v1 = c_ulonglong;
pub const CUmemGenericAllocationHandle = CUmemGenericAllocationHandle_v1;
pub const CU_MEM_HANDLE_TYPE_NONE: c_int = 0;
pub const CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR: c_int = 1;
pub const CU_MEM_HANDLE_TYPE_WIN32: c_int = 2;
pub const CU_MEM_HANDLE_TYPE_WIN32_KMT: c_int = 4;
pub const CU_MEM_HANDLE_TYPE_FABRIC: c_int = 8;
pub const CU_MEM_HANDLE_TYPE_MAX: c_int = 2147483647;
pub const enum_CUmemAllocationHandleType_enum = c_uint;
pub const CUmemAllocationHandleType = enum_CUmemAllocationHandleType_enum;
pub const CU_MEM_ACCESS_FLAGS_PROT_NONE: c_int = 0;
pub const CU_MEM_ACCESS_FLAGS_PROT_READ: c_int = 1;
pub const CU_MEM_ACCESS_FLAGS_PROT_READWRITE: c_int = 3;
pub const CU_MEM_ACCESS_FLAGS_PROT_MAX: c_int = 2147483647;
pub const enum_CUmemAccess_flags_enum = c_uint;
pub const CUmemAccess_flags = enum_CUmemAccess_flags_enum;
pub const CU_MEM_LOCATION_TYPE_INVALID: c_int = 0;
pub const CU_MEM_LOCATION_TYPE_DEVICE: c_int = 1;
pub const CU_MEM_LOCATION_TYPE_HOST: c_int = 2;
pub const CU_MEM_LOCATION_TYPE_HOST_NUMA: c_int = 3;
pub const CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT: c_int = 4;
pub const CU_MEM_LOCATION_TYPE_MAX: c_int = 2147483647;
pub const enum_CUmemLocationType_enum = c_uint;
pub const CUmemLocationType = enum_CUmemLocationType_enum;
pub const CU_MEM_ALLOCATION_TYPE_INVALID: c_int = 0;
pub const CU_MEM_ALLOCATION_TYPE_PINNED: c_int = 1;
pub const CU_MEM_ALLOCATION_TYPE_MAX: c_int = 2147483647;
pub const enum_CUmemAllocationType_enum = c_uint;
pub const CUmemAllocationType = enum_CUmemAllocationType_enum;
pub const CU_MEM_ALLOC_GRANULARITY_MINIMUM: c_int = 0;
pub const CU_MEM_ALLOC_GRANULARITY_RECOMMENDED: c_int = 1;
pub const enum_CUmemAllocationGranularity_flags_enum = c_uint;
pub const CUmemAllocationGranularity_flags = enum_CUmemAllocationGranularity_flags_enum;
pub const CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD: c_int = 1;
pub const CU_MEM_RANGE_HANDLE_TYPE_MAX: c_int = 2147483647;
pub const enum_CUmemRangeHandleType_enum = c_uint;
pub const CUmemRangeHandleType = enum_CUmemRangeHandleType_enum;
pub const CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE: c_int = 1;
pub const enum_CUmemRangeFlags_enum = c_uint;
pub const CUmemRangeFlags = enum_CUmemRangeFlags_enum;
pub const CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL: c_int = 0;
pub const CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL: c_int = 1;
pub const enum_CUarraySparseSubresourceType_enum = c_uint;
pub const CUarraySparseSubresourceType = enum_CUarraySparseSubresourceType_enum;
pub const CU_MEM_OPERATION_TYPE_MAP: c_int = 1;
pub const CU_MEM_OPERATION_TYPE_UNMAP: c_int = 2;
pub const enum_CUmemOperationType_enum = c_uint;
pub const CUmemOperationType = enum_CUmemOperationType_enum;
pub const CU_MEM_HANDLE_TYPE_GENERIC: c_int = 0;
pub const enum_CUmemHandleType_enum = c_uint;
pub const CUmemHandleType = enum_CUmemHandleType_enum;
const union_unnamed_34 = extern union {
    mipmap: CUmipmappedArray,
    array: CUarray,
};
const struct_unnamed_36 = extern struct {
    level: c_uint = @import("std").mem.zeroes(c_uint),
    layer: c_uint = @import("std").mem.zeroes(c_uint),
    offsetX: c_uint = @import("std").mem.zeroes(c_uint),
    offsetY: c_uint = @import("std").mem.zeroes(c_uint),
    offsetZ: c_uint = @import("std").mem.zeroes(c_uint),
    extentWidth: c_uint = @import("std").mem.zeroes(c_uint),
    extentHeight: c_uint = @import("std").mem.zeroes(c_uint),
    extentDepth: c_uint = @import("std").mem.zeroes(c_uint),
};
const struct_unnamed_37 = extern struct {
    layer: c_uint = @import("std").mem.zeroes(c_uint),
    offset: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    size: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
};
const union_unnamed_35 = extern union {
    sparseLevel: struct_unnamed_36,
    miptail: struct_unnamed_37,
};
const union_unnamed_38 = extern union {
    memHandle: CUmemGenericAllocationHandle,
};
pub const struct_CUarrayMapInfo_st = extern struct {
    resourceType: CUresourcetype = @import("std").mem.zeroes(CUresourcetype),
    resource: union_unnamed_34 = @import("std").mem.zeroes(union_unnamed_34),
    subresourceType: CUarraySparseSubresourceType = @import("std").mem.zeroes(CUarraySparseSubresourceType),
    subresource: union_unnamed_35 = @import("std").mem.zeroes(union_unnamed_35),
    memOperationType: CUmemOperationType = @import("std").mem.zeroes(CUmemOperationType),
    memHandleType: CUmemHandleType = @import("std").mem.zeroes(CUmemHandleType),
    memHandle: union_unnamed_38 = @import("std").mem.zeroes(union_unnamed_38),
    offset: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    deviceBitMask: c_uint = @import("std").mem.zeroes(c_uint),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
    reserved: [2]c_uint = @import("std").mem.zeroes([2]c_uint),
};
pub const CUarrayMapInfo_v1 = struct_CUarrayMapInfo_st;
pub const CUarrayMapInfo = CUarrayMapInfo_v1;
pub const struct_CUmemLocation_st = extern struct {
    type: CUmemLocationType = @import("std").mem.zeroes(CUmemLocationType),
    id: c_int = @import("std").mem.zeroes(c_int),
};
pub const CUmemLocation_v1 = struct_CUmemLocation_st;
pub const CUmemLocation = CUmemLocation_v1;
pub const CU_MEM_ALLOCATION_COMP_NONE: c_int = 0;
pub const CU_MEM_ALLOCATION_COMP_GENERIC: c_int = 1;
pub const enum_CUmemAllocationCompType_enum = c_uint;
pub const CUmemAllocationCompType = enum_CUmemAllocationCompType_enum;
const struct_unnamed_39 = extern struct {
    compressionType: u8 = @import("std").mem.zeroes(u8),
    gpuDirectRDMACapable: u8 = @import("std").mem.zeroes(u8),
    usage: c_ushort = @import("std").mem.zeroes(c_ushort),
    reserved: [4]u8 = @import("std").mem.zeroes([4]u8),
};
pub const struct_CUmemAllocationProp_st = extern struct {
    type: CUmemAllocationType = @import("std").mem.zeroes(CUmemAllocationType),
    requestedHandleTypes: CUmemAllocationHandleType = @import("std").mem.zeroes(CUmemAllocationHandleType),
    location: CUmemLocation = @import("std").mem.zeroes(CUmemLocation),
    win32HandleMetaData: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    allocFlags: struct_unnamed_39 = @import("std").mem.zeroes(struct_unnamed_39),
};
pub const CUmemAllocationProp_v1 = struct_CUmemAllocationProp_st;
pub const CUmemAllocationProp = CUmemAllocationProp_v1;
pub const CU_MULTICAST_GRANULARITY_MINIMUM: c_int = 0;
pub const CU_MULTICAST_GRANULARITY_RECOMMENDED: c_int = 1;
pub const enum_CUmulticastGranularity_flags_enum = c_uint;
pub const CUmulticastGranularity_flags = enum_CUmulticastGranularity_flags_enum;
pub const struct_CUmulticastObjectProp_st = extern struct {
    numDevices: c_uint = @import("std").mem.zeroes(c_uint),
    size: usize = @import("std").mem.zeroes(usize),
    handleTypes: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    flags: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
};
pub const CUmulticastObjectProp_v1 = struct_CUmulticastObjectProp_st;
pub const CUmulticastObjectProp = CUmulticastObjectProp_v1;
pub const struct_CUmemAccessDesc_st = extern struct {
    location: CUmemLocation = @import("std").mem.zeroes(CUmemLocation),
    flags: CUmemAccess_flags = @import("std").mem.zeroes(CUmemAccess_flags),
};
pub const CUmemAccessDesc_v1 = struct_CUmemAccessDesc_st;
pub const CUmemAccessDesc = CUmemAccessDesc_v1;
pub const CU_GRAPH_EXEC_UPDATE_SUCCESS: c_int = 0;
pub const CU_GRAPH_EXEC_UPDATE_ERROR: c_int = 1;
pub const CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED: c_int = 2;
pub const CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED: c_int = 3;
pub const CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED: c_int = 4;
pub const CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED: c_int = 5;
pub const CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED: c_int = 6;
pub const CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE: c_int = 7;
pub const CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED: c_int = 8;
pub const enum_CUgraphExecUpdateResult_enum = c_uint;
pub const CUgraphExecUpdateResult = enum_CUgraphExecUpdateResult_enum;
pub const struct_CUgraphExecUpdateResultInfo_st = extern struct {
    result: CUgraphExecUpdateResult = @import("std").mem.zeroes(CUgraphExecUpdateResult),
    errorNode: CUgraphNode = @import("std").mem.zeroes(CUgraphNode),
    errorFromNode: CUgraphNode = @import("std").mem.zeroes(CUgraphNode),
};
pub const CUgraphExecUpdateResultInfo_v1 = struct_CUgraphExecUpdateResultInfo_st;
pub const CUgraphExecUpdateResultInfo = CUgraphExecUpdateResultInfo_v1;
pub const CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES: c_int = 1;
pub const CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC: c_int = 2;
pub const CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES: c_int = 3;
pub const CU_MEMPOOL_ATTR_RELEASE_THRESHOLD: c_int = 4;
pub const CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT: c_int = 5;
pub const CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH: c_int = 6;
pub const CU_MEMPOOL_ATTR_USED_MEM_CURRENT: c_int = 7;
pub const CU_MEMPOOL_ATTR_USED_MEM_HIGH: c_int = 8;
pub const enum_CUmemPool_attribute_enum = c_uint;
pub const CUmemPool_attribute = enum_CUmemPool_attribute_enum;
pub const struct_CUmemPoolProps_st = extern struct {
    allocType: CUmemAllocationType = @import("std").mem.zeroes(CUmemAllocationType),
    handleTypes: CUmemAllocationHandleType = @import("std").mem.zeroes(CUmemAllocationHandleType),
    location: CUmemLocation = @import("std").mem.zeroes(CUmemLocation),
    win32SecurityAttributes: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    maxSize: usize = @import("std").mem.zeroes(usize),
    usage: c_ushort = @import("std").mem.zeroes(c_ushort),
    reserved: [54]u8 = @import("std").mem.zeroes([54]u8),
};
pub const CUmemPoolProps_v1 = struct_CUmemPoolProps_st;
pub const CUmemPoolProps = CUmemPoolProps_v1;
pub const struct_CUmemPoolPtrExportData_st = extern struct {
    reserved: [64]u8 = @import("std").mem.zeroes([64]u8),
};
pub const CUmemPoolPtrExportData_v1 = struct_CUmemPoolPtrExportData_st;
pub const CUmemPoolPtrExportData = CUmemPoolPtrExportData_v1;
pub const struct_CUDA_MEM_ALLOC_NODE_PARAMS_v1_st = extern struct {
    poolProps: CUmemPoolProps = @import("std").mem.zeroes(CUmemPoolProps),
    accessDescs: [*c]const CUmemAccessDesc = @import("std").mem.zeroes([*c]const CUmemAccessDesc),
    accessDescCount: usize = @import("std").mem.zeroes(usize),
    bytesize: usize = @import("std").mem.zeroes(usize),
    dptr: CUdeviceptr = @import("std").mem.zeroes(CUdeviceptr),
};
pub const CUDA_MEM_ALLOC_NODE_PARAMS_v1 = struct_CUDA_MEM_ALLOC_NODE_PARAMS_v1_st;
pub const CUDA_MEM_ALLOC_NODE_PARAMS = CUDA_MEM_ALLOC_NODE_PARAMS_v1;
pub const struct_CUDA_MEM_ALLOC_NODE_PARAMS_v2_st = extern struct {
    poolProps: CUmemPoolProps = @import("std").mem.zeroes(CUmemPoolProps),
    accessDescs: [*c]const CUmemAccessDesc = @import("std").mem.zeroes([*c]const CUmemAccessDesc),
    accessDescCount: usize = @import("std").mem.zeroes(usize),
    bytesize: usize = @import("std").mem.zeroes(usize),
    dptr: CUdeviceptr = @import("std").mem.zeroes(CUdeviceptr),
};
pub const CUDA_MEM_ALLOC_NODE_PARAMS_v2 = struct_CUDA_MEM_ALLOC_NODE_PARAMS_v2_st;
pub const struct_CUDA_MEM_FREE_NODE_PARAMS_st = extern struct {
    dptr: CUdeviceptr = @import("std").mem.zeroes(CUdeviceptr),
};
pub const CUDA_MEM_FREE_NODE_PARAMS = struct_CUDA_MEM_FREE_NODE_PARAMS_st;
pub const CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT: c_int = 0;
pub const CU_GRAPH_MEM_ATTR_USED_MEM_HIGH: c_int = 1;
pub const CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT: c_int = 2;
pub const CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH: c_int = 3;
pub const enum_CUgraphMem_attribute_enum = c_uint;
pub const CUgraphMem_attribute = enum_CUgraphMem_attribute_enum;
pub const struct_CUDA_CHILD_GRAPH_NODE_PARAMS_st = extern struct {
    graph: CUgraph = @import("std").mem.zeroes(CUgraph),
};
pub const CUDA_CHILD_GRAPH_NODE_PARAMS = struct_CUDA_CHILD_GRAPH_NODE_PARAMS_st;
pub const struct_CUDA_EVENT_RECORD_NODE_PARAMS_st = extern struct {
    event: CUevent = @import("std").mem.zeroes(CUevent),
};
pub const CUDA_EVENT_RECORD_NODE_PARAMS = struct_CUDA_EVENT_RECORD_NODE_PARAMS_st;
pub const struct_CUDA_EVENT_WAIT_NODE_PARAMS_st = extern struct {
    event: CUevent = @import("std").mem.zeroes(CUevent),
};
pub const CUDA_EVENT_WAIT_NODE_PARAMS = struct_CUDA_EVENT_WAIT_NODE_PARAMS_st;
const union_unnamed_40 = extern union {
    reserved1: [29]c_longlong,
    kernel: CUDA_KERNEL_NODE_PARAMS_v3,
    memcpy: CUDA_MEMCPY_NODE_PARAMS,
    memset: CUDA_MEMSET_NODE_PARAMS_v2,
    host: CUDA_HOST_NODE_PARAMS_v2,
    graph: CUDA_CHILD_GRAPH_NODE_PARAMS,
    eventWait: CUDA_EVENT_WAIT_NODE_PARAMS,
    eventRecord: CUDA_EVENT_RECORD_NODE_PARAMS,
    extSemSignal: CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2,
    extSemWait: CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2,
    alloc: CUDA_MEM_ALLOC_NODE_PARAMS_v2,
    free: CUDA_MEM_FREE_NODE_PARAMS,
    memOp: CUDA_BATCH_MEM_OP_NODE_PARAMS_v2,
    conditional: CUDA_CONDITIONAL_NODE_PARAMS,
};
pub const struct_CUgraphNodeParams_st = extern struct {
    type: CUgraphNodeType = @import("std").mem.zeroes(CUgraphNodeType),
    reserved0: [3]c_int = @import("std").mem.zeroes([3]c_int),
    unnamed_0: union_unnamed_40 = @import("std").mem.zeroes(union_unnamed_40),
    reserved2: c_longlong = @import("std").mem.zeroes(c_longlong),
};
pub const CUgraphNodeParams = struct_CUgraphNodeParams_st;
pub const CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST: c_int = 1;
pub const CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS: c_int = 2;
pub const enum_CUflushGPUDirectRDMAWritesOptions_enum = c_uint;
pub const CUflushGPUDirectRDMAWritesOptions = enum_CUflushGPUDirectRDMAWritesOptions_enum;
pub const CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE: c_int = 0;
pub const CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER: c_int = 100;
pub const CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES: c_int = 200;
pub const enum_CUGPUDirectRDMAWritesOrdering_enum = c_uint;
pub const CUGPUDirectRDMAWritesOrdering = enum_CUGPUDirectRDMAWritesOrdering_enum;
pub const CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER: c_int = 100;
pub const CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES: c_int = 200;
pub const enum_CUflushGPUDirectRDMAWritesScope_enum = c_uint;
pub const CUflushGPUDirectRDMAWritesScope = enum_CUflushGPUDirectRDMAWritesScope_enum;
pub const CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX: c_int = 0;
pub const enum_CUflushGPUDirectRDMAWritesTarget_enum = c_uint;
pub const CUflushGPUDirectRDMAWritesTarget = enum_CUflushGPUDirectRDMAWritesTarget_enum;
pub const CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE: c_int = 1;
pub const CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES: c_int = 2;
pub const CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS: c_int = 4;
pub const CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS: c_int = 8;
pub const CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS: c_int = 16;
pub const CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS: c_int = 32;
pub const CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS: c_int = 64;
pub const CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS: c_int = 128;
pub const CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS: c_int = 256;
pub const CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES: c_int = 512;
pub const CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES: c_int = 1024;
pub const CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS: c_int = 2048;
pub const CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS: c_int = 4096;
pub const CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS: c_int = 8192;
pub const CU_GRAPH_DEBUG_DOT_FLAGS_EXTRA_TOPO_INFO: c_int = 16384;
pub const CU_GRAPH_DEBUG_DOT_FLAGS_CONDITIONAL_NODE_PARAMS: c_int = 32768;
pub const enum_CUgraphDebugDot_flags_enum = c_uint;
pub const CUgraphDebugDot_flags = enum_CUgraphDebugDot_flags_enum;
pub const CU_USER_OBJECT_NO_DESTRUCTOR_SYNC: c_int = 1;
pub const enum_CUuserObject_flags_enum = c_uint;
pub const CUuserObject_flags = enum_CUuserObject_flags_enum;
pub const CU_GRAPH_USER_OBJECT_MOVE: c_int = 1;
pub const enum_CUuserObjectRetain_flags_enum = c_uint;
pub const CUuserObjectRetain_flags = enum_CUuserObjectRetain_flags_enum;
pub const CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH: c_int = 1;
pub const CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD: c_int = 2;
pub const CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH: c_int = 4;
pub const CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY: c_int = 8;
pub const enum_CUgraphInstantiate_flags_enum = c_uint;
pub const CUgraphInstantiate_flags = enum_CUgraphInstantiate_flags_enum;
pub const CU_DEVICE_NUMA_CONFIG_NONE: c_int = 0;
pub const CU_DEVICE_NUMA_CONFIG_NUMA_NODE: c_int = 1;
pub const enum_CUdeviceNumaConfig_enum = c_uint;
pub const CUdeviceNumaConfig = enum_CUdeviceNumaConfig_enum;
pub const CU_PROCESS_STATE_RUNNING: c_int = 0;
pub const CU_PROCESS_STATE_LOCKED: c_int = 1;
pub const CU_PROCESS_STATE_CHECKPOINTED: c_int = 2;
pub const CU_PROCESS_STATE_FAILED: c_int = 3;
pub const enum_CUprocessState_enum = c_uint;
pub const CUprocessState = enum_CUprocessState_enum;
pub const struct_CUcheckpointLockArgs_st = extern struct {
    timeoutMs: c_uint = @import("std").mem.zeroes(c_uint),
    reserved0: c_uint = @import("std").mem.zeroes(c_uint),
    reserved1: [7]cuuint64_t = @import("std").mem.zeroes([7]cuuint64_t),
};
pub const CUcheckpointLockArgs = struct_CUcheckpointLockArgs_st;
pub const struct_CUcheckpointCheckpointArgs_st = extern struct {
    reserved: [8]cuuint64_t = @import("std").mem.zeroes([8]cuuint64_t),
};
pub const CUcheckpointCheckpointArgs = struct_CUcheckpointCheckpointArgs_st;
pub const struct_CUcheckpointRestoreArgs_st = extern struct {
    reserved: [8]cuuint64_t = @import("std").mem.zeroes([8]cuuint64_t),
};
pub const CUcheckpointRestoreArgs = struct_CUcheckpointRestoreArgs_st;
pub const struct_CUcheckpointUnlockArgs_st = extern struct {
    reserved: [8]cuuint64_t = @import("std").mem.zeroes([8]cuuint64_t),
};
pub const CUcheckpointUnlockArgs = struct_CUcheckpointUnlockArgs_st;
pub const CU_MEMCPY_FLAG_DEFAULT: c_int = 0;
pub const CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE: c_int = 1;
pub const enum_CUmemcpyFlags_enum = c_uint;
pub const CUmemcpyFlags = enum_CUmemcpyFlags_enum;
pub const CU_MEMCPY_SRC_ACCESS_ORDER_INVALID: c_int = 0;
pub const CU_MEMCPY_SRC_ACCESS_ORDER_STREAM: c_int = 1;
pub const CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL: c_int = 2;
pub const CU_MEMCPY_SRC_ACCESS_ORDER_ANY: c_int = 3;
pub const CU_MEMCPY_SRC_ACCESS_ORDER_MAX: c_int = 2147483647;
pub const enum_CUmemcpySrcAccessOrder_enum = c_uint;
pub const CUmemcpySrcAccessOrder = enum_CUmemcpySrcAccessOrder_enum;
pub const struct_CUmemcpyAttributes_st = extern struct {
    srcAccessOrder: CUmemcpySrcAccessOrder = @import("std").mem.zeroes(CUmemcpySrcAccessOrder),
    srcLocHint: CUmemLocation = @import("std").mem.zeroes(CUmemLocation),
    dstLocHint: CUmemLocation = @import("std").mem.zeroes(CUmemLocation),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const CUmemcpyAttributes_v1 = struct_CUmemcpyAttributes_st;
pub const CUmemcpyAttributes = CUmemcpyAttributes_v1;
pub const CU_MEMCPY_OPERAND_TYPE_POINTER: c_int = 1;
pub const CU_MEMCPY_OPERAND_TYPE_ARRAY: c_int = 2;
pub const CU_MEMCPY_OPERAND_TYPE_MAX: c_int = 2147483647;
pub const enum_CUmemcpy3DOperandType_enum = c_uint;
pub const CUmemcpy3DOperandType = enum_CUmemcpy3DOperandType_enum;
pub const struct_CUoffset3D_st = extern struct {
    x: usize = @import("std").mem.zeroes(usize),
    y: usize = @import("std").mem.zeroes(usize),
    z: usize = @import("std").mem.zeroes(usize),
};
pub const CUoffset3D_v1 = struct_CUoffset3D_st;
pub const CUoffset3D = CUoffset3D_v1;
pub const struct_CUextent3D_st = extern struct {
    width: usize = @import("std").mem.zeroes(usize),
    height: usize = @import("std").mem.zeroes(usize),
    depth: usize = @import("std").mem.zeroes(usize),
};
pub const CUextent3D_v1 = struct_CUextent3D_st;
pub const CUextent3D = CUextent3D_v1;
const struct_unnamed_42 = extern struct {
    ptr: CUdeviceptr = @import("std").mem.zeroes(CUdeviceptr),
    rowLength: usize = @import("std").mem.zeroes(usize),
    layerHeight: usize = @import("std").mem.zeroes(usize),
    locHint: CUmemLocation = @import("std").mem.zeroes(CUmemLocation),
};
const struct_unnamed_43 = extern struct {
    array: CUarray = @import("std").mem.zeroes(CUarray),
    offset: CUoffset3D = @import("std").mem.zeroes(CUoffset3D),
};
const union_unnamed_41 = extern union {
    ptr: struct_unnamed_42,
    array: struct_unnamed_43,
};
pub const struct_CUmemcpy3DOperand_st = extern struct {
    type: CUmemcpy3DOperandType = @import("std").mem.zeroes(CUmemcpy3DOperandType),
    op: union_unnamed_41 = @import("std").mem.zeroes(union_unnamed_41),
};
pub const CUmemcpy3DOperand_v1 = struct_CUmemcpy3DOperand_st;
pub const CUmemcpy3DOperand = CUmemcpy3DOperand_v1;
pub const struct_CUDA_MEMCPY3D_BATCH_OP_st = extern struct {
    src: CUmemcpy3DOperand = @import("std").mem.zeroes(CUmemcpy3DOperand),
    dst: CUmemcpy3DOperand = @import("std").mem.zeroes(CUmemcpy3DOperand),
    extent: CUextent3D = @import("std").mem.zeroes(CUextent3D),
    srcAccessOrder: CUmemcpySrcAccessOrder = @import("std").mem.zeroes(CUmemcpySrcAccessOrder),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const CUDA_MEMCPY3D_BATCH_OP_v1 = struct_CUDA_MEMCPY3D_BATCH_OP_st;
pub const CUDA_MEMCPY3D_BATCH_OP = CUDA_MEMCPY3D_BATCH_OP_v1;
pub extern fn cuGetErrorString(@"error": CUresult, pStr: [*c][*c]const u8) CUresult;
pub extern fn cuGetErrorName(@"error": CUresult, pStr: [*c][*c]const u8) CUresult;
pub extern fn cuInit(Flags: c_uint) CUresult;
pub extern fn cuDriverGetVersion(driverVersion: [*c]c_int) CUresult;
pub extern fn cuDeviceGet(device: [*c]CUdevice, ordinal: c_int) CUresult;
pub extern fn cuDeviceGetCount(count: [*c]c_int) CUresult;
pub extern fn cuDeviceGetName(name: [*c]u8, len: c_int, dev: CUdevice) CUresult;
pub extern fn cuDeviceGetUuid(uuid: [*c]CUuuid, dev: CUdevice) CUresult;
pub extern fn cuDeviceGetUuid_v2(uuid: [*c]CUuuid, dev: CUdevice) CUresult;
pub extern fn cuDeviceGetLuid(luid: [*c]u8, deviceNodeMask: [*c]c_uint, dev: CUdevice) CUresult;
pub extern fn cuDeviceTotalMem_v2(bytes: [*c]usize, dev: CUdevice) CUresult;
pub extern fn cuDeviceGetTexture1DLinearMaxWidth(maxWidthInElements: [*c]usize, format: CUarray_format, numChannels: c_uint, dev: CUdevice) CUresult;
pub extern fn cuDeviceGetAttribute(pi: [*c]c_int, attrib: CUdevice_attribute, dev: CUdevice) CUresult;
pub extern fn cuDeviceGetNvSciSyncAttributes(nvSciSyncAttrList: ?*anyopaque, dev: CUdevice, flags: c_int) CUresult;
pub extern fn cuDeviceSetMemPool(dev: CUdevice, pool: CUmemoryPool) CUresult;
pub extern fn cuDeviceGetMemPool(pool: [*c]CUmemoryPool, dev: CUdevice) CUresult;
pub extern fn cuDeviceGetDefaultMemPool(pool_out: [*c]CUmemoryPool, dev: CUdevice) CUresult;
pub extern fn cuDeviceGetExecAffinitySupport(pi: [*c]c_int, @"type": CUexecAffinityType, dev: CUdevice) CUresult;
pub extern fn cuFlushGPUDirectRDMAWrites(target: CUflushGPUDirectRDMAWritesTarget, scope: CUflushGPUDirectRDMAWritesScope) CUresult;
pub extern fn cuDeviceGetProperties(prop: [*c]CUdevprop, dev: CUdevice) CUresult;
pub extern fn cuDeviceComputeCapability(major: [*c]c_int, minor: [*c]c_int, dev: CUdevice) CUresult;
pub extern fn cuDevicePrimaryCtxRetain(pctx: [*c]CUcontext, dev: CUdevice) CUresult;
pub extern fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) CUresult;
pub extern fn cuDevicePrimaryCtxSetFlags_v2(dev: CUdevice, flags: c_uint) CUresult;
pub extern fn cuDevicePrimaryCtxGetState(dev: CUdevice, flags: [*c]c_uint, active: [*c]c_int) CUresult;
pub extern fn cuDevicePrimaryCtxReset_v2(dev: CUdevice) CUresult;
pub extern fn cuCtxCreate_v2(pctx: [*c]CUcontext, flags: c_uint, dev: CUdevice) CUresult;
pub extern fn cuCtxCreate_v3(pctx: [*c]CUcontext, paramsArray: [*c]CUexecAffinityParam, numParams: c_int, flags: c_uint, dev: CUdevice) CUresult;
pub extern fn cuCtxCreate_v4(pctx: [*c]CUcontext, ctxCreateParams: [*c]CUctxCreateParams, flags: c_uint, dev: CUdevice) CUresult;
pub extern fn cuCtxDestroy_v2(ctx: CUcontext) CUresult;
pub extern fn cuCtxPushCurrent_v2(ctx: CUcontext) CUresult;
pub extern fn cuCtxPopCurrent_v2(pctx: [*c]CUcontext) CUresult;
pub extern fn cuCtxSetCurrent(ctx: CUcontext) CUresult;
pub extern fn cuCtxGetCurrent(pctx: [*c]CUcontext) CUresult;
pub extern fn cuCtxGetDevice(device: [*c]CUdevice) CUresult;
pub extern fn cuCtxGetFlags(flags: [*c]c_uint) CUresult;
pub extern fn cuCtxSetFlags(flags: c_uint) CUresult;
pub extern fn cuCtxGetId(ctx: CUcontext, ctxId: [*c]c_ulonglong) CUresult;
pub extern fn cuCtxSynchronize() CUresult;
pub extern fn cuCtxSetLimit(limit: CUlimit, value: usize) CUresult;
pub extern fn cuCtxGetLimit(pvalue: [*c]usize, limit: CUlimit) CUresult;
pub extern fn cuCtxGetCacheConfig(pconfig: [*c]CUfunc_cache) CUresult;
pub extern fn cuCtxSetCacheConfig(config: CUfunc_cache) CUresult;
pub extern fn cuCtxGetApiVersion(ctx: CUcontext, version: [*c]c_uint) CUresult;
pub extern fn cuCtxGetStreamPriorityRange(leastPriority: [*c]c_int, greatestPriority: [*c]c_int) CUresult;
pub extern fn cuCtxResetPersistingL2Cache() CUresult;
pub extern fn cuCtxGetExecAffinity(pExecAffinity: [*c]CUexecAffinityParam, @"type": CUexecAffinityType) CUresult;
pub extern fn cuCtxRecordEvent(hCtx: CUcontext, hEvent: CUevent) CUresult;
pub extern fn cuCtxWaitEvent(hCtx: CUcontext, hEvent: CUevent) CUresult;
pub extern fn cuCtxAttach(pctx: [*c]CUcontext, flags: c_uint) CUresult;
pub extern fn cuCtxDetach(ctx: CUcontext) CUresult;
pub extern fn cuCtxGetSharedMemConfig(pConfig: [*c]CUsharedconfig) CUresult;
pub extern fn cuCtxSetSharedMemConfig(config: CUsharedconfig) CUresult;
pub extern fn cuModuleLoad(module: [*c]CUmodule, fname: [*c]const u8) CUresult;
pub extern fn cuModuleLoadData(module: [*c]CUmodule, image: ?*const anyopaque) CUresult;
pub extern fn cuModuleLoadDataEx(module: [*c]CUmodule, image: ?*const anyopaque, numOptions: c_uint, options: [*c]CUjit_option, optionValues: [*c]?*anyopaque) CUresult;
pub extern fn cuModuleLoadFatBinary(module: [*c]CUmodule, fatCubin: ?*const anyopaque) CUresult;
pub extern fn cuModuleUnload(hmod: CUmodule) CUresult;
pub const CU_MODULE_EAGER_LOADING: c_int = 1;
pub const CU_MODULE_LAZY_LOADING: c_int = 2;
pub const enum_CUmoduleLoadingMode_enum = c_uint;
pub const CUmoduleLoadingMode = enum_CUmoduleLoadingMode_enum;
pub extern fn cuModuleGetLoadingMode(mode: [*c]CUmoduleLoadingMode) CUresult;
pub extern fn cuModuleGetFunction(hfunc: [*c]CUfunction, hmod: CUmodule, name: [*c]const u8) CUresult;
pub extern fn cuModuleGetFunctionCount(count: [*c]c_uint, mod: CUmodule) CUresult;
pub extern fn cuModuleEnumerateFunctions(functions: [*c]CUfunction, numFunctions: c_uint, mod: CUmodule) CUresult;
pub extern fn cuModuleGetGlobal_v2(dptr: [*c]CUdeviceptr, bytes: [*c]usize, hmod: CUmodule, name: [*c]const u8) CUresult;
pub extern fn cuLinkCreate_v2(numOptions: c_uint, options: [*c]CUjit_option, optionValues: [*c]?*anyopaque, stateOut: [*c]CUlinkState) CUresult;
pub extern fn cuLinkAddData_v2(state: CUlinkState, @"type": CUjitInputType, data: ?*anyopaque, size: usize, name: [*c]const u8, numOptions: c_uint, options: [*c]CUjit_option, optionValues: [*c]?*anyopaque) CUresult;
pub extern fn cuLinkAddFile_v2(state: CUlinkState, @"type": CUjitInputType, path: [*c]const u8, numOptions: c_uint, options: [*c]CUjit_option, optionValues: [*c]?*anyopaque) CUresult;
pub extern fn cuLinkComplete(state: CUlinkState, cubinOut: [*c]?*anyopaque, sizeOut: [*c]usize) CUresult;
pub extern fn cuLinkDestroy(state: CUlinkState) CUresult;
pub extern fn cuModuleGetTexRef(pTexRef: [*c]CUtexref, hmod: CUmodule, name: [*c]const u8) CUresult;
pub extern fn cuModuleGetSurfRef(pSurfRef: [*c]CUsurfref, hmod: CUmodule, name: [*c]const u8) CUresult;
pub extern fn cuLibraryLoadData(library: [*c]CUlibrary, code: ?*const anyopaque, jitOptions: [*c]CUjit_option, jitOptionsValues: [*c]?*anyopaque, numJitOptions: c_uint, libraryOptions: [*c]CUlibraryOption, libraryOptionValues: [*c]?*anyopaque, numLibraryOptions: c_uint) CUresult;
pub extern fn cuLibraryLoadFromFile(library: [*c]CUlibrary, fileName: [*c]const u8, jitOptions: [*c]CUjit_option, jitOptionsValues: [*c]?*anyopaque, numJitOptions: c_uint, libraryOptions: [*c]CUlibraryOption, libraryOptionValues: [*c]?*anyopaque, numLibraryOptions: c_uint) CUresult;
pub extern fn cuLibraryUnload(library: CUlibrary) CUresult;
pub extern fn cuLibraryGetKernel(pKernel: [*c]CUkernel, library: CUlibrary, name: [*c]const u8) CUresult;
pub extern fn cuLibraryGetKernelCount(count: [*c]c_uint, lib: CUlibrary) CUresult;
pub extern fn cuLibraryEnumerateKernels(kernels: [*c]CUkernel, numKernels: c_uint, lib: CUlibrary) CUresult;
pub extern fn cuLibraryGetModule(pMod: [*c]CUmodule, library: CUlibrary) CUresult;
pub extern fn cuKernelGetFunction(pFunc: [*c]CUfunction, kernel: CUkernel) CUresult;
pub extern fn cuKernelGetLibrary(pLib: [*c]CUlibrary, kernel: CUkernel) CUresult;
pub extern fn cuLibraryGetGlobal(dptr: [*c]CUdeviceptr, bytes: [*c]usize, library: CUlibrary, name: [*c]const u8) CUresult;
pub extern fn cuLibraryGetManaged(dptr: [*c]CUdeviceptr, bytes: [*c]usize, library: CUlibrary, name: [*c]const u8) CUresult;
pub extern fn cuLibraryGetUnifiedFunction(fptr: [*c]?*anyopaque, library: CUlibrary, symbol: [*c]const u8) CUresult;
pub extern fn cuKernelGetAttribute(pi: [*c]c_int, attrib: CUfunction_attribute, kernel: CUkernel, dev: CUdevice) CUresult;
pub extern fn cuKernelSetAttribute(attrib: CUfunction_attribute, val: c_int, kernel: CUkernel, dev: CUdevice) CUresult;
pub extern fn cuKernelSetCacheConfig(kernel: CUkernel, config: CUfunc_cache, dev: CUdevice) CUresult;
pub extern fn cuKernelGetName(name: [*c][*c]const u8, hfunc: CUkernel) CUresult;
pub extern fn cuKernelGetParamInfo(kernel: CUkernel, paramIndex: usize, paramOffset: [*c]usize, paramSize: [*c]usize) CUresult;
pub extern fn cuMemGetInfo_v2(free: [*c]usize, total: [*c]usize) CUresult;
pub extern fn cuMemAlloc_v2(dptr: [*c]CUdeviceptr, bytesize: usize) CUresult;
pub extern fn cuMemAllocPitch_v2(dptr: [*c]CUdeviceptr, pPitch: [*c]usize, WidthInBytes: usize, Height: usize, ElementSizeBytes: c_uint) CUresult;
pub extern fn cuMemFree_v2(dptr: CUdeviceptr) CUresult;
pub extern fn cuMemGetAddressRange_v2(pbase: [*c]CUdeviceptr, psize: [*c]usize, dptr: CUdeviceptr) CUresult;
pub extern fn cuMemAllocHost_v2(pp: [*c]?*anyopaque, bytesize: usize) CUresult;
pub extern fn cuMemFreeHost(p: ?*anyopaque) CUresult;
pub extern fn cuMemHostAlloc(pp: [*c]?*anyopaque, bytesize: usize, Flags: c_uint) CUresult;
pub extern fn cuMemHostGetDevicePointer_v2(pdptr: [*c]CUdeviceptr, p: ?*anyopaque, Flags: c_uint) CUresult;
pub extern fn cuMemHostGetFlags(pFlags: [*c]c_uint, p: ?*anyopaque) CUresult;
pub extern fn cuMemAllocManaged(dptr: [*c]CUdeviceptr, bytesize: usize, flags: c_uint) CUresult;
pub extern fn cuDeviceRegisterAsyncNotification(device: CUdevice, callbackFunc: CUasyncCallback, userData: ?*anyopaque, callback: [*c]CUasyncCallbackHandle) CUresult;
pub extern fn cuDeviceUnregisterAsyncNotification(device: CUdevice, callback: CUasyncCallbackHandle) CUresult;
pub extern fn cuDeviceGetByPCIBusId(dev: [*c]CUdevice, pciBusId: [*c]const u8) CUresult;
pub extern fn cuDeviceGetPCIBusId(pciBusId: [*c]u8, len: c_int, dev: CUdevice) CUresult;
pub extern fn cuIpcGetEventHandle(pHandle: [*c]CUipcEventHandle, event: CUevent) CUresult;
pub extern fn cuIpcOpenEventHandle(phEvent: [*c]CUevent, handle: CUipcEventHandle) CUresult;
pub extern fn cuIpcGetMemHandle(pHandle: [*c]CUipcMemHandle, dptr: CUdeviceptr) CUresult;
pub extern fn cuIpcOpenMemHandle_v2(pdptr: [*c]CUdeviceptr, handle: CUipcMemHandle, Flags: c_uint) CUresult;
pub extern fn cuIpcCloseMemHandle(dptr: CUdeviceptr) CUresult;
pub extern fn cuMemHostRegister_v2(p: ?*anyopaque, bytesize: usize, Flags: c_uint) CUresult;
pub extern fn cuMemHostUnregister(p: ?*anyopaque) CUresult;
pub extern fn cuMemcpy(dst: CUdeviceptr, src: CUdeviceptr, ByteCount: usize) CUresult;
pub extern fn cuMemcpyPeer(dstDevice: CUdeviceptr, dstContext: CUcontext, srcDevice: CUdeviceptr, srcContext: CUcontext, ByteCount: usize) CUresult;
pub extern fn cuMemcpyHtoD_v2(dstDevice: CUdeviceptr, srcHost: ?*const anyopaque, ByteCount: usize) CUresult;
pub extern fn cuMemcpyDtoH_v2(dstHost: ?*anyopaque, srcDevice: CUdeviceptr, ByteCount: usize) CUresult;
pub extern fn cuMemcpyDtoD_v2(dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: usize) CUresult;
pub extern fn cuMemcpyDtoA_v2(dstArray: CUarray, dstOffset: usize, srcDevice: CUdeviceptr, ByteCount: usize) CUresult;
pub extern fn cuMemcpyAtoD_v2(dstDevice: CUdeviceptr, srcArray: CUarray, srcOffset: usize, ByteCount: usize) CUresult;
pub extern fn cuMemcpyHtoA_v2(dstArray: CUarray, dstOffset: usize, srcHost: ?*const anyopaque, ByteCount: usize) CUresult;
pub extern fn cuMemcpyAtoH_v2(dstHost: ?*anyopaque, srcArray: CUarray, srcOffset: usize, ByteCount: usize) CUresult;
pub extern fn cuMemcpyAtoA_v2(dstArray: CUarray, dstOffset: usize, srcArray: CUarray, srcOffset: usize, ByteCount: usize) CUresult;
pub extern fn cuMemcpy2D_v2(pCopy: [*c]const CUDA_MEMCPY2D) CUresult;
pub extern fn cuMemcpy2DUnaligned_v2(pCopy: [*c]const CUDA_MEMCPY2D) CUresult;
pub extern fn cuMemcpy3D_v2(pCopy: [*c]const CUDA_MEMCPY3D) CUresult;
pub extern fn cuMemcpy3DPeer(pCopy: [*c]const CUDA_MEMCPY3D_PEER) CUresult;
pub extern fn cuMemcpyAsync(dst: CUdeviceptr, src: CUdeviceptr, ByteCount: usize, hStream: CUstream) CUresult;
pub extern fn cuMemcpyPeerAsync(dstDevice: CUdeviceptr, dstContext: CUcontext, srcDevice: CUdeviceptr, srcContext: CUcontext, ByteCount: usize, hStream: CUstream) CUresult;
pub extern fn cuMemcpyHtoDAsync_v2(dstDevice: CUdeviceptr, srcHost: ?*const anyopaque, ByteCount: usize, hStream: CUstream) CUresult;
pub extern fn cuMemcpyDtoHAsync_v2(dstHost: ?*anyopaque, srcDevice: CUdeviceptr, ByteCount: usize, hStream: CUstream) CUresult;
pub extern fn cuMemcpyDtoDAsync_v2(dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: usize, hStream: CUstream) CUresult;
pub extern fn cuMemcpyHtoAAsync_v2(dstArray: CUarray, dstOffset: usize, srcHost: ?*const anyopaque, ByteCount: usize, hStream: CUstream) CUresult;
pub extern fn cuMemcpyAtoHAsync_v2(dstHost: ?*anyopaque, srcArray: CUarray, srcOffset: usize, ByteCount: usize, hStream: CUstream) CUresult;
pub extern fn cuMemcpy2DAsync_v2(pCopy: [*c]const CUDA_MEMCPY2D, hStream: CUstream) CUresult;
pub extern fn cuMemcpy3DAsync_v2(pCopy: [*c]const CUDA_MEMCPY3D, hStream: CUstream) CUresult;
pub extern fn cuMemcpy3DPeerAsync(pCopy: [*c]const CUDA_MEMCPY3D_PEER, hStream: CUstream) CUresult;
pub extern fn cuMemcpyBatchAsync(dsts: [*c]CUdeviceptr, srcs: [*c]CUdeviceptr, sizes: [*c]usize, count: usize, attrs: [*c]CUmemcpyAttributes, attrsIdxs: [*c]usize, numAttrs: usize, failIdx: [*c]usize, hStream: CUstream) CUresult;
pub extern fn cuMemcpy3DBatchAsync(numOps: usize, opList: [*c]CUDA_MEMCPY3D_BATCH_OP, failIdx: [*c]usize, flags: c_ulonglong, hStream: CUstream) CUresult;
pub extern fn cuMemsetD8_v2(dstDevice: CUdeviceptr, uc: u8, N: usize) CUresult;
pub extern fn cuMemsetD16_v2(dstDevice: CUdeviceptr, us: c_ushort, N: usize) CUresult;
pub extern fn cuMemsetD32_v2(dstDevice: CUdeviceptr, ui: c_uint, N: usize) CUresult;
pub extern fn cuMemsetD2D8_v2(dstDevice: CUdeviceptr, dstPitch: usize, uc: u8, Width: usize, Height: usize) CUresult;
pub extern fn cuMemsetD2D16_v2(dstDevice: CUdeviceptr, dstPitch: usize, us: c_ushort, Width: usize, Height: usize) CUresult;
pub extern fn cuMemsetD2D32_v2(dstDevice: CUdeviceptr, dstPitch: usize, ui: c_uint, Width: usize, Height: usize) CUresult;
pub extern fn cuMemsetD8Async(dstDevice: CUdeviceptr, uc: u8, N: usize, hStream: CUstream) CUresult;
pub extern fn cuMemsetD16Async(dstDevice: CUdeviceptr, us: c_ushort, N: usize, hStream: CUstream) CUresult;
pub extern fn cuMemsetD32Async(dstDevice: CUdeviceptr, ui: c_uint, N: usize, hStream: CUstream) CUresult;
pub extern fn cuMemsetD2D8Async(dstDevice: CUdeviceptr, dstPitch: usize, uc: u8, Width: usize, Height: usize, hStream: CUstream) CUresult;
pub extern fn cuMemsetD2D16Async(dstDevice: CUdeviceptr, dstPitch: usize, us: c_ushort, Width: usize, Height: usize, hStream: CUstream) CUresult;
pub extern fn cuMemsetD2D32Async(dstDevice: CUdeviceptr, dstPitch: usize, ui: c_uint, Width: usize, Height: usize, hStream: CUstream) CUresult;
pub extern fn cuArrayCreate_v2(pHandle: [*c]CUarray, pAllocateArray: [*c]const CUDA_ARRAY_DESCRIPTOR) CUresult;
pub extern fn cuArrayGetDescriptor_v2(pArrayDescriptor: [*c]CUDA_ARRAY_DESCRIPTOR, hArray: CUarray) CUresult;
pub extern fn cuArrayGetSparseProperties(sparseProperties: [*c]CUDA_ARRAY_SPARSE_PROPERTIES, array: CUarray) CUresult;
pub extern fn cuMipmappedArrayGetSparseProperties(sparseProperties: [*c]CUDA_ARRAY_SPARSE_PROPERTIES, mipmap: CUmipmappedArray) CUresult;
pub extern fn cuArrayGetMemoryRequirements(memoryRequirements: [*c]CUDA_ARRAY_MEMORY_REQUIREMENTS, array: CUarray, device: CUdevice) CUresult;
pub extern fn cuMipmappedArrayGetMemoryRequirements(memoryRequirements: [*c]CUDA_ARRAY_MEMORY_REQUIREMENTS, mipmap: CUmipmappedArray, device: CUdevice) CUresult;
pub extern fn cuArrayGetPlane(pPlaneArray: [*c]CUarray, hArray: CUarray, planeIdx: c_uint) CUresult;
pub extern fn cuArrayDestroy(hArray: CUarray) CUresult;
pub extern fn cuArray3DCreate_v2(pHandle: [*c]CUarray, pAllocateArray: [*c]const CUDA_ARRAY3D_DESCRIPTOR) CUresult;
pub extern fn cuArray3DGetDescriptor_v2(pArrayDescriptor: [*c]CUDA_ARRAY3D_DESCRIPTOR, hArray: CUarray) CUresult;
pub extern fn cuMipmappedArrayCreate(pHandle: [*c]CUmipmappedArray, pMipmappedArrayDesc: [*c]const CUDA_ARRAY3D_DESCRIPTOR, numMipmapLevels: c_uint) CUresult;
pub extern fn cuMipmappedArrayGetLevel(pLevelArray: [*c]CUarray, hMipmappedArray: CUmipmappedArray, level: c_uint) CUresult;
pub extern fn cuMipmappedArrayDestroy(hMipmappedArray: CUmipmappedArray) CUresult;
pub extern fn cuMemGetHandleForAddressRange(handle: ?*anyopaque, dptr: CUdeviceptr, size: usize, handleType: CUmemRangeHandleType, flags: c_ulonglong) CUresult;
pub const CU_MEM_DECOMPRESS_UNSUPPORTED: c_int = 0;
pub const CU_MEM_DECOMPRESS_ALGORITHM_DEFLATE: c_int = 1;
pub const CU_MEM_DECOMPRESS_ALGORITHM_SNAPPY: c_int = 2;
pub const enum_CUmemDecompressAlgorithm_enum = c_uint;
pub const CUmemDecompressAlgorithm = enum_CUmemDecompressAlgorithm_enum;
pub const struct_CUmemDecompressParams_st = extern struct {
    srcNumBytes: usize = @import("std").mem.zeroes(usize),
    dstNumBytes: usize = @import("std").mem.zeroes(usize),
    dstActBytes: [*c]cuuint32_t = @import("std").mem.zeroes([*c]cuuint32_t),
    src: ?*const anyopaque = @import("std").mem.zeroes(?*const anyopaque),
    dst: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    algo: CUmemDecompressAlgorithm = @import("std").mem.zeroes(CUmemDecompressAlgorithm),
    padding: [20]u8 = @import("std").mem.zeroes([20]u8),
};
pub const CUmemDecompressParams = struct_CUmemDecompressParams_st;
pub extern fn cuMemBatchDecompressAsync(paramsArray: [*c]CUmemDecompressParams, count: usize, flags: c_uint, errorIndex: [*c]usize, stream: CUstream) CUresult;
pub extern fn cuMemAddressReserve(ptr: [*c]CUdeviceptr, size: usize, alignment: usize, addr: CUdeviceptr, flags: c_ulonglong) CUresult;
pub extern fn cuMemAddressFree(ptr: CUdeviceptr, size: usize) CUresult;
pub extern fn cuMemCreate(handle: [*c]CUmemGenericAllocationHandle, size: usize, prop: [*c]const CUmemAllocationProp, flags: c_ulonglong) CUresult;
pub extern fn cuMemRelease(handle: CUmemGenericAllocationHandle) CUresult;
pub extern fn cuMemMap(ptr: CUdeviceptr, size: usize, offset: usize, handle: CUmemGenericAllocationHandle, flags: c_ulonglong) CUresult;
pub extern fn cuMemMapArrayAsync(mapInfoList: [*c]CUarrayMapInfo, count: c_uint, hStream: CUstream) CUresult;
pub extern fn cuMemUnmap(ptr: CUdeviceptr, size: usize) CUresult;
pub extern fn cuMemSetAccess(ptr: CUdeviceptr, size: usize, desc: [*c]const CUmemAccessDesc, count: usize) CUresult;
pub extern fn cuMemGetAccess(flags: [*c]c_ulonglong, location: [*c]const CUmemLocation, ptr: CUdeviceptr) CUresult;
pub extern fn cuMemExportToShareableHandle(shareableHandle: ?*anyopaque, handle: CUmemGenericAllocationHandle, handleType: CUmemAllocationHandleType, flags: c_ulonglong) CUresult;
pub extern fn cuMemImportFromShareableHandle(handle: [*c]CUmemGenericAllocationHandle, osHandle: ?*anyopaque, shHandleType: CUmemAllocationHandleType) CUresult;
pub extern fn cuMemGetAllocationGranularity(granularity: [*c]usize, prop: [*c]const CUmemAllocationProp, option: CUmemAllocationGranularity_flags) CUresult;
pub extern fn cuMemGetAllocationPropertiesFromHandle(prop: [*c]CUmemAllocationProp, handle: CUmemGenericAllocationHandle) CUresult;
pub extern fn cuMemRetainAllocationHandle(handle: [*c]CUmemGenericAllocationHandle, addr: ?*anyopaque) CUresult;
pub extern fn cuMemFreeAsync(dptr: CUdeviceptr, hStream: CUstream) CUresult;
pub extern fn cuMemAllocAsync(dptr: [*c]CUdeviceptr, bytesize: usize, hStream: CUstream) CUresult;
pub extern fn cuMemPoolTrimTo(pool: CUmemoryPool, minBytesToKeep: usize) CUresult;
pub extern fn cuMemPoolSetAttribute(pool: CUmemoryPool, attr: CUmemPool_attribute, value: ?*anyopaque) CUresult;
pub extern fn cuMemPoolGetAttribute(pool: CUmemoryPool, attr: CUmemPool_attribute, value: ?*anyopaque) CUresult;
pub extern fn cuMemPoolSetAccess(pool: CUmemoryPool, map: [*c]const CUmemAccessDesc, count: usize) CUresult;
pub extern fn cuMemPoolGetAccess(flags: [*c]CUmemAccess_flags, memPool: CUmemoryPool, location: [*c]CUmemLocation) CUresult;
pub extern fn cuMemPoolCreate(pool: [*c]CUmemoryPool, poolProps: [*c]const CUmemPoolProps) CUresult;
pub extern fn cuMemPoolDestroy(pool: CUmemoryPool) CUresult;
pub extern fn cuMemAllocFromPoolAsync(dptr: [*c]CUdeviceptr, bytesize: usize, pool: CUmemoryPool, hStream: CUstream) CUresult;
pub extern fn cuMemPoolExportToShareableHandle(handle_out: ?*anyopaque, pool: CUmemoryPool, handleType: CUmemAllocationHandleType, flags: c_ulonglong) CUresult;
pub extern fn cuMemPoolImportFromShareableHandle(pool_out: [*c]CUmemoryPool, handle: ?*anyopaque, handleType: CUmemAllocationHandleType, flags: c_ulonglong) CUresult;
pub extern fn cuMemPoolExportPointer(shareData_out: [*c]CUmemPoolPtrExportData, ptr: CUdeviceptr) CUresult;
pub extern fn cuMemPoolImportPointer(ptr_out: [*c]CUdeviceptr, pool: CUmemoryPool, shareData: [*c]CUmemPoolPtrExportData) CUresult;
pub extern fn cuMulticastCreate(mcHandle: [*c]CUmemGenericAllocationHandle, prop: [*c]const CUmulticastObjectProp) CUresult;
pub extern fn cuMulticastAddDevice(mcHandle: CUmemGenericAllocationHandle, dev: CUdevice) CUresult;
pub extern fn cuMulticastBindMem(mcHandle: CUmemGenericAllocationHandle, mcOffset: usize, memHandle: CUmemGenericAllocationHandle, memOffset: usize, size: usize, flags: c_ulonglong) CUresult;
pub extern fn cuMulticastBindAddr(mcHandle: CUmemGenericAllocationHandle, mcOffset: usize, memptr: CUdeviceptr, size: usize, flags: c_ulonglong) CUresult;
pub extern fn cuMulticastUnbind(mcHandle: CUmemGenericAllocationHandle, dev: CUdevice, mcOffset: usize, size: usize) CUresult;
pub extern fn cuMulticastGetGranularity(granularity: [*c]usize, prop: [*c]const CUmulticastObjectProp, option: CUmulticastGranularity_flags) CUresult;
pub extern fn cuPointerGetAttribute(data: ?*anyopaque, attribute: CUpointer_attribute, ptr: CUdeviceptr) CUresult;
pub extern fn cuMemPrefetchAsync(devPtr: CUdeviceptr, count: usize, dstDevice: CUdevice, hStream: CUstream) CUresult;
pub extern fn cuMemPrefetchAsync_v2(devPtr: CUdeviceptr, count: usize, location: CUmemLocation, flags: c_uint, hStream: CUstream) CUresult;
pub extern fn cuMemAdvise(devPtr: CUdeviceptr, count: usize, advice: CUmem_advise, device: CUdevice) CUresult;
pub extern fn cuMemAdvise_v2(devPtr: CUdeviceptr, count: usize, advice: CUmem_advise, location: CUmemLocation) CUresult;
pub extern fn cuMemRangeGetAttribute(data: ?*anyopaque, dataSize: usize, attribute: CUmem_range_attribute, devPtr: CUdeviceptr, count: usize) CUresult;
pub extern fn cuMemRangeGetAttributes(data: [*c]?*anyopaque, dataSizes: [*c]usize, attributes: [*c]CUmem_range_attribute, numAttributes: usize, devPtr: CUdeviceptr, count: usize) CUresult;
pub extern fn cuPointerSetAttribute(value: ?*const anyopaque, attribute: CUpointer_attribute, ptr: CUdeviceptr) CUresult;
pub extern fn cuPointerGetAttributes(numAttributes: c_uint, attributes: [*c]CUpointer_attribute, data: [*c]?*anyopaque, ptr: CUdeviceptr) CUresult;
pub extern fn cuStreamCreate(phStream: [*c]CUstream, Flags: c_uint) CUresult;
pub extern fn cuStreamCreateWithPriority(phStream: [*c]CUstream, flags: c_uint, priority: c_int) CUresult;
pub extern fn cuStreamGetPriority(hStream: CUstream, priority: [*c]c_int) CUresult;
pub extern fn cuStreamGetDevice(hStream: CUstream, device: [*c]CUdevice) CUresult;
pub extern fn cuStreamGetFlags(hStream: CUstream, flags: [*c]c_uint) CUresult;
pub extern fn cuStreamGetId(hStream: CUstream, streamId: [*c]c_ulonglong) CUresult;
pub extern fn cuStreamGetCtx(hStream: CUstream, pctx: [*c]CUcontext) CUresult;
pub extern fn cuStreamGetCtx_v2(hStream: CUstream, pCtx: [*c]CUcontext, pGreenCtx: [*c]CUgreenCtx) CUresult;
pub extern fn cuStreamWaitEvent(hStream: CUstream, hEvent: CUevent, Flags: c_uint) CUresult;
pub extern fn cuStreamAddCallback(hStream: CUstream, callback: CUstreamCallback, userData: ?*anyopaque, flags: c_uint) CUresult;
pub extern fn cuStreamBeginCapture_v2(hStream: CUstream, mode: CUstreamCaptureMode) CUresult;
pub extern fn cuStreamBeginCaptureToGraph(hStream: CUstream, hGraph: CUgraph, dependencies: [*c]const CUgraphNode, dependencyData: [*c]const CUgraphEdgeData, numDependencies: usize, mode: CUstreamCaptureMode) CUresult;
pub extern fn cuThreadExchangeStreamCaptureMode(mode: [*c]CUstreamCaptureMode) CUresult;
pub extern fn cuStreamEndCapture(hStream: CUstream, phGraph: [*c]CUgraph) CUresult;
pub extern fn cuStreamIsCapturing(hStream: CUstream, captureStatus: [*c]CUstreamCaptureStatus) CUresult;
pub extern fn cuStreamGetCaptureInfo_v2(hStream: CUstream, captureStatus_out: [*c]CUstreamCaptureStatus, id_out: [*c]cuuint64_t, graph_out: [*c]CUgraph, dependencies_out: [*c][*c]const CUgraphNode, numDependencies_out: [*c]usize) CUresult;
pub extern fn cuStreamGetCaptureInfo_v3(hStream: CUstream, captureStatus_out: [*c]CUstreamCaptureStatus, id_out: [*c]cuuint64_t, graph_out: [*c]CUgraph, dependencies_out: [*c][*c]const CUgraphNode, edgeData_out: [*c][*c]const CUgraphEdgeData, numDependencies_out: [*c]usize) CUresult;
pub extern fn cuStreamUpdateCaptureDependencies(hStream: CUstream, dependencies: [*c]CUgraphNode, numDependencies: usize, flags: c_uint) CUresult;
pub extern fn cuStreamUpdateCaptureDependencies_v2(hStream: CUstream, dependencies: [*c]CUgraphNode, dependencyData: [*c]const CUgraphEdgeData, numDependencies: usize, flags: c_uint) CUresult;
pub extern fn cuStreamAttachMemAsync(hStream: CUstream, dptr: CUdeviceptr, length: usize, flags: c_uint) CUresult;
pub extern fn cuStreamQuery(hStream: CUstream) CUresult;
pub extern fn cuStreamSynchronize(hStream: CUstream) CUresult;
pub extern fn cuStreamDestroy_v2(hStream: CUstream) CUresult;
pub extern fn cuStreamCopyAttributes(dst: CUstream, src: CUstream) CUresult;
pub extern fn cuStreamGetAttribute(hStream: CUstream, attr: CUstreamAttrID, value_out: [*c]CUstreamAttrValue) CUresult;
pub extern fn cuStreamSetAttribute(hStream: CUstream, attr: CUstreamAttrID, value: [*c]const CUstreamAttrValue) CUresult;
pub extern fn cuEventCreate(phEvent: [*c]CUevent, Flags: c_uint) CUresult;
pub extern fn cuEventRecord(hEvent: CUevent, hStream: CUstream) CUresult;
pub extern fn cuEventRecordWithFlags(hEvent: CUevent, hStream: CUstream, flags: c_uint) CUresult;
pub extern fn cuEventQuery(hEvent: CUevent) CUresult;
pub extern fn cuEventSynchronize(hEvent: CUevent) CUresult;
pub extern fn cuEventDestroy_v2(hEvent: CUevent) CUresult;
pub extern fn cuEventElapsedTime(pMilliseconds: [*c]f32, hStart: CUevent, hEnd: CUevent) CUresult;
pub extern fn cuEventElapsedTime_v2(pMilliseconds: [*c]f32, hStart: CUevent, hEnd: CUevent) CUresult;
pub extern fn cuImportExternalMemory(extMem_out: [*c]CUexternalMemory, memHandleDesc: [*c]const CUDA_EXTERNAL_MEMORY_HANDLE_DESC) CUresult;
pub extern fn cuExternalMemoryGetMappedBuffer(devPtr: [*c]CUdeviceptr, extMem: CUexternalMemory, bufferDesc: [*c]const CUDA_EXTERNAL_MEMORY_BUFFER_DESC) CUresult;
pub extern fn cuExternalMemoryGetMappedMipmappedArray(mipmap: [*c]CUmipmappedArray, extMem: CUexternalMemory, mipmapDesc: [*c]const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC) CUresult;
pub extern fn cuDestroyExternalMemory(extMem: CUexternalMemory) CUresult;
pub extern fn cuImportExternalSemaphore(extSem_out: [*c]CUexternalSemaphore, semHandleDesc: [*c]const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC) CUresult;
pub extern fn cuSignalExternalSemaphoresAsync(extSemArray: [*c]const CUexternalSemaphore, paramsArray: [*c]const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS, numExtSems: c_uint, stream: CUstream) CUresult;
pub extern fn cuWaitExternalSemaphoresAsync(extSemArray: [*c]const CUexternalSemaphore, paramsArray: [*c]const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS, numExtSems: c_uint, stream: CUstream) CUresult;
pub extern fn cuDestroyExternalSemaphore(extSem: CUexternalSemaphore) CUresult;
pub extern fn cuStreamWaitValue32_v2(stream: CUstream, addr: CUdeviceptr, value: cuuint32_t, flags: c_uint) CUresult;
pub extern fn cuStreamWaitValue64_v2(stream: CUstream, addr: CUdeviceptr, value: cuuint64_t, flags: c_uint) CUresult;
pub extern fn cuStreamWriteValue32_v2(stream: CUstream, addr: CUdeviceptr, value: cuuint32_t, flags: c_uint) CUresult;
pub extern fn cuStreamWriteValue64_v2(stream: CUstream, addr: CUdeviceptr, value: cuuint64_t, flags: c_uint) CUresult;
pub extern fn cuStreamBatchMemOp_v2(stream: CUstream, count: c_uint, paramArray: [*c]CUstreamBatchMemOpParams, flags: c_uint) CUresult;
pub extern fn cuFuncGetAttribute(pi: [*c]c_int, attrib: CUfunction_attribute, hfunc: CUfunction) CUresult;
pub extern fn cuFuncSetAttribute(hfunc: CUfunction, attrib: CUfunction_attribute, value: c_int) CUresult;
pub extern fn cuFuncSetCacheConfig(hfunc: CUfunction, config: CUfunc_cache) CUresult;
pub extern fn cuFuncGetModule(hmod: [*c]CUmodule, hfunc: CUfunction) CUresult;
pub extern fn cuFuncGetName(name: [*c][*c]const u8, hfunc: CUfunction) CUresult;
pub extern fn cuFuncGetParamInfo(func: CUfunction, paramIndex: usize, paramOffset: [*c]usize, paramSize: [*c]usize) CUresult;
pub const CU_FUNCTION_LOADING_STATE_UNLOADED: c_int = 0;
pub const CU_FUNCTION_LOADING_STATE_LOADED: c_int = 1;
pub const CU_FUNCTION_LOADING_STATE_MAX: c_int = 2;
pub const enum_CUfunctionLoadingState_enum = c_uint;
pub const CUfunctionLoadingState = enum_CUfunctionLoadingState_enum;
pub extern fn cuFuncIsLoaded(state: [*c]CUfunctionLoadingState, function: CUfunction) CUresult;
pub extern fn cuFuncLoad(function: CUfunction) CUresult;
pub extern fn cuLaunchKernel(f: CUfunction, gridDimX: c_uint, gridDimY: c_uint, gridDimZ: c_uint, blockDimX: c_uint, blockDimY: c_uint, blockDimZ: c_uint, sharedMemBytes: c_uint, hStream: CUstream, kernelParams: [*c]?*anyopaque, extra: [*c]?*anyopaque) CUresult;
pub extern fn cuLaunchKernelEx(config: [*c]const CUlaunchConfig, f: CUfunction, kernelParams: [*c]?*anyopaque, extra: [*c]?*anyopaque) CUresult;
pub extern fn cuLaunchCooperativeKernel(f: CUfunction, gridDimX: c_uint, gridDimY: c_uint, gridDimZ: c_uint, blockDimX: c_uint, blockDimY: c_uint, blockDimZ: c_uint, sharedMemBytes: c_uint, hStream: CUstream, kernelParams: [*c]?*anyopaque) CUresult;
pub extern fn cuLaunchCooperativeKernelMultiDevice(launchParamsList: [*c]CUDA_LAUNCH_PARAMS, numDevices: c_uint, flags: c_uint) CUresult;
pub extern fn cuLaunchHostFunc(hStream: CUstream, @"fn": CUhostFn, userData: ?*anyopaque) CUresult;
pub extern fn cuFuncSetBlockShape(hfunc: CUfunction, x: c_int, y: c_int, z: c_int) CUresult;
pub extern fn cuFuncSetSharedSize(hfunc: CUfunction, bytes: c_uint) CUresult;
pub extern fn cuParamSetSize(hfunc: CUfunction, numbytes: c_uint) CUresult;
pub extern fn cuParamSeti(hfunc: CUfunction, offset: c_int, value: c_uint) CUresult;
pub extern fn cuParamSetf(hfunc: CUfunction, offset: c_int, value: f32) CUresult;
pub extern fn cuParamSetv(hfunc: CUfunction, offset: c_int, ptr: ?*anyopaque, numbytes: c_uint) CUresult;
pub extern fn cuLaunch(f: CUfunction) CUresult;
pub extern fn cuLaunchGrid(f: CUfunction, grid_width: c_int, grid_height: c_int) CUresult;
pub extern fn cuLaunchGridAsync(f: CUfunction, grid_width: c_int, grid_height: c_int, hStream: CUstream) CUresult;
pub extern fn cuParamSetTexRef(hfunc: CUfunction, texunit: c_int, hTexRef: CUtexref) CUresult;
pub extern fn cuFuncSetSharedMemConfig(hfunc: CUfunction, config: CUsharedconfig) CUresult;
pub extern fn cuGraphCreate(phGraph: [*c]CUgraph, flags: c_uint) CUresult;
pub extern fn cuGraphAddKernelNode_v2(phGraphNode: [*c]CUgraphNode, hGraph: CUgraph, dependencies: [*c]const CUgraphNode, numDependencies: usize, nodeParams: [*c]const CUDA_KERNEL_NODE_PARAMS) CUresult;
pub extern fn cuGraphKernelNodeGetParams_v2(hNode: CUgraphNode, nodeParams: [*c]CUDA_KERNEL_NODE_PARAMS) CUresult;
pub extern fn cuGraphKernelNodeSetParams_v2(hNode: CUgraphNode, nodeParams: [*c]const CUDA_KERNEL_NODE_PARAMS) CUresult;
pub extern fn cuGraphAddMemcpyNode(phGraphNode: [*c]CUgraphNode, hGraph: CUgraph, dependencies: [*c]const CUgraphNode, numDependencies: usize, copyParams: [*c]const CUDA_MEMCPY3D, ctx: CUcontext) CUresult;
pub extern fn cuGraphMemcpyNodeGetParams(hNode: CUgraphNode, nodeParams: [*c]CUDA_MEMCPY3D) CUresult;
pub extern fn cuGraphMemcpyNodeSetParams(hNode: CUgraphNode, nodeParams: [*c]const CUDA_MEMCPY3D) CUresult;
pub extern fn cuGraphAddMemsetNode(phGraphNode: [*c]CUgraphNode, hGraph: CUgraph, dependencies: [*c]const CUgraphNode, numDependencies: usize, memsetParams: [*c]const CUDA_MEMSET_NODE_PARAMS, ctx: CUcontext) CUresult;
pub extern fn cuGraphMemsetNodeGetParams(hNode: CUgraphNode, nodeParams: [*c]CUDA_MEMSET_NODE_PARAMS) CUresult;
pub extern fn cuGraphMemsetNodeSetParams(hNode: CUgraphNode, nodeParams: [*c]const CUDA_MEMSET_NODE_PARAMS) CUresult;
pub extern fn cuGraphAddHostNode(phGraphNode: [*c]CUgraphNode, hGraph: CUgraph, dependencies: [*c]const CUgraphNode, numDependencies: usize, nodeParams: [*c]const CUDA_HOST_NODE_PARAMS) CUresult;
pub extern fn cuGraphHostNodeGetParams(hNode: CUgraphNode, nodeParams: [*c]CUDA_HOST_NODE_PARAMS) CUresult;
pub extern fn cuGraphHostNodeSetParams(hNode: CUgraphNode, nodeParams: [*c]const CUDA_HOST_NODE_PARAMS) CUresult;
pub extern fn cuGraphAddChildGraphNode(phGraphNode: [*c]CUgraphNode, hGraph: CUgraph, dependencies: [*c]const CUgraphNode, numDependencies: usize, childGraph: CUgraph) CUresult;
pub extern fn cuGraphChildGraphNodeGetGraph(hNode: CUgraphNode, phGraph: [*c]CUgraph) CUresult;
pub extern fn cuGraphAddEmptyNode(phGraphNode: [*c]CUgraphNode, hGraph: CUgraph, dependencies: [*c]const CUgraphNode, numDependencies: usize) CUresult;
pub extern fn cuGraphAddEventRecordNode(phGraphNode: [*c]CUgraphNode, hGraph: CUgraph, dependencies: [*c]const CUgraphNode, numDependencies: usize, event: CUevent) CUresult;
pub extern fn cuGraphEventRecordNodeGetEvent(hNode: CUgraphNode, event_out: [*c]CUevent) CUresult;
pub extern fn cuGraphEventRecordNodeSetEvent(hNode: CUgraphNode, event: CUevent) CUresult;
pub extern fn cuGraphAddEventWaitNode(phGraphNode: [*c]CUgraphNode, hGraph: CUgraph, dependencies: [*c]const CUgraphNode, numDependencies: usize, event: CUevent) CUresult;
pub extern fn cuGraphEventWaitNodeGetEvent(hNode: CUgraphNode, event_out: [*c]CUevent) CUresult;
pub extern fn cuGraphEventWaitNodeSetEvent(hNode: CUgraphNode, event: CUevent) CUresult;
pub extern fn cuGraphAddExternalSemaphoresSignalNode(phGraphNode: [*c]CUgraphNode, hGraph: CUgraph, dependencies: [*c]const CUgraphNode, numDependencies: usize, nodeParams: [*c]const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) CUresult;
pub extern fn cuGraphExternalSemaphoresSignalNodeGetParams(hNode: CUgraphNode, params_out: [*c]CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) CUresult;
pub extern fn cuGraphExternalSemaphoresSignalNodeSetParams(hNode: CUgraphNode, nodeParams: [*c]const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) CUresult;
pub extern fn cuGraphAddExternalSemaphoresWaitNode(phGraphNode: [*c]CUgraphNode, hGraph: CUgraph, dependencies: [*c]const CUgraphNode, numDependencies: usize, nodeParams: [*c]const CUDA_EXT_SEM_WAIT_NODE_PARAMS) CUresult;
pub extern fn cuGraphExternalSemaphoresWaitNodeGetParams(hNode: CUgraphNode, params_out: [*c]CUDA_EXT_SEM_WAIT_NODE_PARAMS) CUresult;
pub extern fn cuGraphExternalSemaphoresWaitNodeSetParams(hNode: CUgraphNode, nodeParams: [*c]const CUDA_EXT_SEM_WAIT_NODE_PARAMS) CUresult;
pub extern fn cuGraphAddBatchMemOpNode(phGraphNode: [*c]CUgraphNode, hGraph: CUgraph, dependencies: [*c]const CUgraphNode, numDependencies: usize, nodeParams: [*c]const CUDA_BATCH_MEM_OP_NODE_PARAMS) CUresult;
pub extern fn cuGraphBatchMemOpNodeGetParams(hNode: CUgraphNode, nodeParams_out: [*c]CUDA_BATCH_MEM_OP_NODE_PARAMS) CUresult;
pub extern fn cuGraphBatchMemOpNodeSetParams(hNode: CUgraphNode, nodeParams: [*c]const CUDA_BATCH_MEM_OP_NODE_PARAMS) CUresult;
pub extern fn cuGraphExecBatchMemOpNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: [*c]const CUDA_BATCH_MEM_OP_NODE_PARAMS) CUresult;
pub extern fn cuGraphAddMemAllocNode(phGraphNode: [*c]CUgraphNode, hGraph: CUgraph, dependencies: [*c]const CUgraphNode, numDependencies: usize, nodeParams: [*c]CUDA_MEM_ALLOC_NODE_PARAMS) CUresult;
pub extern fn cuGraphMemAllocNodeGetParams(hNode: CUgraphNode, params_out: [*c]CUDA_MEM_ALLOC_NODE_PARAMS) CUresult;
pub extern fn cuGraphAddMemFreeNode(phGraphNode: [*c]CUgraphNode, hGraph: CUgraph, dependencies: [*c]const CUgraphNode, numDependencies: usize, dptr: CUdeviceptr) CUresult;
pub extern fn cuGraphMemFreeNodeGetParams(hNode: CUgraphNode, dptr_out: [*c]CUdeviceptr) CUresult;
pub extern fn cuDeviceGraphMemTrim(device: CUdevice) CUresult;
pub extern fn cuDeviceGetGraphMemAttribute(device: CUdevice, attr: CUgraphMem_attribute, value: ?*anyopaque) CUresult;
pub extern fn cuDeviceSetGraphMemAttribute(device: CUdevice, attr: CUgraphMem_attribute, value: ?*anyopaque) CUresult;
pub extern fn cuGraphClone(phGraphClone: [*c]CUgraph, originalGraph: CUgraph) CUresult;
pub extern fn cuGraphNodeFindInClone(phNode: [*c]CUgraphNode, hOriginalNode: CUgraphNode, hClonedGraph: CUgraph) CUresult;
pub extern fn cuGraphNodeGetType(hNode: CUgraphNode, @"type": [*c]CUgraphNodeType) CUresult;
pub extern fn cuGraphGetNodes(hGraph: CUgraph, nodes: [*c]CUgraphNode, numNodes: [*c]usize) CUresult;
pub extern fn cuGraphGetRootNodes(hGraph: CUgraph, rootNodes: [*c]CUgraphNode, numRootNodes: [*c]usize) CUresult;
pub extern fn cuGraphGetEdges(hGraph: CUgraph, from: [*c]CUgraphNode, to: [*c]CUgraphNode, numEdges: [*c]usize) CUresult;
pub extern fn cuGraphGetEdges_v2(hGraph: CUgraph, from: [*c]CUgraphNode, to: [*c]CUgraphNode, edgeData: [*c]CUgraphEdgeData, numEdges: [*c]usize) CUresult;
pub extern fn cuGraphNodeGetDependencies(hNode: CUgraphNode, dependencies: [*c]CUgraphNode, numDependencies: [*c]usize) CUresult;
pub extern fn cuGraphNodeGetDependencies_v2(hNode: CUgraphNode, dependencies: [*c]CUgraphNode, edgeData: [*c]CUgraphEdgeData, numDependencies: [*c]usize) CUresult;
pub extern fn cuGraphNodeGetDependentNodes(hNode: CUgraphNode, dependentNodes: [*c]CUgraphNode, numDependentNodes: [*c]usize) CUresult;
pub extern fn cuGraphNodeGetDependentNodes_v2(hNode: CUgraphNode, dependentNodes: [*c]CUgraphNode, edgeData: [*c]CUgraphEdgeData, numDependentNodes: [*c]usize) CUresult;
pub extern fn cuGraphAddDependencies(hGraph: CUgraph, from: [*c]const CUgraphNode, to: [*c]const CUgraphNode, numDependencies: usize) CUresult;
pub extern fn cuGraphAddDependencies_v2(hGraph: CUgraph, from: [*c]const CUgraphNode, to: [*c]const CUgraphNode, edgeData: [*c]const CUgraphEdgeData, numDependencies: usize) CUresult;
pub extern fn cuGraphRemoveDependencies(hGraph: CUgraph, from: [*c]const CUgraphNode, to: [*c]const CUgraphNode, numDependencies: usize) CUresult;
pub extern fn cuGraphRemoveDependencies_v2(hGraph: CUgraph, from: [*c]const CUgraphNode, to: [*c]const CUgraphNode, edgeData: [*c]const CUgraphEdgeData, numDependencies: usize) CUresult;
pub extern fn cuGraphDestroyNode(hNode: CUgraphNode) CUresult;
pub extern fn cuGraphInstantiateWithFlags(phGraphExec: [*c]CUgraphExec, hGraph: CUgraph, flags: c_ulonglong) CUresult;
pub extern fn cuGraphInstantiateWithParams(phGraphExec: [*c]CUgraphExec, hGraph: CUgraph, instantiateParams: [*c]CUDA_GRAPH_INSTANTIATE_PARAMS) CUresult;
pub extern fn cuGraphExecGetFlags(hGraphExec: CUgraphExec, flags: [*c]cuuint64_t) CUresult;
pub extern fn cuGraphExecKernelNodeSetParams_v2(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: [*c]const CUDA_KERNEL_NODE_PARAMS) CUresult;
pub extern fn cuGraphExecMemcpyNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, copyParams: [*c]const CUDA_MEMCPY3D, ctx: CUcontext) CUresult;
pub extern fn cuGraphExecMemsetNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, memsetParams: [*c]const CUDA_MEMSET_NODE_PARAMS, ctx: CUcontext) CUresult;
pub extern fn cuGraphExecHostNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: [*c]const CUDA_HOST_NODE_PARAMS) CUresult;
pub extern fn cuGraphExecChildGraphNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, childGraph: CUgraph) CUresult;
pub extern fn cuGraphExecEventRecordNodeSetEvent(hGraphExec: CUgraphExec, hNode: CUgraphNode, event: CUevent) CUresult;
pub extern fn cuGraphExecEventWaitNodeSetEvent(hGraphExec: CUgraphExec, hNode: CUgraphNode, event: CUevent) CUresult;
pub extern fn cuGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: [*c]const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) CUresult;
pub extern fn cuGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: [*c]const CUDA_EXT_SEM_WAIT_NODE_PARAMS) CUresult;
pub extern fn cuGraphNodeSetEnabled(hGraphExec: CUgraphExec, hNode: CUgraphNode, isEnabled: c_uint) CUresult;
pub extern fn cuGraphNodeGetEnabled(hGraphExec: CUgraphExec, hNode: CUgraphNode, isEnabled: [*c]c_uint) CUresult;
pub extern fn cuGraphUpload(hGraphExec: CUgraphExec, hStream: CUstream) CUresult;
pub extern fn cuGraphLaunch(hGraphExec: CUgraphExec, hStream: CUstream) CUresult;
pub extern fn cuGraphExecDestroy(hGraphExec: CUgraphExec) CUresult;
pub extern fn cuGraphDestroy(hGraph: CUgraph) CUresult;
pub extern fn cuGraphExecUpdate_v2(hGraphExec: CUgraphExec, hGraph: CUgraph, resultInfo: [*c]CUgraphExecUpdateResultInfo) CUresult;
pub extern fn cuGraphKernelNodeCopyAttributes(dst: CUgraphNode, src: CUgraphNode) CUresult;
pub extern fn cuGraphKernelNodeGetAttribute(hNode: CUgraphNode, attr: CUkernelNodeAttrID, value_out: [*c]CUkernelNodeAttrValue) CUresult;
pub extern fn cuGraphKernelNodeSetAttribute(hNode: CUgraphNode, attr: CUkernelNodeAttrID, value: [*c]const CUkernelNodeAttrValue) CUresult;
pub extern fn cuGraphDebugDotPrint(hGraph: CUgraph, path: [*c]const u8, flags: c_uint) CUresult;
pub extern fn cuUserObjectCreate(object_out: [*c]CUuserObject, ptr: ?*anyopaque, destroy: CUhostFn, initialRefcount: c_uint, flags: c_uint) CUresult;
pub extern fn cuUserObjectRetain(object: CUuserObject, count: c_uint) CUresult;
pub extern fn cuUserObjectRelease(object: CUuserObject, count: c_uint) CUresult;
pub extern fn cuGraphRetainUserObject(graph: CUgraph, object: CUuserObject, count: c_uint, flags: c_uint) CUresult;
pub extern fn cuGraphReleaseUserObject(graph: CUgraph, object: CUuserObject, count: c_uint) CUresult;
pub extern fn cuGraphAddNode(phGraphNode: [*c]CUgraphNode, hGraph: CUgraph, dependencies: [*c]const CUgraphNode, numDependencies: usize, nodeParams: [*c]CUgraphNodeParams) CUresult;
pub extern fn cuGraphAddNode_v2(phGraphNode: [*c]CUgraphNode, hGraph: CUgraph, dependencies: [*c]const CUgraphNode, dependencyData: [*c]const CUgraphEdgeData, numDependencies: usize, nodeParams: [*c]CUgraphNodeParams) CUresult;
pub extern fn cuGraphNodeSetParams(hNode: CUgraphNode, nodeParams: [*c]CUgraphNodeParams) CUresult;
pub extern fn cuGraphExecNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: [*c]CUgraphNodeParams) CUresult;
pub extern fn cuGraphConditionalHandleCreate(pHandle_out: [*c]CUgraphConditionalHandle, hGraph: CUgraph, ctx: CUcontext, defaultLaunchValue: c_uint, flags: c_uint) CUresult;
pub extern fn cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks: [*c]c_int, func: CUfunction, blockSize: c_int, dynamicSMemSize: usize) CUresult;
pub extern fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks: [*c]c_int, func: CUfunction, blockSize: c_int, dynamicSMemSize: usize, flags: c_uint) CUresult;
pub extern fn cuOccupancyMaxPotentialBlockSize(minGridSize: [*c]c_int, blockSize: [*c]c_int, func: CUfunction, blockSizeToDynamicSMemSize: CUoccupancyB2DSize, dynamicSMemSize: usize, blockSizeLimit: c_int) CUresult;
pub extern fn cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize: [*c]c_int, blockSize: [*c]c_int, func: CUfunction, blockSizeToDynamicSMemSize: CUoccupancyB2DSize, dynamicSMemSize: usize, blockSizeLimit: c_int, flags: c_uint) CUresult;
pub extern fn cuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize: [*c]usize, func: CUfunction, numBlocks: c_int, blockSize: c_int) CUresult;
pub extern fn cuOccupancyMaxPotentialClusterSize(clusterSize: [*c]c_int, func: CUfunction, config: [*c]const CUlaunchConfig) CUresult;
pub extern fn cuOccupancyMaxActiveClusters(numClusters: [*c]c_int, func: CUfunction, config: [*c]const CUlaunchConfig) CUresult;
pub extern fn cuTexRefSetArray(hTexRef: CUtexref, hArray: CUarray, Flags: c_uint) CUresult;
pub extern fn cuTexRefSetMipmappedArray(hTexRef: CUtexref, hMipmappedArray: CUmipmappedArray, Flags: c_uint) CUresult;
pub extern fn cuTexRefSetAddress_v2(ByteOffset: [*c]usize, hTexRef: CUtexref, dptr: CUdeviceptr, bytes: usize) CUresult;
pub extern fn cuTexRefSetAddress2D_v3(hTexRef: CUtexref, desc: [*c]const CUDA_ARRAY_DESCRIPTOR, dptr: CUdeviceptr, Pitch: usize) CUresult;
pub extern fn cuTexRefSetFormat(hTexRef: CUtexref, fmt: CUarray_format, NumPackedComponents: c_int) CUresult;
pub extern fn cuTexRefSetAddressMode(hTexRef: CUtexref, dim: c_int, am: CUaddress_mode) CUresult;
pub extern fn cuTexRefSetFilterMode(hTexRef: CUtexref, fm: CUfilter_mode) CUresult;
pub extern fn cuTexRefSetMipmapFilterMode(hTexRef: CUtexref, fm: CUfilter_mode) CUresult;
pub extern fn cuTexRefSetMipmapLevelBias(hTexRef: CUtexref, bias: f32) CUresult;
pub extern fn cuTexRefSetMipmapLevelClamp(hTexRef: CUtexref, minMipmapLevelClamp: f32, maxMipmapLevelClamp: f32) CUresult;
pub extern fn cuTexRefSetMaxAnisotropy(hTexRef: CUtexref, maxAniso: c_uint) CUresult;
pub extern fn cuTexRefSetBorderColor(hTexRef: CUtexref, pBorderColor: [*c]f32) CUresult;
pub extern fn cuTexRefSetFlags(hTexRef: CUtexref, Flags: c_uint) CUresult;
pub extern fn cuTexRefGetAddress_v2(pdptr: [*c]CUdeviceptr, hTexRef: CUtexref) CUresult;
pub extern fn cuTexRefGetArray(phArray: [*c]CUarray, hTexRef: CUtexref) CUresult;
pub extern fn cuTexRefGetMipmappedArray(phMipmappedArray: [*c]CUmipmappedArray, hTexRef: CUtexref) CUresult;
pub extern fn cuTexRefGetAddressMode(pam: [*c]CUaddress_mode, hTexRef: CUtexref, dim: c_int) CUresult;
pub extern fn cuTexRefGetFilterMode(pfm: [*c]CUfilter_mode, hTexRef: CUtexref) CUresult;
pub extern fn cuTexRefGetFormat(pFormat: [*c]CUarray_format, pNumChannels: [*c]c_int, hTexRef: CUtexref) CUresult;
pub extern fn cuTexRefGetMipmapFilterMode(pfm: [*c]CUfilter_mode, hTexRef: CUtexref) CUresult;
pub extern fn cuTexRefGetMipmapLevelBias(pbias: [*c]f32, hTexRef: CUtexref) CUresult;
pub extern fn cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp: [*c]f32, pmaxMipmapLevelClamp: [*c]f32, hTexRef: CUtexref) CUresult;
pub extern fn cuTexRefGetMaxAnisotropy(pmaxAniso: [*c]c_int, hTexRef: CUtexref) CUresult;
pub extern fn cuTexRefGetBorderColor(pBorderColor: [*c]f32, hTexRef: CUtexref) CUresult;
pub extern fn cuTexRefGetFlags(pFlags: [*c]c_uint, hTexRef: CUtexref) CUresult;
pub extern fn cuTexRefCreate(pTexRef: [*c]CUtexref) CUresult;
pub extern fn cuTexRefDestroy(hTexRef: CUtexref) CUresult;
pub extern fn cuSurfRefSetArray(hSurfRef: CUsurfref, hArray: CUarray, Flags: c_uint) CUresult;
pub extern fn cuSurfRefGetArray(phArray: [*c]CUarray, hSurfRef: CUsurfref) CUresult;
pub extern fn cuTexObjectCreate(pTexObject: [*c]CUtexObject, pResDesc: [*c]const CUDA_RESOURCE_DESC, pTexDesc: [*c]const CUDA_TEXTURE_DESC, pResViewDesc: [*c]const CUDA_RESOURCE_VIEW_DESC) CUresult;
pub extern fn cuTexObjectDestroy(texObject: CUtexObject) CUresult;
pub extern fn cuTexObjectGetResourceDesc(pResDesc: [*c]CUDA_RESOURCE_DESC, texObject: CUtexObject) CUresult;
pub extern fn cuTexObjectGetTextureDesc(pTexDesc: [*c]CUDA_TEXTURE_DESC, texObject: CUtexObject) CUresult;
pub extern fn cuTexObjectGetResourceViewDesc(pResViewDesc: [*c]CUDA_RESOURCE_VIEW_DESC, texObject: CUtexObject) CUresult;
pub extern fn cuSurfObjectCreate(pSurfObject: [*c]CUsurfObject, pResDesc: [*c]const CUDA_RESOURCE_DESC) CUresult;
pub extern fn cuSurfObjectDestroy(surfObject: CUsurfObject) CUresult;
pub extern fn cuSurfObjectGetResourceDesc(pResDesc: [*c]CUDA_RESOURCE_DESC, surfObject: CUsurfObject) CUresult;
pub extern fn cuTensorMapEncodeTiled(tensorMap: [*c]CUtensorMap, tensorDataType: CUtensorMapDataType, tensorRank: cuuint32_t, globalAddress: ?*anyopaque, globalDim: [*c]const cuuint64_t, globalStrides: [*c]const cuuint64_t, boxDim: [*c]const cuuint32_t, elementStrides: [*c]const cuuint32_t, interleave: CUtensorMapInterleave, swizzle: CUtensorMapSwizzle, l2Promotion: CUtensorMapL2promotion, oobFill: CUtensorMapFloatOOBfill) CUresult;
pub extern fn cuTensorMapEncodeIm2col(tensorMap: [*c]CUtensorMap, tensorDataType: CUtensorMapDataType, tensorRank: cuuint32_t, globalAddress: ?*anyopaque, globalDim: [*c]const cuuint64_t, globalStrides: [*c]const cuuint64_t, pixelBoxLowerCorner: [*c]const c_int, pixelBoxUpperCorner: [*c]const c_int, channelsPerPixel: cuuint32_t, pixelsPerColumn: cuuint32_t, elementStrides: [*c]const cuuint32_t, interleave: CUtensorMapInterleave, swizzle: CUtensorMapSwizzle, l2Promotion: CUtensorMapL2promotion, oobFill: CUtensorMapFloatOOBfill) CUresult;
pub extern fn cuTensorMapEncodeIm2colWide(tensorMap: [*c]CUtensorMap, tensorDataType: CUtensorMapDataType, tensorRank: cuuint32_t, globalAddress: ?*anyopaque, globalDim: [*c]const cuuint64_t, globalStrides: [*c]const cuuint64_t, pixelBoxLowerCornerWidth: c_int, pixelBoxUpperCornerWidth: c_int, channelsPerPixel: cuuint32_t, pixelsPerColumn: cuuint32_t, elementStrides: [*c]const cuuint32_t, interleave: CUtensorMapInterleave, mode: CUtensorMapIm2ColWideMode, swizzle: CUtensorMapSwizzle, l2Promotion: CUtensorMapL2promotion, oobFill: CUtensorMapFloatOOBfill) CUresult;
pub extern fn cuTensorMapReplaceAddress(tensorMap: [*c]CUtensorMap, globalAddress: ?*anyopaque) CUresult;
pub extern fn cuDeviceCanAccessPeer(canAccessPeer: [*c]c_int, dev: CUdevice, peerDev: CUdevice) CUresult;
pub extern fn cuCtxEnablePeerAccess(peerContext: CUcontext, Flags: c_uint) CUresult;
pub extern fn cuCtxDisablePeerAccess(peerContext: CUcontext) CUresult;
pub extern fn cuDeviceGetP2PAttribute(value: [*c]c_int, attrib: CUdevice_P2PAttribute, srcDevice: CUdevice, dstDevice: CUdevice) CUresult;
pub extern fn cuGraphicsUnregisterResource(resource: CUgraphicsResource) CUresult;
pub extern fn cuGraphicsSubResourceGetMappedArray(pArray: [*c]CUarray, resource: CUgraphicsResource, arrayIndex: c_uint, mipLevel: c_uint) CUresult;
pub extern fn cuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray: [*c]CUmipmappedArray, resource: CUgraphicsResource) CUresult;
pub extern fn cuGraphicsResourceGetMappedPointer_v2(pDevPtr: [*c]CUdeviceptr, pSize: [*c]usize, resource: CUgraphicsResource) CUresult;
pub extern fn cuGraphicsResourceSetMapFlags_v2(resource: CUgraphicsResource, flags: c_uint) CUresult;
pub extern fn cuGraphicsMapResources(count: c_uint, resources: [*c]CUgraphicsResource, hStream: CUstream) CUresult;
pub extern fn cuGraphicsUnmapResources(count: c_uint, resources: [*c]CUgraphicsResource, hStream: CUstream) CUresult;
pub extern fn cuGetProcAddress_v2(symbol: [*c]const u8, pfn: [*c]?*anyopaque, cudaVersion: c_int, flags: cuuint64_t, symbolStatus: [*c]CUdriverProcAddressQueryResult) CUresult;
pub const CU_COREDUMP_ENABLE_ON_EXCEPTION: c_int = 1;
pub const CU_COREDUMP_TRIGGER_HOST: c_int = 2;
pub const CU_COREDUMP_LIGHTWEIGHT: c_int = 3;
pub const CU_COREDUMP_ENABLE_USER_TRIGGER: c_int = 4;
pub const CU_COREDUMP_FILE: c_int = 5;
pub const CU_COREDUMP_PIPE: c_int = 6;
pub const CU_COREDUMP_GENERATION_FLAGS: c_int = 7;
pub const CU_COREDUMP_MAX: c_int = 8;
pub const enum_CUcoredumpSettings_enum = c_uint;
pub const CUcoredumpSettings = enum_CUcoredumpSettings_enum;
pub const CU_COREDUMP_DEFAULT_FLAGS: c_int = 0;
pub const CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES: c_int = 1;
pub const CU_COREDUMP_SKIP_GLOBAL_MEMORY: c_int = 2;
pub const CU_COREDUMP_SKIP_SHARED_MEMORY: c_int = 4;
pub const CU_COREDUMP_SKIP_LOCAL_MEMORY: c_int = 8;
pub const CU_COREDUMP_SKIP_ABORT: c_int = 16;
pub const CU_COREDUMP_SKIP_CONSTBANK_MEMORY: c_int = 32;
pub const CU_COREDUMP_LIGHTWEIGHT_FLAGS: c_int = 47;
pub const enum_CUCoredumpGenerationFlags = c_uint;
pub const CUCoredumpGenerationFlags = enum_CUCoredumpGenerationFlags;
pub extern fn cuCoredumpGetAttribute(attrib: CUcoredumpSettings, value: ?*anyopaque, size: [*c]usize) CUresult;
pub extern fn cuCoredumpGetAttributeGlobal(attrib: CUcoredumpSettings, value: ?*anyopaque, size: [*c]usize) CUresult;
pub extern fn cuCoredumpSetAttribute(attrib: CUcoredumpSettings, value: ?*anyopaque, size: [*c]usize) CUresult;
pub extern fn cuCoredumpSetAttributeGlobal(attrib: CUcoredumpSettings, value: ?*anyopaque, size: [*c]usize) CUresult;
pub extern fn cuGetExportTable(ppExportTable: [*c]?*const anyopaque, pExportTableId: [*c]const CUuuid) CUresult;
pub const struct_CUdevResourceDesc_st = opaque {};
pub const CUdevResourceDesc = ?*struct_CUdevResourceDesc_st;
pub const CU_GREEN_CTX_DEFAULT_STREAM: c_int = 1;
pub const CUgreenCtxCreate_flags = c_uint;
pub const CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING: c_int = 1;
pub const CU_DEV_SM_RESOURCE_SPLIT_MAX_POTENTIAL_CLUSTER_SIZE: c_int = 2;
pub const CUdevSmResourceSplit_flags = c_uint;
pub const CU_DEV_RESOURCE_TYPE_INVALID: c_int = 0;
pub const CU_DEV_RESOURCE_TYPE_SM: c_int = 1;
pub const CUdevResourceType = c_uint;
pub const struct_CUdevSmResource_st = extern struct {
    smCount: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const CUdevSmResource = struct_CUdevSmResource_st;
const union_unnamed_44 = extern union {
    sm: CUdevSmResource,
    _oversize: [48]u8,
};
pub const struct_CUdevResource_st = extern struct {
    type: CUdevResourceType = @import("std").mem.zeroes(CUdevResourceType),
    _internal_padding: [92]u8 = @import("std").mem.zeroes([92]u8),
    unnamed_0: union_unnamed_44 = @import("std").mem.zeroes(union_unnamed_44),
};
pub const CUdevResource_v1 = struct_CUdevResource_st;
pub const CUdevResource = CUdevResource_v1;
pub extern fn cuGreenCtxCreate(phCtx: [*c]CUgreenCtx, desc: CUdevResourceDesc, dev: CUdevice, flags: c_uint) CUresult;
pub extern fn cuGreenCtxDestroy(hCtx: CUgreenCtx) CUresult;
pub extern fn cuCtxFromGreenCtx(pContext: [*c]CUcontext, hCtx: CUgreenCtx) CUresult;
pub extern fn cuDeviceGetDevResource(device: CUdevice, resource: [*c]CUdevResource, @"type": CUdevResourceType) CUresult;
pub extern fn cuCtxGetDevResource(hCtx: CUcontext, resource: [*c]CUdevResource, @"type": CUdevResourceType) CUresult;
pub extern fn cuGreenCtxGetDevResource(hCtx: CUgreenCtx, resource: [*c]CUdevResource, @"type": CUdevResourceType) CUresult;
pub extern fn cuDevSmResourceSplitByCount(result: [*c]CUdevResource, nbGroups: [*c]c_uint, input: [*c]const CUdevResource, remaining: [*c]CUdevResource, useFlags: c_uint, minCount: c_uint) CUresult;
pub extern fn cuDevResourceGenerateDesc(phDesc: [*c]CUdevResourceDesc, resources: [*c]CUdevResource, nbResources: c_uint) CUresult;
pub extern fn cuGreenCtxRecordEvent(hCtx: CUgreenCtx, hEvent: CUevent) CUresult;
pub extern fn cuGreenCtxWaitEvent(hCtx: CUgreenCtx, hEvent: CUevent) CUresult;
pub extern fn cuStreamGetGreenCtx(hStream: CUstream, phCtx: [*c]CUgreenCtx) CUresult;
pub extern fn cuGreenCtxStreamCreate(phStream: [*c]CUstream, greenCtx: CUgreenCtx, flags: c_uint, priority: c_int) CUresult;
pub extern fn cuCheckpointProcessGetRestoreThreadId(pid: c_int, tid: [*c]c_int) CUresult;
pub extern fn cuCheckpointProcessGetState(pid: c_int, state: [*c]CUprocessState) CUresult;
pub extern fn cuCheckpointProcessLock(pid: c_int, args: [*c]CUcheckpointLockArgs) CUresult;
pub extern fn cuCheckpointProcessCheckpoint(pid: c_int, args: [*c]CUcheckpointCheckpointArgs) CUresult;
pub extern fn cuCheckpointProcessRestore(pid: c_int, args: [*c]CUcheckpointRestoreArgs) CUresult;
pub extern fn cuCheckpointProcessUnlock(pid: c_int, args: [*c]CUcheckpointUnlockArgs) CUresult;
pub const cudaRoundNearest: c_int = 0;
pub const cudaRoundZero: c_int = 1;
pub const cudaRoundPosInf: c_int = 2;
pub const cudaRoundMinInf: c_int = 3;
pub const enum_cudaRoundMode = c_uint;
pub const struct_char1 = extern struct {
    x: i8 = @import("std").mem.zeroes(i8),
};
pub const struct_uchar1 = extern struct {
    x: u8 = @import("std").mem.zeroes(u8),
};
pub const struct_char2 = extern struct {
    x: i8 = @import("std").mem.zeroes(i8),
    y: i8 = @import("std").mem.zeroes(i8),
};
pub const struct_uchar2 = extern struct {
    x: u8 = @import("std").mem.zeroes(u8),
    y: u8 = @import("std").mem.zeroes(u8),
};
pub const struct_char3 = extern struct {
    x: i8 = @import("std").mem.zeroes(i8),
    y: i8 = @import("std").mem.zeroes(i8),
    z: i8 = @import("std").mem.zeroes(i8),
};
pub const struct_uchar3 = extern struct {
    x: u8 = @import("std").mem.zeroes(u8),
    y: u8 = @import("std").mem.zeroes(u8),
    z: u8 = @import("std").mem.zeroes(u8),
};
pub const struct_char4 = extern struct {
    x: i8 = @import("std").mem.zeroes(i8),
    y: i8 = @import("std").mem.zeroes(i8),
    z: i8 = @import("std").mem.zeroes(i8),
    w: i8 = @import("std").mem.zeroes(i8),
};
pub const struct_uchar4 = extern struct {
    x: u8 = @import("std").mem.zeroes(u8),
    y: u8 = @import("std").mem.zeroes(u8),
    z: u8 = @import("std").mem.zeroes(u8),
    w: u8 = @import("std").mem.zeroes(u8),
};
pub const struct_short1 = extern struct {
    x: c_short = @import("std").mem.zeroes(c_short),
};
pub const struct_ushort1 = extern struct {
    x: c_ushort = @import("std").mem.zeroes(c_ushort),
};
pub const struct_short2 = extern struct {
    x: c_short = @import("std").mem.zeroes(c_short),
    y: c_short = @import("std").mem.zeroes(c_short),
};
pub const struct_ushort2 = extern struct {
    x: c_ushort = @import("std").mem.zeroes(c_ushort),
    y: c_ushort = @import("std").mem.zeroes(c_ushort),
};
pub const struct_short3 = extern struct {
    x: c_short = @import("std").mem.zeroes(c_short),
    y: c_short = @import("std").mem.zeroes(c_short),
    z: c_short = @import("std").mem.zeroes(c_short),
};
pub const struct_ushort3 = extern struct {
    x: c_ushort = @import("std").mem.zeroes(c_ushort),
    y: c_ushort = @import("std").mem.zeroes(c_ushort),
    z: c_ushort = @import("std").mem.zeroes(c_ushort),
};
pub const struct_short4 = extern struct {
    x: c_short = @import("std").mem.zeroes(c_short),
    y: c_short = @import("std").mem.zeroes(c_short),
    z: c_short = @import("std").mem.zeroes(c_short),
    w: c_short = @import("std").mem.zeroes(c_short),
};
pub const struct_ushort4 = extern struct {
    x: c_ushort = @import("std").mem.zeroes(c_ushort),
    y: c_ushort = @import("std").mem.zeroes(c_ushort),
    z: c_ushort = @import("std").mem.zeroes(c_ushort),
    w: c_ushort = @import("std").mem.zeroes(c_ushort),
};
pub const struct_int1 = extern struct {
    x: c_int = @import("std").mem.zeroes(c_int),
};
pub const struct_uint1 = extern struct {
    x: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const struct_int2 = extern struct {
    x: c_int = @import("std").mem.zeroes(c_int),
    y: c_int = @import("std").mem.zeroes(c_int),
};
pub const struct_uint2 = extern struct {
    x: c_uint = @import("std").mem.zeroes(c_uint),
    y: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const struct_int3 = extern struct {
    x: c_int = @import("std").mem.zeroes(c_int),
    y: c_int = @import("std").mem.zeroes(c_int),
    z: c_int = @import("std").mem.zeroes(c_int),
};
pub const struct_uint3 = extern struct {
    x: c_uint = @import("std").mem.zeroes(c_uint),
    y: c_uint = @import("std").mem.zeroes(c_uint),
    z: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const struct_int4 = extern struct {
    x: c_int = @import("std").mem.zeroes(c_int),
    y: c_int = @import("std").mem.zeroes(c_int),
    z: c_int = @import("std").mem.zeroes(c_int),
    w: c_int = @import("std").mem.zeroes(c_int),
};
pub const struct_uint4 = extern struct {
    x: c_uint = @import("std").mem.zeroes(c_uint),
    y: c_uint = @import("std").mem.zeroes(c_uint),
    z: c_uint = @import("std").mem.zeroes(c_uint),
    w: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const struct_long1 = extern struct {
    x: c_long = @import("std").mem.zeroes(c_long),
};
pub const struct_ulong1 = extern struct {
    x: c_ulong = @import("std").mem.zeroes(c_ulong),
};
pub const struct_long2 = extern struct {
    x: c_long = @import("std").mem.zeroes(c_long),
    y: c_long = @import("std").mem.zeroes(c_long),
};
pub const struct_ulong2 = extern struct {
    x: c_ulong = @import("std").mem.zeroes(c_ulong),
    y: c_ulong = @import("std").mem.zeroes(c_ulong),
};
pub const struct_long3 = extern struct {
    x: c_long = @import("std").mem.zeroes(c_long),
    y: c_long = @import("std").mem.zeroes(c_long),
    z: c_long = @import("std").mem.zeroes(c_long),
};
pub const struct_ulong3 = extern struct {
    x: c_ulong = @import("std").mem.zeroes(c_ulong),
    y: c_ulong = @import("std").mem.zeroes(c_ulong),
    z: c_ulong = @import("std").mem.zeroes(c_ulong),
};
pub const struct_long4 = extern struct {
    x: c_long = @import("std").mem.zeroes(c_long),
    y: c_long = @import("std").mem.zeroes(c_long),
    z: c_long = @import("std").mem.zeroes(c_long),
    w: c_long = @import("std").mem.zeroes(c_long),
};
pub const struct_ulong4 = extern struct {
    x: c_ulong = @import("std").mem.zeroes(c_ulong),
    y: c_ulong = @import("std").mem.zeroes(c_ulong),
    z: c_ulong = @import("std").mem.zeroes(c_ulong),
    w: c_ulong = @import("std").mem.zeroes(c_ulong),
};
pub const struct_float1 = extern struct {
    x: f32 = @import("std").mem.zeroes(f32),
};
pub const struct_float2 = extern struct {
    x: f32 = @import("std").mem.zeroes(f32),
    y: f32 = @import("std").mem.zeroes(f32),
};
pub const struct_float3 = extern struct {
    x: f32 = @import("std").mem.zeroes(f32),
    y: f32 = @import("std").mem.zeroes(f32),
    z: f32 = @import("std").mem.zeroes(f32),
};
pub const struct_float4 = extern struct {
    x: f32 = @import("std").mem.zeroes(f32),
    y: f32 = @import("std").mem.zeroes(f32),
    z: f32 = @import("std").mem.zeroes(f32),
    w: f32 = @import("std").mem.zeroes(f32),
};
pub const struct_longlong1 = extern struct {
    x: c_longlong = @import("std").mem.zeroes(c_longlong),
};
pub const struct_ulonglong1 = extern struct {
    x: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
};
pub const struct_longlong2 = extern struct {
    x: c_longlong = @import("std").mem.zeroes(c_longlong),
    y: c_longlong = @import("std").mem.zeroes(c_longlong),
};
pub const struct_ulonglong2 = extern struct {
    x: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    y: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
};
pub const struct_longlong3 = extern struct {
    x: c_longlong = @import("std").mem.zeroes(c_longlong),
    y: c_longlong = @import("std").mem.zeroes(c_longlong),
    z: c_longlong = @import("std").mem.zeroes(c_longlong),
};
pub const struct_ulonglong3 = extern struct {
    x: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    y: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    z: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
};
pub const struct_longlong4 = extern struct {
    x: c_longlong = @import("std").mem.zeroes(c_longlong),
    y: c_longlong = @import("std").mem.zeroes(c_longlong),
    z: c_longlong = @import("std").mem.zeroes(c_longlong),
    w: c_longlong = @import("std").mem.zeroes(c_longlong),
};
pub const struct_ulonglong4 = extern struct {
    x: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    y: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    z: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    w: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
};
pub const struct_double1 = extern struct {
    x: f64 = @import("std").mem.zeroes(f64),
};
pub const struct_double2 = extern struct {
    x: f64 = @import("std").mem.zeroes(f64),
    y: f64 = @import("std").mem.zeroes(f64),
};
pub const struct_double3 = extern struct {
    x: f64 = @import("std").mem.zeroes(f64),
    y: f64 = @import("std").mem.zeroes(f64),
    z: f64 = @import("std").mem.zeroes(f64),
};
pub const struct_double4 = extern struct {
    x: f64 = @import("std").mem.zeroes(f64),
    y: f64 = @import("std").mem.zeroes(f64),
    z: f64 = @import("std").mem.zeroes(f64),
    w: f64 = @import("std").mem.zeroes(f64),
};
pub const char1 = struct_char1;
pub const uchar1 = struct_uchar1;
pub const char2 = struct_char2;
pub const uchar2 = struct_uchar2;
pub const char3 = struct_char3;
pub const uchar3 = struct_uchar3;
pub const char4 = struct_char4;
pub const uchar4 = struct_uchar4;
pub const short1 = struct_short1;
pub const ushort1 = struct_ushort1;
pub const short2 = struct_short2;
pub const ushort2 = struct_ushort2;
pub const short3 = struct_short3;
pub const ushort3 = struct_ushort3;
pub const short4 = struct_short4;
pub const ushort4 = struct_ushort4;
pub const int1 = struct_int1;
pub const uint1 = struct_uint1;
pub const int2 = struct_int2;
pub const uint2 = struct_uint2;
pub const int3 = struct_int3;
pub const uint3 = struct_uint3;
pub const int4 = struct_int4;
pub const uint4 = struct_uint4;
pub const long1 = struct_long1;
pub const ulong1 = struct_ulong1;
pub const long2 = struct_long2;
pub const ulong2 = struct_ulong2;
pub const long3 = struct_long3;
pub const ulong3 = struct_ulong3;
pub const long4 = struct_long4;
pub const ulong4 = struct_ulong4;
pub const float1 = struct_float1;
pub const float2 = struct_float2;
pub const float3 = struct_float3;
pub const float4 = struct_float4;
pub const longlong1 = struct_longlong1;
pub const ulonglong1 = struct_ulonglong1;
pub const longlong2 = struct_longlong2;
pub const ulonglong2 = struct_ulonglong2;
pub const longlong3 = struct_longlong3;
pub const ulonglong3 = struct_ulonglong3;
pub const longlong4 = struct_longlong4;
pub const ulonglong4 = struct_ulonglong4;
pub const double1 = struct_double1;
pub const double2 = struct_double2;
pub const double3 = struct_double3;
pub const double4 = struct_double4;
pub const struct_dim3 = extern struct {
    x: c_uint = @import("std").mem.zeroes(c_uint),
    y: c_uint = @import("std").mem.zeroes(c_uint),
    z: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const dim3 = struct_dim3;
pub const max_align_t = extern struct {
    __clang_max_align_nonce1: c_longlong align(8) = @import("std").mem.zeroes(c_longlong),
    __clang_max_align_nonce2: c_longdouble align(16) = @import("std").mem.zeroes(c_longdouble),
};
pub const cudaSuccess: c_int = 0;
pub const cudaErrorInvalidValue: c_int = 1;
pub const cudaErrorMemoryAllocation: c_int = 2;
pub const cudaErrorInitializationError: c_int = 3;
pub const cudaErrorCudartUnloading: c_int = 4;
pub const cudaErrorProfilerDisabled: c_int = 5;
pub const cudaErrorProfilerNotInitialized: c_int = 6;
pub const cudaErrorProfilerAlreadyStarted: c_int = 7;
pub const cudaErrorProfilerAlreadyStopped: c_int = 8;
pub const cudaErrorInvalidConfiguration: c_int = 9;
pub const cudaErrorInvalidPitchValue: c_int = 12;
pub const cudaErrorInvalidSymbol: c_int = 13;
pub const cudaErrorInvalidHostPointer: c_int = 16;
pub const cudaErrorInvalidDevicePointer: c_int = 17;
pub const cudaErrorInvalidTexture: c_int = 18;
pub const cudaErrorInvalidTextureBinding: c_int = 19;
pub const cudaErrorInvalidChannelDescriptor: c_int = 20;
pub const cudaErrorInvalidMemcpyDirection: c_int = 21;
pub const cudaErrorAddressOfConstant: c_int = 22;
pub const cudaErrorTextureFetchFailed: c_int = 23;
pub const cudaErrorTextureNotBound: c_int = 24;
pub const cudaErrorSynchronizationError: c_int = 25;
pub const cudaErrorInvalidFilterSetting: c_int = 26;
pub const cudaErrorInvalidNormSetting: c_int = 27;
pub const cudaErrorMixedDeviceExecution: c_int = 28;
pub const cudaErrorNotYetImplemented: c_int = 31;
pub const cudaErrorMemoryValueTooLarge: c_int = 32;
pub const cudaErrorStubLibrary: c_int = 34;
pub const cudaErrorInsufficientDriver: c_int = 35;
pub const cudaErrorCallRequiresNewerDriver: c_int = 36;
pub const cudaErrorInvalidSurface: c_int = 37;
pub const cudaErrorDuplicateVariableName: c_int = 43;
pub const cudaErrorDuplicateTextureName: c_int = 44;
pub const cudaErrorDuplicateSurfaceName: c_int = 45;
pub const cudaErrorDevicesUnavailable: c_int = 46;
pub const cudaErrorIncompatibleDriverContext: c_int = 49;
pub const cudaErrorMissingConfiguration: c_int = 52;
pub const cudaErrorPriorLaunchFailure: c_int = 53;
pub const cudaErrorLaunchMaxDepthExceeded: c_int = 65;
pub const cudaErrorLaunchFileScopedTex: c_int = 66;
pub const cudaErrorLaunchFileScopedSurf: c_int = 67;
pub const cudaErrorSyncDepthExceeded: c_int = 68;
pub const cudaErrorLaunchPendingCountExceeded: c_int = 69;
pub const cudaErrorInvalidDeviceFunction: c_int = 98;
pub const cudaErrorNoDevice: c_int = 100;
pub const cudaErrorInvalidDevice: c_int = 101;
pub const cudaErrorDeviceNotLicensed: c_int = 102;
pub const cudaErrorSoftwareValidityNotEstablished: c_int = 103;
pub const cudaErrorStartupFailure: c_int = 127;
pub const cudaErrorInvalidKernelImage: c_int = 200;
pub const cudaErrorDeviceUninitialized: c_int = 201;
pub const cudaErrorMapBufferObjectFailed: c_int = 205;
pub const cudaErrorUnmapBufferObjectFailed: c_int = 206;
pub const cudaErrorArrayIsMapped: c_int = 207;
pub const cudaErrorAlreadyMapped: c_int = 208;
pub const cudaErrorNoKernelImageForDevice: c_int = 209;
pub const cudaErrorAlreadyAcquired: c_int = 210;
pub const cudaErrorNotMapped: c_int = 211;
pub const cudaErrorNotMappedAsArray: c_int = 212;
pub const cudaErrorNotMappedAsPointer: c_int = 213;
pub const cudaErrorECCUncorrectable: c_int = 214;
pub const cudaErrorUnsupportedLimit: c_int = 215;
pub const cudaErrorDeviceAlreadyInUse: c_int = 216;
pub const cudaErrorPeerAccessUnsupported: c_int = 217;
pub const cudaErrorInvalidPtx: c_int = 218;
pub const cudaErrorInvalidGraphicsContext: c_int = 219;
pub const cudaErrorNvlinkUncorrectable: c_int = 220;
pub const cudaErrorJitCompilerNotFound: c_int = 221;
pub const cudaErrorUnsupportedPtxVersion: c_int = 222;
pub const cudaErrorJitCompilationDisabled: c_int = 223;
pub const cudaErrorUnsupportedExecAffinity: c_int = 224;
pub const cudaErrorUnsupportedDevSideSync: c_int = 225;
pub const cudaErrorContained: c_int = 226;
pub const cudaErrorInvalidSource: c_int = 300;
pub const cudaErrorFileNotFound: c_int = 301;
pub const cudaErrorSharedObjectSymbolNotFound: c_int = 302;
pub const cudaErrorSharedObjectInitFailed: c_int = 303;
pub const cudaErrorOperatingSystem: c_int = 304;
pub const cudaErrorInvalidResourceHandle: c_int = 400;
pub const cudaErrorIllegalState: c_int = 401;
pub const cudaErrorLossyQuery: c_int = 402;
pub const cudaErrorSymbolNotFound: c_int = 500;
pub const cudaErrorNotReady: c_int = 600;
pub const cudaErrorIllegalAddress: c_int = 700;
pub const cudaErrorLaunchOutOfResources: c_int = 701;
pub const cudaErrorLaunchTimeout: c_int = 702;
pub const cudaErrorLaunchIncompatibleTexturing: c_int = 703;
pub const cudaErrorPeerAccessAlreadyEnabled: c_int = 704;
pub const cudaErrorPeerAccessNotEnabled: c_int = 705;
pub const cudaErrorSetOnActiveProcess: c_int = 708;
pub const cudaErrorContextIsDestroyed: c_int = 709;
pub const cudaErrorAssert: c_int = 710;
pub const cudaErrorTooManyPeers: c_int = 711;
pub const cudaErrorHostMemoryAlreadyRegistered: c_int = 712;
pub const cudaErrorHostMemoryNotRegistered: c_int = 713;
pub const cudaErrorHardwareStackError: c_int = 714;
pub const cudaErrorIllegalInstruction: c_int = 715;
pub const cudaErrorMisalignedAddress: c_int = 716;
pub const cudaErrorInvalidAddressSpace: c_int = 717;
pub const cudaErrorInvalidPc: c_int = 718;
pub const cudaErrorLaunchFailure: c_int = 719;
pub const cudaErrorCooperativeLaunchTooLarge: c_int = 720;
pub const cudaErrorTensorMemoryLeak: c_int = 721;
pub const cudaErrorNotPermitted: c_int = 800;
pub const cudaErrorNotSupported: c_int = 801;
pub const cudaErrorSystemNotReady: c_int = 802;
pub const cudaErrorSystemDriverMismatch: c_int = 803;
pub const cudaErrorCompatNotSupportedOnDevice: c_int = 804;
pub const cudaErrorMpsConnectionFailed: c_int = 805;
pub const cudaErrorMpsRpcFailure: c_int = 806;
pub const cudaErrorMpsServerNotReady: c_int = 807;
pub const cudaErrorMpsMaxClientsReached: c_int = 808;
pub const cudaErrorMpsMaxConnectionsReached: c_int = 809;
pub const cudaErrorMpsClientTerminated: c_int = 810;
pub const cudaErrorCdpNotSupported: c_int = 811;
pub const cudaErrorCdpVersionMismatch: c_int = 812;
pub const cudaErrorStreamCaptureUnsupported: c_int = 900;
pub const cudaErrorStreamCaptureInvalidated: c_int = 901;
pub const cudaErrorStreamCaptureMerge: c_int = 902;
pub const cudaErrorStreamCaptureUnmatched: c_int = 903;
pub const cudaErrorStreamCaptureUnjoined: c_int = 904;
pub const cudaErrorStreamCaptureIsolation: c_int = 905;
pub const cudaErrorStreamCaptureImplicit: c_int = 906;
pub const cudaErrorCapturedEvent: c_int = 907;
pub const cudaErrorStreamCaptureWrongThread: c_int = 908;
pub const cudaErrorTimeout: c_int = 909;
pub const cudaErrorGraphExecUpdateFailure: c_int = 910;
pub const cudaErrorExternalDevice: c_int = 911;
pub const cudaErrorInvalidClusterSize: c_int = 912;
pub const cudaErrorFunctionNotLoaded: c_int = 913;
pub const cudaErrorInvalidResourceType: c_int = 914;
pub const cudaErrorInvalidResourceConfiguration: c_int = 915;
pub const cudaErrorUnknown: c_int = 999;
pub const cudaErrorApiFailureBase: c_int = 10000;
pub const enum_cudaError = c_uint;
pub const cudaChannelFormatKindSigned: c_int = 0;
pub const cudaChannelFormatKindUnsigned: c_int = 1;
pub const cudaChannelFormatKindFloat: c_int = 2;
pub const cudaChannelFormatKindNone: c_int = 3;
pub const cudaChannelFormatKindNV12: c_int = 4;
pub const cudaChannelFormatKindUnsignedNormalized8X1: c_int = 5;
pub const cudaChannelFormatKindUnsignedNormalized8X2: c_int = 6;
pub const cudaChannelFormatKindUnsignedNormalized8X4: c_int = 7;
pub const cudaChannelFormatKindUnsignedNormalized16X1: c_int = 8;
pub const cudaChannelFormatKindUnsignedNormalized16X2: c_int = 9;
pub const cudaChannelFormatKindUnsignedNormalized16X4: c_int = 10;
pub const cudaChannelFormatKindSignedNormalized8X1: c_int = 11;
pub const cudaChannelFormatKindSignedNormalized8X2: c_int = 12;
pub const cudaChannelFormatKindSignedNormalized8X4: c_int = 13;
pub const cudaChannelFormatKindSignedNormalized16X1: c_int = 14;
pub const cudaChannelFormatKindSignedNormalized16X2: c_int = 15;
pub const cudaChannelFormatKindSignedNormalized16X4: c_int = 16;
pub const cudaChannelFormatKindUnsignedBlockCompressed1: c_int = 17;
pub const cudaChannelFormatKindUnsignedBlockCompressed1SRGB: c_int = 18;
pub const cudaChannelFormatKindUnsignedBlockCompressed2: c_int = 19;
pub const cudaChannelFormatKindUnsignedBlockCompressed2SRGB: c_int = 20;
pub const cudaChannelFormatKindUnsignedBlockCompressed3: c_int = 21;
pub const cudaChannelFormatKindUnsignedBlockCompressed3SRGB: c_int = 22;
pub const cudaChannelFormatKindUnsignedBlockCompressed4: c_int = 23;
pub const cudaChannelFormatKindSignedBlockCompressed4: c_int = 24;
pub const cudaChannelFormatKindUnsignedBlockCompressed5: c_int = 25;
pub const cudaChannelFormatKindSignedBlockCompressed5: c_int = 26;
pub const cudaChannelFormatKindUnsignedBlockCompressed6H: c_int = 27;
pub const cudaChannelFormatKindSignedBlockCompressed6H: c_int = 28;
pub const cudaChannelFormatKindUnsignedBlockCompressed7: c_int = 29;
pub const cudaChannelFormatKindUnsignedBlockCompressed7SRGB: c_int = 30;
pub const cudaChannelFormatKindUnsignedNormalized1010102: c_int = 31;
pub const enum_cudaChannelFormatKind = c_uint;
pub const struct_cudaChannelFormatDesc = extern struct {
    x: c_int = @import("std").mem.zeroes(c_int),
    y: c_int = @import("std").mem.zeroes(c_int),
    z: c_int = @import("std").mem.zeroes(c_int),
    w: c_int = @import("std").mem.zeroes(c_int),
    f: enum_cudaChannelFormatKind = @import("std").mem.zeroes(enum_cudaChannelFormatKind),
};
pub const struct_cudaArray = opaque {};
pub const cudaArray_t = ?*struct_cudaArray;
pub const cudaArray_const_t = ?*const struct_cudaArray;
pub const struct_cudaMipmappedArray = opaque {};
pub const cudaMipmappedArray_t = ?*struct_cudaMipmappedArray;
pub const cudaMipmappedArray_const_t = ?*const struct_cudaMipmappedArray;
const struct_unnamed_45 = extern struct {
    width: c_uint = @import("std").mem.zeroes(c_uint),
    height: c_uint = @import("std").mem.zeroes(c_uint),
    depth: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const struct_cudaArraySparseProperties = extern struct {
    tileExtent: struct_unnamed_45 = @import("std").mem.zeroes(struct_unnamed_45),
    miptailFirstLevel: c_uint = @import("std").mem.zeroes(c_uint),
    miptailSize: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
    reserved: [4]c_uint = @import("std").mem.zeroes([4]c_uint),
};
pub const struct_cudaArrayMemoryRequirements = extern struct {
    size: usize = @import("std").mem.zeroes(usize),
    alignment: usize = @import("std").mem.zeroes(usize),
    reserved: [4]c_uint = @import("std").mem.zeroes([4]c_uint),
};
pub const cudaMemoryTypeUnregistered: c_int = 0;
pub const cudaMemoryTypeHost: c_int = 1;
pub const cudaMemoryTypeDevice: c_int = 2;
pub const cudaMemoryTypeManaged: c_int = 3;
pub const enum_cudaMemoryType = c_uint;
pub const cudaMemcpyHostToHost: c_int = 0;
pub const cudaMemcpyHostToDevice: c_int = 1;
pub const cudaMemcpyDeviceToHost: c_int = 2;
pub const cudaMemcpyDeviceToDevice: c_int = 3;
pub const cudaMemcpyDefault: c_int = 4;
pub const enum_cudaMemcpyKind = c_uint;
pub const struct_cudaPitchedPtr = extern struct {
    ptr: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    pitch: usize = @import("std").mem.zeroes(usize),
    xsize: usize = @import("std").mem.zeroes(usize),
    ysize: usize = @import("std").mem.zeroes(usize),
};
pub const struct_cudaExtent = extern struct {
    width: usize = @import("std").mem.zeroes(usize),
    height: usize = @import("std").mem.zeroes(usize),
    depth: usize = @import("std").mem.zeroes(usize),
};
pub const struct_cudaPos = extern struct {
    x: usize = @import("std").mem.zeroes(usize),
    y: usize = @import("std").mem.zeroes(usize),
    z: usize = @import("std").mem.zeroes(usize),
};
pub const struct_cudaMemcpy3DParms = extern struct {
    srcArray: cudaArray_t = @import("std").mem.zeroes(cudaArray_t),
    srcPos: struct_cudaPos = @import("std").mem.zeroes(struct_cudaPos),
    srcPtr: struct_cudaPitchedPtr = @import("std").mem.zeroes(struct_cudaPitchedPtr),
    dstArray: cudaArray_t = @import("std").mem.zeroes(cudaArray_t),
    dstPos: struct_cudaPos = @import("std").mem.zeroes(struct_cudaPos),
    dstPtr: struct_cudaPitchedPtr = @import("std").mem.zeroes(struct_cudaPitchedPtr),
    extent: struct_cudaExtent = @import("std").mem.zeroes(struct_cudaExtent),
    kind: enum_cudaMemcpyKind = @import("std").mem.zeroes(enum_cudaMemcpyKind),
};
pub const struct_cudaMemcpyNodeParams = extern struct {
    flags: c_int = @import("std").mem.zeroes(c_int),
    reserved: [3]c_int = @import("std").mem.zeroes([3]c_int),
    copyParams: struct_cudaMemcpy3DParms = @import("std").mem.zeroes(struct_cudaMemcpy3DParms),
};
pub const struct_cudaMemcpy3DPeerParms = extern struct {
    srcArray: cudaArray_t = @import("std").mem.zeroes(cudaArray_t),
    srcPos: struct_cudaPos = @import("std").mem.zeroes(struct_cudaPos),
    srcPtr: struct_cudaPitchedPtr = @import("std").mem.zeroes(struct_cudaPitchedPtr),
    srcDevice: c_int = @import("std").mem.zeroes(c_int),
    dstArray: cudaArray_t = @import("std").mem.zeroes(cudaArray_t),
    dstPos: struct_cudaPos = @import("std").mem.zeroes(struct_cudaPos),
    dstPtr: struct_cudaPitchedPtr = @import("std").mem.zeroes(struct_cudaPitchedPtr),
    dstDevice: c_int = @import("std").mem.zeroes(c_int),
    extent: struct_cudaExtent = @import("std").mem.zeroes(struct_cudaExtent),
};
pub const struct_cudaMemsetParams = extern struct {
    dst: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    pitch: usize = @import("std").mem.zeroes(usize),
    value: c_uint = @import("std").mem.zeroes(c_uint),
    elementSize: c_uint = @import("std").mem.zeroes(c_uint),
    width: usize = @import("std").mem.zeroes(usize),
    height: usize = @import("std").mem.zeroes(usize),
};
pub const struct_cudaMemsetParamsV2 = extern struct {
    dst: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    pitch: usize = @import("std").mem.zeroes(usize),
    value: c_uint = @import("std").mem.zeroes(c_uint),
    elementSize: c_uint = @import("std").mem.zeroes(c_uint),
    width: usize = @import("std").mem.zeroes(usize),
    height: usize = @import("std").mem.zeroes(usize),
};
pub const cudaAccessPropertyNormal: c_int = 0;
pub const cudaAccessPropertyStreaming: c_int = 1;
pub const cudaAccessPropertyPersisting: c_int = 2;
pub const enum_cudaAccessProperty = c_uint;
pub const struct_cudaAccessPolicyWindow = extern struct {
    base_ptr: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    num_bytes: usize = @import("std").mem.zeroes(usize),
    hitRatio: f32 = @import("std").mem.zeroes(f32),
    hitProp: enum_cudaAccessProperty = @import("std").mem.zeroes(enum_cudaAccessProperty),
    missProp: enum_cudaAccessProperty = @import("std").mem.zeroes(enum_cudaAccessProperty),
};
pub const cudaHostFn_t = ?*const fn (?*anyopaque) callconv(.c) void;
pub const struct_cudaHostNodeParams = extern struct {
    @"fn": cudaHostFn_t = @import("std").mem.zeroes(cudaHostFn_t),
    userData: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
};
pub const struct_cudaHostNodeParamsV2 = extern struct {
    @"fn": cudaHostFn_t = @import("std").mem.zeroes(cudaHostFn_t),
    userData: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
};
pub const cudaStreamCaptureStatusNone: c_int = 0;
pub const cudaStreamCaptureStatusActive: c_int = 1;
pub const cudaStreamCaptureStatusInvalidated: c_int = 2;
pub const enum_cudaStreamCaptureStatus = c_uint;
pub const cudaStreamCaptureModeGlobal: c_int = 0;
pub const cudaStreamCaptureModeThreadLocal: c_int = 1;
pub const cudaStreamCaptureModeRelaxed: c_int = 2;
pub const enum_cudaStreamCaptureMode = c_uint;
pub const cudaSyncPolicyAuto: c_int = 1;
pub const cudaSyncPolicySpin: c_int = 2;
pub const cudaSyncPolicyYield: c_int = 3;
pub const cudaSyncPolicyBlockingSync: c_int = 4;
pub const enum_cudaSynchronizationPolicy = c_uint;
pub const cudaClusterSchedulingPolicyDefault: c_int = 0;
pub const cudaClusterSchedulingPolicySpread: c_int = 1;
pub const cudaClusterSchedulingPolicyLoadBalancing: c_int = 2;
pub const enum_cudaClusterSchedulingPolicy = c_uint;
pub const cudaStreamAddCaptureDependencies: c_int = 0;
pub const cudaStreamSetCaptureDependencies: c_int = 1;
pub const enum_cudaStreamUpdateCaptureDependenciesFlags = c_uint;
pub const cudaUserObjectNoDestructorSync: c_int = 1;
pub const enum_cudaUserObjectFlags = c_uint;
pub const cudaGraphUserObjectMove: c_int = 1;
pub const enum_cudaUserObjectRetainFlags = c_uint;
pub const struct_cudaGraphicsResource = opaque {};
pub const cudaGraphicsRegisterFlagsNone: c_int = 0;
pub const cudaGraphicsRegisterFlagsReadOnly: c_int = 1;
pub const cudaGraphicsRegisterFlagsWriteDiscard: c_int = 2;
pub const cudaGraphicsRegisterFlagsSurfaceLoadStore: c_int = 4;
pub const cudaGraphicsRegisterFlagsTextureGather: c_int = 8;
pub const enum_cudaGraphicsRegisterFlags = c_uint;
pub const cudaGraphicsMapFlagsNone: c_int = 0;
pub const cudaGraphicsMapFlagsReadOnly: c_int = 1;
pub const cudaGraphicsMapFlagsWriteDiscard: c_int = 2;
pub const enum_cudaGraphicsMapFlags = c_uint;
pub const cudaGraphicsCubeFacePositiveX: c_int = 0;
pub const cudaGraphicsCubeFaceNegativeX: c_int = 1;
pub const cudaGraphicsCubeFacePositiveY: c_int = 2;
pub const cudaGraphicsCubeFaceNegativeY: c_int = 3;
pub const cudaGraphicsCubeFacePositiveZ: c_int = 4;
pub const cudaGraphicsCubeFaceNegativeZ: c_int = 5;
pub const enum_cudaGraphicsCubeFace = c_uint;
pub const cudaResourceTypeArray: c_int = 0;
pub const cudaResourceTypeMipmappedArray: c_int = 1;
pub const cudaResourceTypeLinear: c_int = 2;
pub const cudaResourceTypePitch2D: c_int = 3;
pub const enum_cudaResourceType = c_uint;
pub const cudaResViewFormatNone: c_int = 0;
pub const cudaResViewFormatUnsignedChar1: c_int = 1;
pub const cudaResViewFormatUnsignedChar2: c_int = 2;
pub const cudaResViewFormatUnsignedChar4: c_int = 3;
pub const cudaResViewFormatSignedChar1: c_int = 4;
pub const cudaResViewFormatSignedChar2: c_int = 5;
pub const cudaResViewFormatSignedChar4: c_int = 6;
pub const cudaResViewFormatUnsignedShort1: c_int = 7;
pub const cudaResViewFormatUnsignedShort2: c_int = 8;
pub const cudaResViewFormatUnsignedShort4: c_int = 9;
pub const cudaResViewFormatSignedShort1: c_int = 10;
pub const cudaResViewFormatSignedShort2: c_int = 11;
pub const cudaResViewFormatSignedShort4: c_int = 12;
pub const cudaResViewFormatUnsignedInt1: c_int = 13;
pub const cudaResViewFormatUnsignedInt2: c_int = 14;
pub const cudaResViewFormatUnsignedInt4: c_int = 15;
pub const cudaResViewFormatSignedInt1: c_int = 16;
pub const cudaResViewFormatSignedInt2: c_int = 17;
pub const cudaResViewFormatSignedInt4: c_int = 18;
pub const cudaResViewFormatHalf1: c_int = 19;
pub const cudaResViewFormatHalf2: c_int = 20;
pub const cudaResViewFormatHalf4: c_int = 21;
pub const cudaResViewFormatFloat1: c_int = 22;
pub const cudaResViewFormatFloat2: c_int = 23;
pub const cudaResViewFormatFloat4: c_int = 24;
pub const cudaResViewFormatUnsignedBlockCompressed1: c_int = 25;
pub const cudaResViewFormatUnsignedBlockCompressed2: c_int = 26;
pub const cudaResViewFormatUnsignedBlockCompressed3: c_int = 27;
pub const cudaResViewFormatUnsignedBlockCompressed4: c_int = 28;
pub const cudaResViewFormatSignedBlockCompressed4: c_int = 29;
pub const cudaResViewFormatUnsignedBlockCompressed5: c_int = 30;
pub const cudaResViewFormatSignedBlockCompressed5: c_int = 31;
pub const cudaResViewFormatUnsignedBlockCompressed6H: c_int = 32;
pub const cudaResViewFormatSignedBlockCompressed6H: c_int = 33;
pub const cudaResViewFormatUnsignedBlockCompressed7: c_int = 34;
pub const enum_cudaResourceViewFormat = c_uint;
const struct_unnamed_47 = extern struct {
    array: cudaArray_t = @import("std").mem.zeroes(cudaArray_t),
};
const struct_unnamed_48 = extern struct {
    mipmap: cudaMipmappedArray_t = @import("std").mem.zeroes(cudaMipmappedArray_t),
};
const struct_unnamed_49 = extern struct {
    devPtr: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    desc: struct_cudaChannelFormatDesc = @import("std").mem.zeroes(struct_cudaChannelFormatDesc),
    sizeInBytes: usize = @import("std").mem.zeroes(usize),
};
const struct_unnamed_50 = extern struct {
    devPtr: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    desc: struct_cudaChannelFormatDesc = @import("std").mem.zeroes(struct_cudaChannelFormatDesc),
    width: usize = @import("std").mem.zeroes(usize),
    height: usize = @import("std").mem.zeroes(usize),
    pitchInBytes: usize = @import("std").mem.zeroes(usize),
};
const union_unnamed_46 = extern union {
    array: struct_unnamed_47,
    mipmap: struct_unnamed_48,
    linear: struct_unnamed_49,
    pitch2D: struct_unnamed_50,
};
pub const struct_cudaResourceDesc = extern struct {
    resType: enum_cudaResourceType = @import("std").mem.zeroes(enum_cudaResourceType),
    res: union_unnamed_46 = @import("std").mem.zeroes(union_unnamed_46),
};
pub const struct_cudaResourceViewDesc = extern struct {
    format: enum_cudaResourceViewFormat = @import("std").mem.zeroes(enum_cudaResourceViewFormat),
    width: usize = @import("std").mem.zeroes(usize),
    height: usize = @import("std").mem.zeroes(usize),
    depth: usize = @import("std").mem.zeroes(usize),
    firstMipmapLevel: c_uint = @import("std").mem.zeroes(c_uint),
    lastMipmapLevel: c_uint = @import("std").mem.zeroes(c_uint),
    firstLayer: c_uint = @import("std").mem.zeroes(c_uint),
    lastLayer: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const struct_cudaPointerAttributes = extern struct {
    type: enum_cudaMemoryType = @import("std").mem.zeroes(enum_cudaMemoryType),
    device: c_int = @import("std").mem.zeroes(c_int),
    devicePointer: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    hostPointer: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
};
pub const struct_cudaFuncAttributes = extern struct {
    sharedSizeBytes: usize = @import("std").mem.zeroes(usize),
    constSizeBytes: usize = @import("std").mem.zeroes(usize),
    localSizeBytes: usize = @import("std").mem.zeroes(usize),
    maxThreadsPerBlock: c_int = @import("std").mem.zeroes(c_int),
    numRegs: c_int = @import("std").mem.zeroes(c_int),
    ptxVersion: c_int = @import("std").mem.zeroes(c_int),
    binaryVersion: c_int = @import("std").mem.zeroes(c_int),
    cacheModeCA: c_int = @import("std").mem.zeroes(c_int),
    maxDynamicSharedSizeBytes: c_int = @import("std").mem.zeroes(c_int),
    preferredShmemCarveout: c_int = @import("std").mem.zeroes(c_int),
    clusterDimMustBeSet: c_int = @import("std").mem.zeroes(c_int),
    requiredClusterWidth: c_int = @import("std").mem.zeroes(c_int),
    requiredClusterHeight: c_int = @import("std").mem.zeroes(c_int),
    requiredClusterDepth: c_int = @import("std").mem.zeroes(c_int),
    clusterSchedulingPolicyPreference: c_int = @import("std").mem.zeroes(c_int),
    nonPortableClusterSizeAllowed: c_int = @import("std").mem.zeroes(c_int),
    reserved: [16]c_int = @import("std").mem.zeroes([16]c_int),
};
pub const cudaFuncAttributeMaxDynamicSharedMemorySize: c_int = 8;
pub const cudaFuncAttributePreferredSharedMemoryCarveout: c_int = 9;
pub const cudaFuncAttributeClusterDimMustBeSet: c_int = 10;
pub const cudaFuncAttributeRequiredClusterWidth: c_int = 11;
pub const cudaFuncAttributeRequiredClusterHeight: c_int = 12;
pub const cudaFuncAttributeRequiredClusterDepth: c_int = 13;
pub const cudaFuncAttributeNonPortableClusterSizeAllowed: c_int = 14;
pub const cudaFuncAttributeClusterSchedulingPolicyPreference: c_int = 15;
pub const cudaFuncAttributeMax: c_int = 16;
pub const enum_cudaFuncAttribute = c_uint;
pub const cudaFuncCachePreferNone: c_int = 0;
pub const cudaFuncCachePreferShared: c_int = 1;
pub const cudaFuncCachePreferL1: c_int = 2;
pub const cudaFuncCachePreferEqual: c_int = 3;
pub const enum_cudaFuncCache = c_uint;
pub const cudaSharedMemBankSizeDefault: c_int = 0;
pub const cudaSharedMemBankSizeFourByte: c_int = 1;
pub const cudaSharedMemBankSizeEightByte: c_int = 2;
pub const enum_cudaSharedMemConfig = c_uint;
pub const cudaSharedmemCarveoutDefault: c_int = -1;
pub const cudaSharedmemCarveoutMaxShared: c_int = 100;
pub const cudaSharedmemCarveoutMaxL1: c_int = 0;
pub const enum_cudaSharedCarveout = c_int;
pub const cudaComputeModeDefault: c_int = 0;
pub const cudaComputeModeExclusive: c_int = 1;
pub const cudaComputeModeProhibited: c_int = 2;
pub const cudaComputeModeExclusiveProcess: c_int = 3;
pub const enum_cudaComputeMode = c_uint;
pub const cudaLimitStackSize: c_int = 0;
pub const cudaLimitPrintfFifoSize: c_int = 1;
pub const cudaLimitMallocHeapSize: c_int = 2;
pub const cudaLimitDevRuntimeSyncDepth: c_int = 3;
pub const cudaLimitDevRuntimePendingLaunchCount: c_int = 4;
pub const cudaLimitMaxL2FetchGranularity: c_int = 5;
pub const cudaLimitPersistingL2CacheSize: c_int = 6;
pub const enum_cudaLimit = c_uint;
pub const cudaMemAdviseSetReadMostly: c_int = 1;
pub const cudaMemAdviseUnsetReadMostly: c_int = 2;
pub const cudaMemAdviseSetPreferredLocation: c_int = 3;
pub const cudaMemAdviseUnsetPreferredLocation: c_int = 4;
pub const cudaMemAdviseSetAccessedBy: c_int = 5;
pub const cudaMemAdviseUnsetAccessedBy: c_int = 6;
pub const enum_cudaMemoryAdvise = c_uint;
pub const cudaMemRangeAttributeReadMostly: c_int = 1;
pub const cudaMemRangeAttributePreferredLocation: c_int = 2;
pub const cudaMemRangeAttributeAccessedBy: c_int = 3;
pub const cudaMemRangeAttributeLastPrefetchLocation: c_int = 4;
pub const cudaMemRangeAttributePreferredLocationType: c_int = 5;
pub const cudaMemRangeAttributePreferredLocationId: c_int = 6;
pub const cudaMemRangeAttributeLastPrefetchLocationType: c_int = 7;
pub const cudaMemRangeAttributeLastPrefetchLocationId: c_int = 8;
pub const enum_cudaMemRangeAttribute = c_uint;
pub const cudaFlushGPUDirectRDMAWritesOptionHost: c_int = 1;
pub const cudaFlushGPUDirectRDMAWritesOptionMemOps: c_int = 2;
pub const enum_cudaFlushGPUDirectRDMAWritesOptions = c_uint;
pub const cudaGPUDirectRDMAWritesOrderingNone: c_int = 0;
pub const cudaGPUDirectRDMAWritesOrderingOwner: c_int = 100;
pub const cudaGPUDirectRDMAWritesOrderingAllDevices: c_int = 200;
pub const enum_cudaGPUDirectRDMAWritesOrdering = c_uint;
pub const cudaFlushGPUDirectRDMAWritesToOwner: c_int = 100;
pub const cudaFlushGPUDirectRDMAWritesToAllDevices: c_int = 200;
pub const enum_cudaFlushGPUDirectRDMAWritesScope = c_uint;
pub const cudaFlushGPUDirectRDMAWritesTargetCurrentDevice: c_int = 0;
pub const enum_cudaFlushGPUDirectRDMAWritesTarget = c_uint;
pub const cudaDevAttrMaxThreadsPerBlock: c_int = 1;
pub const cudaDevAttrMaxBlockDimX: c_int = 2;
pub const cudaDevAttrMaxBlockDimY: c_int = 3;
pub const cudaDevAttrMaxBlockDimZ: c_int = 4;
pub const cudaDevAttrMaxGridDimX: c_int = 5;
pub const cudaDevAttrMaxGridDimY: c_int = 6;
pub const cudaDevAttrMaxGridDimZ: c_int = 7;
pub const cudaDevAttrMaxSharedMemoryPerBlock: c_int = 8;
pub const cudaDevAttrTotalConstantMemory: c_int = 9;
pub const cudaDevAttrWarpSize: c_int = 10;
pub const cudaDevAttrMaxPitch: c_int = 11;
pub const cudaDevAttrMaxRegistersPerBlock: c_int = 12;
pub const cudaDevAttrClockRate: c_int = 13;
pub const cudaDevAttrTextureAlignment: c_int = 14;
pub const cudaDevAttrGpuOverlap: c_int = 15;
pub const cudaDevAttrMultiProcessorCount: c_int = 16;
pub const cudaDevAttrKernelExecTimeout: c_int = 17;
pub const cudaDevAttrIntegrated: c_int = 18;
pub const cudaDevAttrCanMapHostMemory: c_int = 19;
pub const cudaDevAttrComputeMode: c_int = 20;
pub const cudaDevAttrMaxTexture1DWidth: c_int = 21;
pub const cudaDevAttrMaxTexture2DWidth: c_int = 22;
pub const cudaDevAttrMaxTexture2DHeight: c_int = 23;
pub const cudaDevAttrMaxTexture3DWidth: c_int = 24;
pub const cudaDevAttrMaxTexture3DHeight: c_int = 25;
pub const cudaDevAttrMaxTexture3DDepth: c_int = 26;
pub const cudaDevAttrMaxTexture2DLayeredWidth: c_int = 27;
pub const cudaDevAttrMaxTexture2DLayeredHeight: c_int = 28;
pub const cudaDevAttrMaxTexture2DLayeredLayers: c_int = 29;
pub const cudaDevAttrSurfaceAlignment: c_int = 30;
pub const cudaDevAttrConcurrentKernels: c_int = 31;
pub const cudaDevAttrEccEnabled: c_int = 32;
pub const cudaDevAttrPciBusId: c_int = 33;
pub const cudaDevAttrPciDeviceId: c_int = 34;
pub const cudaDevAttrTccDriver: c_int = 35;
pub const cudaDevAttrMemoryClockRate: c_int = 36;
pub const cudaDevAttrGlobalMemoryBusWidth: c_int = 37;
pub const cudaDevAttrL2CacheSize: c_int = 38;
pub const cudaDevAttrMaxThreadsPerMultiProcessor: c_int = 39;
pub const cudaDevAttrAsyncEngineCount: c_int = 40;
pub const cudaDevAttrUnifiedAddressing: c_int = 41;
pub const cudaDevAttrMaxTexture1DLayeredWidth: c_int = 42;
pub const cudaDevAttrMaxTexture1DLayeredLayers: c_int = 43;
pub const cudaDevAttrMaxTexture2DGatherWidth: c_int = 45;
pub const cudaDevAttrMaxTexture2DGatherHeight: c_int = 46;
pub const cudaDevAttrMaxTexture3DWidthAlt: c_int = 47;
pub const cudaDevAttrMaxTexture3DHeightAlt: c_int = 48;
pub const cudaDevAttrMaxTexture3DDepthAlt: c_int = 49;
pub const cudaDevAttrPciDomainId: c_int = 50;
pub const cudaDevAttrTexturePitchAlignment: c_int = 51;
pub const cudaDevAttrMaxTextureCubemapWidth: c_int = 52;
pub const cudaDevAttrMaxTextureCubemapLayeredWidth: c_int = 53;
pub const cudaDevAttrMaxTextureCubemapLayeredLayers: c_int = 54;
pub const cudaDevAttrMaxSurface1DWidth: c_int = 55;
pub const cudaDevAttrMaxSurface2DWidth: c_int = 56;
pub const cudaDevAttrMaxSurface2DHeight: c_int = 57;
pub const cudaDevAttrMaxSurface3DWidth: c_int = 58;
pub const cudaDevAttrMaxSurface3DHeight: c_int = 59;
pub const cudaDevAttrMaxSurface3DDepth: c_int = 60;
pub const cudaDevAttrMaxSurface1DLayeredWidth: c_int = 61;
pub const cudaDevAttrMaxSurface1DLayeredLayers: c_int = 62;
pub const cudaDevAttrMaxSurface2DLayeredWidth: c_int = 63;
pub const cudaDevAttrMaxSurface2DLayeredHeight: c_int = 64;
pub const cudaDevAttrMaxSurface2DLayeredLayers: c_int = 65;
pub const cudaDevAttrMaxSurfaceCubemapWidth: c_int = 66;
pub const cudaDevAttrMaxSurfaceCubemapLayeredWidth: c_int = 67;
pub const cudaDevAttrMaxSurfaceCubemapLayeredLayers: c_int = 68;
pub const cudaDevAttrMaxTexture1DLinearWidth: c_int = 69;
pub const cudaDevAttrMaxTexture2DLinearWidth: c_int = 70;
pub const cudaDevAttrMaxTexture2DLinearHeight: c_int = 71;
pub const cudaDevAttrMaxTexture2DLinearPitch: c_int = 72;
pub const cudaDevAttrMaxTexture2DMipmappedWidth: c_int = 73;
pub const cudaDevAttrMaxTexture2DMipmappedHeight: c_int = 74;
pub const cudaDevAttrComputeCapabilityMajor: c_int = 75;
pub const cudaDevAttrComputeCapabilityMinor: c_int = 76;
pub const cudaDevAttrMaxTexture1DMipmappedWidth: c_int = 77;
pub const cudaDevAttrStreamPrioritiesSupported: c_int = 78;
pub const cudaDevAttrGlobalL1CacheSupported: c_int = 79;
pub const cudaDevAttrLocalL1CacheSupported: c_int = 80;
pub const cudaDevAttrMaxSharedMemoryPerMultiprocessor: c_int = 81;
pub const cudaDevAttrMaxRegistersPerMultiprocessor: c_int = 82;
pub const cudaDevAttrManagedMemory: c_int = 83;
pub const cudaDevAttrIsMultiGpuBoard: c_int = 84;
pub const cudaDevAttrMultiGpuBoardGroupID: c_int = 85;
pub const cudaDevAttrHostNativeAtomicSupported: c_int = 86;
pub const cudaDevAttrSingleToDoublePrecisionPerfRatio: c_int = 87;
pub const cudaDevAttrPageableMemoryAccess: c_int = 88;
pub const cudaDevAttrConcurrentManagedAccess: c_int = 89;
pub const cudaDevAttrComputePreemptionSupported: c_int = 90;
pub const cudaDevAttrCanUseHostPointerForRegisteredMem: c_int = 91;
pub const cudaDevAttrReserved92: c_int = 92;
pub const cudaDevAttrReserved93: c_int = 93;
pub const cudaDevAttrReserved94: c_int = 94;
pub const cudaDevAttrCooperativeLaunch: c_int = 95;
pub const cudaDevAttrCooperativeMultiDeviceLaunch: c_int = 96;
pub const cudaDevAttrMaxSharedMemoryPerBlockOptin: c_int = 97;
pub const cudaDevAttrCanFlushRemoteWrites: c_int = 98;
pub const cudaDevAttrHostRegisterSupported: c_int = 99;
pub const cudaDevAttrPageableMemoryAccessUsesHostPageTables: c_int = 100;
pub const cudaDevAttrDirectManagedMemAccessFromHost: c_int = 101;
pub const cudaDevAttrMaxBlocksPerMultiprocessor: c_int = 106;
pub const cudaDevAttrMaxPersistingL2CacheSize: c_int = 108;
pub const cudaDevAttrMaxAccessPolicyWindowSize: c_int = 109;
pub const cudaDevAttrReservedSharedMemoryPerBlock: c_int = 111;
pub const cudaDevAttrSparseCudaArraySupported: c_int = 112;
pub const cudaDevAttrHostRegisterReadOnlySupported: c_int = 113;
pub const cudaDevAttrTimelineSemaphoreInteropSupported: c_int = 114;
pub const cudaDevAttrMaxTimelineSemaphoreInteropSupported: c_int = 114;
pub const cudaDevAttrMemoryPoolsSupported: c_int = 115;
pub const cudaDevAttrGPUDirectRDMASupported: c_int = 116;
pub const cudaDevAttrGPUDirectRDMAFlushWritesOptions: c_int = 117;
pub const cudaDevAttrGPUDirectRDMAWritesOrdering: c_int = 118;
pub const cudaDevAttrMemoryPoolSupportedHandleTypes: c_int = 119;
pub const cudaDevAttrClusterLaunch: c_int = 120;
pub const cudaDevAttrDeferredMappingCudaArraySupported: c_int = 121;
pub const cudaDevAttrReserved122: c_int = 122;
pub const cudaDevAttrReserved123: c_int = 123;
pub const cudaDevAttrReserved124: c_int = 124;
pub const cudaDevAttrIpcEventSupport: c_int = 125;
pub const cudaDevAttrMemSyncDomainCount: c_int = 126;
pub const cudaDevAttrReserved127: c_int = 127;
pub const cudaDevAttrReserved128: c_int = 128;
pub const cudaDevAttrReserved129: c_int = 129;
pub const cudaDevAttrNumaConfig: c_int = 130;
pub const cudaDevAttrNumaId: c_int = 131;
pub const cudaDevAttrReserved132: c_int = 132;
pub const cudaDevAttrMpsEnabled: c_int = 133;
pub const cudaDevAttrHostNumaId: c_int = 134;
pub const cudaDevAttrD3D12CigSupported: c_int = 135;
pub const cudaDevAttrGpuPciDeviceId: c_int = 139;
pub const cudaDevAttrGpuPciSubsystemId: c_int = 140;
pub const cudaDevAttrHostNumaMultinodeIpcSupported: c_int = 143;
pub const cudaDevAttrMax: c_int = 144;
pub const enum_cudaDeviceAttr = c_uint;
pub const cudaMemPoolReuseFollowEventDependencies: c_int = 1;
pub const cudaMemPoolReuseAllowOpportunistic: c_int = 2;
pub const cudaMemPoolReuseAllowInternalDependencies: c_int = 3;
pub const cudaMemPoolAttrReleaseThreshold: c_int = 4;
pub const cudaMemPoolAttrReservedMemCurrent: c_int = 5;
pub const cudaMemPoolAttrReservedMemHigh: c_int = 6;
pub const cudaMemPoolAttrUsedMemCurrent: c_int = 7;
pub const cudaMemPoolAttrUsedMemHigh: c_int = 8;
pub const enum_cudaMemPoolAttr = c_uint;
pub const cudaMemLocationTypeInvalid: c_int = 0;
pub const cudaMemLocationTypeDevice: c_int = 1;
pub const cudaMemLocationTypeHost: c_int = 2;
pub const cudaMemLocationTypeHostNuma: c_int = 3;
pub const cudaMemLocationTypeHostNumaCurrent: c_int = 4;
pub const enum_cudaMemLocationType = c_uint;
pub const struct_cudaMemLocation = extern struct {
    type: enum_cudaMemLocationType = @import("std").mem.zeroes(enum_cudaMemLocationType),
    id: c_int = @import("std").mem.zeroes(c_int),
};
pub const cudaMemAccessFlagsProtNone: c_int = 0;
pub const cudaMemAccessFlagsProtRead: c_int = 1;
pub const cudaMemAccessFlagsProtReadWrite: c_int = 3;
pub const enum_cudaMemAccessFlags = c_uint;
pub const struct_cudaMemAccessDesc = extern struct {
    location: struct_cudaMemLocation = @import("std").mem.zeroes(struct_cudaMemLocation),
    flags: enum_cudaMemAccessFlags = @import("std").mem.zeroes(enum_cudaMemAccessFlags),
};
pub const cudaMemAllocationTypeInvalid: c_int = 0;
pub const cudaMemAllocationTypePinned: c_int = 1;
pub const cudaMemAllocationTypeMax: c_int = 2147483647;
pub const enum_cudaMemAllocationType = c_uint;
pub const cudaMemHandleTypeNone: c_int = 0;
pub const cudaMemHandleTypePosixFileDescriptor: c_int = 1;
pub const cudaMemHandleTypeWin32: c_int = 2;
pub const cudaMemHandleTypeWin32Kmt: c_int = 4;
pub const cudaMemHandleTypeFabric: c_int = 8;
pub const enum_cudaMemAllocationHandleType = c_uint;
pub const struct_cudaMemPoolProps = extern struct {
    allocType: enum_cudaMemAllocationType = @import("std").mem.zeroes(enum_cudaMemAllocationType),
    handleTypes: enum_cudaMemAllocationHandleType = @import("std").mem.zeroes(enum_cudaMemAllocationHandleType),
    location: struct_cudaMemLocation = @import("std").mem.zeroes(struct_cudaMemLocation),
    win32SecurityAttributes: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    maxSize: usize = @import("std").mem.zeroes(usize),
    usage: c_ushort = @import("std").mem.zeroes(c_ushort),
    reserved: [54]u8 = @import("std").mem.zeroes([54]u8),
};
pub const struct_cudaMemPoolPtrExportData = extern struct {
    reserved: [64]u8 = @import("std").mem.zeroes([64]u8),
};
pub const struct_cudaMemAllocNodeParams = extern struct {
    poolProps: struct_cudaMemPoolProps = @import("std").mem.zeroes(struct_cudaMemPoolProps),
    accessDescs: [*c]const struct_cudaMemAccessDesc = @import("std").mem.zeroes([*c]const struct_cudaMemAccessDesc),
    accessDescCount: usize = @import("std").mem.zeroes(usize),
    bytesize: usize = @import("std").mem.zeroes(usize),
    dptr: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
};
pub const struct_cudaMemAllocNodeParamsV2 = extern struct {
    poolProps: struct_cudaMemPoolProps = @import("std").mem.zeroes(struct_cudaMemPoolProps),
    accessDescs: [*c]const struct_cudaMemAccessDesc = @import("std").mem.zeroes([*c]const struct_cudaMemAccessDesc),
    accessDescCount: usize = @import("std").mem.zeroes(usize),
    bytesize: usize = @import("std").mem.zeroes(usize),
    dptr: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
};
pub const struct_cudaMemFreeNodeParams = extern struct {
    dptr: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
};
pub const cudaGraphMemAttrUsedMemCurrent: c_int = 0;
pub const cudaGraphMemAttrUsedMemHigh: c_int = 1;
pub const cudaGraphMemAttrReservedMemCurrent: c_int = 2;
pub const cudaGraphMemAttrReservedMemHigh: c_int = 3;
pub const enum_cudaGraphMemAttributeType = c_uint;
pub const cudaMemcpyFlagDefault: c_int = 0;
pub const cudaMemcpyFlagPreferOverlapWithCompute: c_int = 1;
pub const enum_cudaMemcpyFlags = c_uint;
pub const cudaMemcpySrcAccessOrderInvalid: c_int = 0;
pub const cudaMemcpySrcAccessOrderStream: c_int = 1;
pub const cudaMemcpySrcAccessOrderDuringApiCall: c_int = 2;
pub const cudaMemcpySrcAccessOrderAny: c_int = 3;
pub const cudaMemcpySrcAccessOrderMax: c_int = 2147483647;
pub const enum_cudaMemcpySrcAccessOrder = c_uint;
pub const struct_cudaMemcpyAttributes = extern struct {
    srcAccessOrder: enum_cudaMemcpySrcAccessOrder = @import("std").mem.zeroes(enum_cudaMemcpySrcAccessOrder),
    srcLocHint: struct_cudaMemLocation = @import("std").mem.zeroes(struct_cudaMemLocation),
    dstLocHint: struct_cudaMemLocation = @import("std").mem.zeroes(struct_cudaMemLocation),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const cudaMemcpyOperandTypePointer: c_int = 1;
pub const cudaMemcpyOperandTypeArray: c_int = 2;
pub const cudaMemcpyOperandTypeMax: c_int = 2147483647;
pub const enum_cudaMemcpy3DOperandType = c_uint;
pub const struct_cudaOffset3D = extern struct {
    x: usize = @import("std").mem.zeroes(usize),
    y: usize = @import("std").mem.zeroes(usize),
    z: usize = @import("std").mem.zeroes(usize),
};
const struct_unnamed_52 = extern struct {
    ptr: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    rowLength: usize = @import("std").mem.zeroes(usize),
    layerHeight: usize = @import("std").mem.zeroes(usize),
    locHint: struct_cudaMemLocation = @import("std").mem.zeroes(struct_cudaMemLocation),
};
const struct_unnamed_53 = extern struct {
    array: cudaArray_t = @import("std").mem.zeroes(cudaArray_t),
    offset: struct_cudaOffset3D = @import("std").mem.zeroes(struct_cudaOffset3D),
};
const union_unnamed_51 = extern union {
    ptr: struct_unnamed_52,
    array: struct_unnamed_53,
};
pub const struct_cudaMemcpy3DOperand = extern struct {
    type: enum_cudaMemcpy3DOperandType = @import("std").mem.zeroes(enum_cudaMemcpy3DOperandType),
    op: union_unnamed_51 = @import("std").mem.zeroes(union_unnamed_51),
};
pub const struct_cudaMemcpy3DBatchOp = extern struct {
    src: struct_cudaMemcpy3DOperand = @import("std").mem.zeroes(struct_cudaMemcpy3DOperand),
    dst: struct_cudaMemcpy3DOperand = @import("std").mem.zeroes(struct_cudaMemcpy3DOperand),
    extent: struct_cudaExtent = @import("std").mem.zeroes(struct_cudaExtent),
    srcAccessOrder: enum_cudaMemcpySrcAccessOrder = @import("std").mem.zeroes(enum_cudaMemcpySrcAccessOrder),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const cudaDevP2PAttrPerformanceRank: c_int = 1;
pub const cudaDevP2PAttrAccessSupported: c_int = 2;
pub const cudaDevP2PAttrNativeAtomicSupported: c_int = 3;
pub const cudaDevP2PAttrCudaArrayAccessSupported: c_int = 4;
pub const enum_cudaDeviceP2PAttr = c_uint;
pub const cudaUUID_t = struct_CUuuid_st;
pub const struct_cudaDeviceProp = extern struct {
    name: [256]u8 = @import("std").mem.zeroes([256]u8),
    uuid: cudaUUID_t = @import("std").mem.zeroes(cudaUUID_t),
    luid: [8]u8 = @import("std").mem.zeroes([8]u8),
    luidDeviceNodeMask: c_uint = @import("std").mem.zeroes(c_uint),
    totalGlobalMem: usize = @import("std").mem.zeroes(usize),
    sharedMemPerBlock: usize = @import("std").mem.zeroes(usize),
    regsPerBlock: c_int = @import("std").mem.zeroes(c_int),
    warpSize: c_int = @import("std").mem.zeroes(c_int),
    memPitch: usize = @import("std").mem.zeroes(usize),
    maxThreadsPerBlock: c_int = @import("std").mem.zeroes(c_int),
    maxThreadsDim: [3]c_int = @import("std").mem.zeroes([3]c_int),
    maxGridSize: [3]c_int = @import("std").mem.zeroes([3]c_int),
    clockRate: c_int = @import("std").mem.zeroes(c_int),
    totalConstMem: usize = @import("std").mem.zeroes(usize),
    major: c_int = @import("std").mem.zeroes(c_int),
    minor: c_int = @import("std").mem.zeroes(c_int),
    textureAlignment: usize = @import("std").mem.zeroes(usize),
    texturePitchAlignment: usize = @import("std").mem.zeroes(usize),
    deviceOverlap: c_int = @import("std").mem.zeroes(c_int),
    multiProcessorCount: c_int = @import("std").mem.zeroes(c_int),
    kernelExecTimeoutEnabled: c_int = @import("std").mem.zeroes(c_int),
    integrated: c_int = @import("std").mem.zeroes(c_int),
    canMapHostMemory: c_int = @import("std").mem.zeroes(c_int),
    computeMode: c_int = @import("std").mem.zeroes(c_int),
    maxTexture1D: c_int = @import("std").mem.zeroes(c_int),
    maxTexture1DMipmap: c_int = @import("std").mem.zeroes(c_int),
    maxTexture1DLinear: c_int = @import("std").mem.zeroes(c_int),
    maxTexture2D: [2]c_int = @import("std").mem.zeroes([2]c_int),
    maxTexture2DMipmap: [2]c_int = @import("std").mem.zeroes([2]c_int),
    maxTexture2DLinear: [3]c_int = @import("std").mem.zeroes([3]c_int),
    maxTexture2DGather: [2]c_int = @import("std").mem.zeroes([2]c_int),
    maxTexture3D: [3]c_int = @import("std").mem.zeroes([3]c_int),
    maxTexture3DAlt: [3]c_int = @import("std").mem.zeroes([3]c_int),
    maxTextureCubemap: c_int = @import("std").mem.zeroes(c_int),
    maxTexture1DLayered: [2]c_int = @import("std").mem.zeroes([2]c_int),
    maxTexture2DLayered: [3]c_int = @import("std").mem.zeroes([3]c_int),
    maxTextureCubemapLayered: [2]c_int = @import("std").mem.zeroes([2]c_int),
    maxSurface1D: c_int = @import("std").mem.zeroes(c_int),
    maxSurface2D: [2]c_int = @import("std").mem.zeroes([2]c_int),
    maxSurface3D: [3]c_int = @import("std").mem.zeroes([3]c_int),
    maxSurface1DLayered: [2]c_int = @import("std").mem.zeroes([2]c_int),
    maxSurface2DLayered: [3]c_int = @import("std").mem.zeroes([3]c_int),
    maxSurfaceCubemap: c_int = @import("std").mem.zeroes(c_int),
    maxSurfaceCubemapLayered: [2]c_int = @import("std").mem.zeroes([2]c_int),
    surfaceAlignment: usize = @import("std").mem.zeroes(usize),
    concurrentKernels: c_int = @import("std").mem.zeroes(c_int),
    ECCEnabled: c_int = @import("std").mem.zeroes(c_int),
    pciBusID: c_int = @import("std").mem.zeroes(c_int),
    pciDeviceID: c_int = @import("std").mem.zeroes(c_int),
    pciDomainID: c_int = @import("std").mem.zeroes(c_int),
    tccDriver: c_int = @import("std").mem.zeroes(c_int),
    asyncEngineCount: c_int = @import("std").mem.zeroes(c_int),
    unifiedAddressing: c_int = @import("std").mem.zeroes(c_int),
    memoryClockRate: c_int = @import("std").mem.zeroes(c_int),
    memoryBusWidth: c_int = @import("std").mem.zeroes(c_int),
    l2CacheSize: c_int = @import("std").mem.zeroes(c_int),
    persistingL2CacheMaxSize: c_int = @import("std").mem.zeroes(c_int),
    maxThreadsPerMultiProcessor: c_int = @import("std").mem.zeroes(c_int),
    streamPrioritiesSupported: c_int = @import("std").mem.zeroes(c_int),
    globalL1CacheSupported: c_int = @import("std").mem.zeroes(c_int),
    localL1CacheSupported: c_int = @import("std").mem.zeroes(c_int),
    sharedMemPerMultiprocessor: usize = @import("std").mem.zeroes(usize),
    regsPerMultiprocessor: c_int = @import("std").mem.zeroes(c_int),
    managedMemory: c_int = @import("std").mem.zeroes(c_int),
    isMultiGpuBoard: c_int = @import("std").mem.zeroes(c_int),
    multiGpuBoardGroupID: c_int = @import("std").mem.zeroes(c_int),
    hostNativeAtomicSupported: c_int = @import("std").mem.zeroes(c_int),
    singleToDoublePrecisionPerfRatio: c_int = @import("std").mem.zeroes(c_int),
    pageableMemoryAccess: c_int = @import("std").mem.zeroes(c_int),
    concurrentManagedAccess: c_int = @import("std").mem.zeroes(c_int),
    computePreemptionSupported: c_int = @import("std").mem.zeroes(c_int),
    canUseHostPointerForRegisteredMem: c_int = @import("std").mem.zeroes(c_int),
    cooperativeLaunch: c_int = @import("std").mem.zeroes(c_int),
    cooperativeMultiDeviceLaunch: c_int = @import("std").mem.zeroes(c_int),
    sharedMemPerBlockOptin: usize = @import("std").mem.zeroes(usize),
    pageableMemoryAccessUsesHostPageTables: c_int = @import("std").mem.zeroes(c_int),
    directManagedMemAccessFromHost: c_int = @import("std").mem.zeroes(c_int),
    maxBlocksPerMultiProcessor: c_int = @import("std").mem.zeroes(c_int),
    accessPolicyMaxWindowSize: c_int = @import("std").mem.zeroes(c_int),
    reservedSharedMemPerBlock: usize = @import("std").mem.zeroes(usize),
    hostRegisterSupported: c_int = @import("std").mem.zeroes(c_int),
    sparseCudaArraySupported: c_int = @import("std").mem.zeroes(c_int),
    hostRegisterReadOnlySupported: c_int = @import("std").mem.zeroes(c_int),
    timelineSemaphoreInteropSupported: c_int = @import("std").mem.zeroes(c_int),
    memoryPoolsSupported: c_int = @import("std").mem.zeroes(c_int),
    gpuDirectRDMASupported: c_int = @import("std").mem.zeroes(c_int),
    gpuDirectRDMAFlushWritesOptions: c_uint = @import("std").mem.zeroes(c_uint),
    gpuDirectRDMAWritesOrdering: c_int = @import("std").mem.zeroes(c_int),
    memoryPoolSupportedHandleTypes: c_uint = @import("std").mem.zeroes(c_uint),
    deferredMappingCudaArraySupported: c_int = @import("std").mem.zeroes(c_int),
    ipcEventSupported: c_int = @import("std").mem.zeroes(c_int),
    clusterLaunch: c_int = @import("std").mem.zeroes(c_int),
    unifiedFunctionPointers: c_int = @import("std").mem.zeroes(c_int),
    reserved: [63]c_int = @import("std").mem.zeroes([63]c_int),
};
pub const struct_cudaIpcEventHandle_st = extern struct {
    reserved: [64]u8 = @import("std").mem.zeroes([64]u8),
};
pub const cudaIpcEventHandle_t = struct_cudaIpcEventHandle_st;
pub const struct_cudaIpcMemHandle_st = extern struct {
    reserved: [64]u8 = @import("std").mem.zeroes([64]u8),
};
pub const cudaIpcMemHandle_t = struct_cudaIpcMemHandle_st;
pub const struct_cudaMemFabricHandle_st = extern struct {
    reserved: [64]u8 = @import("std").mem.zeroes([64]u8),
};
pub const cudaMemFabricHandle_t = struct_cudaMemFabricHandle_st;
pub const cudaExternalMemoryHandleTypeOpaqueFd: c_int = 1;
pub const cudaExternalMemoryHandleTypeOpaqueWin32: c_int = 2;
pub const cudaExternalMemoryHandleTypeOpaqueWin32Kmt: c_int = 3;
pub const cudaExternalMemoryHandleTypeD3D12Heap: c_int = 4;
pub const cudaExternalMemoryHandleTypeD3D12Resource: c_int = 5;
pub const cudaExternalMemoryHandleTypeD3D11Resource: c_int = 6;
pub const cudaExternalMemoryHandleTypeD3D11ResourceKmt: c_int = 7;
pub const cudaExternalMemoryHandleTypeNvSciBuf: c_int = 8;
pub const enum_cudaExternalMemoryHandleType = c_uint;
const struct_unnamed_55 = extern struct {
    handle: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    name: ?*const anyopaque = @import("std").mem.zeroes(?*const anyopaque),
};
const union_unnamed_54 = extern union {
    fd: c_int,
    win32: struct_unnamed_55,
    nvSciBufObject: ?*const anyopaque,
};
pub const struct_cudaExternalMemoryHandleDesc = extern struct {
    type: enum_cudaExternalMemoryHandleType = @import("std").mem.zeroes(enum_cudaExternalMemoryHandleType),
    handle: union_unnamed_54 = @import("std").mem.zeroes(union_unnamed_54),
    size: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const struct_cudaExternalMemoryBufferDesc = extern struct {
    offset: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    size: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const struct_cudaExternalMemoryMipmappedArrayDesc = extern struct {
    offset: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    formatDesc: struct_cudaChannelFormatDesc = @import("std").mem.zeroes(struct_cudaChannelFormatDesc),
    extent: struct_cudaExtent = @import("std").mem.zeroes(struct_cudaExtent),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
    numLevels: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const cudaExternalSemaphoreHandleTypeOpaqueFd: c_int = 1;
pub const cudaExternalSemaphoreHandleTypeOpaqueWin32: c_int = 2;
pub const cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt: c_int = 3;
pub const cudaExternalSemaphoreHandleTypeD3D12Fence: c_int = 4;
pub const cudaExternalSemaphoreHandleTypeD3D11Fence: c_int = 5;
pub const cudaExternalSemaphoreHandleTypeNvSciSync: c_int = 6;
pub const cudaExternalSemaphoreHandleTypeKeyedMutex: c_int = 7;
pub const cudaExternalSemaphoreHandleTypeKeyedMutexKmt: c_int = 8;
pub const cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd: c_int = 9;
pub const cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32: c_int = 10;
pub const enum_cudaExternalSemaphoreHandleType = c_uint;
const struct_unnamed_57 = extern struct {
    handle: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    name: ?*const anyopaque = @import("std").mem.zeroes(?*const anyopaque),
};
const union_unnamed_56 = extern union {
    fd: c_int,
    win32: struct_unnamed_57,
    nvSciSyncObj: ?*const anyopaque,
};
pub const struct_cudaExternalSemaphoreHandleDesc = extern struct {
    type: enum_cudaExternalSemaphoreHandleType = @import("std").mem.zeroes(enum_cudaExternalSemaphoreHandleType),
    handle: union_unnamed_56 = @import("std").mem.zeroes(union_unnamed_56),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
};
const struct_unnamed_59 = extern struct {
    value: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
};
const union_unnamed_60 = extern union {
    fence: ?*anyopaque,
    reserved: c_ulonglong,
};
const struct_unnamed_61 = extern struct {
    key: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
};
const struct_unnamed_58 = extern struct {
    fence: struct_unnamed_59 = @import("std").mem.zeroes(struct_unnamed_59),
    nvSciSync: union_unnamed_60 = @import("std").mem.zeroes(union_unnamed_60),
    keyedMutex: struct_unnamed_61 = @import("std").mem.zeroes(struct_unnamed_61),
};
pub const struct_cudaExternalSemaphoreSignalParams_v1 = extern struct {
    params: struct_unnamed_58 = @import("std").mem.zeroes(struct_unnamed_58),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
};
const struct_unnamed_63 = extern struct {
    value: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
};
const union_unnamed_64 = extern union {
    fence: ?*anyopaque,
    reserved: c_ulonglong,
};
const struct_unnamed_65 = extern struct {
    key: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    timeoutMs: c_uint = @import("std").mem.zeroes(c_uint),
};
const struct_unnamed_62 = extern struct {
    fence: struct_unnamed_63 = @import("std").mem.zeroes(struct_unnamed_63),
    nvSciSync: union_unnamed_64 = @import("std").mem.zeroes(union_unnamed_64),
    keyedMutex: struct_unnamed_65 = @import("std").mem.zeroes(struct_unnamed_65),
};
pub const struct_cudaExternalSemaphoreWaitParams_v1 = extern struct {
    params: struct_unnamed_62 = @import("std").mem.zeroes(struct_unnamed_62),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
};
const struct_unnamed_67 = extern struct {
    value: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
};
const union_unnamed_68 = extern union {
    fence: ?*anyopaque,
    reserved: c_ulonglong,
};
const struct_unnamed_69 = extern struct {
    key: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
};
const struct_unnamed_66 = extern struct {
    fence: struct_unnamed_67 = @import("std").mem.zeroes(struct_unnamed_67),
    nvSciSync: union_unnamed_68 = @import("std").mem.zeroes(union_unnamed_68),
    keyedMutex: struct_unnamed_69 = @import("std").mem.zeroes(struct_unnamed_69),
    reserved: [12]c_uint = @import("std").mem.zeroes([12]c_uint),
};
pub const struct_cudaExternalSemaphoreSignalParams = extern struct {
    params: struct_unnamed_66 = @import("std").mem.zeroes(struct_unnamed_66),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
    reserved: [16]c_uint = @import("std").mem.zeroes([16]c_uint),
};
const struct_unnamed_71 = extern struct {
    value: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
};
const union_unnamed_72 = extern union {
    fence: ?*anyopaque,
    reserved: c_ulonglong,
};
const struct_unnamed_73 = extern struct {
    key: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    timeoutMs: c_uint = @import("std").mem.zeroes(c_uint),
};
const struct_unnamed_70 = extern struct {
    fence: struct_unnamed_71 = @import("std").mem.zeroes(struct_unnamed_71),
    nvSciSync: union_unnamed_72 = @import("std").mem.zeroes(union_unnamed_72),
    keyedMutex: struct_unnamed_73 = @import("std").mem.zeroes(struct_unnamed_73),
    reserved: [10]c_uint = @import("std").mem.zeroes([10]c_uint),
};
pub const struct_cudaExternalSemaphoreWaitParams = extern struct {
    params: struct_unnamed_70 = @import("std").mem.zeroes(struct_unnamed_70),
    flags: c_uint = @import("std").mem.zeroes(c_uint),
    reserved: [16]c_uint = @import("std").mem.zeroes([16]c_uint),
};
pub const cudaError_t = enum_cudaError;
pub const cudaStream_t = ?*struct_CUstream_st;
pub const cudaEvent_t = ?*struct_CUevent_st;
pub const cudaGraphicsResource_t = ?*struct_cudaGraphicsResource;
pub const struct_CUexternalMemory_st = opaque {};
pub const cudaExternalMemory_t = ?*struct_CUexternalMemory_st;
pub const struct_CUexternalSemaphore_st = opaque {};
pub const cudaExternalSemaphore_t = ?*struct_CUexternalSemaphore_st;
pub const cudaGraph_t = ?*struct_CUgraph_st;
pub const cudaGraphNode_t = ?*struct_CUgraphNode_st;
pub const cudaUserObject_t = ?*struct_CUuserObject_st;
pub const cudaGraphConditionalHandle = c_ulonglong;
pub const cudaFunction_t = ?*struct_CUfunc_st;
pub const cudaKernel_t = ?*struct_CUkern_st;
pub const cudaJitMaxRegisters: c_int = 0;
pub const cudaJitThreadsPerBlock: c_int = 1;
pub const cudaJitWallTime: c_int = 2;
pub const cudaJitInfoLogBuffer: c_int = 3;
pub const cudaJitInfoLogBufferSizeBytes: c_int = 4;
pub const cudaJitErrorLogBuffer: c_int = 5;
pub const cudaJitErrorLogBufferSizeBytes: c_int = 6;
pub const cudaJitOptimizationLevel: c_int = 7;
pub const cudaJitFallbackStrategy: c_int = 10;
pub const cudaJitGenerateDebugInfo: c_int = 11;
pub const cudaJitLogVerbose: c_int = 12;
pub const cudaJitGenerateLineInfo: c_int = 13;
pub const cudaJitCacheMode: c_int = 14;
pub const cudaJitPositionIndependentCode: c_int = 30;
pub const cudaJitMinCtaPerSm: c_int = 31;
pub const cudaJitMaxThreadsPerBlock: c_int = 32;
pub const cudaJitOverrideDirectiveValues: c_int = 33;
pub const enum_cudaJitOption = c_uint;
pub const cudaLibraryHostUniversalFunctionAndDataTable: c_int = 0;
pub const cudaLibraryBinaryIsPreserved: c_int = 1;
pub const enum_cudaLibraryOption = c_uint;
pub const struct_cudalibraryHostUniversalFunctionAndDataTable = extern struct {
    functionTable: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    functionWindowSize: usize = @import("std").mem.zeroes(usize),
    dataTable: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    dataWindowSize: usize = @import("std").mem.zeroes(usize),
};
pub const cudaJitCacheOptionNone: c_int = 0;
pub const cudaJitCacheOptionCG: c_int = 1;
pub const cudaJitCacheOptionCA: c_int = 2;
pub const enum_cudaJit_CacheMode = c_uint;
pub const cudaPreferPtx: c_int = 0;
pub const cudaPreferBinary: c_int = 1;
pub const enum_cudaJit_Fallback = c_uint;
pub const cudaLibrary_t = ?*struct_CUlib_st;
pub const cudaMemPool_t = ?*struct_CUmemPoolHandle_st;
pub const cudaCGScopeInvalid: c_int = 0;
pub const cudaCGScopeGrid: c_int = 1;
pub const cudaCGScopeMultiGrid: c_int = 2;
pub const enum_cudaCGScope = c_uint;
pub const struct_cudaLaunchParams = extern struct {
    func: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    gridDim: dim3 = @import("std").mem.zeroes(dim3),
    blockDim: dim3 = @import("std").mem.zeroes(dim3),
    args: [*c]?*anyopaque = @import("std").mem.zeroes([*c]?*anyopaque),
    sharedMem: usize = @import("std").mem.zeroes(usize),
    stream: cudaStream_t = @import("std").mem.zeroes(cudaStream_t),
};
pub const struct_cudaKernelNodeParams = extern struct {
    func: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    gridDim: dim3 = @import("std").mem.zeroes(dim3),
    blockDim: dim3 = @import("std").mem.zeroes(dim3),
    sharedMemBytes: c_uint = @import("std").mem.zeroes(c_uint),
    kernelParams: [*c]?*anyopaque = @import("std").mem.zeroes([*c]?*anyopaque),
    extra: [*c]?*anyopaque = @import("std").mem.zeroes([*c]?*anyopaque),
};
pub const struct_cudaKernelNodeParamsV2 = extern struct {
    func: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    gridDim: dim3 = @import("std").mem.zeroes(dim3),
    blockDim: dim3 = @import("std").mem.zeroes(dim3),
    sharedMemBytes: c_uint = @import("std").mem.zeroes(c_uint),
    kernelParams: [*c]?*anyopaque = @import("std").mem.zeroes([*c]?*anyopaque),
    extra: [*c]?*anyopaque = @import("std").mem.zeroes([*c]?*anyopaque),
};
pub const struct_cudaExternalSemaphoreSignalNodeParams = extern struct {
    extSemArray: [*c]cudaExternalSemaphore_t = @import("std").mem.zeroes([*c]cudaExternalSemaphore_t),
    paramsArray: [*c]const struct_cudaExternalSemaphoreSignalParams = @import("std").mem.zeroes([*c]const struct_cudaExternalSemaphoreSignalParams),
    numExtSems: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const struct_cudaExternalSemaphoreSignalNodeParamsV2 = extern struct {
    extSemArray: [*c]cudaExternalSemaphore_t = @import("std").mem.zeroes([*c]cudaExternalSemaphore_t),
    paramsArray: [*c]const struct_cudaExternalSemaphoreSignalParams = @import("std").mem.zeroes([*c]const struct_cudaExternalSemaphoreSignalParams),
    numExtSems: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const struct_cudaExternalSemaphoreWaitNodeParams = extern struct {
    extSemArray: [*c]cudaExternalSemaphore_t = @import("std").mem.zeroes([*c]cudaExternalSemaphore_t),
    paramsArray: [*c]const struct_cudaExternalSemaphoreWaitParams = @import("std").mem.zeroes([*c]const struct_cudaExternalSemaphoreWaitParams),
    numExtSems: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const struct_cudaExternalSemaphoreWaitNodeParamsV2 = extern struct {
    extSemArray: [*c]cudaExternalSemaphore_t = @import("std").mem.zeroes([*c]cudaExternalSemaphore_t),
    paramsArray: [*c]const struct_cudaExternalSemaphoreWaitParams = @import("std").mem.zeroes([*c]const struct_cudaExternalSemaphoreWaitParams),
    numExtSems: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const cudaGraphCondAssignDefault: c_int = 1;
pub const enum_cudaGraphConditionalHandleFlags = c_uint;
pub const cudaGraphCondTypeIf: c_int = 0;
pub const cudaGraphCondTypeWhile: c_int = 1;
pub const cudaGraphCondTypeSwitch: c_int = 2;
pub const enum_cudaGraphConditionalNodeType = c_uint;
pub const struct_cudaConditionalNodeParams = extern struct {
    handle: cudaGraphConditionalHandle = @import("std").mem.zeroes(cudaGraphConditionalHandle),
    type: enum_cudaGraphConditionalNodeType = @import("std").mem.zeroes(enum_cudaGraphConditionalNodeType),
    size: c_uint = @import("std").mem.zeroes(c_uint),
    phGraph_out: [*c]cudaGraph_t = @import("std").mem.zeroes([*c]cudaGraph_t),
};
pub const cudaGraphNodeTypeKernel: c_int = 0;
pub const cudaGraphNodeTypeMemcpy: c_int = 1;
pub const cudaGraphNodeTypeMemset: c_int = 2;
pub const cudaGraphNodeTypeHost: c_int = 3;
pub const cudaGraphNodeTypeGraph: c_int = 4;
pub const cudaGraphNodeTypeEmpty: c_int = 5;
pub const cudaGraphNodeTypeWaitEvent: c_int = 6;
pub const cudaGraphNodeTypeEventRecord: c_int = 7;
pub const cudaGraphNodeTypeExtSemaphoreSignal: c_int = 8;
pub const cudaGraphNodeTypeExtSemaphoreWait: c_int = 9;
pub const cudaGraphNodeTypeMemAlloc: c_int = 10;
pub const cudaGraphNodeTypeMemFree: c_int = 11;
pub const cudaGraphNodeTypeConditional: c_int = 13;
pub const cudaGraphNodeTypeCount: c_int = 14;
pub const enum_cudaGraphNodeType = c_uint;
pub const struct_cudaChildGraphNodeParams = extern struct {
    graph: cudaGraph_t = @import("std").mem.zeroes(cudaGraph_t),
};
pub const struct_cudaEventRecordNodeParams = extern struct {
    event: cudaEvent_t = @import("std").mem.zeroes(cudaEvent_t),
};
pub const struct_cudaEventWaitNodeParams = extern struct {
    event: cudaEvent_t = @import("std").mem.zeroes(cudaEvent_t),
};
const union_unnamed_74 = extern union {
    reserved1: [29]c_longlong,
    kernel: struct_cudaKernelNodeParamsV2,
    memcpy: struct_cudaMemcpyNodeParams,
    memset: struct_cudaMemsetParamsV2,
    host: struct_cudaHostNodeParamsV2,
    graph: struct_cudaChildGraphNodeParams,
    eventWait: struct_cudaEventWaitNodeParams,
    eventRecord: struct_cudaEventRecordNodeParams,
    extSemSignal: struct_cudaExternalSemaphoreSignalNodeParamsV2,
    extSemWait: struct_cudaExternalSemaphoreWaitNodeParamsV2,
    alloc: struct_cudaMemAllocNodeParamsV2,
    free: struct_cudaMemFreeNodeParams,
    conditional: struct_cudaConditionalNodeParams,
};
pub const struct_cudaGraphNodeParams = extern struct {
    type: enum_cudaGraphNodeType = @import("std").mem.zeroes(enum_cudaGraphNodeType),
    reserved0: [3]c_int = @import("std").mem.zeroes([3]c_int),
    unnamed_0: union_unnamed_74 = @import("std").mem.zeroes(union_unnamed_74),
    reserved2: c_longlong = @import("std").mem.zeroes(c_longlong),
};
pub const cudaGraphDependencyTypeDefault: c_int = 0;
pub const cudaGraphDependencyTypeProgrammatic: c_int = 1;
pub const enum_cudaGraphDependencyType_enum = c_uint;
pub const cudaGraphDependencyType = enum_cudaGraphDependencyType_enum;
pub const struct_cudaGraphEdgeData_st = extern struct {
    from_port: u8 = @import("std").mem.zeroes(u8),
    to_port: u8 = @import("std").mem.zeroes(u8),
    type: u8 = @import("std").mem.zeroes(u8),
    reserved: [5]u8 = @import("std").mem.zeroes([5]u8),
};
pub const cudaGraphEdgeData = struct_cudaGraphEdgeData_st;
pub const cudaGraphExec_t = ?*struct_CUgraphExec_st;
pub const cudaGraphExecUpdateSuccess: c_int = 0;
pub const cudaGraphExecUpdateError: c_int = 1;
pub const cudaGraphExecUpdateErrorTopologyChanged: c_int = 2;
pub const cudaGraphExecUpdateErrorNodeTypeChanged: c_int = 3;
pub const cudaGraphExecUpdateErrorFunctionChanged: c_int = 4;
pub const cudaGraphExecUpdateErrorParametersChanged: c_int = 5;
pub const cudaGraphExecUpdateErrorNotSupported: c_int = 6;
pub const cudaGraphExecUpdateErrorUnsupportedFunctionChange: c_int = 7;
pub const cudaGraphExecUpdateErrorAttributesChanged: c_int = 8;
pub const enum_cudaGraphExecUpdateResult = c_uint;
pub const cudaGraphInstantiateSuccess: c_int = 0;
pub const cudaGraphInstantiateError: c_int = 1;
pub const cudaGraphInstantiateInvalidStructure: c_int = 2;
pub const cudaGraphInstantiateNodeOperationNotSupported: c_int = 3;
pub const cudaGraphInstantiateMultipleDevicesNotSupported: c_int = 4;
pub const cudaGraphInstantiateConditionalHandleUnused: c_int = 5;
pub const enum_cudaGraphInstantiateResult = c_uint;
pub const cudaGraphInstantiateResult = enum_cudaGraphInstantiateResult;
pub const struct_cudaGraphInstantiateParams_st = extern struct {
    flags: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
    uploadStream: cudaStream_t = @import("std").mem.zeroes(cudaStream_t),
    errNode_out: cudaGraphNode_t = @import("std").mem.zeroes(cudaGraphNode_t),
    result_out: cudaGraphInstantiateResult = @import("std").mem.zeroes(cudaGraphInstantiateResult),
};
pub const cudaGraphInstantiateParams = struct_cudaGraphInstantiateParams_st;
pub const struct_cudaGraphExecUpdateResultInfo_st = extern struct {
    result: enum_cudaGraphExecUpdateResult = @import("std").mem.zeroes(enum_cudaGraphExecUpdateResult),
    errorNode: cudaGraphNode_t = @import("std").mem.zeroes(cudaGraphNode_t),
    errorFromNode: cudaGraphNode_t = @import("std").mem.zeroes(cudaGraphNode_t),
};
pub const cudaGraphExecUpdateResultInfo = struct_cudaGraphExecUpdateResultInfo_st;
pub const cudaGraphDeviceNode_t = ?*struct_CUgraphDeviceUpdatableNode_st;
pub const cudaGraphKernelNodeFieldInvalid: c_int = 0;
pub const cudaGraphKernelNodeFieldGridDim: c_int = 1;
pub const cudaGraphKernelNodeFieldParam: c_int = 2;
pub const cudaGraphKernelNodeFieldEnabled: c_int = 3;
pub const enum_cudaGraphKernelNodeField = c_uint;
const struct_unnamed_76 = extern struct {
    pValue: ?*const anyopaque = @import("std").mem.zeroes(?*const anyopaque),
    offset: usize = @import("std").mem.zeroes(usize),
    size: usize = @import("std").mem.zeroes(usize),
};
const union_unnamed_75 = extern union {
    gridDim: dim3,
    param: struct_unnamed_76,
    isEnabled: c_uint,
};
pub const struct_cudaGraphKernelNodeUpdate = extern struct {
    node: cudaGraphDeviceNode_t = @import("std").mem.zeroes(cudaGraphDeviceNode_t),
    field: enum_cudaGraphKernelNodeField = @import("std").mem.zeroes(enum_cudaGraphKernelNodeField),
    updateData: union_unnamed_75 = @import("std").mem.zeroes(union_unnamed_75),
};
pub const cudaEnableDefault: c_int = 0;
pub const cudaEnableLegacyStream: c_int = 1;
pub const cudaEnablePerThreadDefaultStream: c_int = 2;
pub const enum_cudaGetDriverEntryPointFlags = c_uint;
pub const cudaDriverEntryPointSuccess: c_int = 0;
pub const cudaDriverEntryPointSymbolNotFound: c_int = 1;
pub const cudaDriverEntryPointVersionNotSufficent: c_int = 2;
pub const enum_cudaDriverEntryPointQueryResult = c_uint;
pub const cudaGraphDebugDotFlagsVerbose: c_int = 1;
pub const cudaGraphDebugDotFlagsKernelNodeParams: c_int = 4;
pub const cudaGraphDebugDotFlagsMemcpyNodeParams: c_int = 8;
pub const cudaGraphDebugDotFlagsMemsetNodeParams: c_int = 16;
pub const cudaGraphDebugDotFlagsHostNodeParams: c_int = 32;
pub const cudaGraphDebugDotFlagsEventNodeParams: c_int = 64;
pub const cudaGraphDebugDotFlagsExtSemasSignalNodeParams: c_int = 128;
pub const cudaGraphDebugDotFlagsExtSemasWaitNodeParams: c_int = 256;
pub const cudaGraphDebugDotFlagsKernelNodeAttributes: c_int = 512;
pub const cudaGraphDebugDotFlagsHandles: c_int = 1024;
pub const cudaGraphDebugDotFlagsConditionalNodeParams: c_int = 32768;
pub const enum_cudaGraphDebugDotFlags = c_uint;
pub const cudaGraphInstantiateFlagAutoFreeOnLaunch: c_int = 1;
pub const cudaGraphInstantiateFlagUpload: c_int = 2;
pub const cudaGraphInstantiateFlagDeviceLaunch: c_int = 4;
pub const cudaGraphInstantiateFlagUseNodePriority: c_int = 8;
pub const enum_cudaGraphInstantiateFlags = c_uint;
pub const cudaLaunchMemSyncDomainDefault: c_int = 0;
pub const cudaLaunchMemSyncDomainRemote: c_int = 1;
pub const enum_cudaLaunchMemSyncDomain = c_uint;
pub const cudaLaunchMemSyncDomain = enum_cudaLaunchMemSyncDomain;
pub const struct_cudaLaunchMemSyncDomainMap_st = extern struct {
    default_: u8 = @import("std").mem.zeroes(u8),
    remote: u8 = @import("std").mem.zeroes(u8),
};
pub const cudaLaunchMemSyncDomainMap = struct_cudaLaunchMemSyncDomainMap_st;
pub const cudaLaunchAttributeIgnore: c_int = 0;
pub const cudaLaunchAttributeAccessPolicyWindow: c_int = 1;
pub const cudaLaunchAttributeCooperative: c_int = 2;
pub const cudaLaunchAttributeSynchronizationPolicy: c_int = 3;
pub const cudaLaunchAttributeClusterDimension: c_int = 4;
pub const cudaLaunchAttributeClusterSchedulingPolicyPreference: c_int = 5;
pub const cudaLaunchAttributeProgrammaticStreamSerialization: c_int = 6;
pub const cudaLaunchAttributeProgrammaticEvent: c_int = 7;
pub const cudaLaunchAttributePriority: c_int = 8;
pub const cudaLaunchAttributeMemSyncDomainMap: c_int = 9;
pub const cudaLaunchAttributeMemSyncDomain: c_int = 10;
pub const cudaLaunchAttributePreferredClusterDimension: c_int = 11;
pub const cudaLaunchAttributeLaunchCompletionEvent: c_int = 12;
pub const cudaLaunchAttributeDeviceUpdatableKernelNode: c_int = 13;
pub const cudaLaunchAttributePreferredSharedMemoryCarveout: c_int = 14;
pub const enum_cudaLaunchAttributeID = c_uint;
pub const cudaLaunchAttributeID = enum_cudaLaunchAttributeID;
const struct_unnamed_77 = extern struct {
    x: c_uint = @import("std").mem.zeroes(c_uint),
    y: c_uint = @import("std").mem.zeroes(c_uint),
    z: c_uint = @import("std").mem.zeroes(c_uint),
};
const struct_unnamed_78 = extern struct {
    event: cudaEvent_t = @import("std").mem.zeroes(cudaEvent_t),
    flags: c_int = @import("std").mem.zeroes(c_int),
    triggerAtBlockStart: c_int = @import("std").mem.zeroes(c_int),
};
const struct_unnamed_79 = extern struct {
    x: c_uint = @import("std").mem.zeroes(c_uint),
    y: c_uint = @import("std").mem.zeroes(c_uint),
    z: c_uint = @import("std").mem.zeroes(c_uint),
};
const struct_unnamed_80 = extern struct {
    event: cudaEvent_t = @import("std").mem.zeroes(cudaEvent_t),
    flags: c_int = @import("std").mem.zeroes(c_int),
};
const struct_unnamed_81 = extern struct {
    deviceUpdatable: c_int = @import("std").mem.zeroes(c_int),
    devNode: cudaGraphDeviceNode_t = @import("std").mem.zeroes(cudaGraphDeviceNode_t),
};
pub const union_cudaLaunchAttributeValue = extern union {
    pad: [64]u8,
    accessPolicyWindow: struct_cudaAccessPolicyWindow,
    cooperative: c_int,
    syncPolicy: enum_cudaSynchronizationPolicy,
    clusterDim: struct_unnamed_77,
    clusterSchedulingPolicyPreference: enum_cudaClusterSchedulingPolicy,
    programmaticStreamSerializationAllowed: c_int,
    programmaticEvent: struct_unnamed_78,
    priority: c_int,
    memSyncDomainMap: cudaLaunchMemSyncDomainMap,
    memSyncDomain: cudaLaunchMemSyncDomain,
    preferredClusterDim: struct_unnamed_79,
    launchCompletionEvent: struct_unnamed_80,
    deviceUpdatableKernelNode: struct_unnamed_81,
    sharedMemCarveout: c_uint,
};
pub const cudaLaunchAttributeValue = union_cudaLaunchAttributeValue;
pub const struct_cudaLaunchAttribute_st = extern struct {
    id: cudaLaunchAttributeID = @import("std").mem.zeroes(cudaLaunchAttributeID),
    pad: [4]u8 = @import("std").mem.zeroes([4]u8),
    val: cudaLaunchAttributeValue = @import("std").mem.zeroes(cudaLaunchAttributeValue),
};
pub const cudaLaunchAttribute = struct_cudaLaunchAttribute_st;
pub const struct_cudaLaunchConfig_st = extern struct {
    gridDim: dim3 = @import("std").mem.zeroes(dim3),
    blockDim: dim3 = @import("std").mem.zeroes(dim3),
    dynamicSmemBytes: usize = @import("std").mem.zeroes(usize),
    stream: cudaStream_t = @import("std").mem.zeroes(cudaStream_t),
    attrs: [*c]cudaLaunchAttribute = @import("std").mem.zeroes([*c]cudaLaunchAttribute),
    numAttrs: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const cudaLaunchConfig_t = struct_cudaLaunchConfig_st;
pub const cudaDeviceNumaConfigNone: c_int = 0;
pub const cudaDeviceNumaConfigNumaNode: c_int = 1;
pub const enum_cudaDeviceNumaConfig = c_uint;
pub const struct_cudaAsyncCallbackEntry = opaque {};
pub const cudaAsyncCallbackHandle_t = ?*struct_cudaAsyncCallbackEntry;
pub const cudaAsyncNotificationTypeOverBudget: c_int = 1;
pub const enum_cudaAsyncNotificationType_enum = c_uint;
pub const cudaAsyncNotificationType = enum_cudaAsyncNotificationType_enum;
const struct_unnamed_83 = extern struct {
    bytesOverBudget: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
};
const union_unnamed_82 = extern union {
    overBudget: struct_unnamed_83,
};
pub const struct_cudaAsyncNotificationInfo = extern struct {
    type: cudaAsyncNotificationType = @import("std").mem.zeroes(cudaAsyncNotificationType),
    info: union_unnamed_82 = @import("std").mem.zeroes(union_unnamed_82),
};
pub const cudaAsyncNotificationInfo_t = struct_cudaAsyncNotificationInfo;
pub const cudaAsyncCallback = ?*const fn ([*c]cudaAsyncNotificationInfo_t, ?*anyopaque, cudaAsyncCallbackHandle_t) callconv(.c) void;
pub const cudaBoundaryModeZero: c_int = 0;
pub const cudaBoundaryModeClamp: c_int = 1;
pub const cudaBoundaryModeTrap: c_int = 2;
pub const enum_cudaSurfaceBoundaryMode = c_uint;
pub const cudaFormatModeForced: c_int = 0;
pub const cudaFormatModeAuto: c_int = 1;
pub const enum_cudaSurfaceFormatMode = c_uint;
pub const cudaSurfaceObject_t = c_ulonglong;
pub const cudaAddressModeWrap: c_int = 0;
pub const cudaAddressModeClamp: c_int = 1;
pub const cudaAddressModeMirror: c_int = 2;
pub const cudaAddressModeBorder: c_int = 3;
pub const enum_cudaTextureAddressMode = c_uint;
pub const cudaFilterModePoint: c_int = 0;
pub const cudaFilterModeLinear: c_int = 1;
pub const enum_cudaTextureFilterMode = c_uint;
pub const cudaReadModeElementType: c_int = 0;
pub const cudaReadModeNormalizedFloat: c_int = 1;
pub const enum_cudaTextureReadMode = c_uint;
pub const struct_cudaTextureDesc = extern struct {
    addressMode: [3]enum_cudaTextureAddressMode = @import("std").mem.zeroes([3]enum_cudaTextureAddressMode),
    filterMode: enum_cudaTextureFilterMode = @import("std").mem.zeroes(enum_cudaTextureFilterMode),
    readMode: enum_cudaTextureReadMode = @import("std").mem.zeroes(enum_cudaTextureReadMode),
    sRGB: c_int = @import("std").mem.zeroes(c_int),
    borderColor: [4]f32 = @import("std").mem.zeroes([4]f32),
    normalizedCoords: c_int = @import("std").mem.zeroes(c_int),
    maxAnisotropy: c_uint = @import("std").mem.zeroes(c_uint),
    mipmapFilterMode: enum_cudaTextureFilterMode = @import("std").mem.zeroes(enum_cudaTextureFilterMode),
    mipmapLevelBias: f32 = @import("std").mem.zeroes(f32),
    minMipmapLevelClamp: f32 = @import("std").mem.zeroes(f32),
    maxMipmapLevelClamp: f32 = @import("std").mem.zeroes(f32),
    disableTrilinearOptimization: c_int = @import("std").mem.zeroes(c_int),
    seamlessCubemap: c_int = @import("std").mem.zeroes(c_int),
};
pub const cudaTextureObject_t = c_ulonglong;
pub const CUDA_R_16F: c_int = 2;
pub const CUDA_C_16F: c_int = 6;
pub const CUDA_R_16BF: c_int = 14;
pub const CUDA_C_16BF: c_int = 15;
pub const CUDA_R_32F: c_int = 0;
pub const CUDA_C_32F: c_int = 4;
pub const CUDA_R_64F: c_int = 1;
pub const CUDA_C_64F: c_int = 5;
pub const CUDA_R_4I: c_int = 16;
pub const CUDA_C_4I: c_int = 17;
pub const CUDA_R_4U: c_int = 18;
pub const CUDA_C_4U: c_int = 19;
pub const CUDA_R_8I: c_int = 3;
pub const CUDA_C_8I: c_int = 7;
pub const CUDA_R_8U: c_int = 8;
pub const CUDA_C_8U: c_int = 9;
pub const CUDA_R_16I: c_int = 20;
pub const CUDA_C_16I: c_int = 21;
pub const CUDA_R_16U: c_int = 22;
pub const CUDA_C_16U: c_int = 23;
pub const CUDA_R_32I: c_int = 10;
pub const CUDA_C_32I: c_int = 11;
pub const CUDA_R_32U: c_int = 12;
pub const CUDA_C_32U: c_int = 13;
pub const CUDA_R_64I: c_int = 24;
pub const CUDA_C_64I: c_int = 25;
pub const CUDA_R_64U: c_int = 26;
pub const CUDA_C_64U: c_int = 27;
pub const CUDA_R_8F_E4M3: c_int = 28;
pub const CUDA_R_8F_UE4M3: c_int = 28;
pub const CUDA_R_8F_E5M2: c_int = 29;
pub const CUDA_R_8F_UE8M0: c_int = 30;
pub const CUDA_R_6F_E2M3: c_int = 31;
pub const CUDA_R_6F_E3M2: c_int = 32;
pub const CUDA_R_4F_E2M1: c_int = 33;
pub const enum_cudaDataType_t = c_uint;
pub const cudaDataType = enum_cudaDataType_t;
pub const MAJOR_VERSION: c_int = 0;
pub const MINOR_VERSION: c_int = 1;
pub const PATCH_LEVEL: c_int = 2;
pub const enum_libraryPropertyType_t = c_uint;
pub const libraryPropertyType = enum_libraryPropertyType_t;
pub const cudaDataType_t = enum_cudaDataType_t;
pub const libraryPropertyType_t = enum_libraryPropertyType_t;
pub extern fn cudaDeviceReset() cudaError_t;
pub extern fn cudaDeviceSynchronize() cudaError_t;
pub extern fn cudaDeviceSetLimit(limit: enum_cudaLimit, value: usize) cudaError_t;
pub extern fn cudaDeviceGetLimit(pValue: [*c]usize, limit: enum_cudaLimit) cudaError_t;
pub extern fn cudaDeviceGetTexture1DLinearMaxWidth(maxWidthInElements: [*c]usize, fmtDesc: [*c]const struct_cudaChannelFormatDesc, device: c_int) cudaError_t;
pub extern fn cudaDeviceGetCacheConfig(pCacheConfig: [*c]enum_cudaFuncCache) cudaError_t;
pub extern fn cudaDeviceGetStreamPriorityRange(leastPriority: [*c]c_int, greatestPriority: [*c]c_int) cudaError_t;
pub extern fn cudaDeviceSetCacheConfig(cacheConfig: enum_cudaFuncCache) cudaError_t;
pub extern fn cudaDeviceGetByPCIBusId(device: [*c]c_int, pciBusId: [*c]const u8) cudaError_t;
pub extern fn cudaDeviceGetPCIBusId(pciBusId: [*c]u8, len: c_int, device: c_int) cudaError_t;
pub extern fn cudaIpcGetEventHandle(handle: [*c]cudaIpcEventHandle_t, event: cudaEvent_t) cudaError_t;
pub extern fn cudaIpcOpenEventHandle(event: [*c]cudaEvent_t, handle: cudaIpcEventHandle_t) cudaError_t;
pub extern fn cudaIpcGetMemHandle(handle: [*c]cudaIpcMemHandle_t, devPtr: ?*anyopaque) cudaError_t;
pub extern fn cudaIpcOpenMemHandle(devPtr: [*c]?*anyopaque, handle: cudaIpcMemHandle_t, flags: c_uint) cudaError_t;
pub extern fn cudaIpcCloseMemHandle(devPtr: ?*anyopaque) cudaError_t;
pub extern fn cudaDeviceFlushGPUDirectRDMAWrites(target: enum_cudaFlushGPUDirectRDMAWritesTarget, scope: enum_cudaFlushGPUDirectRDMAWritesScope) cudaError_t;
pub extern fn cudaDeviceRegisterAsyncNotification(device: c_int, callbackFunc: cudaAsyncCallback, userData: ?*anyopaque, callback: [*c]cudaAsyncCallbackHandle_t) cudaError_t;
pub extern fn cudaDeviceUnregisterAsyncNotification(device: c_int, callback: cudaAsyncCallbackHandle_t) cudaError_t;
pub extern fn cudaDeviceGetSharedMemConfig(pConfig: [*c]enum_cudaSharedMemConfig) cudaError_t;
pub extern fn cudaDeviceSetSharedMemConfig(config: enum_cudaSharedMemConfig) cudaError_t;
pub extern fn cudaThreadExit() cudaError_t;
pub extern fn cudaThreadSynchronize() cudaError_t;
pub extern fn cudaThreadSetLimit(limit: enum_cudaLimit, value: usize) cudaError_t;
pub extern fn cudaThreadGetLimit(pValue: [*c]usize, limit: enum_cudaLimit) cudaError_t;
pub extern fn cudaThreadGetCacheConfig(pCacheConfig: [*c]enum_cudaFuncCache) cudaError_t;
pub extern fn cudaThreadSetCacheConfig(cacheConfig: enum_cudaFuncCache) cudaError_t;
pub extern fn cudaGetLastError() cudaError_t;
pub extern fn cudaPeekAtLastError() cudaError_t;
pub extern fn cudaGetErrorName(@"error": cudaError_t) [*c]const u8;
pub extern fn cudaGetErrorString(@"error": cudaError_t) [*c]const u8;
pub extern fn cudaGetDeviceCount(count: [*c]c_int) cudaError_t;
pub extern fn cudaGetDeviceProperties_v2(prop: [*c]struct_cudaDeviceProp, device: c_int) cudaError_t;
pub extern fn cudaDeviceGetAttribute(value: [*c]c_int, attr: enum_cudaDeviceAttr, device: c_int) cudaError_t;
pub extern fn cudaDeviceGetDefaultMemPool(memPool: [*c]cudaMemPool_t, device: c_int) cudaError_t;
pub extern fn cudaDeviceSetMemPool(device: c_int, memPool: cudaMemPool_t) cudaError_t;
pub extern fn cudaDeviceGetMemPool(memPool: [*c]cudaMemPool_t, device: c_int) cudaError_t;
pub extern fn cudaDeviceGetNvSciSyncAttributes(nvSciSyncAttrList: ?*anyopaque, device: c_int, flags: c_int) cudaError_t;
pub extern fn cudaDeviceGetP2PAttribute(value: [*c]c_int, attr: enum_cudaDeviceP2PAttr, srcDevice: c_int, dstDevice: c_int) cudaError_t;
pub extern fn cudaChooseDevice(device: [*c]c_int, prop: [*c]const struct_cudaDeviceProp) cudaError_t;
pub extern fn cudaInitDevice(device: c_int, deviceFlags: c_uint, flags: c_uint) cudaError_t;
pub extern fn cudaSetDevice(device: c_int) cudaError_t;
pub extern fn cudaGetDevice(device: [*c]c_int) cudaError_t;
pub extern fn cudaSetValidDevices(device_arr: [*c]c_int, len: c_int) cudaError_t;
pub extern fn cudaSetDeviceFlags(flags: c_uint) cudaError_t;
pub extern fn cudaGetDeviceFlags(flags: [*c]c_uint) cudaError_t;
pub extern fn cudaStreamCreate(pStream: [*c]cudaStream_t) cudaError_t;
pub extern fn cudaStreamCreateWithFlags(pStream: [*c]cudaStream_t, flags: c_uint) cudaError_t;
pub extern fn cudaStreamCreateWithPriority(pStream: [*c]cudaStream_t, flags: c_uint, priority: c_int) cudaError_t;
pub extern fn cudaStreamGetPriority(hStream: cudaStream_t, priority: [*c]c_int) cudaError_t;
pub extern fn cudaStreamGetFlags(hStream: cudaStream_t, flags: [*c]c_uint) cudaError_t;
pub extern fn cudaStreamGetId(hStream: cudaStream_t, streamId: [*c]c_ulonglong) cudaError_t;
pub extern fn cudaStreamGetDevice(hStream: cudaStream_t, device: [*c]c_int) cudaError_t;
pub extern fn cudaCtxResetPersistingL2Cache() cudaError_t;
pub extern fn cudaStreamCopyAttributes(dst: cudaStream_t, src: cudaStream_t) cudaError_t;
pub extern fn cudaStreamGetAttribute(hStream: cudaStream_t, attr: cudaLaunchAttributeID, value_out: [*c]cudaLaunchAttributeValue) cudaError_t;
pub extern fn cudaStreamSetAttribute(hStream: cudaStream_t, attr: cudaLaunchAttributeID, value: [*c]const cudaLaunchAttributeValue) cudaError_t;
pub extern fn cudaStreamDestroy(stream: cudaStream_t) cudaError_t;
pub extern fn cudaStreamWaitEvent(stream: cudaStream_t, event: cudaEvent_t, flags: c_uint) cudaError_t;
pub const cudaStreamCallback_t = ?*const fn (cudaStream_t, cudaError_t, ?*anyopaque) callconv(.c) void;
pub extern fn cudaStreamAddCallback(stream: cudaStream_t, callback: cudaStreamCallback_t, userData: ?*anyopaque, flags: c_uint) cudaError_t;
pub extern fn cudaStreamSynchronize(stream: cudaStream_t) cudaError_t;
pub extern fn cudaStreamQuery(stream: cudaStream_t) cudaError_t;
pub extern fn cudaStreamAttachMemAsync(stream: cudaStream_t, devPtr: ?*anyopaque, length: usize, flags: c_uint) cudaError_t;
pub extern fn cudaStreamBeginCapture(stream: cudaStream_t, mode: enum_cudaStreamCaptureMode) cudaError_t;
pub extern fn cudaStreamBeginCaptureToGraph(stream: cudaStream_t, graph: cudaGraph_t, dependencies: [*c]const cudaGraphNode_t, dependencyData: [*c]const cudaGraphEdgeData, numDependencies: usize, mode: enum_cudaStreamCaptureMode) cudaError_t;
pub extern fn cudaThreadExchangeStreamCaptureMode(mode: [*c]enum_cudaStreamCaptureMode) cudaError_t;
pub extern fn cudaStreamEndCapture(stream: cudaStream_t, pGraph: [*c]cudaGraph_t) cudaError_t;
pub extern fn cudaStreamIsCapturing(stream: cudaStream_t, pCaptureStatus: [*c]enum_cudaStreamCaptureStatus) cudaError_t;
pub extern fn cudaStreamGetCaptureInfo_v2(stream: cudaStream_t, captureStatus_out: [*c]enum_cudaStreamCaptureStatus, id_out: [*c]c_ulonglong, graph_out: [*c]cudaGraph_t, dependencies_out: [*c][*c]const cudaGraphNode_t, numDependencies_out: [*c]usize) cudaError_t;
pub extern fn cudaStreamGetCaptureInfo_v3(stream: cudaStream_t, captureStatus_out: [*c]enum_cudaStreamCaptureStatus, id_out: [*c]c_ulonglong, graph_out: [*c]cudaGraph_t, dependencies_out: [*c][*c]const cudaGraphNode_t, edgeData_out: [*c][*c]const cudaGraphEdgeData, numDependencies_out: [*c]usize) cudaError_t;
pub extern fn cudaStreamUpdateCaptureDependencies(stream: cudaStream_t, dependencies: [*c]cudaGraphNode_t, numDependencies: usize, flags: c_uint) cudaError_t;
pub extern fn cudaStreamUpdateCaptureDependencies_v2(stream: cudaStream_t, dependencies: [*c]cudaGraphNode_t, dependencyData: [*c]const cudaGraphEdgeData, numDependencies: usize, flags: c_uint) cudaError_t;
pub extern fn cudaEventCreate(event: [*c]cudaEvent_t) cudaError_t;
pub extern fn cudaEventCreateWithFlags(event: [*c]cudaEvent_t, flags: c_uint) cudaError_t;
pub extern fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) cudaError_t;
pub extern fn cudaEventRecordWithFlags(event: cudaEvent_t, stream: cudaStream_t, flags: c_uint) cudaError_t;
pub extern fn cudaEventQuery(event: cudaEvent_t) cudaError_t;
pub extern fn cudaEventSynchronize(event: cudaEvent_t) cudaError_t;
pub extern fn cudaEventDestroy(event: cudaEvent_t) cudaError_t;
pub extern fn cudaEventElapsedTime(ms: [*c]f32, start: cudaEvent_t, end: cudaEvent_t) cudaError_t;
pub extern fn cudaEventElapsedTime_v2(ms: [*c]f32, start: cudaEvent_t, end: cudaEvent_t) cudaError_t;
pub extern fn cudaImportExternalMemory(extMem_out: [*c]cudaExternalMemory_t, memHandleDesc: [*c]const struct_cudaExternalMemoryHandleDesc) cudaError_t;
pub extern fn cudaExternalMemoryGetMappedBuffer(devPtr: [*c]?*anyopaque, extMem: cudaExternalMemory_t, bufferDesc: [*c]const struct_cudaExternalMemoryBufferDesc) cudaError_t;
pub extern fn cudaExternalMemoryGetMappedMipmappedArray(mipmap: [*c]cudaMipmappedArray_t, extMem: cudaExternalMemory_t, mipmapDesc: [*c]const struct_cudaExternalMemoryMipmappedArrayDesc) cudaError_t;
pub extern fn cudaDestroyExternalMemory(extMem: cudaExternalMemory_t) cudaError_t;
pub extern fn cudaImportExternalSemaphore(extSem_out: [*c]cudaExternalSemaphore_t, semHandleDesc: [*c]const struct_cudaExternalSemaphoreHandleDesc) cudaError_t;
pub extern fn cudaSignalExternalSemaphoresAsync_v2(extSemArray: [*c]const cudaExternalSemaphore_t, paramsArray: [*c]const struct_cudaExternalSemaphoreSignalParams, numExtSems: c_uint, stream: cudaStream_t) cudaError_t;
pub extern fn cudaWaitExternalSemaphoresAsync_v2(extSemArray: [*c]const cudaExternalSemaphore_t, paramsArray: [*c]const struct_cudaExternalSemaphoreWaitParams, numExtSems: c_uint, stream: cudaStream_t) cudaError_t;
pub extern fn cudaDestroyExternalSemaphore(extSem: cudaExternalSemaphore_t) cudaError_t;
pub extern fn cudaLaunchKernel(func: ?*const anyopaque, gridDim: dim3, blockDim: dim3, args: [*c]?*anyopaque, sharedMem: usize, stream: cudaStream_t) cudaError_t;
pub extern fn cudaLaunchKernelExC(config: [*c]const cudaLaunchConfig_t, func: ?*const anyopaque, args: [*c]?*anyopaque) cudaError_t;
pub extern fn cudaLaunchCooperativeKernel(func: ?*const anyopaque, gridDim: dim3, blockDim: dim3, args: [*c]?*anyopaque, sharedMem: usize, stream: cudaStream_t) cudaError_t;
pub extern fn cudaLaunchCooperativeKernelMultiDevice(launchParamsList: [*c]struct_cudaLaunchParams, numDevices: c_uint, flags: c_uint) cudaError_t;
pub extern fn cudaFuncSetCacheConfig(func: ?*const anyopaque, cacheConfig: enum_cudaFuncCache) cudaError_t;
pub extern fn cudaFuncGetAttributes(attr: [*c]struct_cudaFuncAttributes, func: ?*const anyopaque) cudaError_t;
pub extern fn cudaFuncSetAttribute(func: ?*const anyopaque, attr: enum_cudaFuncAttribute, value: c_int) cudaError_t;
pub extern fn cudaFuncGetName(name: [*c][*c]const u8, func: ?*const anyopaque) cudaError_t;
pub extern fn cudaFuncGetParamInfo(func: ?*const anyopaque, paramIndex: usize, paramOffset: [*c]usize, paramSize: [*c]usize) cudaError_t;
pub extern fn cudaSetDoubleForDevice(d: [*c]f64) cudaError_t;
pub extern fn cudaSetDoubleForHost(d: [*c]f64) cudaError_t;
pub extern fn cudaLaunchHostFunc(stream: cudaStream_t, @"fn": cudaHostFn_t, userData: ?*anyopaque) cudaError_t;
pub extern fn cudaFuncSetSharedMemConfig(func: ?*const anyopaque, config: enum_cudaSharedMemConfig) cudaError_t;
pub extern fn cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks: [*c]c_int, func: ?*const anyopaque, blockSize: c_int, dynamicSMemSize: usize) cudaError_t;
pub extern fn cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize: [*c]usize, func: ?*const anyopaque, numBlocks: c_int, blockSize: c_int) cudaError_t;
pub extern fn cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks: [*c]c_int, func: ?*const anyopaque, blockSize: c_int, dynamicSMemSize: usize, flags: c_uint) cudaError_t;
pub extern fn cudaOccupancyMaxPotentialClusterSize(clusterSize: [*c]c_int, func: ?*const anyopaque, launchConfig: [*c]const cudaLaunchConfig_t) cudaError_t;
pub extern fn cudaOccupancyMaxActiveClusters(numClusters: [*c]c_int, func: ?*const anyopaque, launchConfig: [*c]const cudaLaunchConfig_t) cudaError_t;
pub extern fn cudaMallocManaged(devPtr: [*c]?*anyopaque, size: usize, flags: c_uint) cudaError_t;
pub extern fn cudaMalloc(devPtr: [*c]?*anyopaque, size: usize) cudaError_t;
pub extern fn cudaMallocHost(ptr: [*c]?*anyopaque, size: usize) cudaError_t;
pub extern fn cudaMallocPitch(devPtr: [*c]?*anyopaque, pitch: [*c]usize, width: usize, height: usize) cudaError_t;
pub extern fn cudaMallocArray(array: [*c]cudaArray_t, desc: [*c]const struct_cudaChannelFormatDesc, width: usize, height: usize, flags: c_uint) cudaError_t;
pub extern fn cudaFree(devPtr: ?*anyopaque) cudaError_t;
pub extern fn cudaFreeHost(ptr: ?*anyopaque) cudaError_t;
pub extern fn cudaFreeArray(array: cudaArray_t) cudaError_t;
pub extern fn cudaFreeMipmappedArray(mipmappedArray: cudaMipmappedArray_t) cudaError_t;
pub extern fn cudaHostAlloc(pHost: [*c]?*anyopaque, size: usize, flags: c_uint) cudaError_t;
pub extern fn cudaHostRegister(ptr: ?*anyopaque, size: usize, flags: c_uint) cudaError_t;
pub extern fn cudaHostUnregister(ptr: ?*anyopaque) cudaError_t;
pub extern fn cudaHostGetDevicePointer(pDevice: [*c]?*anyopaque, pHost: ?*anyopaque, flags: c_uint) cudaError_t;
pub extern fn cudaHostGetFlags(pFlags: [*c]c_uint, pHost: ?*anyopaque) cudaError_t;
pub extern fn cudaMalloc3D(pitchedDevPtr: [*c]struct_cudaPitchedPtr, extent: struct_cudaExtent) cudaError_t;
pub extern fn cudaMalloc3DArray(array: [*c]cudaArray_t, desc: [*c]const struct_cudaChannelFormatDesc, extent: struct_cudaExtent, flags: c_uint) cudaError_t;
pub extern fn cudaMallocMipmappedArray(mipmappedArray: [*c]cudaMipmappedArray_t, desc: [*c]const struct_cudaChannelFormatDesc, extent: struct_cudaExtent, numLevels: c_uint, flags: c_uint) cudaError_t;
pub extern fn cudaGetMipmappedArrayLevel(levelArray: [*c]cudaArray_t, mipmappedArray: cudaMipmappedArray_const_t, level: c_uint) cudaError_t;
pub extern fn cudaMemcpy3D(p: [*c]const struct_cudaMemcpy3DParms) cudaError_t;
pub extern fn cudaMemcpy3DPeer(p: [*c]const struct_cudaMemcpy3DPeerParms) cudaError_t;
pub extern fn cudaMemcpy3DAsync(p: [*c]const struct_cudaMemcpy3DParms, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpy3DPeerAsync(p: [*c]const struct_cudaMemcpy3DPeerParms, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemGetInfo(free: [*c]usize, total: [*c]usize) cudaError_t;
pub extern fn cudaArrayGetInfo(desc: [*c]struct_cudaChannelFormatDesc, extent: [*c]struct_cudaExtent, flags: [*c]c_uint, array: cudaArray_t) cudaError_t;
pub extern fn cudaArrayGetPlane(pPlaneArray: [*c]cudaArray_t, hArray: cudaArray_t, planeIdx: c_uint) cudaError_t;
pub extern fn cudaArrayGetMemoryRequirements(memoryRequirements: [*c]struct_cudaArrayMemoryRequirements, array: cudaArray_t, device: c_int) cudaError_t;
pub extern fn cudaMipmappedArrayGetMemoryRequirements(memoryRequirements: [*c]struct_cudaArrayMemoryRequirements, mipmap: cudaMipmappedArray_t, device: c_int) cudaError_t;
pub extern fn cudaArrayGetSparseProperties(sparseProperties: [*c]struct_cudaArraySparseProperties, array: cudaArray_t) cudaError_t;
pub extern fn cudaMipmappedArrayGetSparseProperties(sparseProperties: [*c]struct_cudaArraySparseProperties, mipmap: cudaMipmappedArray_t) cudaError_t;
pub extern fn cudaMemcpy(dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaMemcpyPeer(dst: ?*anyopaque, dstDevice: c_int, src: ?*const anyopaque, srcDevice: c_int, count: usize) cudaError_t;
pub extern fn cudaMemcpy2D(dst: ?*anyopaque, dpitch: usize, src: ?*const anyopaque, spitch: usize, width: usize, height: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaMemcpy2DToArray(dst: cudaArray_t, wOffset: usize, hOffset: usize, src: ?*const anyopaque, spitch: usize, width: usize, height: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaMemcpy2DFromArray(dst: ?*anyopaque, dpitch: usize, src: cudaArray_const_t, wOffset: usize, hOffset: usize, width: usize, height: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaMemcpy2DArrayToArray(dst: cudaArray_t, wOffsetDst: usize, hOffsetDst: usize, src: cudaArray_const_t, wOffsetSrc: usize, hOffsetSrc: usize, width: usize, height: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaMemcpyToSymbol(symbol: ?*const anyopaque, src: ?*const anyopaque, count: usize, offset: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaMemcpyFromSymbol(dst: ?*anyopaque, symbol: ?*const anyopaque, count: usize, offset: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaMemcpyAsync(dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: enum_cudaMemcpyKind, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpyPeerAsync(dst: ?*anyopaque, dstDevice: c_int, src: ?*const anyopaque, srcDevice: c_int, count: usize, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpyBatchAsync(dsts: [*c]?*anyopaque, srcs: [*c]?*anyopaque, sizes: [*c]usize, count: usize, attrs: [*c]struct_cudaMemcpyAttributes, attrsIdxs: [*c]usize, numAttrs: usize, failIdx: [*c]usize, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpy3DBatchAsync(numOps: usize, opList: [*c]struct_cudaMemcpy3DBatchOp, failIdx: [*c]usize, flags: c_ulonglong, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpy2DAsync(dst: ?*anyopaque, dpitch: usize, src: ?*const anyopaque, spitch: usize, width: usize, height: usize, kind: enum_cudaMemcpyKind, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpy2DToArrayAsync(dst: cudaArray_t, wOffset: usize, hOffset: usize, src: ?*const anyopaque, spitch: usize, width: usize, height: usize, kind: enum_cudaMemcpyKind, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpy2DFromArrayAsync(dst: ?*anyopaque, dpitch: usize, src: cudaArray_const_t, wOffset: usize, hOffset: usize, width: usize, height: usize, kind: enum_cudaMemcpyKind, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpyToSymbolAsync(symbol: ?*const anyopaque, src: ?*const anyopaque, count: usize, offset: usize, kind: enum_cudaMemcpyKind, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpyFromSymbolAsync(dst: ?*anyopaque, symbol: ?*const anyopaque, count: usize, offset: usize, kind: enum_cudaMemcpyKind, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemset(devPtr: ?*anyopaque, value: c_int, count: usize) cudaError_t;
pub extern fn cudaMemset2D(devPtr: ?*anyopaque, pitch: usize, value: c_int, width: usize, height: usize) cudaError_t;
pub extern fn cudaMemset3D(pitchedDevPtr: struct_cudaPitchedPtr, value: c_int, extent: struct_cudaExtent) cudaError_t;
pub extern fn cudaMemsetAsync(devPtr: ?*anyopaque, value: c_int, count: usize, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemset2DAsync(devPtr: ?*anyopaque, pitch: usize, value: c_int, width: usize, height: usize, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemset3DAsync(pitchedDevPtr: struct_cudaPitchedPtr, value: c_int, extent: struct_cudaExtent, stream: cudaStream_t) cudaError_t;
pub extern fn cudaGetSymbolAddress(devPtr: [*c]?*anyopaque, symbol: ?*const anyopaque) cudaError_t;
pub extern fn cudaGetSymbolSize(size: [*c]usize, symbol: ?*const anyopaque) cudaError_t;
pub extern fn cudaMemPrefetchAsync(devPtr: ?*const anyopaque, count: usize, dstDevice: c_int, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemPrefetchAsync_v2(devPtr: ?*const anyopaque, count: usize, location: struct_cudaMemLocation, flags: c_uint, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemAdvise(devPtr: ?*const anyopaque, count: usize, advice: enum_cudaMemoryAdvise, device: c_int) cudaError_t;
pub extern fn cudaMemAdvise_v2(devPtr: ?*const anyopaque, count: usize, advice: enum_cudaMemoryAdvise, location: struct_cudaMemLocation) cudaError_t;
pub extern fn cudaMemRangeGetAttribute(data: ?*anyopaque, dataSize: usize, attribute: enum_cudaMemRangeAttribute, devPtr: ?*const anyopaque, count: usize) cudaError_t;
pub extern fn cudaMemRangeGetAttributes(data: [*c]?*anyopaque, dataSizes: [*c]usize, attributes: [*c]enum_cudaMemRangeAttribute, numAttributes: usize, devPtr: ?*const anyopaque, count: usize) cudaError_t;
pub extern fn cudaMemcpyToArray(dst: cudaArray_t, wOffset: usize, hOffset: usize, src: ?*const anyopaque, count: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaMemcpyFromArray(dst: ?*anyopaque, src: cudaArray_const_t, wOffset: usize, hOffset: usize, count: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaMemcpyArrayToArray(dst: cudaArray_t, wOffsetDst: usize, hOffsetDst: usize, src: cudaArray_const_t, wOffsetSrc: usize, hOffsetSrc: usize, count: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaMemcpyToArrayAsync(dst: cudaArray_t, wOffset: usize, hOffset: usize, src: ?*const anyopaque, count: usize, kind: enum_cudaMemcpyKind, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemcpyFromArrayAsync(dst: ?*anyopaque, src: cudaArray_const_t, wOffset: usize, hOffset: usize, count: usize, kind: enum_cudaMemcpyKind, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMallocAsync(devPtr: [*c]?*anyopaque, size: usize, hStream: cudaStream_t) cudaError_t;
pub extern fn cudaFreeAsync(devPtr: ?*anyopaque, hStream: cudaStream_t) cudaError_t;
pub extern fn cudaMemPoolTrimTo(memPool: cudaMemPool_t, minBytesToKeep: usize) cudaError_t;
pub extern fn cudaMemPoolSetAttribute(memPool: cudaMemPool_t, attr: enum_cudaMemPoolAttr, value: ?*anyopaque) cudaError_t;
pub extern fn cudaMemPoolGetAttribute(memPool: cudaMemPool_t, attr: enum_cudaMemPoolAttr, value: ?*anyopaque) cudaError_t;
pub extern fn cudaMemPoolSetAccess(memPool: cudaMemPool_t, descList: [*c]const struct_cudaMemAccessDesc, count: usize) cudaError_t;
pub extern fn cudaMemPoolGetAccess(flags: [*c]enum_cudaMemAccessFlags, memPool: cudaMemPool_t, location: [*c]struct_cudaMemLocation) cudaError_t;
pub extern fn cudaMemPoolCreate(memPool: [*c]cudaMemPool_t, poolProps: [*c]const struct_cudaMemPoolProps) cudaError_t;
pub extern fn cudaMemPoolDestroy(memPool: cudaMemPool_t) cudaError_t;
pub extern fn cudaMallocFromPoolAsync(ptr: [*c]?*anyopaque, size: usize, memPool: cudaMemPool_t, stream: cudaStream_t) cudaError_t;
pub extern fn cudaMemPoolExportToShareableHandle(shareableHandle: ?*anyopaque, memPool: cudaMemPool_t, handleType: enum_cudaMemAllocationHandleType, flags: c_uint) cudaError_t;
pub extern fn cudaMemPoolImportFromShareableHandle(memPool: [*c]cudaMemPool_t, shareableHandle: ?*anyopaque, handleType: enum_cudaMemAllocationHandleType, flags: c_uint) cudaError_t;
pub extern fn cudaMemPoolExportPointer(exportData: [*c]struct_cudaMemPoolPtrExportData, ptr: ?*anyopaque) cudaError_t;
pub extern fn cudaMemPoolImportPointer(ptr: [*c]?*anyopaque, memPool: cudaMemPool_t, exportData: [*c]struct_cudaMemPoolPtrExportData) cudaError_t;
pub extern fn cudaPointerGetAttributes(attributes: [*c]struct_cudaPointerAttributes, ptr: ?*const anyopaque) cudaError_t;
pub extern fn cudaDeviceCanAccessPeer(canAccessPeer: [*c]c_int, device: c_int, peerDevice: c_int) cudaError_t;
pub extern fn cudaDeviceEnablePeerAccess(peerDevice: c_int, flags: c_uint) cudaError_t;
pub extern fn cudaDeviceDisablePeerAccess(peerDevice: c_int) cudaError_t;
pub extern fn cudaGraphicsUnregisterResource(resource: cudaGraphicsResource_t) cudaError_t;
pub extern fn cudaGraphicsResourceSetMapFlags(resource: cudaGraphicsResource_t, flags: c_uint) cudaError_t;
pub extern fn cudaGraphicsMapResources(count: c_int, resources: [*c]cudaGraphicsResource_t, stream: cudaStream_t) cudaError_t;
pub extern fn cudaGraphicsUnmapResources(count: c_int, resources: [*c]cudaGraphicsResource_t, stream: cudaStream_t) cudaError_t;
pub extern fn cudaGraphicsResourceGetMappedPointer(devPtr: [*c]?*anyopaque, size: [*c]usize, resource: cudaGraphicsResource_t) cudaError_t;
pub extern fn cudaGraphicsSubResourceGetMappedArray(array: [*c]cudaArray_t, resource: cudaGraphicsResource_t, arrayIndex: c_uint, mipLevel: c_uint) cudaError_t;
pub extern fn cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray: [*c]cudaMipmappedArray_t, resource: cudaGraphicsResource_t) cudaError_t;
pub extern fn cudaGetChannelDesc(desc: [*c]struct_cudaChannelFormatDesc, array: cudaArray_const_t) cudaError_t;
pub extern fn cudaCreateChannelDesc(x: c_int, y: c_int, z: c_int, w: c_int, f: enum_cudaChannelFormatKind) struct_cudaChannelFormatDesc;
pub extern fn cudaCreateTextureObject(pTexObject: [*c]cudaTextureObject_t, pResDesc: [*c]const struct_cudaResourceDesc, pTexDesc: [*c]const struct_cudaTextureDesc, pResViewDesc: [*c]const struct_cudaResourceViewDesc) cudaError_t;
pub extern fn cudaDestroyTextureObject(texObject: cudaTextureObject_t) cudaError_t;
pub extern fn cudaGetTextureObjectResourceDesc(pResDesc: [*c]struct_cudaResourceDesc, texObject: cudaTextureObject_t) cudaError_t;
pub extern fn cudaGetTextureObjectTextureDesc(pTexDesc: [*c]struct_cudaTextureDesc, texObject: cudaTextureObject_t) cudaError_t;
pub extern fn cudaGetTextureObjectResourceViewDesc(pResViewDesc: [*c]struct_cudaResourceViewDesc, texObject: cudaTextureObject_t) cudaError_t;
pub extern fn cudaCreateSurfaceObject(pSurfObject: [*c]cudaSurfaceObject_t, pResDesc: [*c]const struct_cudaResourceDesc) cudaError_t;
pub extern fn cudaDestroySurfaceObject(surfObject: cudaSurfaceObject_t) cudaError_t;
pub extern fn cudaGetSurfaceObjectResourceDesc(pResDesc: [*c]struct_cudaResourceDesc, surfObject: cudaSurfaceObject_t) cudaError_t;
pub extern fn cudaDriverGetVersion(driverVersion: [*c]c_int) cudaError_t;
pub extern fn cudaRuntimeGetVersion(runtimeVersion: [*c]c_int) cudaError_t;
pub extern fn cudaGraphCreate(pGraph: [*c]cudaGraph_t, flags: c_uint) cudaError_t;
pub extern fn cudaGraphAddKernelNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, pNodeParams: [*c]const struct_cudaKernelNodeParams) cudaError_t;
pub extern fn cudaGraphKernelNodeGetParams(node: cudaGraphNode_t, pNodeParams: [*c]struct_cudaKernelNodeParams) cudaError_t;
pub extern fn cudaGraphKernelNodeSetParams(node: cudaGraphNode_t, pNodeParams: [*c]const struct_cudaKernelNodeParams) cudaError_t;
pub extern fn cudaGraphKernelNodeCopyAttributes(hSrc: cudaGraphNode_t, hDst: cudaGraphNode_t) cudaError_t;
pub extern fn cudaGraphKernelNodeGetAttribute(hNode: cudaGraphNode_t, attr: cudaLaunchAttributeID, value_out: [*c]cudaLaunchAttributeValue) cudaError_t;
pub extern fn cudaGraphKernelNodeSetAttribute(hNode: cudaGraphNode_t, attr: cudaLaunchAttributeID, value: [*c]const cudaLaunchAttributeValue) cudaError_t;
pub extern fn cudaGraphAddMemcpyNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, pCopyParams: [*c]const struct_cudaMemcpy3DParms) cudaError_t;
pub extern fn cudaGraphAddMemcpyNodeToSymbol(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, symbol: ?*const anyopaque, src: ?*const anyopaque, count: usize, offset: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaGraphAddMemcpyNodeFromSymbol(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, dst: ?*anyopaque, symbol: ?*const anyopaque, count: usize, offset: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaGraphAddMemcpyNode1D(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaGraphMemcpyNodeGetParams(node: cudaGraphNode_t, pNodeParams: [*c]struct_cudaMemcpy3DParms) cudaError_t;
pub extern fn cudaGraphMemcpyNodeSetParams(node: cudaGraphNode_t, pNodeParams: [*c]const struct_cudaMemcpy3DParms) cudaError_t;
pub extern fn cudaGraphMemcpyNodeSetParamsToSymbol(node: cudaGraphNode_t, symbol: ?*const anyopaque, src: ?*const anyopaque, count: usize, offset: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaGraphMemcpyNodeSetParamsFromSymbol(node: cudaGraphNode_t, dst: ?*anyopaque, symbol: ?*const anyopaque, count: usize, offset: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaGraphMemcpyNodeSetParams1D(node: cudaGraphNode_t, dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaGraphAddMemsetNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, pMemsetParams: [*c]const struct_cudaMemsetParams) cudaError_t;
pub extern fn cudaGraphMemsetNodeGetParams(node: cudaGraphNode_t, pNodeParams: [*c]struct_cudaMemsetParams) cudaError_t;
pub extern fn cudaGraphMemsetNodeSetParams(node: cudaGraphNode_t, pNodeParams: [*c]const struct_cudaMemsetParams) cudaError_t;
pub extern fn cudaGraphAddHostNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, pNodeParams: [*c]const struct_cudaHostNodeParams) cudaError_t;
pub extern fn cudaGraphHostNodeGetParams(node: cudaGraphNode_t, pNodeParams: [*c]struct_cudaHostNodeParams) cudaError_t;
pub extern fn cudaGraphHostNodeSetParams(node: cudaGraphNode_t, pNodeParams: [*c]const struct_cudaHostNodeParams) cudaError_t;
pub extern fn cudaGraphAddChildGraphNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, childGraph: cudaGraph_t) cudaError_t;
pub extern fn cudaGraphChildGraphNodeGetGraph(node: cudaGraphNode_t, pGraph: [*c]cudaGraph_t) cudaError_t;
pub extern fn cudaGraphAddEmptyNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize) cudaError_t;
pub extern fn cudaGraphAddEventRecordNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, event: cudaEvent_t) cudaError_t;
pub extern fn cudaGraphEventRecordNodeGetEvent(node: cudaGraphNode_t, event_out: [*c]cudaEvent_t) cudaError_t;
pub extern fn cudaGraphEventRecordNodeSetEvent(node: cudaGraphNode_t, event: cudaEvent_t) cudaError_t;
pub extern fn cudaGraphAddEventWaitNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, event: cudaEvent_t) cudaError_t;
pub extern fn cudaGraphEventWaitNodeGetEvent(node: cudaGraphNode_t, event_out: [*c]cudaEvent_t) cudaError_t;
pub extern fn cudaGraphEventWaitNodeSetEvent(node: cudaGraphNode_t, event: cudaEvent_t) cudaError_t;
pub extern fn cudaGraphAddExternalSemaphoresSignalNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, nodeParams: [*c]const struct_cudaExternalSemaphoreSignalNodeParams) cudaError_t;
pub extern fn cudaGraphExternalSemaphoresSignalNodeGetParams(hNode: cudaGraphNode_t, params_out: [*c]struct_cudaExternalSemaphoreSignalNodeParams) cudaError_t;
pub extern fn cudaGraphExternalSemaphoresSignalNodeSetParams(hNode: cudaGraphNode_t, nodeParams: [*c]const struct_cudaExternalSemaphoreSignalNodeParams) cudaError_t;
pub extern fn cudaGraphAddExternalSemaphoresWaitNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, nodeParams: [*c]const struct_cudaExternalSemaphoreWaitNodeParams) cudaError_t;
pub extern fn cudaGraphExternalSemaphoresWaitNodeGetParams(hNode: cudaGraphNode_t, params_out: [*c]struct_cudaExternalSemaphoreWaitNodeParams) cudaError_t;
pub extern fn cudaGraphExternalSemaphoresWaitNodeSetParams(hNode: cudaGraphNode_t, nodeParams: [*c]const struct_cudaExternalSemaphoreWaitNodeParams) cudaError_t;
pub extern fn cudaGraphAddMemAllocNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, nodeParams: [*c]struct_cudaMemAllocNodeParams) cudaError_t;
pub extern fn cudaGraphMemAllocNodeGetParams(node: cudaGraphNode_t, params_out: [*c]struct_cudaMemAllocNodeParams) cudaError_t;
pub extern fn cudaGraphAddMemFreeNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, dptr: ?*anyopaque) cudaError_t;
pub extern fn cudaGraphMemFreeNodeGetParams(node: cudaGraphNode_t, dptr_out: ?*anyopaque) cudaError_t;
pub extern fn cudaDeviceGraphMemTrim(device: c_int) cudaError_t;
pub extern fn cudaDeviceGetGraphMemAttribute(device: c_int, attr: enum_cudaGraphMemAttributeType, value: ?*anyopaque) cudaError_t;
pub extern fn cudaDeviceSetGraphMemAttribute(device: c_int, attr: enum_cudaGraphMemAttributeType, value: ?*anyopaque) cudaError_t;
pub extern fn cudaGraphClone(pGraphClone: [*c]cudaGraph_t, originalGraph: cudaGraph_t) cudaError_t;
pub extern fn cudaGraphNodeFindInClone(pNode: [*c]cudaGraphNode_t, originalNode: cudaGraphNode_t, clonedGraph: cudaGraph_t) cudaError_t;
pub extern fn cudaGraphNodeGetType(node: cudaGraphNode_t, pType: [*c]enum_cudaGraphNodeType) cudaError_t;
pub extern fn cudaGraphGetNodes(graph: cudaGraph_t, nodes: [*c]cudaGraphNode_t, numNodes: [*c]usize) cudaError_t;
pub extern fn cudaGraphGetRootNodes(graph: cudaGraph_t, pRootNodes: [*c]cudaGraphNode_t, pNumRootNodes: [*c]usize) cudaError_t;
pub extern fn cudaGraphGetEdges(graph: cudaGraph_t, from: [*c]cudaGraphNode_t, to: [*c]cudaGraphNode_t, numEdges: [*c]usize) cudaError_t;
pub extern fn cudaGraphGetEdges_v2(graph: cudaGraph_t, from: [*c]cudaGraphNode_t, to: [*c]cudaGraphNode_t, edgeData: [*c]cudaGraphEdgeData, numEdges: [*c]usize) cudaError_t;
pub extern fn cudaGraphNodeGetDependencies(node: cudaGraphNode_t, pDependencies: [*c]cudaGraphNode_t, pNumDependencies: [*c]usize) cudaError_t;
pub extern fn cudaGraphNodeGetDependencies_v2(node: cudaGraphNode_t, pDependencies: [*c]cudaGraphNode_t, edgeData: [*c]cudaGraphEdgeData, pNumDependencies: [*c]usize) cudaError_t;
pub extern fn cudaGraphNodeGetDependentNodes(node: cudaGraphNode_t, pDependentNodes: [*c]cudaGraphNode_t, pNumDependentNodes: [*c]usize) cudaError_t;
pub extern fn cudaGraphNodeGetDependentNodes_v2(node: cudaGraphNode_t, pDependentNodes: [*c]cudaGraphNode_t, edgeData: [*c]cudaGraphEdgeData, pNumDependentNodes: [*c]usize) cudaError_t;
pub extern fn cudaGraphAddDependencies(graph: cudaGraph_t, from: [*c]const cudaGraphNode_t, to: [*c]const cudaGraphNode_t, numDependencies: usize) cudaError_t;
pub extern fn cudaGraphAddDependencies_v2(graph: cudaGraph_t, from: [*c]const cudaGraphNode_t, to: [*c]const cudaGraphNode_t, edgeData: [*c]const cudaGraphEdgeData, numDependencies: usize) cudaError_t;
pub extern fn cudaGraphRemoveDependencies(graph: cudaGraph_t, from: [*c]const cudaGraphNode_t, to: [*c]const cudaGraphNode_t, numDependencies: usize) cudaError_t;
pub extern fn cudaGraphRemoveDependencies_v2(graph: cudaGraph_t, from: [*c]const cudaGraphNode_t, to: [*c]const cudaGraphNode_t, edgeData: [*c]const cudaGraphEdgeData, numDependencies: usize) cudaError_t;
pub extern fn cudaGraphDestroyNode(node: cudaGraphNode_t) cudaError_t;
pub extern fn cudaGraphInstantiate(pGraphExec: [*c]cudaGraphExec_t, graph: cudaGraph_t, flags: c_ulonglong) cudaError_t;
pub extern fn cudaGraphInstantiateWithFlags(pGraphExec: [*c]cudaGraphExec_t, graph: cudaGraph_t, flags: c_ulonglong) cudaError_t;
pub extern fn cudaGraphInstantiateWithParams(pGraphExec: [*c]cudaGraphExec_t, graph: cudaGraph_t, instantiateParams: [*c]cudaGraphInstantiateParams) cudaError_t;
pub extern fn cudaGraphExecGetFlags(graphExec: cudaGraphExec_t, flags: [*c]c_ulonglong) cudaError_t;
pub extern fn cudaGraphExecKernelNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, pNodeParams: [*c]const struct_cudaKernelNodeParams) cudaError_t;
pub extern fn cudaGraphExecMemcpyNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, pNodeParams: [*c]const struct_cudaMemcpy3DParms) cudaError_t;
pub extern fn cudaGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, symbol: ?*const anyopaque, src: ?*const anyopaque, count: usize, offset: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, dst: ?*anyopaque, symbol: ?*const anyopaque, count: usize, offset: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaGraphExecMemcpyNodeSetParams1D(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: enum_cudaMemcpyKind) cudaError_t;
pub extern fn cudaGraphExecMemsetNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, pNodeParams: [*c]const struct_cudaMemsetParams) cudaError_t;
pub extern fn cudaGraphExecHostNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, pNodeParams: [*c]const struct_cudaHostNodeParams) cudaError_t;
pub extern fn cudaGraphExecChildGraphNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, childGraph: cudaGraph_t) cudaError_t;
pub extern fn cudaGraphExecEventRecordNodeSetEvent(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, event: cudaEvent_t) cudaError_t;
pub extern fn cudaGraphExecEventWaitNodeSetEvent(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, event: cudaEvent_t) cudaError_t;
pub extern fn cudaGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, nodeParams: [*c]const struct_cudaExternalSemaphoreSignalNodeParams) cudaError_t;
pub extern fn cudaGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, nodeParams: [*c]const struct_cudaExternalSemaphoreWaitNodeParams) cudaError_t;
pub extern fn cudaGraphNodeSetEnabled(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, isEnabled: c_uint) cudaError_t;
pub extern fn cudaGraphNodeGetEnabled(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, isEnabled: [*c]c_uint) cudaError_t;
pub extern fn cudaGraphExecUpdate(hGraphExec: cudaGraphExec_t, hGraph: cudaGraph_t, resultInfo: [*c]cudaGraphExecUpdateResultInfo) cudaError_t;
pub extern fn cudaGraphUpload(graphExec: cudaGraphExec_t, stream: cudaStream_t) cudaError_t;
pub extern fn cudaGraphLaunch(graphExec: cudaGraphExec_t, stream: cudaStream_t) cudaError_t;
pub extern fn cudaGraphExecDestroy(graphExec: cudaGraphExec_t) cudaError_t;
pub extern fn cudaGraphDestroy(graph: cudaGraph_t) cudaError_t;
pub extern fn cudaGraphDebugDotPrint(graph: cudaGraph_t, path: [*c]const u8, flags: c_uint) cudaError_t;
pub extern fn cudaUserObjectCreate(object_out: [*c]cudaUserObject_t, ptr: ?*anyopaque, destroy: cudaHostFn_t, initialRefcount: c_uint, flags: c_uint) cudaError_t;
pub extern fn cudaUserObjectRetain(object: cudaUserObject_t, count: c_uint) cudaError_t;
pub extern fn cudaUserObjectRelease(object: cudaUserObject_t, count: c_uint) cudaError_t;
pub extern fn cudaGraphRetainUserObject(graph: cudaGraph_t, object: cudaUserObject_t, count: c_uint, flags: c_uint) cudaError_t;
pub extern fn cudaGraphReleaseUserObject(graph: cudaGraph_t, object: cudaUserObject_t, count: c_uint) cudaError_t;
pub extern fn cudaGraphAddNode(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, numDependencies: usize, nodeParams: [*c]struct_cudaGraphNodeParams) cudaError_t;
pub extern fn cudaGraphAddNode_v2(pGraphNode: [*c]cudaGraphNode_t, graph: cudaGraph_t, pDependencies: [*c]const cudaGraphNode_t, dependencyData: [*c]const cudaGraphEdgeData, numDependencies: usize, nodeParams: [*c]struct_cudaGraphNodeParams) cudaError_t;
pub extern fn cudaGraphNodeSetParams(node: cudaGraphNode_t, nodeParams: [*c]struct_cudaGraphNodeParams) cudaError_t;
pub extern fn cudaGraphExecNodeSetParams(graphExec: cudaGraphExec_t, node: cudaGraphNode_t, nodeParams: [*c]struct_cudaGraphNodeParams) cudaError_t;
pub extern fn cudaGraphConditionalHandleCreate(pHandle_out: [*c]cudaGraphConditionalHandle, graph: cudaGraph_t, defaultLaunchValue: c_uint, flags: c_uint) cudaError_t;
pub extern fn cudaGetDriverEntryPoint(symbol: [*c]const u8, funcPtr: [*c]?*anyopaque, flags: c_ulonglong, driverStatus: [*c]enum_cudaDriverEntryPointQueryResult) cudaError_t;
pub extern fn cudaGetDriverEntryPointByVersion(symbol: [*c]const u8, funcPtr: [*c]?*anyopaque, cudaVersion: c_uint, flags: c_ulonglong, driverStatus: [*c]enum_cudaDriverEntryPointQueryResult) cudaError_t;
pub extern fn cudaLibraryLoadData(library: [*c]cudaLibrary_t, code: ?*const anyopaque, jitOptions: [*c]enum_cudaJitOption, jitOptionsValues: [*c]?*anyopaque, numJitOptions: c_uint, libraryOptions: [*c]enum_cudaLibraryOption, libraryOptionValues: [*c]?*anyopaque, numLibraryOptions: c_uint) cudaError_t;
pub extern fn cudaLibraryLoadFromFile(library: [*c]cudaLibrary_t, fileName: [*c]const u8, jitOptions: [*c]enum_cudaJitOption, jitOptionsValues: [*c]?*anyopaque, numJitOptions: c_uint, libraryOptions: [*c]enum_cudaLibraryOption, libraryOptionValues: [*c]?*anyopaque, numLibraryOptions: c_uint) cudaError_t;
pub extern fn cudaLibraryUnload(library: cudaLibrary_t) cudaError_t;
pub extern fn cudaLibraryGetKernel(pKernel: [*c]cudaKernel_t, library: cudaLibrary_t, name: [*c]const u8) cudaError_t;
pub extern fn cudaLibraryGetGlobal(dptr: [*c]?*anyopaque, bytes: [*c]usize, library: cudaLibrary_t, name: [*c]const u8) cudaError_t;
pub extern fn cudaLibraryGetManaged(dptr: [*c]?*anyopaque, bytes: [*c]usize, library: cudaLibrary_t, name: [*c]const u8) cudaError_t;
pub extern fn cudaLibraryGetUnifiedFunction(fptr: [*c]?*anyopaque, library: cudaLibrary_t, symbol: [*c]const u8) cudaError_t;
pub extern fn cudaLibraryGetKernelCount(count: [*c]c_uint, lib: cudaLibrary_t) cudaError_t;
pub extern fn cudaLibraryEnumerateKernels(kernels: [*c]cudaKernel_t, numKernels: c_uint, lib: cudaLibrary_t) cudaError_t;
pub extern fn cudaKernelSetAttributeForDevice(kernel: cudaKernel_t, attr: enum_cudaFuncAttribute, value: c_int, device: c_int) cudaError_t;
pub extern fn cudaGetExportTable(ppExportTable: [*c]?*const anyopaque, pExportTableId: [*c]const cudaUUID_t) cudaError_t;
pub extern fn cudaGetFuncBySymbol(functionPtr: [*c]cudaFunction_t, symbolPtr: ?*const anyopaque) cudaError_t;
pub extern fn cudaGetKernel(kernelPtr: [*c]cudaKernel_t, entryFuncAddr: ?*const anyopaque) cudaError_t;
pub fn make_cudaPitchedPtr(arg_d: ?*anyopaque, arg_p: usize, arg_xsz: usize, arg_ysz: usize) callconv(.c) struct_cudaPitchedPtr {
    var d = arg_d;
    _ = &d;
    var p = arg_p;
    _ = &p;
    var xsz = arg_xsz;
    _ = &xsz;
    var ysz = arg_ysz;
    _ = &ysz;
    var s: struct_cudaPitchedPtr = undefined;
    _ = &s;
    s.ptr = d;
    s.pitch = p;
    s.xsize = xsz;
    s.ysize = ysz;
    return s;
}
pub fn make_cudaPos(arg_x: usize, arg_y: usize, arg_z: usize) callconv(.c) struct_cudaPos {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var p: struct_cudaPos = undefined;
    _ = &p;
    p.x = x;
    p.y = y;
    p.z = z;
    return p;
}
pub fn make_cudaExtent(arg_w: usize, arg_h: usize, arg_d: usize) callconv(.c) struct_cudaExtent {
    var w = arg_w;
    _ = &w;
    var h = arg_h;
    _ = &h;
    var d = arg_d;
    _ = &d;
    var e: struct_cudaExtent = undefined;
    _ = &e;
    e.width = w;
    e.height = h;
    e.depth = d;
    return e;
}
pub fn make_char1(arg_x: i8) callconv(.c) char1 {
    var x = arg_x;
    _ = &x;
    var t: char1 = undefined;
    _ = &t;
    t.x = x;
    return t;
}
pub fn make_uchar1(arg_x: u8) callconv(.c) uchar1 {
    var x = arg_x;
    _ = &x;
    var t: uchar1 = undefined;
    _ = &t;
    t.x = x;
    return t;
}
pub fn make_char2(arg_x: i8, arg_y: i8) callconv(.c) char2 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var t: char2 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    return t;
}
pub fn make_uchar2(arg_x: u8, arg_y: u8) callconv(.c) uchar2 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var t: uchar2 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    return t;
}
pub fn make_char3(arg_x: i8, arg_y: i8, arg_z: i8) callconv(.c) char3 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var t: char3 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    return t;
}
pub fn make_uchar3(arg_x: u8, arg_y: u8, arg_z: u8) callconv(.c) uchar3 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var t: uchar3 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    return t;
}
pub fn make_char4(arg_x: i8, arg_y: i8, arg_z: i8, arg_w: i8) callconv(.c) char4 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var w = arg_w;
    _ = &w;
    var t: char4 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    t.w = w;
    return t;
}
pub fn make_uchar4(arg_x: u8, arg_y: u8, arg_z: u8, arg_w: u8) callconv(.c) uchar4 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var w = arg_w;
    _ = &w;
    var t: uchar4 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    t.w = w;
    return t;
}
pub fn make_short1(arg_x: c_short) callconv(.c) short1 {
    var x = arg_x;
    _ = &x;
    var t: short1 = undefined;
    _ = &t;
    t.x = x;
    return t;
}
pub fn make_ushort1(arg_x: c_ushort) callconv(.c) ushort1 {
    var x = arg_x;
    _ = &x;
    var t: ushort1 = undefined;
    _ = &t;
    t.x = x;
    return t;
}
pub fn make_short2(arg_x: c_short, arg_y: c_short) callconv(.c) short2 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var t: short2 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    return t;
}
pub fn make_ushort2(arg_x: c_ushort, arg_y: c_ushort) callconv(.c) ushort2 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var t: ushort2 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    return t;
}
pub fn make_short3(arg_x: c_short, arg_y: c_short, arg_z: c_short) callconv(.c) short3 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var t: short3 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    return t;
}
pub fn make_ushort3(arg_x: c_ushort, arg_y: c_ushort, arg_z: c_ushort) callconv(.c) ushort3 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var t: ushort3 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    return t;
}
pub fn make_short4(arg_x: c_short, arg_y: c_short, arg_z: c_short, arg_w: c_short) callconv(.c) short4 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var w = arg_w;
    _ = &w;
    var t: short4 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    t.w = w;
    return t;
}
pub fn make_ushort4(arg_x: c_ushort, arg_y: c_ushort, arg_z: c_ushort, arg_w: c_ushort) callconv(.c) ushort4 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var w = arg_w;
    _ = &w;
    var t: ushort4 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    t.w = w;
    return t;
}
pub fn make_int1(arg_x: c_int) callconv(.c) int1 {
    var x = arg_x;
    _ = &x;
    var t: int1 = undefined;
    _ = &t;
    t.x = x;
    return t;
}
pub fn make_uint1(arg_x: c_uint) callconv(.c) uint1 {
    var x = arg_x;
    _ = &x;
    var t: uint1 = undefined;
    _ = &t;
    t.x = x;
    return t;
}
pub fn make_int2(arg_x: c_int, arg_y: c_int) callconv(.c) int2 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var t: int2 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    return t;
}
pub fn make_uint2(arg_x: c_uint, arg_y: c_uint) callconv(.c) uint2 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var t: uint2 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    return t;
}
pub fn make_int3(arg_x: c_int, arg_y: c_int, arg_z: c_int) callconv(.c) int3 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var t: int3 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    return t;
}
pub fn make_uint3(arg_x: c_uint, arg_y: c_uint, arg_z: c_uint) callconv(.c) uint3 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var t: uint3 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    return t;
}
pub fn make_int4(arg_x: c_int, arg_y: c_int, arg_z: c_int, arg_w: c_int) callconv(.c) int4 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var w = arg_w;
    _ = &w;
    var t: int4 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    t.w = w;
    return t;
}
pub fn make_uint4(arg_x: c_uint, arg_y: c_uint, arg_z: c_uint, arg_w: c_uint) callconv(.c) uint4 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var w = arg_w;
    _ = &w;
    var t: uint4 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    t.w = w;
    return t;
}
pub fn make_long1(arg_x: c_long) callconv(.c) long1 {
    var x = arg_x;
    _ = &x;
    var t: long1 = undefined;
    _ = &t;
    t.x = x;
    return t;
}
pub fn make_ulong1(arg_x: c_ulong) callconv(.c) ulong1 {
    var x = arg_x;
    _ = &x;
    var t: ulong1 = undefined;
    _ = &t;
    t.x = x;
    return t;
}
pub fn make_long2(arg_x: c_long, arg_y: c_long) callconv(.c) long2 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var t: long2 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    return t;
}
pub fn make_ulong2(arg_x: c_ulong, arg_y: c_ulong) callconv(.c) ulong2 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var t: ulong2 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    return t;
}
pub fn make_long3(arg_x: c_long, arg_y: c_long, arg_z: c_long) callconv(.c) long3 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var t: long3 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    return t;
}
pub fn make_ulong3(arg_x: c_ulong, arg_y: c_ulong, arg_z: c_ulong) callconv(.c) ulong3 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var t: ulong3 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    return t;
}
pub fn make_long4(arg_x: c_long, arg_y: c_long, arg_z: c_long, arg_w: c_long) callconv(.c) long4 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var w = arg_w;
    _ = &w;
    var t: long4 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    t.w = w;
    return t;
}
pub fn make_ulong4(arg_x: c_ulong, arg_y: c_ulong, arg_z: c_ulong, arg_w: c_ulong) callconv(.c) ulong4 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var w = arg_w;
    _ = &w;
    var t: ulong4 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    t.w = w;
    return t;
}
pub fn make_float1(arg_x: f32) callconv(.c) float1 {
    var x = arg_x;
    _ = &x;
    var t: float1 = undefined;
    _ = &t;
    t.x = x;
    return t;
}
pub fn make_float2(arg_x: f32, arg_y: f32) callconv(.c) float2 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var t: float2 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    return t;
}
pub fn make_float3(arg_x: f32, arg_y: f32, arg_z: f32) callconv(.c) float3 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var t: float3 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    return t;
}
pub fn make_float4(arg_x: f32, arg_y: f32, arg_z: f32, arg_w: f32) callconv(.c) float4 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var w = arg_w;
    _ = &w;
    var t: float4 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    t.w = w;
    return t;
}
pub fn make_longlong1(arg_x: c_longlong) callconv(.c) longlong1 {
    var x = arg_x;
    _ = &x;
    var t: longlong1 = undefined;
    _ = &t;
    t.x = x;
    return t;
}
pub fn make_ulonglong1(arg_x: c_ulonglong) callconv(.c) ulonglong1 {
    var x = arg_x;
    _ = &x;
    var t: ulonglong1 = undefined;
    _ = &t;
    t.x = x;
    return t;
}
pub fn make_longlong2(arg_x: c_longlong, arg_y: c_longlong) callconv(.c) longlong2 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var t: longlong2 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    return t;
}
pub fn make_ulonglong2(arg_x: c_ulonglong, arg_y: c_ulonglong) callconv(.c) ulonglong2 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var t: ulonglong2 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    return t;
}
pub fn make_longlong3(arg_x: c_longlong, arg_y: c_longlong, arg_z: c_longlong) callconv(.c) longlong3 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var t: longlong3 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    return t;
}
pub fn make_ulonglong3(arg_x: c_ulonglong, arg_y: c_ulonglong, arg_z: c_ulonglong) callconv(.c) ulonglong3 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var t: ulonglong3 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    return t;
}
pub fn make_longlong4(arg_x: c_longlong, arg_y: c_longlong, arg_z: c_longlong, arg_w: c_longlong) callconv(.c) longlong4 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var w = arg_w;
    _ = &w;
    var t: longlong4 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    t.w = w;
    return t;
}
pub fn make_ulonglong4(arg_x: c_ulonglong, arg_y: c_ulonglong, arg_z: c_ulonglong, arg_w: c_ulonglong) callconv(.c) ulonglong4 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var w = arg_w;
    _ = &w;
    var t: ulonglong4 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    t.w = w;
    return t;
}
pub fn make_double1(arg_x: f64) callconv(.c) double1 {
    var x = arg_x;
    _ = &x;
    var t: double1 = undefined;
    _ = &t;
    t.x = x;
    return t;
}
pub fn make_double2(arg_x: f64, arg_y: f64) callconv(.c) double2 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var t: double2 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    return t;
}
pub fn make_double3(arg_x: f64, arg_y: f64, arg_z: f64) callconv(.c) double3 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var t: double3 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    return t;
}
pub fn make_double4(arg_x: f64, arg_y: f64, arg_z: f64, arg_w: f64) callconv(.c) double4 {
    var x = arg_x;
    _ = &x;
    var y = arg_y;
    _ = &y;
    var z = arg_z;
    _ = &z;
    var w = arg_w;
    _ = &w;
    var t: double4 = undefined;
    _ = &t;
    t.x = x;
    t.y = y;
    t.z = z;
    t.w = w;
    return t;
}
pub extern const threadIdx: uint3;
pub extern const blockIdx: uint3;
pub extern const blockDim: dim3;
pub extern const gridDim: dim3;
pub extern const warpSize: c_int;
pub const CURAND_STATUS_SUCCESS: c_int = 0;
pub const CURAND_STATUS_VERSION_MISMATCH: c_int = 100;
pub const CURAND_STATUS_NOT_INITIALIZED: c_int = 101;
pub const CURAND_STATUS_ALLOCATION_FAILED: c_int = 102;
pub const CURAND_STATUS_TYPE_ERROR: c_int = 103;
pub const CURAND_STATUS_OUT_OF_RANGE: c_int = 104;
pub const CURAND_STATUS_LENGTH_NOT_MULTIPLE: c_int = 105;
pub const CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: c_int = 106;
pub const CURAND_STATUS_LAUNCH_FAILURE: c_int = 201;
pub const CURAND_STATUS_PREEXISTING_FAILURE: c_int = 202;
pub const CURAND_STATUS_INITIALIZATION_FAILED: c_int = 203;
pub const CURAND_STATUS_ARCH_MISMATCH: c_int = 204;
pub const CURAND_STATUS_INTERNAL_ERROR: c_int = 999;
pub const enum_curandStatus = c_uint;
pub const curandStatus_t = enum_curandStatus;
pub const CURAND_RNG_TEST: c_int = 0;
pub const CURAND_RNG_PSEUDO_DEFAULT: c_int = 100;
pub const CURAND_RNG_PSEUDO_XORWOW: c_int = 101;
pub const CURAND_RNG_PSEUDO_MRG32K3A: c_int = 121;
pub const CURAND_RNG_PSEUDO_MTGP32: c_int = 141;
pub const CURAND_RNG_PSEUDO_MT19937: c_int = 142;
pub const CURAND_RNG_PSEUDO_PHILOX4_32_10: c_int = 161;
pub const CURAND_RNG_QUASI_DEFAULT: c_int = 200;
pub const CURAND_RNG_QUASI_SOBOL32: c_int = 201;
pub const CURAND_RNG_QUASI_SCRAMBLED_SOBOL32: c_int = 202;
pub const CURAND_RNG_QUASI_SOBOL64: c_int = 203;
pub const CURAND_RNG_QUASI_SCRAMBLED_SOBOL64: c_int = 204;
pub const enum_curandRngType = c_uint;
pub const curandRngType_t = enum_curandRngType;
pub const CURAND_ORDERING_PSEUDO_BEST: c_int = 100;
pub const CURAND_ORDERING_PSEUDO_DEFAULT: c_int = 101;
pub const CURAND_ORDERING_PSEUDO_SEEDED: c_int = 102;
pub const CURAND_ORDERING_PSEUDO_LEGACY: c_int = 103;
pub const CURAND_ORDERING_PSEUDO_DYNAMIC: c_int = 104;
pub const CURAND_ORDERING_QUASI_DEFAULT: c_int = 201;
pub const enum_curandOrdering = c_uint;
pub const curandOrdering_t = enum_curandOrdering;
pub const CURAND_DIRECTION_VECTORS_32_JOEKUO6: c_int = 101;
pub const CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6: c_int = 102;
pub const CURAND_DIRECTION_VECTORS_64_JOEKUO6: c_int = 103;
pub const CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6: c_int = 104;
pub const enum_curandDirectionVectorSet = c_uint;
pub const curandDirectionVectorSet_t = enum_curandDirectionVectorSet;
pub const curandDirectionVectors32_t = [32]c_uint;
pub const curandDirectionVectors64_t = [64]c_ulonglong;
pub const struct_curandGenerator_st = opaque {};
pub const curandGenerator_t = ?*struct_curandGenerator_st;
pub const curandDistribution_st = f64;
pub const curandDistribution_t = [*c]curandDistribution_st;
pub const struct_curandDistributionShift_st = opaque {};
pub const curandDistributionShift_t = ?*struct_curandDistributionShift_st;
pub const struct_curandDistributionM2Shift_st = opaque {};
pub const curandDistributionM2Shift_t = ?*struct_curandDistributionM2Shift_st;
pub const struct_curandHistogramM2_st = opaque {};
pub const curandHistogramM2_t = ?*struct_curandHistogramM2_st;
pub const curandHistogramM2K_st = c_uint;
pub const curandHistogramM2K_t = [*c]curandHistogramM2K_st;
pub const curandHistogramM2V_st = curandDistribution_st;
pub const curandHistogramM2V_t = [*c]curandHistogramM2V_st;
pub const struct_curandDiscreteDistribution_st = opaque {};
pub const curandDiscreteDistribution_t = ?*struct_curandDiscreteDistribution_st;
pub const CURAND_CHOOSE_BEST: c_int = 0;
pub const CURAND_ITR: c_int = 1;
pub const CURAND_KNUTH: c_int = 2;
pub const CURAND_HITR: c_int = 3;
pub const CURAND_M1: c_int = 4;
pub const CURAND_M2: c_int = 5;
pub const CURAND_BINARY_SEARCH: c_int = 6;
pub const CURAND_DISCRETE_GAUSS: c_int = 7;
pub const CURAND_REJECTION: c_int = 8;
pub const CURAND_DEVICE_API: c_int = 9;
pub const CURAND_FAST_REJECTION: c_int = 10;
pub const CURAND_3RD: c_int = 11;
pub const CURAND_DEFINITION: c_int = 12;
pub const CURAND_POISSON: c_int = 13;
pub const enum_curandMethod = c_uint;
pub const curandMethod_t = enum_curandMethod;
pub extern fn curandCreateGenerator(generator: [*c]curandGenerator_t, rng_type: curandRngType_t) curandStatus_t;
pub extern fn curandCreateGeneratorHost(generator: [*c]curandGenerator_t, rng_type: curandRngType_t) curandStatus_t;
pub extern fn curandDestroyGenerator(generator: curandGenerator_t) curandStatus_t;
pub extern fn curandGetVersion(version: [*c]c_int) curandStatus_t;
pub extern fn curandGetProperty(@"type": libraryPropertyType, value: [*c]c_int) curandStatus_t;
pub extern fn curandSetStream(generator: curandGenerator_t, stream: cudaStream_t) curandStatus_t;
pub extern fn curandSetPseudoRandomGeneratorSeed(generator: curandGenerator_t, seed: c_ulonglong) curandStatus_t;
pub extern fn curandSetGeneratorOffset(generator: curandGenerator_t, offset: c_ulonglong) curandStatus_t;
pub extern fn curandSetGeneratorOrdering(generator: curandGenerator_t, order: curandOrdering_t) curandStatus_t;
pub extern fn curandSetQuasiRandomGeneratorDimensions(generator: curandGenerator_t, num_dimensions: c_uint) curandStatus_t;
pub extern fn curandGenerate(generator: curandGenerator_t, outputPtr: [*c]c_uint, num: usize) curandStatus_t;
pub extern fn curandGenerateLongLong(generator: curandGenerator_t, outputPtr: [*c]c_ulonglong, num: usize) curandStatus_t;
pub extern fn curandGenerateUniform(generator: curandGenerator_t, outputPtr: [*c]f32, num: usize) curandStatus_t;
pub extern fn curandGenerateUniformDouble(generator: curandGenerator_t, outputPtr: [*c]f64, num: usize) curandStatus_t;
pub extern fn curandGenerateNormal(generator: curandGenerator_t, outputPtr: [*c]f32, n: usize, mean: f32, stddev: f32) curandStatus_t;
pub extern fn curandGenerateNormalDouble(generator: curandGenerator_t, outputPtr: [*c]f64, n: usize, mean: f64, stddev: f64) curandStatus_t;
pub extern fn curandGenerateLogNormal(generator: curandGenerator_t, outputPtr: [*c]f32, n: usize, mean: f32, stddev: f32) curandStatus_t;
pub extern fn curandGenerateLogNormalDouble(generator: curandGenerator_t, outputPtr: [*c]f64, n: usize, mean: f64, stddev: f64) curandStatus_t;
pub extern fn curandCreatePoissonDistribution(lambda: f64, discrete_distribution: [*c]curandDiscreteDistribution_t) curandStatus_t;
pub extern fn curandDestroyDistribution(discrete_distribution: curandDiscreteDistribution_t) curandStatus_t;
pub extern fn curandGeneratePoisson(generator: curandGenerator_t, outputPtr: [*c]c_uint, n: usize, lambda: f64) curandStatus_t;
pub extern fn curandGeneratePoissonMethod(generator: curandGenerator_t, outputPtr: [*c]c_uint, n: usize, lambda: f64, method: curandMethod_t) curandStatus_t;
pub extern fn curandGenerateBinomial(generator: curandGenerator_t, outputPtr: [*c]c_uint, num: usize, n: c_uint, p: f64) curandStatus_t;
pub extern fn curandGenerateBinomialMethod(generator: curandGenerator_t, outputPtr: [*c]c_uint, num: usize, n: c_uint, p: f64, method: curandMethod_t) curandStatus_t;
pub extern fn curandGenerateSeeds(generator: curandGenerator_t) curandStatus_t;
pub extern fn curandGetDirectionVectors32(vectors: [*c][*c]curandDirectionVectors32_t, set: curandDirectionVectorSet_t) curandStatus_t;
pub extern fn curandGetScrambleConstants32(constants: [*c][*c]c_uint) curandStatus_t;
pub extern fn curandGetDirectionVectors64(vectors: [*c][*c]curandDirectionVectors64_t, set: curandDirectionVectorSet_t) curandStatus_t;
pub extern fn curandGetScrambleConstants64(constants: [*c][*c]c_ulonglong) curandStatus_t;
pub const __half_raw = extern struct {
    x: c_ushort = @import("std").mem.zeroes(c_ushort),
};
pub const __half2_raw = extern struct {
    x: c_ushort = @import("std").mem.zeroes(c_ushort),
    y: c_ushort = @import("std").mem.zeroes(c_ushort),
};
pub const __nv_bfloat16_raw = extern struct {
    x: c_ushort = @import("std").mem.zeroes(c_ushort),
};
pub const __nv_bfloat162_raw = extern struct {
    x: c_ushort = @import("std").mem.zeroes(c_ushort),
    y: c_ushort = @import("std").mem.zeroes(c_ushort),
};
pub extern fn tomoSinH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSinB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSinF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSinD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoCosH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoCosB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoCosF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoCosD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTanH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTanB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTanF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTanD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoReluH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoReluB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoReluF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoReluD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLeakyReluH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLeakyReluB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLeakyReluF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLeakyReluD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoInvH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoInvB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoInvF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoInvD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEluH(a: [*c]__half_raw, len: usize, alpha: __half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEluB(a: [*c]__nv_bfloat16_raw, len: usize, alpha: __nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEluF(a: [*c]f32, len: usize, alpha: f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEluD(a: [*c]f64, len: usize, alpha: f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSeluH(a: [*c]__half_raw, len: usize, alpha: __half_raw, lambda: __half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSeluB(a: [*c]__nv_bfloat16_raw, len: usize, alpha: __nv_bfloat16_raw, lambda: __nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSeluF(a: [*c]f32, len: usize, alpha: f32, lambda: f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSeluD(a: [*c]f64, len: usize, alpha: f64, lambda: f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSoftplusH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSoftplusB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSoftplusF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSoftplusD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSigmoidH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSigmoidB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSigmoidF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSigmoidD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTanhH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTanhB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTanhF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTanhD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSwishH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSwishB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSwishF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSwishD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGeluH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGeluB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGeluF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGeluD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoHardSigmoidH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoHardSigmoidB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoHardSigmoidF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoHardSigmoidD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoHardSwishH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoHardSwishB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoHardSwishF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoHardSwishD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSoftsignH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSoftsignB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSoftsignF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSoftsignD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSquareH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSquareB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSquareF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSquareD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSqrtH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSqrtB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSqrtF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSqrtD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLogH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLogB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLogF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLogD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoExpH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoExpB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoExpF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoExpD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoAbsH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoAbsB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoAbsF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoAbsD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoScaleH(a: [*c]__half_raw, len: usize, scale: __half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoScaleB(a: [*c]__nv_bfloat16_raw, len: usize, scale: __nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoScaleF(a: [*c]f32, len: usize, scale: f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoScaleD(a: [*c]f64, len: usize, scale: f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoPowH(a: [*c]__half_raw, len: usize, exponent: c_int, stream: cudaStream_t) cudaError_t;
pub extern fn tomoPowB(a: [*c]__nv_bfloat16_raw, len: usize, exponent: c_int, stream: cudaStream_t) cudaError_t;
pub extern fn tomoPowF(a: [*c]f32, len: usize, exponent: c_int, stream: cudaStream_t) cudaError_t;
pub extern fn tomoPowD(a: [*c]f64, len: usize, exponent: c_int, stream: cudaStream_t) cudaError_t;
pub extern fn tomoPowfH(a: [*c]__half_raw, len: usize, exponent: __half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoPowfB(a: [*c]__nv_bfloat16_raw, len: usize, exponent: __nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoPowfF(a: [*c]f32, len: usize, exponent: f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoPowfD(a: [*c]f64, len: usize, exponent: f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoClampH(a: [*c]__half_raw, len: usize, lower: __half_raw, upper: __half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoClampB(a: [*c]__nv_bfloat16_raw, len: usize, lower: __nv_bfloat16_raw, upper: __nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoClampF(a: [*c]f32, len: usize, lower: f32, upper: f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoClampD(a: [*c]f64, len: usize, lower: f64, upper: f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoFloorH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoFloorB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoFloorF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoFloorD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoCeilH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoCeilB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoCeilF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoCeilD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoShiftH(a: [*c]__half_raw, len: usize, offset: __half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoShiftB(a: [*c]__nv_bfloat16_raw, len: usize, offset: __nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoShiftF(a: [*c]f32, len: usize, offset: f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoShiftD(a: [*c]f64, len: usize, offset: f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoScaleShiftH(a: [*c]__half_raw, len: usize, scale: __half_raw, offset: __half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoScaleShiftB(a: [*c]__nv_bfloat16_raw, len: usize, scale: __nv_bfloat16_raw, offset: __nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoScaleShiftF(a: [*c]f32, len: usize, scale: f32, offset: f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoScaleShiftD(a: [*c]f64, len: usize, scale: f64, offset: f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGtH(a: [*c]__half_raw, len: usize, num: __half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGtB(a: [*c]__nv_bfloat16_raw, len: usize, num: __nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGtF(a: [*c]f32, len: usize, num: f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGtD(a: [*c]f64, len: usize, num: f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGtUZ(a: [*c]usize, len: usize, num: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGtEqH(a: [*c]__half_raw, len: usize, num: __half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGtEqB(a: [*c]__nv_bfloat16_raw, len: usize, num: __nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGtEqF(a: [*c]f32, len: usize, num: f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGtEqD(a: [*c]f64, len: usize, num: f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGtEqUZ(a: [*c]usize, len: usize, num: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLtH(a: [*c]__half_raw, len: usize, num: __half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLtB(a: [*c]__nv_bfloat16_raw, len: usize, num: __nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLtF(a: [*c]f32, len: usize, num: f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLtD(a: [*c]f64, len: usize, num: f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLtUZ(a: [*c]usize, len: usize, num: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLtEqH(a: [*c]__half_raw, len: usize, num: __half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLtEqB(a: [*c]__nv_bfloat16_raw, len: usize, num: __nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLtEqF(a: [*c]f32, len: usize, num: f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLtEqD(a: [*c]f64, len: usize, num: f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLtEqUZ(a: [*c]usize, len: usize, num: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEqH(a: [*c]__half_raw, len: usize, num: __half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEqB(a: [*c]__nv_bfloat16_raw, len: usize, num: __nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEqF(a: [*c]f32, len: usize, num: f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEqD(a: [*c]f64, len: usize, num: f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEqUZ(a: [*c]usize, len: usize, num: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoNeqH(a: [*c]__half_raw, len: usize, num: __half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoNeqB(a: [*c]__nv_bfloat16_raw, len: usize, num: __nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoNeqF(a: [*c]f32, len: usize, num: f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoNeqD(a: [*c]f64, len: usize, num: f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoNeqUZ(a: [*c]usize, len: usize, num: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMaskedFillH(data: [*c]__half_raw, mask: [*c]const __half_raw, fillValue: __half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMaskedFillB(data: [*c]__nv_bfloat16_raw, mask: [*c]const __nv_bfloat16_raw, fillValue: __nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMaskedFillF(data: [*c]f32, mask: [*c]const f32, fillValue: f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMaskedFillD(data: [*c]f64, mask: [*c]const f64, fillValue: f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTrilH(data: [*c]__half_raw, rows: usize, cols: usize, fillValue: __half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTriuH(data: [*c]__half_raw, rows: usize, cols: usize, fillValue: __half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTrilB(data: [*c]__nv_bfloat16_raw, rows: usize, cols: usize, fillValue: __nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTriuB(data: [*c]__nv_bfloat16_raw, rows: usize, cols: usize, fillValue: __nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTrilF(data: [*c]f32, rows: usize, cols: usize, fillValue: f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTriuF(data: [*c]f32, rows: usize, cols: usize, fillValue: f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTrilD(data: [*c]f64, rows: usize, cols: usize, fillValue: f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTriuD(data: [*c]f64, rows: usize, cols: usize, fillValue: f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoArangeH(output: [*c]__half_raw, start: __half_raw, step: __half_raw, num_element: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoArangeB(output: [*c]__nv_bfloat16_raw, start: __nv_bfloat16_raw, step: __nv_bfloat16_raw, num_element: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoArangeF(output: [*c]f32, start: f32, step: f32, num_element: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoArangeD(output: [*c]f64, start: f64, step: f64, num_element: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoArangeUZ(output: [*c]usize, start: usize, step: usize, num_element: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoFillNormalH(a: [*c]__half_raw, len: usize, mean: f32, stddev: f32, seed: c_ulonglong, stream: cudaStream_t) cudaError_t;
pub extern fn tomoFillNormalB(a: [*c]__nv_bfloat16_raw, len: usize, mean: f32, stddev: f32, seed: c_ulonglong, stream: cudaStream_t) cudaError_t;
pub extern fn tomoFillUniformH(a: [*c]__half_raw, len: usize, seed: c_ulonglong, stream: cudaStream_t) cudaError_t;
pub extern fn tomoFillUniformB(a: [*c]__nv_bfloat16_raw, len: usize, seed: c_ulonglong, stream: cudaStream_t) cudaError_t;
pub extern fn tomoHasNaNH(data: [*c]const __half_raw, len: usize, result: [*c]bool, stream: cudaStream_t) cudaError_t;
pub extern fn tomoHasNaNB(data: [*c]const __nv_bfloat16_raw, len: usize, result: [*c]bool, stream: cudaStream_t) cudaError_t;
pub extern fn tomoHasNaNF(data: [*c]const f32, len: usize, result: [*c]bool, stream: cudaStream_t) cudaError_t;
pub extern fn tomoHasNaND(data: [*c]const f64, len: usize, result: [*c]bool, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSumReduceH(a: [*c]const __half_raw, len: usize, host_out: [*c]__half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSumReduceB(a: [*c]const __nv_bfloat16_raw, len: usize, host_out: [*c]__nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSumReduceF(a: [*c]const f32, len: usize, host_out: [*c]f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSumReduceD(a: [*c]const f64, len: usize, host_out: [*c]f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMinH(in: [*c]const __half_raw, len: usize, host_out: [*c]__half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMinB(in: [*c]const __nv_bfloat16_raw, len: usize, host_out: [*c]__nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMinF(in: [*c]const f32, len: usize, host_out: [*c]f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMinD(in: [*c]const f64, len: usize, host_out: [*c]f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMaxH(in: [*c]const __half_raw, len: usize, host_out: [*c]__half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMaxB(in: [*c]const __nv_bfloat16_raw, len: usize, host_out: [*c]__nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMaxF(in: [*c]const f32, len: usize, host_out: [*c]f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMaxD(in: [*c]const f64, len: usize, host_out: [*c]f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoL1NormH(a: [*c]const __half_raw, len: usize, host_out: [*c]__half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoL1NormB(a: [*c]const __nv_bfloat16_raw, len: usize, host_out: [*c]__nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoL1NormF(a: [*c]const f32, len: usize, host_out: [*c]f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoL1NormD(a: [*c]const f64, len: usize, host_out: [*c]f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoL2NormH(a: [*c]const __half_raw, len: usize, host_out: [*c]__half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoL2NormB(a: [*c]const __nv_bfloat16_raw, len: usize, host_out: [*c]__nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoL2NormF(a: [*c]const f32, len: usize, host_out: [*c]f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoL2NormD(a: [*c]const f64, len: usize, host_out: [*c]f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoFillH(a: [*c]__half_raw, len: usize, val: __half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoFillB(a: [*c]__nv_bfloat16_raw, len: usize, val: __nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoFillF(a: [*c]f32, len: usize, val: f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoFillD(a: [*c]f64, len: usize, val: f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoFillUZ(a: [*c]usize, len: usize, val: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSortDescH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSortDescB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSortDescF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSortDescD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSortAscH(a: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSortAscB(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSortAscF(a: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSortAscD(a: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoFindH(a: [*c]__half_raw, len: usize, val: __half_raw, stream: cudaStream_t, i: [*c]usize) cudaError_t;
pub extern fn tomoFindB(a: [*c]__nv_bfloat16_raw, len: usize, val: __nv_bfloat16_raw, stream: cudaStream_t, i: [*c]usize) cudaError_t;
pub extern fn tomoFindF(a: [*c]f32, len: usize, val: f32, stream: cudaStream_t, i: [*c]usize) cudaError_t;
pub extern fn tomoFindD(a: [*c]f64, len: usize, val: f64, stream: cudaStream_t, i: [*c]usize) cudaError_t;
pub extern fn tomoHtoB(a: [*c]__half_raw, len: usize, stream: cudaStream_t, out: [*c]__nv_bfloat16_raw) cudaError_t;
pub extern fn tomoHtoF(a: [*c]__half_raw, len: usize, stream: cudaStream_t, out: [*c]f32) cudaError_t;
pub extern fn tomoHtoD(a: [*c]__half_raw, len: usize, stream: cudaStream_t, out: [*c]f64) cudaError_t;
pub extern fn tomoBtoH(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t, out: [*c]__half_raw) cudaError_t;
pub extern fn tomoBtoF(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t, out: [*c]f32) cudaError_t;
pub extern fn tomoBtoD(a: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t, out: [*c]f64) cudaError_t;
pub extern fn tomoFtoH(a: [*c]f32, len: usize, stream: cudaStream_t, out: [*c]__half_raw) cudaError_t;
pub extern fn tomoFtoB(a: [*c]f32, len: usize, stream: cudaStream_t, out: [*c]__nv_bfloat16_raw) cudaError_t;
pub extern fn tomoFtoD(a: [*c]f32, len: usize, stream: cudaStream_t, out: [*c]f64) cudaError_t;
pub extern fn tomoDtoH(a: [*c]f64, len: usize, stream: cudaStream_t, out: [*c]__half_raw) cudaError_t;
pub extern fn tomoDtoB(a: [*c]f64, len: usize, stream: cudaStream_t, out: [*c]__nv_bfloat16_raw) cudaError_t;
pub extern fn tomoDtoF(a: [*c]f64, len: usize, stream: cudaStream_t, out: [*c]f32) cudaError_t;
pub extern fn tomoUztoH(a: [*c]usize, len: usize, stream: cudaStream_t, out: [*c]__half_raw) cudaError_t;
pub extern fn tomoUztoB(a: [*c]usize, len: usize, stream: cudaStream_t, out: [*c]__nv_bfloat16_raw) cudaError_t;
pub extern fn tomoUztoF(a: [*c]usize, len: usize, stream: cudaStream_t, out: [*c]f32) cudaError_t;
pub extern fn tomoUztoD(a: [*c]usize, len: usize, stream: cudaStream_t, out: [*c]f64) cudaError_t;
pub extern fn tomoF16ToBf16(val: __half_raw) __nv_bfloat16_raw;
pub extern fn tomoF32ToBf16(val: f32) __nv_bfloat16_raw;
pub extern fn tomoF64ToBf16(val: f64) __nv_bfloat16_raw;
pub extern fn tomoBf16ToF16(val: __nv_bfloat16_raw) __half_raw;
pub extern fn tomoBf16ToF32(val: __nv_bfloat16_raw) f32;
pub extern fn tomoBf16ToF64(val: __nv_bfloat16_raw) f64;
pub extern fn tomoBf16Add(lh: __nv_bfloat16_raw, rh: __nv_bfloat16_raw) __nv_bfloat16_raw;
pub extern fn tomoBf16Neg(lh: __nv_bfloat16_raw) __nv_bfloat16_raw;
pub extern fn tomoBf16Sub(lh: __nv_bfloat16_raw, rh: __nv_bfloat16_raw) __nv_bfloat16_raw;
pub extern fn tomoBf16Mul(lh: __nv_bfloat16_raw, rh: __nv_bfloat16_raw) __nv_bfloat16_raw;
pub extern fn tomoBf16Div(lh: __nv_bfloat16_raw, rh: __nv_bfloat16_raw) __nv_bfloat16_raw;
pub extern fn tomoAddH(a: [*c]__half_raw, b: [*c]const __half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoAddB(a: [*c]__nv_bfloat16_raw, b: [*c]const __nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoAddF(a: [*c]f32, b: [*c]const f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoAddD(a: [*c]f64, b: [*c]const f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSubH(a: [*c]__half_raw, b: [*c]const __half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSubB(a: [*c]__nv_bfloat16_raw, b: [*c]const __nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSubF(a: [*c]f32, b: [*c]const f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSubD(a: [*c]f64, b: [*c]const f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoProductH(a: [*c]__half_raw, b: [*c]const __half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoProductB(a: [*c]__nv_bfloat16_raw, b: [*c]const __nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoProductF(a: [*c]f32, b: [*c]const f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoProductD(a: [*c]f64, b: [*c]const f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoDivideH(a: [*c]__half_raw, b: [*c]const __half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoDivideB(a: [*c]__nv_bfloat16_raw, b: [*c]const __nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoDivideF(a: [*c]f32, b: [*c]const f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoDivideD(a: [*c]f64, b: [*c]const f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEqualH(a: [*c]__half_raw, b: [*c]const __half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEqualB(a: [*c]__nv_bfloat16_raw, b: [*c]const __nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEqualF(a: [*c]f32, b: [*c]const f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEqualD(a: [*c]f64, b: [*c]const f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEqualUz(a: [*c]usize, b: [*c]const usize, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEqualApproxH(a: [*c]__half_raw, b: [*c]const __half_raw, len: usize, eps: __half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEqualApproxB(a: [*c]__nv_bfloat16_raw, b: [*c]const __nv_bfloat16_raw, len: usize, eps: __nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEqualApproxF(a: [*c]f32, b: [*c]const f32, len: usize, eps: f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEqualApproxD(a: [*c]f64, b: [*c]const f64, len: usize, eps: f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoReluBackwardH(x: [*c]const __half_raw, grad: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoReluBackwardB(x: [*c]const __nv_bfloat16_raw, grad: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoReluBackwardF(x: [*c]const f32, grad: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoReluBackwardD(x: [*c]const f64, grad: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLeakyReluBackwardH(x: [*c]const __half_raw, grad: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLeakyReluBackwardB(x: [*c]const __nv_bfloat16_raw, grad: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLeakyReluBackwardF(x: [*c]const f32, grad: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLeakyReluBackwardD(x: [*c]const f64, grad: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGeluBackwardH(x: [*c]const __half_raw, grad: [*c]__half_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGeluBackwardB(x: [*c]const __nv_bfloat16_raw, grad: [*c]__nv_bfloat16_raw, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGeluBackwardF(x: [*c]const f32, grad: [*c]f32, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGeluBackwardD(x: [*c]const f64, grad: [*c]f64, len: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoBroadcastToH(d_in: [*c]const __half_raw, d_out: [*c]__half_raw, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, in_size: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoBroadcastToB(d_in: [*c]const __nv_bfloat16_raw, d_out: [*c]__nv_bfloat16_raw, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, in_size: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoBroadcastToF(d_in: [*c]const f32, d_out: [*c]f32, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, in_size: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoBroadcastToD(d_in: [*c]const f64, d_out: [*c]f64, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, in_size: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoBroadcastToUZ(d_in: [*c]const usize, d_out: [*c]usize, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, in_size: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSumToH(d_in: [*c]const __half_raw, d_out: [*c]__half_raw, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, in_size: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSumToB(d_in: [*c]const __nv_bfloat16_raw, d_out: [*c]__nv_bfloat16_raw, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, in_size: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSumToF(d_in: [*c]const f32, d_out: [*c]f32, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, in_size: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSumToD(d_in: [*c]const f64, d_out: [*c]f64, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, in_size: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLinearH(A: [*c]const __half_raw, B: [*c]const __half_raw, M: usize, K: usize, N: usize, bias: [*c]const __half_raw, C: [*c]__half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLinearB(A: [*c]const __nv_bfloat16_raw, B: [*c]const __nv_bfloat16_raw, M: usize, K: usize, N: usize, bias: [*c]const __nv_bfloat16_raw, C: [*c]__nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLinearF(A: [*c]const f32, B: [*c]const f32, M: usize, K: usize, N: usize, bias: [*c]const f32, C: [*c]f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLinearD(A: [*c]const f64, B: [*c]const f64, M: usize, K: usize, N: usize, bias: [*c]const f64, C: [*c]f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLinearImpH(A: [*c]const __half_raw, B: [*c]const __half_raw, M: usize, K: usize, N: usize, bias: [*c]const __half_raw, C: [*c]__half_raw, batch_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLinearImpB(A: [*c]const __nv_bfloat16_raw, B: [*c]const __nv_bfloat16_raw, M: usize, K: usize, N: usize, bias: [*c]const __nv_bfloat16_raw, C: [*c]__nv_bfloat16_raw, batch_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLinearImpF(A: [*c]const f32, B: [*c]const f32, M: usize, K: usize, N: usize, bias: [*c]const f32, C: [*c]f32, batch_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoLinearImpD(A: [*c]const f64, B: [*c]const f64, M: usize, K: usize, N: usize, bias: [*c]const f64, C: [*c]f64, batch_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTransposeH(A: [*c]const __half_raw, M: usize, N: usize, C: [*c]__half_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTransposeB(A: [*c]const __nv_bfloat16_raw, M: usize, N: usize, C: [*c]__nv_bfloat16_raw, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTransposeF(A: [*c]const f32, M: usize, N: usize, C: [*c]f32, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTransposeD(A: [*c]const f64, M: usize, N: usize, C: [*c]f64, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMaxToH(d_in: [*c]const __half_raw, d_out: [*c]__half_raw, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, in_size: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMaxToB(d_in: [*c]const __nv_bfloat16_raw, d_out: [*c]__nv_bfloat16_raw, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, in_size: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMaxToF(d_in: [*c]const f32, d_out: [*c]f32, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, in_size: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMaxToD(d_in: [*c]const f64, d_out: [*c]f64, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, in_size: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMinToH(d_in: [*c]const __half_raw, d_out: [*c]__half_raw, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, in_size: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMinToB(d_in: [*c]const __nv_bfloat16_raw, d_out: [*c]__nv_bfloat16_raw, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, in_size: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMinToF(d_in: [*c]const f32, d_out: [*c]f32, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, in_size: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMinToD(d_in: [*c]const f64, d_out: [*c]f64, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, in_size: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTensordotH(d_a: [*c]const __half_raw, d_b: [*c]const __half_raw, d_out: [*c]__half_raw, a_shape: [*c]const usize, a_shape_len: usize, b_shape: [*c]const usize, b_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, a_strides: [*c]const usize, a_strides_len: usize, b_strides: [*c]const usize, b_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, contracted_axes_a: [*c]const usize, contracted_axes_a_len: usize, contracted_axes_b: [*c]const usize, contracted_axes_b_len: usize, a_nd: usize, b_nd: usize, out_nd: usize, out_size: usize, num_contracted: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTensordotB(d_a: [*c]const __nv_bfloat16_raw, d_b: [*c]const __nv_bfloat16_raw, d_out: [*c]__nv_bfloat16_raw, a_shape: [*c]const usize, a_shape_len: usize, b_shape: [*c]const usize, b_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, a_strides: [*c]const usize, a_strides_len: usize, b_strides: [*c]const usize, b_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, contracted_axes_a: [*c]const usize, contracted_axes_a_len: usize, contracted_axes_b: [*c]const usize, contracted_axes_b_len: usize, a_nd: usize, b_nd: usize, out_nd: usize, out_size: usize, num_contracted: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTensordotF(d_a: [*c]const f32, d_b: [*c]const f32, d_out: [*c]f32, a_shape: [*c]const usize, a_shape_len: usize, b_shape: [*c]const usize, b_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, a_strides: [*c]const usize, a_strides_len: usize, b_strides: [*c]const usize, b_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, contracted_axes_a: [*c]const usize, contracted_axes_a_len: usize, contracted_axes_b: [*c]const usize, contracted_axes_b_len: usize, a_nd: usize, b_nd: usize, out_nd: usize, out_size: usize, num_contracted: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTensordotD(d_a: [*c]const f64, d_b: [*c]const f64, d_out: [*c]f64, a_shape: [*c]const usize, a_shape_len: usize, b_shape: [*c]const usize, b_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, a_strides: [*c]const usize, a_strides_len: usize, b_strides: [*c]const usize, b_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, contracted_axes_a: [*c]const usize, contracted_axes_a_len: usize, contracted_axes_b: [*c]const usize, contracted_axes_b_len: usize, a_nd: usize, b_nd: usize, out_nd: usize, out_size: usize, num_contracted: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTransposeExH(d_in: [*c]const __half_raw, d_out: [*c]__half_raw, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, perm: [*c]const usize, perm_len: usize, nd: usize, in_size: usize, out_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTransposeExB(d_in: [*c]const __nv_bfloat16_raw, d_out: [*c]__nv_bfloat16_raw, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, perm: [*c]const usize, perm_len: usize, nd: usize, in_size: usize, out_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTransposeExF(d_in: [*c]const f32, d_out: [*c]f32, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, perm: [*c]const usize, perm_len: usize, nd: usize, in_size: usize, out_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoTransposeExD(d_in: [*c]const f64, d_out: [*c]f64, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, perm: [*c]const usize, perm_len: usize, nd: usize, in_size: usize, out_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoRollaxisH(d_in: [*c]const __half_raw, d_out: [*c]__half_raw, in_shape: [*c]const usize, in_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, axis: usize, start: usize, nd: usize, in_size: usize, out_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoRollaxisB(d_in: [*c]const __nv_bfloat16_raw, d_out: [*c]__nv_bfloat16_raw, in_shape: [*c]const usize, in_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, axis: usize, start: usize, nd: usize, in_size: usize, out_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoRollaxisF(d_in: [*c]const f32, d_out: [*c]f32, in_shape: [*c]const usize, in_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, axis: usize, start: usize, nd: usize, in_size: usize, out_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoRollaxisD(d_in: [*c]const f64, d_out: [*c]f64, in_shape: [*c]const usize, in_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, axis: usize, start: usize, nd: usize, in_size: usize, out_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSwapaxesH(d_in: [*c]const __half_raw, d_out: [*c]__half_raw, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, axis1: usize, axis2: usize, nd: usize, in_size: usize, out_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSwapaxesB(d_in: [*c]const __nv_bfloat16_raw, d_out: [*c]__nv_bfloat16_raw, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, axis1: usize, axis2: usize, nd: usize, in_size: usize, out_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSwapaxesF(d_in: [*c]const f32, d_out: [*c]f32, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, axis1: usize, axis2: usize, nd: usize, in_size: usize, out_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSwapaxesD(d_in: [*c]const f64, d_out: [*c]f64, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, axis1: usize, axis2: usize, nd: usize, in_size: usize, out_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoIm2colH(d_img: [*c]const __half_raw, d_col: [*c]__half_raw, n: usize, c: usize, h: usize, w: usize, kh: usize, kw: usize, sy: usize, sx: usize, ph: usize, pw: usize, dy: usize, dx: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoIm2colB(d_img: [*c]const __nv_bfloat16_raw, d_col: [*c]__nv_bfloat16_raw, n: usize, c: usize, h: usize, w: usize, kh: usize, kw: usize, sy: usize, sx: usize, ph: usize, pw: usize, dy: usize, dx: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoIm2colF(d_img: [*c]const f32, d_col: [*c]f32, n: usize, c: usize, h: usize, w: usize, kh: usize, kw: usize, sy: usize, sx: usize, ph: usize, pw: usize, dy: usize, dx: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoIm2colD(d_img: [*c]const f64, d_col: [*c]f64, n: usize, c: usize, h: usize, w: usize, kh: usize, kw: usize, sy: usize, sx: usize, ph: usize, pw: usize, dy: usize, dx: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoCol2imH(d_col: [*c]const __half_raw, d_img: [*c]__half_raw, n: usize, c: usize, h: usize, w: usize, kh: usize, kw: usize, sy: usize, sx: usize, ph: usize, pw: usize, dx: usize, dy: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoCol2imB(d_col: [*c]const __nv_bfloat16_raw, d_img: [*c]__nv_bfloat16_raw, n: usize, c: usize, h: usize, w: usize, kh: usize, kw: usize, sy: usize, sx: usize, ph: usize, pw: usize, dx: usize, dy: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoCol2imF(d_col: [*c]const f32, d_img: [*c]f32, n: usize, c: usize, h: usize, w: usize, kh: usize, kw: usize, sy: usize, sx: usize, ph: usize, pw: usize, dx: usize, dy: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoCol2imD(d_col: [*c]const f64, d_img: [*c]f64, n: usize, c: usize, h: usize, w: usize, kh: usize, kw: usize, sy: usize, sx: usize, ph: usize, pw: usize, dx: usize, dy: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoIm2col1dH(d_in: [*c]const __half_raw, d_col: [*c]__half_raw, n: usize, c: usize, l: usize, k: usize, stride: usize, pad: usize, dilation: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoIm2col1dB(d_in: [*c]const __nv_bfloat16_raw, d_col: [*c]__nv_bfloat16_raw, n: usize, c: usize, l: usize, k: usize, stride: usize, pad: usize, dilation: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoIm2col1dF(d_in: [*c]const f32, d_col: [*c]f32, n: usize, c: usize, l: usize, k: usize, stride: usize, pad: usize, dilation: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoIm2col1dD(d_in: [*c]const f64, d_col: [*c]f64, n: usize, c: usize, l: usize, k: usize, stride: usize, pad: usize, dilation: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoCol2im1dH(d_col: [*c]const __half_raw, d_in: [*c]__half_raw, n: usize, c: usize, l: usize, k: usize, stride: usize, pad: usize, dilation: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoCol2im1dB(d_col: [*c]const __nv_bfloat16_raw, d_in: [*c]__nv_bfloat16_raw, n: usize, c: usize, l: usize, k: usize, stride: usize, pad: usize, dilation: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoCol2im1dF(d_col: [*c]const f32, d_in: [*c]f32, n: usize, c: usize, l: usize, k: usize, stride: usize, pad: usize, dilation: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoCol2im1dD(d_col: [*c]const f64, d_in: [*c]f64, n: usize, c: usize, l: usize, k: usize, stride: usize, pad: usize, dilation: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoArgmaxH(d_in: [*c]const __half_raw, d_out: [*c]usize, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoArgmaxB(d_in: [*c]const __nv_bfloat16_raw, d_out: [*c]usize, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoArgmaxF(d_in: [*c]const f32, d_out: [*c]usize, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoArgmaxD(d_in: [*c]const f64, d_out: [*c]usize, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoArgminH(d_in: [*c]const __half_raw, d_out: [*c]usize, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoArgminB(d_in: [*c]const __nv_bfloat16_raw, d_out: [*c]usize, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoArgminF(d_in: [*c]const f32, d_out: [*c]usize, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoArgminD(d_in: [*c]const f64, d_out: [*c]usize, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, out_size: usize, nd: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMaxPool2dForwardH(d_in: [*c]const __half_raw, d_out: [*c]__half_raw, N: usize, C: usize, H: usize, W: usize, outH: usize, outW: usize, kH: usize, kW: usize, sH: usize, sW: usize, pH: usize, pW: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMaxPool2dForwardB(d_in: [*c]const __nv_bfloat16_raw, d_out: [*c]__nv_bfloat16_raw, N: usize, C: usize, H: usize, W: usize, outH: usize, outW: usize, kH: usize, kW: usize, sH: usize, sW: usize, pH: usize, pW: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMaxPool2dForwardF(d_in: [*c]const f32, d_out: [*c]f32, N: usize, C: usize, H: usize, W: usize, outH: usize, outW: usize, kH: usize, kW: usize, sH: usize, sW: usize, pH: usize, pW: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMaxPool2dForwardD(d_in: [*c]const f64, d_out: [*c]f64, N: usize, C: usize, H: usize, W: usize, outH: usize, outW: usize, kH: usize, kW: usize, sH: usize, sW: usize, pH: usize, pW: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMaxPool2dBackwardH(input: [*c]const __half_raw, gradOut: [*c]const __half_raw, gradIn: [*c]__half_raw, N: usize, C: usize, H: usize, W: usize, outH: usize, outW: usize, kernelH: usize, kernelW: usize, strideH: usize, strideW: usize, padH: usize, padW: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMaxPool2dBackwardB(input: [*c]const __nv_bfloat16_raw, gradOut: [*c]const __nv_bfloat16_raw, gradIn: [*c]__nv_bfloat16_raw, N: usize, C: usize, H: usize, W: usize, outH: usize, outW: usize, kernelH: usize, kernelW: usize, strideH: usize, strideW: usize, padH: usize, padW: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMaxPool2dBackwardF(input: [*c]const f32, gradOut: [*c]const f32, gradIn: [*c]f32, N: usize, C: usize, H: usize, W: usize, outH: usize, outW: usize, kernelH: usize, kernelW: usize, strideH: usize, strideW: usize, padH: usize, padW: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoMaxPool2dBackwardD(input: [*c]const f64, gradOut: [*c]const f64, gradIn: [*c]f64, N: usize, C: usize, H: usize, W: usize, outH: usize, outW: usize, kernelH: usize, kernelW: usize, strideH: usize, strideW: usize, padH: usize, padW: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoAvgPool2dForwardH(d_in: [*c]const __half_raw, d_out: [*c]__half_raw, N: usize, C: usize, H: usize, W: usize, outH: usize, outW: usize, kH: usize, kW: usize, sH: usize, sW: usize, pH: usize, pW: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoAvgPool2dForwardB(d_in: [*c]const __nv_bfloat16_raw, d_out: [*c]__nv_bfloat16_raw, N: usize, C: usize, H: usize, W: usize, outH: usize, outW: usize, kH: usize, kW: usize, sH: usize, sW: usize, pH: usize, pW: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoAvgPool2dForwardF(d_in: [*c]const f32, d_out: [*c]f32, N: usize, C: usize, H: usize, W: usize, outH: usize, outW: usize, kH: usize, kW: usize, sH: usize, sW: usize, pH: usize, pW: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoAvgPool2dForwardD(d_in: [*c]const f64, d_out: [*c]f64, N: usize, C: usize, H: usize, W: usize, outH: usize, outW: usize, kH: usize, kW: usize, sH: usize, sW: usize, pH: usize, pW: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoAvgPool2dBackwardH(d_in: [*c]const __half_raw, d_out: [*c]__half_raw, N: usize, C: usize, H: usize, W: usize, outH: usize, outW: usize, kH: usize, kW: usize, sH: usize, sW: usize, pH: usize, pW: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoAvgPool2dBackwardB(d_in: [*c]const __nv_bfloat16_raw, d_out: [*c]__nv_bfloat16_raw, N: usize, C: usize, H: usize, W: usize, outH: usize, outW: usize, kH: usize, kW: usize, sH: usize, sW: usize, pH: usize, pW: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoAvgPool2dBackwardF(d_in: [*c]const f32, d_out: [*c]f32, N: usize, C: usize, H: usize, W: usize, outH: usize, outW: usize, kH: usize, kW: usize, sH: usize, sW: usize, pH: usize, pW: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoAvgPool2dBackwardD(d_in: [*c]const f64, d_out: [*c]f64, N: usize, C: usize, H: usize, W: usize, outH: usize, outW: usize, kH: usize, kW: usize, sH: usize, sW: usize, pH: usize, pW: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEmbeddingForwardH(weight: [*c]__half_raw, indices: [*c]const usize, output: [*c]__half_raw, num_embeddings: usize, embedding_dim: usize, batch_size: usize, sequence_length: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEmbeddingForwardB(weight: [*c]__nv_bfloat16_raw, indices: [*c]const usize, output: [*c]__nv_bfloat16_raw, num_embeddings: usize, embedding_dim: usize, batch_size: usize, sequence_length: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEmbeddingForwardF(weight: [*c]f32, indices: [*c]const usize, output: [*c]f32, num_embeddings: usize, embedding_dim: usize, batch_size: usize, sequence_length: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEmbeddingForwardD(weight: [*c]f64, indices: [*c]const usize, output: [*c]f64, num_embeddings: usize, embedding_dim: usize, batch_size: usize, sequence_length: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEmbeddingBackwardH(grad_output: [*c]const __half_raw, indices: [*c]const usize, grad_weight: [*c]__half_raw, num_embeddings: usize, embedding_dim: usize, batch_size: usize, sequence_length: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEmbeddingBackwardB(grad_output: [*c]const __nv_bfloat16_raw, indices: [*c]const usize, grad_weight: [*c]__nv_bfloat16_raw, num_embeddings: usize, embedding_dim: usize, batch_size: usize, sequence_length: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEmbeddingBackwardF(grad_output: [*c]const f32, indices: [*c]const usize, grad_weight: [*c]f32, num_embeddings: usize, embedding_dim: usize, batch_size: usize, sequence_length: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoEmbeddingBackwardD(grad_output: [*c]const f64, indices: [*c]const usize, grad_weight: [*c]f64, num_embeddings: usize, embedding_dim: usize, batch_size: usize, sequence_length: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGetItemH(x: [*c]const __half_raw, y: [*c]__half_raw, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, starts: [*c]const usize, starts_len: usize, steps: [*c]const usize, steps_len: usize, nd: usize, out_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGetItemB(x: [*c]const __nv_bfloat16_raw, y: [*c]__nv_bfloat16_raw, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, starts: [*c]const usize, starts_len: usize, steps: [*c]const usize, steps_len: usize, nd: usize, out_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGetItemF(x: [*c]const f32, y: [*c]f32, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, starts: [*c]const usize, starts_len: usize, steps: [*c]const usize, steps_len: usize, nd: usize, out_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGetItemD(x: [*c]const f64, y: [*c]f64, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, starts: [*c]const usize, starts_len: usize, steps: [*c]const usize, steps_len: usize, nd: usize, out_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSetItemH(src: [*c]const __half_raw, dest: [*c]__half_raw, src_shape: [*c]const usize, src_shape_len: usize, dest_shape: [*c]const usize, dest_shape_len: usize, src_strides: [*c]const usize, src_strides_len: usize, dest_strides: [*c]const usize, dest_strides_len: usize, starts: [*c]const usize, starts_len: usize, steps: [*c]const usize, steps_len: usize, nd: usize, src_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSetItemB(src: [*c]const __nv_bfloat16_raw, dest: [*c]__nv_bfloat16_raw, src_shape: [*c]const usize, src_shape_len: usize, dest_shape: [*c]const usize, dest_shape_len: usize, src_strides: [*c]const usize, src_strides_len: usize, dest_strides: [*c]const usize, dest_strides_len: usize, starts: [*c]const usize, starts_len: usize, steps: [*c]const usize, steps_len: usize, nd: usize, src_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSetItemF(src: [*c]const f32, dest: [*c]f32, src_shape: [*c]const usize, src_shape_len: usize, dest_shape: [*c]const usize, dest_shape_len: usize, src_strides: [*c]const usize, src_strides_len: usize, dest_strides: [*c]const usize, dest_strides_len: usize, starts: [*c]const usize, starts_len: usize, steps: [*c]const usize, steps_len: usize, nd: usize, src_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoSetItemD(src: [*c]const f64, dest: [*c]f64, src_shape: [*c]const usize, src_shape_len: usize, dest_shape: [*c]const usize, dest_shape_len: usize, src_strides: [*c]const usize, src_strides_len: usize, dest_strides: [*c]const usize, dest_strides_len: usize, starts: [*c]const usize, starts_len: usize, steps: [*c]const usize, steps_len: usize, nd: usize, src_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGetItemGradH(gy: [*c]const __half_raw, gx: [*c]__half_raw, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, starts: [*c]const usize, starts_len: usize, steps: [*c]const usize, steps_len: usize, nd: usize, out_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGetItemGradB(gy: [*c]const __nv_bfloat16_raw, gx: [*c]__nv_bfloat16_raw, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, starts: [*c]const usize, starts_len: usize, steps: [*c]const usize, steps_len: usize, nd: usize, out_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGetItemGradF(gy: [*c]const f32, gx: [*c]f32, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, starts: [*c]const usize, starts_len: usize, steps: [*c]const usize, steps_len: usize, nd: usize, out_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoGetItemGradD(gy: [*c]const f64, gx: [*c]f64, in_shape: [*c]const usize, in_shape_len: usize, out_shape: [*c]const usize, out_shape_len: usize, in_strides: [*c]const usize, in_strides_len: usize, out_strides: [*c]const usize, out_strides_len: usize, starts: [*c]const usize, starts_len: usize, steps: [*c]const usize, steps_len: usize, nd: usize, out_size: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoOneHotH(indices: [*c]const usize, one_hot: [*c]__half_raw, batch_size: usize, num_classes: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoOneHotB(indices: [*c]const usize, one_hot: [*c]__nv_bfloat16_raw, batch_size: usize, num_classes: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoOneHotF(indices: [*c]const usize, one_hot: [*c]f32, batch_size: usize, num_classes: usize, stream: cudaStream_t) cudaError_t;
pub extern fn tomoOneHotD(indices: [*c]const usize, one_hot: [*c]f64, batch_size: usize, num_classes: usize, stream: cudaStream_t) cudaError_t;
pub const __llvm__ = @as(c_int, 1);
pub const __clang__ = @as(c_int, 1);
pub const __clang_major__ = @as(c_int, 19);
pub const __clang_minor__ = @as(c_int, 1);
pub const __clang_patchlevel__ = @as(c_int, 7);
pub const __clang_version__ = "19.1.7 (https://github.com/ziglang/zig-bootstrap 11e20c3717ffdc8b2d44aaea24703c434bbdea6c)";
pub const __GNUC__ = @as(c_int, 4);
pub const __GNUC_MINOR__ = @as(c_int, 2);
pub const __GNUC_PATCHLEVEL__ = @as(c_int, 1);
pub const __GXX_ABI_VERSION = @as(c_int, 1002);
pub const __ATOMIC_RELAXED = @as(c_int, 0);
pub const __ATOMIC_CONSUME = @as(c_int, 1);
pub const __ATOMIC_ACQUIRE = @as(c_int, 2);
pub const __ATOMIC_RELEASE = @as(c_int, 3);
pub const __ATOMIC_ACQ_REL = @as(c_int, 4);
pub const __ATOMIC_SEQ_CST = @as(c_int, 5);
pub const __MEMORY_SCOPE_SYSTEM = @as(c_int, 0);
pub const __MEMORY_SCOPE_DEVICE = @as(c_int, 1);
pub const __MEMORY_SCOPE_WRKGRP = @as(c_int, 2);
pub const __MEMORY_SCOPE_WVFRNT = @as(c_int, 3);
pub const __MEMORY_SCOPE_SINGLE = @as(c_int, 4);
pub const __OPENCL_MEMORY_SCOPE_WORK_ITEM = @as(c_int, 0);
pub const __OPENCL_MEMORY_SCOPE_WORK_GROUP = @as(c_int, 1);
pub const __OPENCL_MEMORY_SCOPE_DEVICE = @as(c_int, 2);
pub const __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES = @as(c_int, 3);
pub const __OPENCL_MEMORY_SCOPE_SUB_GROUP = @as(c_int, 4);
pub const __FPCLASS_SNAN = @as(c_int, 0x0001);
pub const __FPCLASS_QNAN = @as(c_int, 0x0002);
pub const __FPCLASS_NEGINF = @as(c_int, 0x0004);
pub const __FPCLASS_NEGNORMAL = @as(c_int, 0x0008);
pub const __FPCLASS_NEGSUBNORMAL = @as(c_int, 0x0010);
pub const __FPCLASS_NEGZERO = @as(c_int, 0x0020);
pub const __FPCLASS_POSZERO = @as(c_int, 0x0040);
pub const __FPCLASS_POSSUBNORMAL = @as(c_int, 0x0080);
pub const __FPCLASS_POSNORMAL = @as(c_int, 0x0100);
pub const __FPCLASS_POSINF = @as(c_int, 0x0200);
pub const __PRAGMA_REDEFINE_EXTNAME = @as(c_int, 1);
pub const __VERSION__ = "Clang 19.1.7 (https://github.com/ziglang/zig-bootstrap 11e20c3717ffdc8b2d44aaea24703c434bbdea6c)";
pub const __GXX_TYPEINFO_EQUALITY_INLINE = @as(c_int, 0);
pub const __OBJC_BOOL_IS_BOOL = @as(c_int, 0);
pub const __CONSTANT_CFSTRINGS__ = @as(c_int, 1);
pub const __SEH__ = @as(c_int, 1);
pub const __clang_literal_encoding__ = "UTF-8";
pub const __clang_wide_literal_encoding__ = "UTF-16";
pub const __ORDER_LITTLE_ENDIAN__ = @as(c_int, 1234);
pub const __ORDER_BIG_ENDIAN__ = @as(c_int, 4321);
pub const __ORDER_PDP_ENDIAN__ = @as(c_int, 3412);
pub const __BYTE_ORDER__ = __ORDER_LITTLE_ENDIAN__;
pub const __LITTLE_ENDIAN__ = @as(c_int, 1);
pub const __CHAR_BIT__ = @as(c_int, 8);
pub const __BOOL_WIDTH__ = @as(c_int, 8);
pub const __SHRT_WIDTH__ = @as(c_int, 16);
pub const __INT_WIDTH__ = @as(c_int, 32);
pub const __LONG_WIDTH__ = @as(c_int, 32);
pub const __LLONG_WIDTH__ = @as(c_int, 64);
pub const __BITINT_MAXWIDTH__ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 8388608, .decimal);
pub const __SCHAR_MAX__ = @as(c_int, 127);
pub const __SHRT_MAX__ = @as(c_int, 32767);
pub const __INT_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 2147483647, .decimal);
pub const __LONG_MAX__ = @as(c_long, 2147483647);
pub const __LONG_LONG_MAX__ = @as(c_longlong, 9223372036854775807);
pub const __WCHAR_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 65535, .decimal);
pub const __WCHAR_WIDTH__ = @as(c_int, 16);
pub const __WINT_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 65535, .decimal);
pub const __WINT_WIDTH__ = @as(c_int, 16);
pub const __INTMAX_MAX__ = @as(c_longlong, 9223372036854775807);
pub const __INTMAX_WIDTH__ = @as(c_int, 64);
pub const __SIZE_MAX__ = @as(c_ulonglong, 18446744073709551615);
pub const __SIZE_WIDTH__ = @as(c_int, 64);
pub const __UINTMAX_MAX__ = @as(c_ulonglong, 18446744073709551615);
pub const __UINTMAX_WIDTH__ = @as(c_int, 64);
pub const __PTRDIFF_MAX__ = @as(c_longlong, 9223372036854775807);
pub const __PTRDIFF_WIDTH__ = @as(c_int, 64);
pub const __INTPTR_MAX__ = @as(c_longlong, 9223372036854775807);
pub const __INTPTR_WIDTH__ = @as(c_int, 64);
pub const __UINTPTR_MAX__ = @as(c_ulonglong, 18446744073709551615);
pub const __UINTPTR_WIDTH__ = @as(c_int, 64);
pub const __SIZEOF_DOUBLE__ = @as(c_int, 8);
pub const __SIZEOF_FLOAT__ = @as(c_int, 4);
pub const __SIZEOF_INT__ = @as(c_int, 4);
pub const __SIZEOF_LONG__ = @as(c_int, 4);
pub const __SIZEOF_LONG_DOUBLE__ = @as(c_int, 16);
pub const __SIZEOF_LONG_LONG__ = @as(c_int, 8);
pub const __SIZEOF_POINTER__ = @as(c_int, 8);
pub const __SIZEOF_SHORT__ = @as(c_int, 2);
pub const __SIZEOF_PTRDIFF_T__ = @as(c_int, 8);
pub const __SIZEOF_SIZE_T__ = @as(c_int, 8);
pub const __SIZEOF_WCHAR_T__ = @as(c_int, 2);
pub const __SIZEOF_WINT_T__ = @as(c_int, 2);
pub const __SIZEOF_INT128__ = @as(c_int, 16);
pub const __INTMAX_TYPE__ = c_longlong;
pub const __INTMAX_FMTd__ = "lld";
pub const __INTMAX_FMTi__ = "lli";
pub const __INTMAX_C_SUFFIX__ = @compileError("unable to translate macro: undefined identifier `LL`");
// (no file):95:9
pub const __UINTMAX_TYPE__ = c_ulonglong;
pub const __UINTMAX_FMTo__ = "llo";
pub const __UINTMAX_FMTu__ = "llu";
pub const __UINTMAX_FMTx__ = "llx";
pub const __UINTMAX_FMTX__ = "llX";
pub const __UINTMAX_C_SUFFIX__ = @compileError("unable to translate macro: undefined identifier `ULL`");
// (no file):101:9
pub const __PTRDIFF_TYPE__ = c_longlong;
pub const __PTRDIFF_FMTd__ = "lld";
pub const __PTRDIFF_FMTi__ = "lli";
pub const __INTPTR_TYPE__ = c_longlong;
pub const __INTPTR_FMTd__ = "lld";
pub const __INTPTR_FMTi__ = "lli";
pub const __SIZE_TYPE__ = c_ulonglong;
pub const __SIZE_FMTo__ = "llo";
pub const __SIZE_FMTu__ = "llu";
pub const __SIZE_FMTx__ = "llx";
pub const __SIZE_FMTX__ = "llX";
pub const __WCHAR_TYPE__ = c_ushort;
pub const __WINT_TYPE__ = c_ushort;
pub const __SIG_ATOMIC_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 2147483647, .decimal);
pub const __SIG_ATOMIC_WIDTH__ = @as(c_int, 32);
pub const __CHAR16_TYPE__ = c_ushort;
pub const __CHAR32_TYPE__ = c_uint;
pub const __UINTPTR_TYPE__ = c_ulonglong;
pub const __UINTPTR_FMTo__ = "llo";
pub const __UINTPTR_FMTu__ = "llu";
pub const __UINTPTR_FMTx__ = "llx";
pub const __UINTPTR_FMTX__ = "llX";
pub const __FLT16_DENORM_MIN__ = @as(f16, 5.9604644775390625e-8);
pub const __FLT16_NORM_MAX__ = @as(f16, 6.5504e+4);
pub const __FLT16_HAS_DENORM__ = @as(c_int, 1);
pub const __FLT16_DIG__ = @as(c_int, 3);
pub const __FLT16_DECIMAL_DIG__ = @as(c_int, 5);
pub const __FLT16_EPSILON__ = @as(f16, 9.765625e-4);
pub const __FLT16_HAS_INFINITY__ = @as(c_int, 1);
pub const __FLT16_HAS_QUIET_NAN__ = @as(c_int, 1);
pub const __FLT16_MANT_DIG__ = @as(c_int, 11);
pub const __FLT16_MAX_10_EXP__ = @as(c_int, 4);
pub const __FLT16_MAX_EXP__ = @as(c_int, 16);
pub const __FLT16_MAX__ = @as(f16, 6.5504e+4);
pub const __FLT16_MIN_10_EXP__ = -@as(c_int, 4);
pub const __FLT16_MIN_EXP__ = -@as(c_int, 13);
pub const __FLT16_MIN__ = @as(f16, 6.103515625e-5);
pub const __FLT_DENORM_MIN__ = @as(f32, 1.40129846e-45);
pub const __FLT_NORM_MAX__ = @as(f32, 3.40282347e+38);
pub const __FLT_HAS_DENORM__ = @as(c_int, 1);
pub const __FLT_DIG__ = @as(c_int, 6);
pub const __FLT_DECIMAL_DIG__ = @as(c_int, 9);
pub const __FLT_EPSILON__ = @as(f32, 1.19209290e-7);
pub const __FLT_HAS_INFINITY__ = @as(c_int, 1);
pub const __FLT_HAS_QUIET_NAN__ = @as(c_int, 1);
pub const __FLT_MANT_DIG__ = @as(c_int, 24);
pub const __FLT_MAX_10_EXP__ = @as(c_int, 38);
pub const __FLT_MAX_EXP__ = @as(c_int, 128);
pub const __FLT_MAX__ = @as(f32, 3.40282347e+38);
pub const __FLT_MIN_10_EXP__ = -@as(c_int, 37);
pub const __FLT_MIN_EXP__ = -@as(c_int, 125);
pub const __FLT_MIN__ = @as(f32, 1.17549435e-38);
pub const __DBL_DENORM_MIN__ = @as(f64, 4.9406564584124654e-324);
pub const __DBL_NORM_MAX__ = @as(f64, 1.7976931348623157e+308);
pub const __DBL_HAS_DENORM__ = @as(c_int, 1);
pub const __DBL_DIG__ = @as(c_int, 15);
pub const __DBL_DECIMAL_DIG__ = @as(c_int, 17);
pub const __DBL_EPSILON__ = @as(f64, 2.2204460492503131e-16);
pub const __DBL_HAS_INFINITY__ = @as(c_int, 1);
pub const __DBL_HAS_QUIET_NAN__ = @as(c_int, 1);
pub const __DBL_MANT_DIG__ = @as(c_int, 53);
pub const __DBL_MAX_10_EXP__ = @as(c_int, 308);
pub const __DBL_MAX_EXP__ = @as(c_int, 1024);
pub const __DBL_MAX__ = @as(f64, 1.7976931348623157e+308);
pub const __DBL_MIN_10_EXP__ = -@as(c_int, 307);
pub const __DBL_MIN_EXP__ = -@as(c_int, 1021);
pub const __DBL_MIN__ = @as(f64, 2.2250738585072014e-308);
pub const __LDBL_DENORM_MIN__ = @as(c_longdouble, 3.64519953188247460253e-4951);
pub const __LDBL_NORM_MAX__ = @as(c_longdouble, 1.18973149535723176502e+4932);
pub const __LDBL_HAS_DENORM__ = @as(c_int, 1);
pub const __LDBL_DIG__ = @as(c_int, 18);
pub const __LDBL_DECIMAL_DIG__ = @as(c_int, 21);
pub const __LDBL_EPSILON__ = @as(c_longdouble, 1.08420217248550443401e-19);
pub const __LDBL_HAS_INFINITY__ = @as(c_int, 1);
pub const __LDBL_HAS_QUIET_NAN__ = @as(c_int, 1);
pub const __LDBL_MANT_DIG__ = @as(c_int, 64);
pub const __LDBL_MAX_10_EXP__ = @as(c_int, 4932);
pub const __LDBL_MAX_EXP__ = @as(c_int, 16384);
pub const __LDBL_MAX__ = @as(c_longdouble, 1.18973149535723176502e+4932);
pub const __LDBL_MIN_10_EXP__ = -@as(c_int, 4931);
pub const __LDBL_MIN_EXP__ = -@as(c_int, 16381);
pub const __LDBL_MIN__ = @as(c_longdouble, 3.36210314311209350626e-4932);
pub const __POINTER_WIDTH__ = @as(c_int, 64);
pub const __BIGGEST_ALIGNMENT__ = @as(c_int, 16);
pub const __WCHAR_UNSIGNED__ = @as(c_int, 1);
pub const __WINT_UNSIGNED__ = @as(c_int, 1);
pub const __INT8_TYPE__ = i8;
pub const __INT8_FMTd__ = "hhd";
pub const __INT8_FMTi__ = "hhi";
pub const __INT8_C_SUFFIX__ = "";
pub const __INT16_TYPE__ = c_short;
pub const __INT16_FMTd__ = "hd";
pub const __INT16_FMTi__ = "hi";
pub const __INT16_C_SUFFIX__ = "";
pub const __INT32_TYPE__ = c_int;
pub const __INT32_FMTd__ = "d";
pub const __INT32_FMTi__ = "i";
pub const __INT32_C_SUFFIX__ = "";
pub const __INT64_TYPE__ = c_longlong;
pub const __INT64_FMTd__ = "lld";
pub const __INT64_FMTi__ = "lli";
pub const __INT64_C_SUFFIX__ = @compileError("unable to translate macro: undefined identifier `LL`");
// (no file):203:9
pub const __UINT8_TYPE__ = u8;
pub const __UINT8_FMTo__ = "hho";
pub const __UINT8_FMTu__ = "hhu";
pub const __UINT8_FMTx__ = "hhx";
pub const __UINT8_FMTX__ = "hhX";
pub const __UINT8_C_SUFFIX__ = "";
pub const __UINT8_MAX__ = @as(c_int, 255);
pub const __INT8_MAX__ = @as(c_int, 127);
pub const __UINT16_TYPE__ = c_ushort;
pub const __UINT16_FMTo__ = "ho";
pub const __UINT16_FMTu__ = "hu";
pub const __UINT16_FMTx__ = "hx";
pub const __UINT16_FMTX__ = "hX";
pub const __UINT16_C_SUFFIX__ = "";
pub const __UINT16_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 65535, .decimal);
pub const __INT16_MAX__ = @as(c_int, 32767);
pub const __UINT32_TYPE__ = c_uint;
pub const __UINT32_FMTo__ = "o";
pub const __UINT32_FMTu__ = "u";
pub const __UINT32_FMTx__ = "x";
pub const __UINT32_FMTX__ = "X";
pub const __UINT32_C_SUFFIX__ = @compileError("unable to translate macro: undefined identifier `U`");
// (no file):225:9
pub const __UINT32_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_uint, 4294967295, .decimal);
pub const __INT32_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 2147483647, .decimal);
pub const __UINT64_TYPE__ = c_ulonglong;
pub const __UINT64_FMTo__ = "llo";
pub const __UINT64_FMTu__ = "llu";
pub const __UINT64_FMTx__ = "llx";
pub const __UINT64_FMTX__ = "llX";
pub const __UINT64_C_SUFFIX__ = @compileError("unable to translate macro: undefined identifier `ULL`");
// (no file):233:9
pub const __UINT64_MAX__ = @as(c_ulonglong, 18446744073709551615);
pub const __INT64_MAX__ = @as(c_longlong, 9223372036854775807);
pub const __INT_LEAST8_TYPE__ = i8;
pub const __INT_LEAST8_MAX__ = @as(c_int, 127);
pub const __INT_LEAST8_WIDTH__ = @as(c_int, 8);
pub const __INT_LEAST8_FMTd__ = "hhd";
pub const __INT_LEAST8_FMTi__ = "hhi";
pub const __UINT_LEAST8_TYPE__ = u8;
pub const __UINT_LEAST8_MAX__ = @as(c_int, 255);
pub const __UINT_LEAST8_FMTo__ = "hho";
pub const __UINT_LEAST8_FMTu__ = "hhu";
pub const __UINT_LEAST8_FMTx__ = "hhx";
pub const __UINT_LEAST8_FMTX__ = "hhX";
pub const __INT_LEAST16_TYPE__ = c_short;
pub const __INT_LEAST16_MAX__ = @as(c_int, 32767);
pub const __INT_LEAST16_WIDTH__ = @as(c_int, 16);
pub const __INT_LEAST16_FMTd__ = "hd";
pub const __INT_LEAST16_FMTi__ = "hi";
pub const __UINT_LEAST16_TYPE__ = c_ushort;
pub const __UINT_LEAST16_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 65535, .decimal);
pub const __UINT_LEAST16_FMTo__ = "ho";
pub const __UINT_LEAST16_FMTu__ = "hu";
pub const __UINT_LEAST16_FMTx__ = "hx";
pub const __UINT_LEAST16_FMTX__ = "hX";
pub const __INT_LEAST32_TYPE__ = c_int;
pub const __INT_LEAST32_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 2147483647, .decimal);
pub const __INT_LEAST32_WIDTH__ = @as(c_int, 32);
pub const __INT_LEAST32_FMTd__ = "d";
pub const __INT_LEAST32_FMTi__ = "i";
pub const __UINT_LEAST32_TYPE__ = c_uint;
pub const __UINT_LEAST32_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_uint, 4294967295, .decimal);
pub const __UINT_LEAST32_FMTo__ = "o";
pub const __UINT_LEAST32_FMTu__ = "u";
pub const __UINT_LEAST32_FMTx__ = "x";
pub const __UINT_LEAST32_FMTX__ = "X";
pub const __INT_LEAST64_TYPE__ = c_longlong;
pub const __INT_LEAST64_MAX__ = @as(c_longlong, 9223372036854775807);
pub const __INT_LEAST64_WIDTH__ = @as(c_int, 64);
pub const __INT_LEAST64_FMTd__ = "lld";
pub const __INT_LEAST64_FMTi__ = "lli";
pub const __UINT_LEAST64_TYPE__ = c_ulonglong;
pub const __UINT_LEAST64_MAX__ = @as(c_ulonglong, 18446744073709551615);
pub const __UINT_LEAST64_FMTo__ = "llo";
pub const __UINT_LEAST64_FMTu__ = "llu";
pub const __UINT_LEAST64_FMTx__ = "llx";
pub const __UINT_LEAST64_FMTX__ = "llX";
pub const __INT_FAST8_TYPE__ = i8;
pub const __INT_FAST8_MAX__ = @as(c_int, 127);
pub const __INT_FAST8_WIDTH__ = @as(c_int, 8);
pub const __INT_FAST8_FMTd__ = "hhd";
pub const __INT_FAST8_FMTi__ = "hhi";
pub const __UINT_FAST8_TYPE__ = u8;
pub const __UINT_FAST8_MAX__ = @as(c_int, 255);
pub const __UINT_FAST8_FMTo__ = "hho";
pub const __UINT_FAST8_FMTu__ = "hhu";
pub const __UINT_FAST8_FMTx__ = "hhx";
pub const __UINT_FAST8_FMTX__ = "hhX";
pub const __INT_FAST16_TYPE__ = c_short;
pub const __INT_FAST16_MAX__ = @as(c_int, 32767);
pub const __INT_FAST16_WIDTH__ = @as(c_int, 16);
pub const __INT_FAST16_FMTd__ = "hd";
pub const __INT_FAST16_FMTi__ = "hi";
pub const __UINT_FAST16_TYPE__ = c_ushort;
pub const __UINT_FAST16_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 65535, .decimal);
pub const __UINT_FAST16_FMTo__ = "ho";
pub const __UINT_FAST16_FMTu__ = "hu";
pub const __UINT_FAST16_FMTx__ = "hx";
pub const __UINT_FAST16_FMTX__ = "hX";
pub const __INT_FAST32_TYPE__ = c_int;
pub const __INT_FAST32_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 2147483647, .decimal);
pub const __INT_FAST32_WIDTH__ = @as(c_int, 32);
pub const __INT_FAST32_FMTd__ = "d";
pub const __INT_FAST32_FMTi__ = "i";
pub const __UINT_FAST32_TYPE__ = c_uint;
pub const __UINT_FAST32_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_uint, 4294967295, .decimal);
pub const __UINT_FAST32_FMTo__ = "o";
pub const __UINT_FAST32_FMTu__ = "u";
pub const __UINT_FAST32_FMTx__ = "x";
pub const __UINT_FAST32_FMTX__ = "X";
pub const __INT_FAST64_TYPE__ = c_longlong;
pub const __INT_FAST64_MAX__ = @as(c_longlong, 9223372036854775807);
pub const __INT_FAST64_WIDTH__ = @as(c_int, 64);
pub const __INT_FAST64_FMTd__ = "lld";
pub const __INT_FAST64_FMTi__ = "lli";
pub const __UINT_FAST64_TYPE__ = c_ulonglong;
pub const __UINT_FAST64_MAX__ = @as(c_ulonglong, 18446744073709551615);
pub const __UINT_FAST64_FMTo__ = "llo";
pub const __UINT_FAST64_FMTu__ = "llu";
pub const __UINT_FAST64_FMTx__ = "llx";
pub const __UINT_FAST64_FMTX__ = "llX";
pub const __USER_LABEL_PREFIX__ = "";
pub const __FINITE_MATH_ONLY__ = @as(c_int, 0);
pub const __GNUC_STDC_INLINE__ = @as(c_int, 1);
pub const __GCC_ATOMIC_TEST_AND_SET_TRUEVAL = @as(c_int, 1);
pub const __GCC_DESTRUCTIVE_SIZE = @as(c_int, 64);
pub const __GCC_CONSTRUCTIVE_SIZE = @as(c_int, 64);
pub const __CLANG_ATOMIC_BOOL_LOCK_FREE = @as(c_int, 2);
pub const __CLANG_ATOMIC_CHAR_LOCK_FREE = @as(c_int, 2);
pub const __CLANG_ATOMIC_CHAR16_T_LOCK_FREE = @as(c_int, 2);
pub const __CLANG_ATOMIC_CHAR32_T_LOCK_FREE = @as(c_int, 2);
pub const __CLANG_ATOMIC_WCHAR_T_LOCK_FREE = @as(c_int, 2);
pub const __CLANG_ATOMIC_SHORT_LOCK_FREE = @as(c_int, 2);
pub const __CLANG_ATOMIC_INT_LOCK_FREE = @as(c_int, 2);
pub const __CLANG_ATOMIC_LONG_LOCK_FREE = @as(c_int, 2);
pub const __CLANG_ATOMIC_LLONG_LOCK_FREE = @as(c_int, 2);
pub const __CLANG_ATOMIC_POINTER_LOCK_FREE = @as(c_int, 2);
pub const __GCC_ATOMIC_BOOL_LOCK_FREE = @as(c_int, 2);
pub const __GCC_ATOMIC_CHAR_LOCK_FREE = @as(c_int, 2);
pub const __GCC_ATOMIC_CHAR16_T_LOCK_FREE = @as(c_int, 2);
pub const __GCC_ATOMIC_CHAR32_T_LOCK_FREE = @as(c_int, 2);
pub const __GCC_ATOMIC_WCHAR_T_LOCK_FREE = @as(c_int, 2);
pub const __GCC_ATOMIC_SHORT_LOCK_FREE = @as(c_int, 2);
pub const __GCC_ATOMIC_INT_LOCK_FREE = @as(c_int, 2);
pub const __GCC_ATOMIC_LONG_LOCK_FREE = @as(c_int, 2);
pub const __GCC_ATOMIC_LLONG_LOCK_FREE = @as(c_int, 2);
pub const __GCC_ATOMIC_POINTER_LOCK_FREE = @as(c_int, 2);
pub const __NO_INLINE__ = @as(c_int, 1);
pub const __PIC__ = @as(c_int, 2);
pub const __pic__ = @as(c_int, 2);
pub const __FLT_RADIX__ = @as(c_int, 2);
pub const __DECIMAL_DIG__ = __LDBL_DECIMAL_DIG__;
pub const __SSP_STRONG__ = @as(c_int, 2);
pub const __GCC_ASM_FLAG_OUTPUTS__ = @as(c_int, 1);
pub const __code_model_small__ = @as(c_int, 1);
pub const __amd64__ = @as(c_int, 1);
pub const __amd64 = @as(c_int, 1);
pub const __x86_64 = @as(c_int, 1);
pub const __x86_64__ = @as(c_int, 1);
pub const __SEG_GS = @as(c_int, 1);
pub const __SEG_FS = @as(c_int, 1);
pub const __seg_gs = @compileError("unable to translate macro: undefined identifier `address_space`");
// (no file):366:9
pub const __seg_fs = @compileError("unable to translate macro: undefined identifier `address_space`");
// (no file):367:9
pub const __corei7 = @as(c_int, 1);
pub const __corei7__ = @as(c_int, 1);
pub const __tune_corei7__ = @as(c_int, 1);
pub const __REGISTER_PREFIX__ = "";
pub const __NO_MATH_INLINES = @as(c_int, 1);
pub const __AES__ = @as(c_int, 1);
pub const __VAES__ = @as(c_int, 1);
pub const __PCLMUL__ = @as(c_int, 1);
pub const __VPCLMULQDQ__ = @as(c_int, 1);
pub const __LAHF_SAHF__ = @as(c_int, 1);
pub const __LZCNT__ = @as(c_int, 1);
pub const __RDRND__ = @as(c_int, 1);
pub const __FSGSBASE__ = @as(c_int, 1);
pub const __BMI__ = @as(c_int, 1);
pub const __BMI2__ = @as(c_int, 1);
pub const __POPCNT__ = @as(c_int, 1);
pub const __PRFCHW__ = @as(c_int, 1);
pub const __RDSEED__ = @as(c_int, 1);
pub const __ADX__ = @as(c_int, 1);
pub const __MOVBE__ = @as(c_int, 1);
pub const __FMA__ = @as(c_int, 1);
pub const __F16C__ = @as(c_int, 1);
pub const __GFNI__ = @as(c_int, 1);
pub const __SHA__ = @as(c_int, 1);
pub const __FXSR__ = @as(c_int, 1);
pub const __XSAVE__ = @as(c_int, 1);
pub const __XSAVEOPT__ = @as(c_int, 1);
pub const __XSAVEC__ = @as(c_int, 1);
pub const __XSAVES__ = @as(c_int, 1);
pub const __CLFLUSHOPT__ = @as(c_int, 1);
pub const __CLWB__ = @as(c_int, 1);
pub const __SHSTK__ = @as(c_int, 1);
pub const __KL__ = @as(c_int, 1);
pub const __WIDEKL__ = @as(c_int, 1);
pub const __RDPID__ = @as(c_int, 1);
pub const __WAITPKG__ = @as(c_int, 1);
pub const __MOVDIRI__ = @as(c_int, 1);
pub const __MOVDIR64B__ = @as(c_int, 1);
pub const __PTWRITE__ = @as(c_int, 1);
pub const __INVPCID__ = @as(c_int, 1);
pub const __HRESET__ = @as(c_int, 1);
pub const __AVXVNNI__ = @as(c_int, 1);
pub const __SERIALIZE__ = @as(c_int, 1);
pub const __CRC32__ = @as(c_int, 1);
pub const __AVX2__ = @as(c_int, 1);
pub const __AVX__ = @as(c_int, 1);
pub const __SSE4_2__ = @as(c_int, 1);
pub const __SSE4_1__ = @as(c_int, 1);
pub const __SSSE3__ = @as(c_int, 1);
pub const __SSE3__ = @as(c_int, 1);
pub const __SSE2__ = @as(c_int, 1);
pub const __SSE2_MATH__ = @as(c_int, 1);
pub const __SSE__ = @as(c_int, 1);
pub const __SSE_MATH__ = @as(c_int, 1);
pub const __MMX__ = @as(c_int, 1);
pub const __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 = @as(c_int, 1);
pub const __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 = @as(c_int, 1);
pub const __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 = @as(c_int, 1);
pub const __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 = @as(c_int, 1);
pub const __GCC_HAVE_SYNC_COMPARE_AND_SWAP_16 = @as(c_int, 1);
pub const __SIZEOF_FLOAT128__ = @as(c_int, 16);
pub const _WIN32 = @as(c_int, 1);
pub const _WIN64 = @as(c_int, 1);
pub const WIN32 = @as(c_int, 1);
pub const __WIN32 = @as(c_int, 1);
pub const __WIN32__ = @as(c_int, 1);
pub const WINNT = @as(c_int, 1);
pub const __WINNT = @as(c_int, 1);
pub const __WINNT__ = @as(c_int, 1);
pub const WIN64 = @as(c_int, 1);
pub const __WIN64 = @as(c_int, 1);
pub const __WIN64__ = @as(c_int, 1);
pub const __MINGW64__ = @as(c_int, 1);
pub const __MSVCRT__ = @as(c_int, 1);
pub const __MINGW32__ = @as(c_int, 1);
pub const __declspec = @compileError("unable to translate C expr: unexpected token '__attribute__'");
// (no file):443:9
pub const _cdecl = @compileError("unable to translate macro: undefined identifier `__cdecl__`");
// (no file):444:9
pub const __cdecl = @compileError("unable to translate macro: undefined identifier `__cdecl__`");
// (no file):445:9
pub const _stdcall = @compileError("unable to translate macro: undefined identifier `__stdcall__`");
// (no file):446:9
pub const __stdcall = @compileError("unable to translate macro: undefined identifier `__stdcall__`");
// (no file):447:9
pub const _fastcall = @compileError("unable to translate macro: undefined identifier `__fastcall__`");
// (no file):448:9
pub const __fastcall = @compileError("unable to translate macro: undefined identifier `__fastcall__`");
// (no file):449:9
pub const _thiscall = @compileError("unable to translate macro: undefined identifier `__thiscall__`");
// (no file):450:9
pub const __thiscall = @compileError("unable to translate macro: undefined identifier `__thiscall__`");
// (no file):451:9
pub const _pascal = @compileError("unable to translate macro: undefined identifier `__pascal__`");
// (no file):452:9
pub const __pascal = @compileError("unable to translate macro: undefined identifier `__pascal__`");
// (no file):453:9
pub const __STDC__ = @as(c_int, 1);
pub const __STDC_HOSTED__ = @as(c_int, 1);
pub const __STDC_VERSION__ = @as(c_long, 201710);
pub const __STDC_UTF_16__ = @as(c_int, 1);
pub const __STDC_UTF_32__ = @as(c_int, 1);
pub const __STDC_EMBED_NOT_FOUND__ = @as(c_int, 0);
pub const __STDC_EMBED_FOUND__ = @as(c_int, 1);
pub const __STDC_EMBED_EMPTY__ = @as(c_int, 2);
pub const __MSVCRT_VERSION__ = @as(c_int, 0xE00);
pub const _WIN32_WINNT = @as(c_int, 0x0a00);
pub const _LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS = @as(c_int, 1);
pub const _LIBCPP_HAS_NO_VENDOR_AVAILABILITY_ANNOTATIONS = @as(c_int, 1);
pub const _LIBCXXABI_DISABLE_VISIBILITY_ANNOTATIONS = @as(c_int, 1);
pub const _LIBCPP_PSTL_BACKEND_SERIAL = @as(c_int, 1);
pub const _LIBCPP_ABI_VERSION = @as(c_int, 1);
pub const _LIBCPP_ABI_NAMESPACE = @compileError("unable to translate macro: undefined identifier `__1`");
// (no file):470:9
pub const _LIBCPP_HARDENING_MODE = @compileError("unable to translate macro: undefined identifier `_LIBCPP_HARDENING_MODE_DEBUG`");
// (no file):471:9
pub const _DEBUG = @as(c_int, 1);
pub const __cuda_cuda_h__ = "";
pub const _LIBCPP_STDLIB_H = "";
pub const _LIBCPP___CONFIG = "";
pub const _LIBCPP___CONFIGURATION_ABI_H = "";
pub const _LIBCPP___CONFIGURATION_COMPILER_H = "";
pub const _LIBCPP_COMPILER_CLANG_BASED = "";
pub const _LIBCPP_CLANG_VER = (__clang_major__ * @as(c_int, 100)) + __clang_minor__;
pub const _LIBCPP___CONFIGURATION_PLATFORM_H = "";
pub const _LIBCPP_OBJECT_FORMAT_COFF = @as(c_int, 1);
pub const _LIBCPP_LITTLE_ENDIAN = "";
pub const _LIBCPP_ABI_LLVM18_NO_UNIQUE_ADDRESS = @compileError("unable to translate macro: undefined identifier `__abi_tag__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\libcxx\include/__configuration/abi.h:132:9
pub const _LIBCPP___CONFIGURATION_AVAILABILITY_H = "";
pub const _LIBCPP___CONFIGURATION_LANGUAGE_H = "";
pub const _LIBCPP_HAS_NO_RTTI = "";
pub const _LIBCPP_HAS_NO_EXCEPTIONS = "";
pub const _LIBCPP_INTRODUCED_IN_LLVM_19 = @as(c_int, 1);
pub const _LIBCPP_INTRODUCED_IN_LLVM_19_ATTRIBUTE = "";
pub const _LIBCPP_INTRODUCED_IN_LLVM_18 = @as(c_int, 1);
pub const _LIBCPP_INTRODUCED_IN_LLVM_18_ATTRIBUTE = "";
pub const _LIBCPP_INTRODUCED_IN_LLVM_17 = @as(c_int, 1);
pub const _LIBCPP_INTRODUCED_IN_LLVM_17_ATTRIBUTE = "";
pub const _LIBCPP_INTRODUCED_IN_LLVM_16 = @as(c_int, 1);
pub const _LIBCPP_INTRODUCED_IN_LLVM_16_ATTRIBUTE = "";
pub const _LIBCPP_INTRODUCED_IN_LLVM_15 = @as(c_int, 1);
pub const _LIBCPP_INTRODUCED_IN_LLVM_15_ATTRIBUTE = "";
pub const _LIBCPP_INTRODUCED_IN_LLVM_14 = @as(c_int, 1);
pub const _LIBCPP_INTRODUCED_IN_LLVM_14_ATTRIBUTE = "";
pub const _LIBCPP_INTRODUCED_IN_LLVM_13 = @as(c_int, 1);
pub const _LIBCPP_INTRODUCED_IN_LLVM_13_ATTRIBUTE = "";
pub const _LIBCPP_INTRODUCED_IN_LLVM_12 = @as(c_int, 1);
pub const _LIBCPP_INTRODUCED_IN_LLVM_12_ATTRIBUTE = "";
pub const _LIBCPP_INTRODUCED_IN_LLVM_11 = @as(c_int, 1);
pub const _LIBCPP_INTRODUCED_IN_LLVM_11_ATTRIBUTE = "";
pub const _LIBCPP_INTRODUCED_IN_LLVM_10 = @as(c_int, 1);
pub const _LIBCPP_INTRODUCED_IN_LLVM_10_ATTRIBUTE = "";
pub const _LIBCPP_INTRODUCED_IN_LLVM_9 = @as(c_int, 1);
pub const _LIBCPP_INTRODUCED_IN_LLVM_9_ATTRIBUTE = "";
pub const _LIBCPP_INTRODUCED_IN_LLVM_9_ATTRIBUTE_PUSH = "";
pub const _LIBCPP_INTRODUCED_IN_LLVM_9_ATTRIBUTE_POP = "";
pub const _LIBCPP_INTRODUCED_IN_LLVM_8 = @as(c_int, 1);
pub const _LIBCPP_INTRODUCED_IN_LLVM_8_ATTRIBUTE = "";
pub const _LIBCPP_INTRODUCED_IN_LLVM_4 = @as(c_int, 1);
pub const _LIBCPP_INTRODUCED_IN_LLVM_4_ATTRIBUTE = "";
pub const _LIBCPP_AVAILABILITY_HAS_BAD_OPTIONAL_ACCESS = _LIBCPP_INTRODUCED_IN_LLVM_4;
pub const _LIBCPP_AVAILABILITY_BAD_OPTIONAL_ACCESS = "";
pub const _LIBCPP_AVAILABILITY_HAS_BAD_VARIANT_ACCESS = _LIBCPP_INTRODUCED_IN_LLVM_4;
pub const _LIBCPP_AVAILABILITY_BAD_VARIANT_ACCESS = "";
pub const _LIBCPP_AVAILABILITY_HAS_BAD_ANY_CAST = _LIBCPP_INTRODUCED_IN_LLVM_4;
pub const _LIBCPP_AVAILABILITY_BAD_ANY_CAST = "";
pub const _LIBCPP_AVAILABILITY_HAS_FILESYSTEM_LIBRARY = _LIBCPP_INTRODUCED_IN_LLVM_9;
pub const _LIBCPP_AVAILABILITY_FILESYSTEM_LIBRARY = "";
pub const _LIBCPP_AVAILABILITY_FILESYSTEM_LIBRARY_PUSH = "";
pub const _LIBCPP_AVAILABILITY_FILESYSTEM_LIBRARY_POP = "";
pub const _LIBCPP_AVAILABILITY_HAS_SYNC = _LIBCPP_INTRODUCED_IN_LLVM_11;
pub const _LIBCPP_AVAILABILITY_SYNC = "";
pub const _LIBCPP_AVAILABILITY_HAS_ADDITIONAL_IOSTREAM_EXPLICIT_INSTANTIATIONS_1 = @as(c_int, 0);
pub const _LIBCPP_AVAILABILITY_HAS_TO_CHARS_FLOATING_POINT = _LIBCPP_INTRODUCED_IN_LLVM_14;
pub const _LIBCPP_AVAILABILITY_TO_CHARS_FLOATING_POINT = "";
pub const _LIBCPP_AVAILABILITY_HAS_VERBOSE_ABORT = _LIBCPP_INTRODUCED_IN_LLVM_15;
pub const _LIBCPP_AVAILABILITY_VERBOSE_ABORT = "";
pub const _LIBCPP_AVAILABILITY_HAS_PMR = _LIBCPP_INTRODUCED_IN_LLVM_16;
pub const _LIBCPP_AVAILABILITY_PMR = "";
pub const _LIBCPP_AVAILABILITY_HAS_INIT_PRIMARY_EXCEPTION = _LIBCPP_INTRODUCED_IN_LLVM_18;
pub const _LIBCPP_AVAILABILITY_INIT_PRIMARY_EXCEPTION = "";
pub const _LIBCPP_AVAILABILITY_HAS_PRINT = _LIBCPP_INTRODUCED_IN_LLVM_18;
pub const _LIBCPP_AVAILABILITY_PRINT = "";
pub const _LIBCPP_AVAILABILITY_HAS_TZDB = _LIBCPP_INTRODUCED_IN_LLVM_19;
pub const _LIBCPP_AVAILABILITY_TZDB = "";
pub const _LIBCPP_AVAILABILITY_HAS_BAD_FUNCTION_CALL_KEY_FUNCTION = _LIBCPP_INTRODUCED_IN_LLVM_19;
pub const _LIBCPP_AVAILABILITY_BAD_FUNCTION_CALL_KEY_FUNCTION = "";
pub const _LIBCPP_AVAILABILITY_HAS_BAD_EXPECTED_ACCESS_KEY_FUNCTION = _LIBCPP_INTRODUCED_IN_LLVM_19;
pub const _LIBCPP_AVAILABILITY_BAD_EXPECTED_ACCESS_KEY_FUNCTION = "";
pub const _LIBCPP_AVAILABILITY_THROW_BAD_ANY_CAST = "";
pub const _LIBCPP_AVAILABILITY_THROW_BAD_OPTIONAL_ACCESS = "";
pub const _LIBCPP_AVAILABILITY_THROW_BAD_VARIANT_ACCESS = "";
pub const _INC_STDLIB = "";
pub const _INC_CORECRT = "";
pub const _INC__MINGW_H = "";
pub const _INC_CRTDEFS_MACRO = "";
pub const __MINGW64_PASTE2 = @compileError("unable to translate C expr: unexpected token '##'");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw_mac.h:10:9
pub inline fn __MINGW64_PASTE(x: anytype, y: anytype) @TypeOf(__MINGW64_PASTE2(x, y)) {
    _ = &x;
    _ = &y;
    return __MINGW64_PASTE2(x, y);
}
pub const __STRINGIFY = @compileError("unable to translate C expr: unexpected token '#'");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw_mac.h:13:9
pub inline fn __MINGW64_STRINGIFY(x: anytype) @TypeOf(__STRINGIFY(x)) {
    _ = &x;
    return __STRINGIFY(x);
}
pub const __MINGW64_VERSION_MAJOR = @as(c_int, 13);
pub const __MINGW64_VERSION_MINOR = @as(c_int, 0);
pub const __MINGW64_VERSION_BUGFIX = @as(c_int, 0);
pub const __MINGW64_VERSION_RC = @as(c_int, 0);
pub const __MINGW64_VERSION_STR = __MINGW64_STRINGIFY(__MINGW64_VERSION_MAJOR) ++ "." ++ __MINGW64_STRINGIFY(__MINGW64_VERSION_MINOR) ++ "." ++ __MINGW64_STRINGIFY(__MINGW64_VERSION_BUGFIX);
pub const __MINGW64_VERSION_STATE = "alpha";
pub const __MINGW32_MAJOR_VERSION = @as(c_int, 3);
pub const __MINGW32_MINOR_VERSION = @as(c_int, 11);
pub const _M_AMD64 = @as(c_int, 100);
pub const _M_X64 = @as(c_int, 100);
pub const @"_" = @as(c_int, 1);
pub const __MINGW_USE_UNDERSCORE_PREFIX = @as(c_int, 0);
pub const __MINGW_IMP_SYMBOL = @compileError("unable to translate macro: undefined identifier `__imp_`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw_mac.h:129:11
pub const __MINGW_IMP_LSYMBOL = @compileError("unable to translate macro: undefined identifier `__imp_`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw_mac.h:130:11
pub inline fn __MINGW_USYMBOL(sym: anytype) @TypeOf(sym) {
    _ = &sym;
    return sym;
}
pub inline fn __MINGW_LSYMBOL(sym: anytype) @TypeOf(__MINGW64_PASTE(@"_", sym)) {
    _ = &sym;
    return __MINGW64_PASTE(@"_", sym);
}
pub const __MINGW_ASM_CALL = @compileError("unable to translate C expr: unexpected token '__asm__'");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw_mac.h:140:9
pub const __MINGW_ASM_CRT_CALL = @compileError("unable to translate C expr: unexpected token '__asm__'");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw_mac.h:141:9
pub const __MINGW_EXTENSION = @compileError("unable to translate C expr: unexpected token '__extension__'");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw_mac.h:173:13
pub const __C89_NAMELESS = __MINGW_EXTENSION;
pub const __C89_NAMELESSSTRUCTNAME = "";
pub const __C89_NAMELESSSTRUCTNAME1 = "";
pub const __C89_NAMELESSSTRUCTNAME2 = "";
pub const __C89_NAMELESSSTRUCTNAME3 = "";
pub const __C89_NAMELESSSTRUCTNAME4 = "";
pub const __C89_NAMELESSSTRUCTNAME5 = "";
pub const __C89_NAMELESSUNIONNAME = "";
pub const __C89_NAMELESSUNIONNAME1 = "";
pub const __C89_NAMELESSUNIONNAME2 = "";
pub const __C89_NAMELESSUNIONNAME3 = "";
pub const __C89_NAMELESSUNIONNAME4 = "";
pub const __C89_NAMELESSUNIONNAME5 = "";
pub const __C89_NAMELESSUNIONNAME6 = "";
pub const __C89_NAMELESSUNIONNAME7 = "";
pub const __C89_NAMELESSUNIONNAME8 = "";
pub const __GNU_EXTENSION = __MINGW_EXTENSION;
pub const __MINGW_HAVE_ANSI_C99_PRINTF = @as(c_int, 1);
pub const __MINGW_HAVE_WIDE_C99_PRINTF = @as(c_int, 1);
pub const __MINGW_HAVE_ANSI_C99_SCANF = @as(c_int, 1);
pub const __MINGW_HAVE_WIDE_C99_SCANF = @as(c_int, 1);
pub const __MINGW_POISON_NAME = @compileError("unable to translate macro: undefined identifier `_layout_has_not_been_verified_and_its_declaration_is_most_likely_incorrect`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw_mac.h:213:11
pub const __MSABI_LONG = @import("std").zig.c_translation.Macros.L_SUFFIX;
pub const __MINGW_GCC_VERSION = ((__GNUC__ * @as(c_int, 10000)) + (__GNUC_MINOR__ * @as(c_int, 100))) + __GNUC_PATCHLEVEL__;
pub inline fn __MINGW_GNUC_PREREQ(major: anytype, minor: anytype) @TypeOf((__GNUC__ > major) or ((__GNUC__ == major) and (__GNUC_MINOR__ >= minor))) {
    _ = &major;
    _ = &minor;
    return (__GNUC__ > major) or ((__GNUC__ == major) and (__GNUC_MINOR__ >= minor));
}
pub inline fn __MINGW_MSC_PREREQ(major: anytype, minor: anytype) @TypeOf(@as(c_int, 0)) {
    _ = &major;
    _ = &minor;
    return @as(c_int, 0);
}
pub const __MINGW_ATTRIB_DEPRECATED_STR = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw_mac.h:257:11
pub const __MINGW_SEC_WARN_STR = "This function or variable may be unsafe, use _CRT_SECURE_NO_WARNINGS to disable deprecation";
pub const __MINGW_MSVC2005_DEPREC_STR = "This POSIX function is deprecated beginning in Visual C++ 2005, use _CRT_NONSTDC_NO_DEPRECATE to disable deprecation";
pub const __MINGW_ATTRIB_DEPRECATED_MSVC2005 = __MINGW_ATTRIB_DEPRECATED_STR(__MINGW_MSVC2005_DEPREC_STR);
pub const __MINGW_ATTRIB_DEPRECATED_SEC_WARN = __MINGW_ATTRIB_DEPRECATED_STR(__MINGW_SEC_WARN_STR);
pub const __MINGW_MS_PRINTF = @compileError("unable to translate macro: undefined identifier `__format__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw_mac.h:281:9
pub const __MINGW_MS_SCANF = @compileError("unable to translate macro: undefined identifier `__format__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw_mac.h:284:9
pub const __MINGW_GNU_PRINTF = @compileError("unable to translate macro: undefined identifier `__format__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw_mac.h:287:9
pub const __MINGW_GNU_SCANF = @compileError("unable to translate macro: undefined identifier `__format__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw_mac.h:290:9
pub const __mingw_ovr = @compileError("unable to translate macro: undefined identifier `__unused__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw_mac.h:311:11
pub const __mingw_attribute_artificial = @compileError("unable to translate macro: undefined identifier `__artificial__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw_mac.h:318:11
pub const __MINGW_SELECTANY = @compileError("unable to translate macro: undefined identifier `__selectany__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw_mac.h:324:9
pub const __MINGW_FORTIFY_LEVEL = @as(c_int, 0);
pub const __mingw_bos_ovr = __mingw_ovr;
pub const __MINGW_FORTIFY_VA_ARG = @as(c_int, 0);
pub const _INC_MINGW_SECAPI = "";
pub const _CRT_SECURE_CPP_OVERLOAD_SECURE_NAMES = @as(c_int, 0);
pub const _CRT_SECURE_CPP_OVERLOAD_SECURE_NAMES_MEMORY = @as(c_int, 0);
pub const _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES = @as(c_int, 0);
pub const _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES_COUNT = @as(c_int, 0);
pub const _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES_MEMORY = @as(c_int, 0);
pub const __MINGW_CRT_NAME_CONCAT2 = @compileError("unable to translate macro: undefined identifier `_s`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw_secapi.h:41:9
pub const __CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES_MEMORY_0_3_ = @compileError("unable to translate C expr: unexpected token ';'");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw_secapi.h:69:9
pub const __LONG32 = c_long;
pub const __MINGW_IMPORT = @compileError("unable to translate macro: undefined identifier `__dllimport__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:44:12
pub const __USE_CRTIMP = @as(c_int, 1);
pub const _CRTIMP = @compileError("unable to translate macro: undefined identifier `__dllimport__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:52:15
pub const __DECLSPEC_SUPPORTED = "";
pub const USE___UUIDOF = @as(c_int, 0);
pub const _inline = @compileError("unable to translate C expr: unexpected token '__inline'");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:74:9
pub const __CRT_INLINE = @compileError("unable to translate macro: undefined identifier `__gnu_inline__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:83:11
pub const __MINGW_INTRIN_INLINE = @compileError("unable to translate macro: undefined identifier `__always_inline__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:90:9
pub const __CRT__NO_INLINE = @as(c_int, 1);
pub const __MINGW_CXX11_CONSTEXPR = "";
pub const __MINGW_CXX14_CONSTEXPR = "";
pub const __UNUSED_PARAM = @compileError("unable to translate macro: undefined identifier `__unused__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:118:11
pub const __restrict_arr = @compileError("unable to translate C expr: unexpected token '__restrict'");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:133:10
pub const __MINGW_ATTRIB_NORETURN = @compileError("unable to translate macro: undefined identifier `__noreturn__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:149:9
pub const __MINGW_ATTRIB_CONST = @compileError("unable to translate C expr: unexpected token '__attribute__'");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:150:9
pub const __MINGW_ATTRIB_MALLOC = @compileError("unable to translate macro: undefined identifier `__malloc__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:160:9
pub const __MINGW_ATTRIB_PURE = @compileError("unable to translate macro: undefined identifier `__pure__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:161:9
pub const __MINGW_ATTRIB_NONNULL = @compileError("unable to translate macro: undefined identifier `__nonnull__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:174:9
pub const __MINGW_ATTRIB_UNUSED = @compileError("unable to translate macro: undefined identifier `__unused__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:180:9
pub const __MINGW_ATTRIB_USED = @compileError("unable to translate macro: undefined identifier `__used__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:186:9
pub const __MINGW_ATTRIB_DEPRECATED = @compileError("unable to translate macro: undefined identifier `__deprecated__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:187:9
pub const __MINGW_ATTRIB_DEPRECATED_MSG = @compileError("unable to translate macro: undefined identifier `__deprecated__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:189:9
pub const __MINGW_NOTHROW = @compileError("unable to translate macro: undefined identifier `__nothrow__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:204:9
pub const __MINGW_ATTRIB_NO_OPTIMIZE = "";
pub const __MINGW_PRAGMA_PARAM = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:222:9
pub const __MINGW_BROKEN_INTERFACE = @compileError("unable to translate macro: undefined identifier `message`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:225:9
pub const _UCRT = "";
pub const _INT128_DEFINED = "";
pub const __int8 = u8;
pub const __int16 = c_short;
pub const __int32 = c_int;
pub const __int64 = c_longlong;
pub const __ptr32 = "";
pub const __ptr64 = "";
pub const __unaligned = "";
pub const __w64 = "";
pub const __forceinline = @compileError("unable to translate macro: undefined identifier `__always_inline__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:284:9
pub const __nothrow = "";
pub const _INC_VADEFS = "";
pub const MINGW_SDK_INIT = "";
pub const MINGW_HAS_SECURE_API = @as(c_int, 1);
pub const __STDC_SECURE_LIB__ = @as(c_long, 200411);
pub const __GOT_SECURE_LIB__ = __STDC_SECURE_LIB__;
pub const MINGW_DDK_H = "";
pub const MINGW_HAS_DDK_H = @as(c_int, 1);
pub const _CRT_PACKING = @as(c_int, 8);
pub const __GNUC_VA_LIST = "";
pub const _VA_LIST_DEFINED = "";
pub inline fn _ADDRESSOF(v: anytype) @TypeOf(&v) {
    _ = &v;
    return &v;
}
pub const _crt_va_start = @compileError("unable to translate macro: undefined identifier `__builtin_va_start`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/vadefs.h:48:9
pub const _crt_va_arg = @compileError("unable to translate C expr: unexpected token 'an identifier'");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/vadefs.h:49:9
pub const _crt_va_end = @compileError("unable to translate macro: undefined identifier `__builtin_va_end`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/vadefs.h:50:9
pub const _crt_va_copy = @compileError("unable to translate macro: undefined identifier `__builtin_va_copy`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/vadefs.h:51:9
pub const __CRT_STRINGIZE = @compileError("unable to translate C expr: unexpected token '#'");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:303:9
pub inline fn _CRT_STRINGIZE(_Value: anytype) @TypeOf(__CRT_STRINGIZE(_Value)) {
    _ = &_Value;
    return __CRT_STRINGIZE(_Value);
}
pub const __CRT_WIDE = @compileError("unable to translate macro: undefined identifier `L`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:308:9
pub inline fn _CRT_WIDE(_String: anytype) @TypeOf(__CRT_WIDE(_String)) {
    _ = &_String;
    return __CRT_WIDE(_String);
}
pub const _W64 = "";
pub const _CRTIMP_NOIA64 = _CRTIMP;
pub const _CRTIMP2 = _CRTIMP;
pub const _CRTIMP_ALTERNATIVE = _CRTIMP;
pub const _CRT_ALTERNATIVE_IMPORTED = "";
pub const _MRTIMP2 = _CRTIMP;
pub const _DLL = "";
pub const _MT = "";
pub const _MCRTIMP = _CRTIMP;
pub const _CRTIMP_PURE = _CRTIMP;
pub const _PGLOBAL = "";
pub const _AGLOBAL = "";
pub const _SECURECRT_FILL_BUFFER_PATTERN = @as(c_int, 0xFD);
pub const _CRT_DEPRECATE_TEXT = @compileError("unable to translate macro: undefined identifier `deprecated`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:367:9
pub const _CRT_INSECURE_DEPRECATE_MEMORY = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:370:9
pub const _CRT_INSECURE_DEPRECATE_GLOBALS = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:374:9
pub const _CRT_MANAGED_HEAP_DEPRECATE = "";
pub const _CRT_OBSOLETE = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:382:9
pub const _CONST_RETURN = "";
pub const UNALIGNED = "";
pub const _CRT_ALIGN = @compileError("unable to translate macro: undefined identifier `__aligned__`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:409:9
pub const __CRTDECL = __cdecl;
pub const _ARGMAX = @as(c_int, 100);
pub const _TRUNCATE = @import("std").zig.c_translation.cast(usize, -@as(c_int, 1));
pub inline fn _CRT_UNUSED(x: anytype) anyopaque {
    _ = &x;
    return @import("std").zig.c_translation.cast(anyopaque, x);
}
pub const __USE_MINGW_ANSI_STDIO = @as(c_int, 0);
pub const _CRT_glob = @compileError("unable to translate macro: undefined identifier `_dowildcard`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:473:9
pub const __ANONYMOUS_DEFINED = "";
pub const _ANONYMOUS_UNION = __MINGW_EXTENSION;
pub const _ANONYMOUS_STRUCT = __MINGW_EXTENSION;
pub const _UNION_NAME = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:493:9
pub const _STRUCT_NAME = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:494:9
pub const DUMMYUNIONNAME = "";
pub const DUMMYUNIONNAME1 = "";
pub const DUMMYUNIONNAME2 = "";
pub const DUMMYUNIONNAME3 = "";
pub const DUMMYUNIONNAME4 = "";
pub const DUMMYUNIONNAME5 = "";
pub const DUMMYUNIONNAME6 = "";
pub const DUMMYUNIONNAME7 = "";
pub const DUMMYUNIONNAME8 = "";
pub const DUMMYUNIONNAME9 = "";
pub const DUMMYSTRUCTNAME = "";
pub const DUMMYSTRUCTNAME1 = "";
pub const DUMMYSTRUCTNAME2 = "";
pub const DUMMYSTRUCTNAME3 = "";
pub const DUMMYSTRUCTNAME4 = "";
pub const DUMMYSTRUCTNAME5 = "";
pub const __CRT_UUID_DECL = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:581:9
pub const __MINGW_DEBUGBREAK_IMPL = !(__has_builtin(__debugbreak) != 0);
pub const __MINGW_FASTFAIL_IMPL = !(__has_builtin(__fastfail) != 0);
pub const __MINGW_PREFETCH_IMPL = @compileError("unable to translate macro: undefined identifier `__prefetch`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/_mingw.h:638:9
pub const _CRTNOALIAS = "";
pub const _CRTRESTRICT = "";
pub const _SIZE_T_DEFINED = "";
pub const _SSIZE_T_DEFINED = "";
pub const _RSIZE_T_DEFINED = "";
pub const _INTPTR_T_DEFINED = "";
pub const __intptr_t_defined = "";
pub const _UINTPTR_T_DEFINED = "";
pub const __uintptr_t_defined = "";
pub const _PTRDIFF_T_DEFINED = "";
pub const _PTRDIFF_T_ = "";
pub const _WCHAR_T_DEFINED = "";
pub const _WCTYPE_T_DEFINED = "";
pub const _WINT_T = "";
pub const _ERRCODE_DEFINED = "";
pub const _TIME32_T_DEFINED = "";
pub const _TIME64_T_DEFINED = "";
pub const _TIME_T_DEFINED = "";
pub const _CRT_SECURE_CPP_NOTHROW = @compileError("unable to translate macro: undefined identifier `throw`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:143:9
pub const __DEFINE_CPP_OVERLOAD_SECURE_FUNC_0_0 = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:262:9
pub const __DEFINE_CPP_OVERLOAD_SECURE_FUNC_0_1 = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:263:9
pub const __DEFINE_CPP_OVERLOAD_SECURE_FUNC_0_2 = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:264:9
pub const __DEFINE_CPP_OVERLOAD_SECURE_FUNC_0_3 = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:265:9
pub const __DEFINE_CPP_OVERLOAD_SECURE_FUNC_0_4 = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:266:9
pub const __DEFINE_CPP_OVERLOAD_SECURE_FUNC_1_1 = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:267:9
pub const __DEFINE_CPP_OVERLOAD_SECURE_FUNC_1_2 = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:268:9
pub const __DEFINE_CPP_OVERLOAD_SECURE_FUNC_1_3 = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:269:9
pub const __DEFINE_CPP_OVERLOAD_SECURE_FUNC_2_0 = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:270:9
pub const __DEFINE_CPP_OVERLOAD_SECURE_FUNC_0_1_ARGLIST = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:271:9
pub const __DEFINE_CPP_OVERLOAD_SECURE_FUNC_0_2_ARGLIST = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:272:9
pub const __DEFINE_CPP_OVERLOAD_SECURE_FUNC_SPLITPATH = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:273:9
pub const __DEFINE_CPP_OVERLOAD_STANDARD_FUNC_0_0 = @compileError("unable to translate macro: undefined identifier `__func_name`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:277:9
pub const __DEFINE_CPP_OVERLOAD_STANDARD_FUNC_0_1 = @compileError("unable to translate macro: undefined identifier `__func_name`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:279:9
pub const __DEFINE_CPP_OVERLOAD_STANDARD_FUNC_0_2 = @compileError("unable to translate macro: undefined identifier `__func_name`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:281:9
pub const __DEFINE_CPP_OVERLOAD_STANDARD_FUNC_0_3 = @compileError("unable to translate macro: undefined identifier `__func_name`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:283:9
pub const __DEFINE_CPP_OVERLOAD_STANDARD_FUNC_0_4 = @compileError("unable to translate macro: undefined identifier `__func_name`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:285:9
pub const __DEFINE_CPP_OVERLOAD_STANDARD_FUNC_0_0_EX = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:422:9
pub const __DEFINE_CPP_OVERLOAD_STANDARD_FUNC_0_1_EX = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:423:9
pub const __DEFINE_CPP_OVERLOAD_STANDARD_FUNC_0_2_EX = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:424:9
pub const __DEFINE_CPP_OVERLOAD_STANDARD_FUNC_0_3_EX = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:425:9
pub const __DEFINE_CPP_OVERLOAD_STANDARD_FUNC_0_4_EX = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:426:9
pub const _TAGLC_ID_DEFINED = "";
pub const _THREADLOCALEINFO = "";
pub const __crt_typefix = @compileError("unable to translate C expr: unexpected token ''");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/corecrt.h:486:9
pub const _CRT_USE_WINAPI_FAMILY_DESKTOP_APP = "";
pub const _INC_CORECRT_WSTDLIB = "";
pub const __CLANG_LIMITS_H = "";
pub const _GCC_LIMITS_H_ = "";
pub const _INC_CRTDEFS = "";
pub const _INC_LIMITS = "";
pub const PATH_MAX = @as(c_int, 260);
pub const CHAR_BIT = @as(c_int, 8);
pub const SCHAR_MIN = -@as(c_int, 128);
pub const SCHAR_MAX = @as(c_int, 127);
pub const UCHAR_MAX = @as(c_int, 0xff);
pub const CHAR_MIN = SCHAR_MIN;
pub const CHAR_MAX = SCHAR_MAX;
pub const MB_LEN_MAX = @as(c_int, 5);
pub const SHRT_MIN = -@import("std").zig.c_translation.promoteIntLiteral(c_int, 32768, .decimal);
pub const SHRT_MAX = @as(c_int, 32767);
pub const USHRT_MAX = @as(c_uint, 0xffff);
pub const INT_MIN = -@import("std").zig.c_translation.promoteIntLiteral(c_int, 2147483647, .decimal) - @as(c_int, 1);
pub const INT_MAX = @import("std").zig.c_translation.promoteIntLiteral(c_int, 2147483647, .decimal);
pub const UINT_MAX = @import("std").zig.c_translation.promoteIntLiteral(c_uint, 0xffffffff, .hex);
pub const LONG_MIN = -@as(c_long, 2147483647) - @as(c_int, 1);
pub const LONG_MAX = @as(c_long, 2147483647);
pub const ULONG_MAX = @as(c_ulong, 0xffffffff);
pub const LLONG_MAX = @as(c_longlong, 9223372036854775807);
pub const LLONG_MIN = -@as(c_longlong, 9223372036854775807) - @as(c_int, 1);
pub const ULLONG_MAX = @as(c_ulonglong, 0xffffffffffffffff);
pub const _I8_MIN = -@as(c_int, 127) - @as(c_int, 1);
pub const _I8_MAX = @as(c_int, 127);
pub const _UI8_MAX = @as(c_uint, 0xff);
pub const _I16_MIN = -@as(c_int, 32767) - @as(c_int, 1);
pub const _I16_MAX = @as(c_int, 32767);
pub const _UI16_MAX = @as(c_uint, 0xffff);
pub const _I32_MIN = -@import("std").zig.c_translation.promoteIntLiteral(c_int, 2147483647, .decimal) - @as(c_int, 1);
pub const _I32_MAX = @import("std").zig.c_translation.promoteIntLiteral(c_int, 2147483647, .decimal);
pub const _UI32_MAX = @import("std").zig.c_translation.promoteIntLiteral(c_uint, 0xffffffff, .hex);
pub const LONG_LONG_MAX = @as(c_longlong, 9223372036854775807);
pub const LONG_LONG_MIN = -LONG_LONG_MAX - @as(c_int, 1);
pub const ULONG_LONG_MAX = (@as(c_ulonglong, 2) * LONG_LONG_MAX) + @as(c_ulonglong, 1);
pub const _I64_MIN = -@as(c_longlong, 9223372036854775807) - @as(c_int, 1);
pub const _I64_MAX = @as(c_longlong, 9223372036854775807);
pub const _UI64_MAX = @as(c_ulonglong, 0xffffffffffffffff);
pub const SIZE_MAX = _UI64_MAX;
pub const SSIZE_MAX = _I64_MAX;
pub const _SECIMP = @compileError("unable to translate macro: undefined identifier `dllimport`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/stdlib.h:22:9
pub const NULL = @import("std").zig.c_translation.cast(?*anyopaque, @as(c_int, 0));
pub const EXIT_SUCCESS = @as(c_int, 0);
pub const EXIT_FAILURE = @as(c_int, 1);
pub const _ONEXIT_T_DEFINED = "";
pub const onexit_t = _onexit_t;
pub const _DIV_T_DEFINED = "";
pub const _CRT_DOUBLE_DEC = "";
pub inline fn _PTR_LD(x: anytype) [*c]u8 {
    _ = &x;
    return @import("std").zig.c_translation.cast([*c]u8, &x.*.ld);
}
pub const RAND_MAX = @as(c_int, 0x7fff);
pub const MB_CUR_MAX = ___mb_cur_max_func();
pub const __mb_cur_max = ___mb_cur_max_func();
pub inline fn __max(a: anytype, b: anytype) @TypeOf(if (a > b) a else b) {
    _ = &a;
    _ = &b;
    return if (a > b) a else b;
}
pub inline fn __min(a: anytype, b: anytype) @TypeOf(if (a < b) a else b) {
    _ = &a;
    _ = &b;
    return if (a < b) a else b;
}
pub const _MAX_PATH = @as(c_int, 260);
pub const _MAX_DRIVE = @as(c_int, 3);
pub const _MAX_DIR = @as(c_int, 256);
pub const _MAX_FNAME = @as(c_int, 256);
pub const _MAX_EXT = @as(c_int, 256);
pub const _OUT_TO_DEFAULT = @as(c_int, 0);
pub const _OUT_TO_STDERR = @as(c_int, 1);
pub const _OUT_TO_MSGBOX = @as(c_int, 2);
pub const _REPORT_ERRMODE = @as(c_int, 3);
pub const _WRITE_ABORT_MSG = @as(c_int, 0x1);
pub const _CALL_REPORTFAULT = @as(c_int, 0x2);
pub const _MAX_ENV = @as(c_int, 32767);
pub const _CRT_ERRNO_DEFINED = "";
pub const errno = _errno().*;
pub const _doserrno = __doserrno().*;
pub const _sys_nerr = __sys_nerr().*;
pub const _sys_errlist = __sys_errlist();
pub const _fmode = __p__fmode().*;
pub const __argc = __p___argc().*;
pub const __argv = __p___argv().*;
pub const __wargv = __p___wargv().*;
pub const _pgmptr = __p__pgmptr().*;
pub const _wpgmptr = __p__wpgmptr().*;
pub const _environ = __p__environ().*;
pub const _wenviron = __p__wenviron().*;
pub const _osplatform = __p__osplatform().*;
pub const _osver = __p__osver().*;
pub const _winver = __p__winver().*;
pub const _winmajor = __p__winmajor().*;
pub const _winminor = __p__winminor().*;
pub const _countof = @compileError("unable to translate C expr: expected ')' instead got '['");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/stdlib.h:263:9
pub const _CRT_TERMINATE_DEFINED = "";
pub const _CRT_ABS_DEFINED = "";
pub const _CRT_ATOF_DEFINED = "";
pub const _CRT_ALGO_DEFINED = "";
pub const _CRT_SYSTEM_DEFINED = "";
pub const _CRT_ALLOCATION_DEFINED = "";
pub const _WSTDLIB_DEFINED = "";
pub const _CRT_WSYSTEM_DEFINED = "";
pub const _CVTBUFSIZE = @as(c_int, 309) + @as(c_int, 40);
pub const _CRT_PERROR_DEFINED = "";
pub const _WSTDLIBP_DEFINED = "";
pub const _CRT_WPERROR_DEFINED = "";
pub const sys_errlist = _sys_errlist;
pub const sys_nerr = _sys_nerr;
pub const environ = _environ;
pub const _CRT_SWAB_DEFINED = "";
pub const _INC_STDLIB_S = "";
pub const _QSORT_S_DEFINED = "";
pub const _MALLOC_H_ = "";
pub const _HEAP_MAXREQ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 0xFFFFFFFFFFFFFFE0, .hex);
pub const _STATIC_ASSERT = @compileError("unable to translate C expr: unexpected token '_Static_assert'");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/malloc.h:29:9
pub const _HEAPEMPTY = -@as(c_int, 1);
pub const _HEAPOK = -@as(c_int, 2);
pub const _HEAPBADBEGIN = -@as(c_int, 3);
pub const _HEAPBADNODE = -@as(c_int, 4);
pub const _HEAPEND = -@as(c_int, 5);
pub const _HEAPBADPTR = -@as(c_int, 6);
pub const _FREEENTRY = @as(c_int, 0);
pub const _USEDENTRY = @as(c_int, 1);
pub const _HEAPINFO_DEFINED = "";
pub const _amblksiz = __p__amblksiz().*;
pub const __MM_MALLOC_H = "";
pub const _MAX_WAIT_MALLOC_CRT = @import("std").zig.c_translation.promoteIntLiteral(c_int, 60000, .decimal);
pub const _alloca = @compileError("unable to translate macro: undefined identifier `__builtin_alloca`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/malloc.h:163:9
pub const _ALLOCA_S_THRESHOLD = @as(c_int, 1024);
pub const _ALLOCA_S_STACK_MARKER = @import("std").zig.c_translation.promoteIntLiteral(c_int, 0xCCCC, .hex);
pub const _ALLOCA_S_HEAP_MARKER = @import("std").zig.c_translation.promoteIntLiteral(c_int, 0xDDDD, .hex);
pub const _ALLOCA_S_MARKER_SIZE = @as(c_int, 16);
pub inline fn _malloca(size: anytype) @TypeOf(_MarkAllocaS(malloc(size + _ALLOCA_S_MARKER_SIZE), _ALLOCA_S_HEAP_MARKER)) {
    _ = &size;
    return _MarkAllocaS(malloc(size + _ALLOCA_S_MARKER_SIZE), _ALLOCA_S_HEAP_MARKER);
}
pub const _FREEA_INLINE = "";
pub const alloca = @compileError("unable to translate macro: undefined identifier `__builtin_alloca`");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\\libc\include\any-windows-any/malloc.h:238:9
pub const _LIBCPP_STDINT_H = "";
pub const __CLANG_STDINT_H = "";
pub const _STDINT_H = "";
pub const __need_wint_t = "";
pub const __need_wchar_t = "";
pub const _WCHAR_T = "";
pub const _LIBCPP_STDDEF_H = "";
pub const INT8_MIN = -@as(c_int, 128);
pub const INT16_MIN = -@import("std").zig.c_translation.promoteIntLiteral(c_int, 32768, .decimal);
pub const INT32_MIN = -@import("std").zig.c_translation.promoteIntLiteral(c_int, 2147483647, .decimal) - @as(c_int, 1);
pub const INT64_MIN = -@as(c_longlong, 9223372036854775807) - @as(c_int, 1);
pub const INT8_MAX = @as(c_int, 127);
pub const INT16_MAX = @as(c_int, 32767);
pub const INT32_MAX = @import("std").zig.c_translation.promoteIntLiteral(c_int, 2147483647, .decimal);
pub const INT64_MAX = @as(c_longlong, 9223372036854775807);
pub const UINT8_MAX = @as(c_int, 255);
pub const UINT16_MAX = @import("std").zig.c_translation.promoteIntLiteral(c_int, 65535, .decimal);
pub const UINT32_MAX = @import("std").zig.c_translation.promoteIntLiteral(c_uint, 0xffffffff, .hex);
pub const UINT64_MAX = @as(c_ulonglong, 0xffffffffffffffff);
pub const INT_LEAST8_MIN = INT8_MIN;
pub const INT_LEAST16_MIN = INT16_MIN;
pub const INT_LEAST32_MIN = INT32_MIN;
pub const INT_LEAST64_MIN = INT64_MIN;
pub const INT_LEAST8_MAX = INT8_MAX;
pub const INT_LEAST16_MAX = INT16_MAX;
pub const INT_LEAST32_MAX = INT32_MAX;
pub const INT_LEAST64_MAX = INT64_MAX;
pub const UINT_LEAST8_MAX = UINT8_MAX;
pub const UINT_LEAST16_MAX = UINT16_MAX;
pub const UINT_LEAST32_MAX = UINT32_MAX;
pub const UINT_LEAST64_MAX = UINT64_MAX;
pub const INT_FAST8_MIN = INT8_MIN;
pub const INT_FAST16_MIN = INT16_MIN;
pub const INT_FAST32_MIN = INT32_MIN;
pub const INT_FAST64_MIN = INT64_MIN;
pub const INT_FAST8_MAX = INT8_MAX;
pub const INT_FAST16_MAX = INT16_MAX;
pub const INT_FAST32_MAX = INT32_MAX;
pub const INT_FAST64_MAX = INT64_MAX;
pub const UINT_FAST8_MAX = UINT8_MAX;
pub const UINT_FAST16_MAX = UINT16_MAX;
pub const UINT_FAST32_MAX = UINT32_MAX;
pub const UINT_FAST64_MAX = UINT64_MAX;
pub const INTPTR_MIN = INT64_MIN;
pub const INTPTR_MAX = INT64_MAX;
pub const UINTPTR_MAX = UINT64_MAX;
pub const INTMAX_MIN = INT64_MIN;
pub const INTMAX_MAX = INT64_MAX;
pub const UINTMAX_MAX = UINT64_MAX;
pub const PTRDIFF_MIN = INT64_MIN;
pub const PTRDIFF_MAX = INT64_MAX;
pub const SIG_ATOMIC_MIN = INT32_MIN;
pub const SIG_ATOMIC_MAX = INT32_MAX;
pub const WCHAR_MIN = @as(c_uint, 0);
pub const WCHAR_MAX = @as(c_uint, 0xffff);
pub const WINT_MIN = @as(c_uint, 0);
pub const WINT_MAX = @as(c_uint, 0xffff);
pub inline fn INT8_C(val: anytype) @TypeOf((INT_LEAST8_MAX - INT_LEAST8_MAX) + val) {
    _ = &val;
    return (INT_LEAST8_MAX - INT_LEAST8_MAX) + val;
}
pub inline fn INT16_C(val: anytype) @TypeOf((INT_LEAST16_MAX - INT_LEAST16_MAX) + val) {
    _ = &val;
    return (INT_LEAST16_MAX - INT_LEAST16_MAX) + val;
}
pub inline fn INT32_C(val: anytype) @TypeOf((INT_LEAST32_MAX - INT_LEAST32_MAX) + val) {
    _ = &val;
    return (INT_LEAST32_MAX - INT_LEAST32_MAX) + val;
}
pub const INT64_C = @import("std").zig.c_translation.Macros.LL_SUFFIX;
pub inline fn UINT8_C(val: anytype) @TypeOf(val) {
    _ = &val;
    return val;
}
pub inline fn UINT16_C(val: anytype) @TypeOf(val) {
    _ = &val;
    return val;
}
pub const UINT32_C = @import("std").zig.c_translation.Macros.U_SUFFIX;
pub const UINT64_C = @import("std").zig.c_translation.Macros.ULL_SUFFIX;
pub const INTMAX_C = @import("std").zig.c_translation.Macros.LL_SUFFIX;
pub const UINTMAX_C = @import("std").zig.c_translation.Macros.ULL_SUFFIX;
pub const __CUDA_DEPRECATED = @compileError("unable to translate macro: undefined identifier `deprecated`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/cuda.h:71:9
pub inline fn __CUDA_API_PTDS(api: anytype) @TypeOf(api) {
    _ = &api;
    return api;
}
pub inline fn __CUDA_API_PTSZ(api: anytype) @TypeOf(api) {
    _ = &api;
    return api;
}
pub const cuDeviceTotalMem = cuDeviceTotalMem_v2;
pub const cuCtxCreate = cuCtxCreate_v2;
pub const cuModuleGetGlobal = cuModuleGetGlobal_v2;
pub const cuMemGetInfo = cuMemGetInfo_v2;
pub const cuMemAlloc = cuMemAlloc_v2;
pub const cuMemAllocPitch = cuMemAllocPitch_v2;
pub const cuMemFree = cuMemFree_v2;
pub const cuMemGetAddressRange = cuMemGetAddressRange_v2;
pub const cuMemAllocHost = cuMemAllocHost_v2;
pub const cuMemHostGetDevicePointer = cuMemHostGetDevicePointer_v2;
pub const cuMemcpyHtoD = __CUDA_API_PTDS(cuMemcpyHtoD_v2);
pub const cuMemcpyDtoH = __CUDA_API_PTDS(cuMemcpyDtoH_v2);
pub const cuMemcpyDtoD = __CUDA_API_PTDS(cuMemcpyDtoD_v2);
pub const cuMemcpyDtoA = __CUDA_API_PTDS(cuMemcpyDtoA_v2);
pub const cuMemcpyAtoD = __CUDA_API_PTDS(cuMemcpyAtoD_v2);
pub const cuMemcpyHtoA = __CUDA_API_PTDS(cuMemcpyHtoA_v2);
pub const cuMemcpyAtoH = __CUDA_API_PTDS(cuMemcpyAtoH_v2);
pub const cuMemcpyAtoA = __CUDA_API_PTDS(cuMemcpyAtoA_v2);
pub const cuMemcpyHtoAAsync = __CUDA_API_PTSZ(cuMemcpyHtoAAsync_v2);
pub const cuMemcpyAtoHAsync = __CUDA_API_PTSZ(cuMemcpyAtoHAsync_v2);
pub const cuMemcpy2D = __CUDA_API_PTDS(cuMemcpy2D_v2);
pub const cuMemcpy2DUnaligned = __CUDA_API_PTDS(cuMemcpy2DUnaligned_v2);
pub const cuMemcpy3D = __CUDA_API_PTDS(cuMemcpy3D_v2);
pub const cuMemcpyHtoDAsync = __CUDA_API_PTSZ(cuMemcpyHtoDAsync_v2);
pub const cuMemcpyDtoHAsync = __CUDA_API_PTSZ(cuMemcpyDtoHAsync_v2);
pub const cuMemcpyDtoDAsync = __CUDA_API_PTSZ(cuMemcpyDtoDAsync_v2);
pub const cuMemcpy2DAsync = __CUDA_API_PTSZ(cuMemcpy2DAsync_v2);
pub const cuMemcpy3DAsync = __CUDA_API_PTSZ(cuMemcpy3DAsync_v2);
pub const cuMemsetD8 = __CUDA_API_PTDS(cuMemsetD8_v2);
pub const cuMemsetD16 = __CUDA_API_PTDS(cuMemsetD16_v2);
pub const cuMemsetD32 = __CUDA_API_PTDS(cuMemsetD32_v2);
pub const cuMemsetD2D8 = __CUDA_API_PTDS(cuMemsetD2D8_v2);
pub const cuMemsetD2D16 = __CUDA_API_PTDS(cuMemsetD2D16_v2);
pub const cuMemsetD2D32 = __CUDA_API_PTDS(cuMemsetD2D32_v2);
pub const cuArrayCreate = cuArrayCreate_v2;
pub const cuArrayGetDescriptor = cuArrayGetDescriptor_v2;
pub const cuArray3DCreate = cuArray3DCreate_v2;
pub const cuArray3DGetDescriptor = cuArray3DGetDescriptor_v2;
pub const cuTexRefSetAddress = cuTexRefSetAddress_v2;
pub const cuTexRefGetAddress = cuTexRefGetAddress_v2;
pub const cuGraphicsResourceGetMappedPointer = cuGraphicsResourceGetMappedPointer_v2;
pub const cuCtxDestroy = cuCtxDestroy_v2;
pub const cuCtxPopCurrent = cuCtxPopCurrent_v2;
pub const cuCtxPushCurrent = cuCtxPushCurrent_v2;
pub const cuStreamDestroy = cuStreamDestroy_v2;
pub const cuEventDestroy = cuEventDestroy_v2;
pub const cuTexRefSetAddress2D = cuTexRefSetAddress2D_v3;
pub const cuLinkCreate = cuLinkCreate_v2;
pub const cuLinkAddData = cuLinkAddData_v2;
pub const cuLinkAddFile = cuLinkAddFile_v2;
pub const cuMemHostRegister = cuMemHostRegister_v2;
pub const cuGraphicsResourceSetMapFlags = cuGraphicsResourceSetMapFlags_v2;
pub const cuStreamBeginCapture = __CUDA_API_PTSZ(cuStreamBeginCapture_v2);
pub const cuDevicePrimaryCtxRelease = cuDevicePrimaryCtxRelease_v2;
pub const cuDevicePrimaryCtxReset = cuDevicePrimaryCtxReset_v2;
pub const cuDevicePrimaryCtxSetFlags = cuDevicePrimaryCtxSetFlags_v2;
pub const cuIpcOpenMemHandle = cuIpcOpenMemHandle_v2;
pub const cuGraphInstantiate = cuGraphInstantiateWithFlags;
pub const cuGraphExecUpdate = cuGraphExecUpdate_v2;
pub const cuGetProcAddress = cuGetProcAddress_v2;
pub const cuGraphAddKernelNode = cuGraphAddKernelNode_v2;
pub const cuGraphKernelNodeGetParams = cuGraphKernelNodeGetParams_v2;
pub const cuGraphKernelNodeSetParams = cuGraphKernelNodeSetParams_v2;
pub const cuGraphExecKernelNodeSetParams = cuGraphExecKernelNodeSetParams_v2;
pub const cuStreamWriteValue32 = __CUDA_API_PTSZ(cuStreamWriteValue32_v2);
pub const cuStreamWaitValue32 = __CUDA_API_PTSZ(cuStreamWaitValue32_v2);
pub const cuStreamWriteValue64 = __CUDA_API_PTSZ(cuStreamWriteValue64_v2);
pub const cuStreamWaitValue64 = __CUDA_API_PTSZ(cuStreamWaitValue64_v2);
pub const cuStreamBatchMemOp = __CUDA_API_PTSZ(cuStreamBatchMemOp_v2);
pub const cuStreamGetCaptureInfo = __CUDA_API_PTSZ(cuStreamGetCaptureInfo_v2);
pub const CUDA_VERSION = @as(c_int, 12080);
pub const CU_UUID_HAS_BEEN_DEFINED = "";
pub const CU_IPC_HANDLE_SIZE = @as(c_int, 64);
pub const CU_STREAM_LEGACY = @import("std").zig.c_translation.cast(CUstream, @as(c_int, 0x1));
pub const CU_STREAM_PER_THREAD = @import("std").zig.c_translation.cast(CUstream, @as(c_int, 0x2));
pub const CU_COMPUTE_ACCELERATED_TARGET_BASE = @import("std").zig.c_translation.promoteIntLiteral(c_int, 0x10000, .hex);
pub const CUDA_CB = @compileError("unable to translate C expr: unexpected token '__stdcall'");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/cuda.h:1652:9
pub const CU_GRAPH_COND_ASSIGN_DEFAULT = @as(c_int, 0x1);
pub const CU_GRAPH_KERNEL_NODE_PORT_DEFAULT = @as(c_int, 0);
pub const CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC = @as(c_int, 1);
pub const CU_GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER = @as(c_int, 2);
pub const CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW = CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW;
pub const CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
pub const CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
pub const CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
pub const CU_KERNEL_NODE_ATTRIBUTE_PRIORITY = CU_LAUNCH_ATTRIBUTE_PRIORITY;
pub const CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP;
pub const CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN;
pub const CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION = CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION;
pub const CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE = CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE;
pub const CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT;
pub const CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW = CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW;
pub const CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY = CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY;
pub const CU_STREAM_ATTRIBUTE_PRIORITY = CU_LAUNCH_ATTRIBUTE_PRIORITY;
pub const CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP;
pub const CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN;
pub const CU_MEMHOSTALLOC_PORTABLE = @as(c_int, 0x01);
pub const CU_MEMHOSTALLOC_DEVICEMAP = @as(c_int, 0x02);
pub const CU_MEMHOSTALLOC_WRITECOMBINED = @as(c_int, 0x04);
pub const CU_MEMHOSTREGISTER_PORTABLE = @as(c_int, 0x01);
pub const CU_MEMHOSTREGISTER_DEVICEMAP = @as(c_int, 0x02);
pub const CU_MEMHOSTREGISTER_IOMEMORY = @as(c_int, 0x04);
pub const CU_MEMHOSTREGISTER_READ_ONLY = @as(c_int, 0x08);
pub const CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL = @as(c_int, 0x1);
pub const CU_TENSOR_MAP_NUM_QWORDS = @as(c_int, 16);
pub const CUDA_EXTERNAL_MEMORY_DEDICATED = @as(c_int, 0x1);
pub const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC = @as(c_int, 0x01);
pub const CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC = @as(c_int, 0x02);
pub const CUDA_NVSCISYNC_ATTR_SIGNAL = @as(c_int, 0x1);
pub const CUDA_NVSCISYNC_ATTR_WAIT = @as(c_int, 0x2);
pub const CU_MEM_CREATE_USAGE_TILE_POOL = @as(c_int, 0x1);
pub const CU_MEM_CREATE_USAGE_HW_DECOMPRESS = @as(c_int, 0x2);
pub const CU_MEM_POOL_CREATE_USAGE_HW_DECOMPRESS = @as(c_int, 0x2);
pub const CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC = @as(c_int, 0x01);
pub const CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC = @as(c_int, 0x02);
pub const CUDA_ARRAY3D_LAYERED = @as(c_int, 0x01);
pub const CUDA_ARRAY3D_2DARRAY = @as(c_int, 0x01);
pub const CUDA_ARRAY3D_SURFACE_LDST = @as(c_int, 0x02);
pub const CUDA_ARRAY3D_CUBEMAP = @as(c_int, 0x04);
pub const CUDA_ARRAY3D_TEXTURE_GATHER = @as(c_int, 0x08);
pub const CUDA_ARRAY3D_DEPTH_TEXTURE = @as(c_int, 0x10);
pub const CUDA_ARRAY3D_COLOR_ATTACHMENT = @as(c_int, 0x20);
pub const CUDA_ARRAY3D_SPARSE = @as(c_int, 0x40);
pub const CUDA_ARRAY3D_DEFERRED_MAPPING = @as(c_int, 0x80);
pub const CUDA_ARRAY3D_VIDEO_ENCODE_DECODE = @as(c_int, 0x100);
pub const CU_TRSA_OVERRIDE_FORMAT = @as(c_int, 0x01);
pub const CU_TRSF_READ_AS_INTEGER = @as(c_int, 0x01);
pub const CU_TRSF_NORMALIZED_COORDINATES = @as(c_int, 0x02);
pub const CU_TRSF_SRGB = @as(c_int, 0x10);
pub const CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION = @as(c_int, 0x20);
pub const CU_TRSF_SEAMLESS_CUBEMAP = @as(c_int, 0x40);
pub const CU_LAUNCH_PARAM_END_AS_INT = @as(c_int, 0x00);
pub const CU_LAUNCH_PARAM_END = @import("std").zig.c_translation.cast(?*anyopaque, CU_LAUNCH_PARAM_END_AS_INT);
pub const CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT = @as(c_int, 0x01);
pub const CU_LAUNCH_PARAM_BUFFER_POINTER = @import("std").zig.c_translation.cast(?*anyopaque, CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT);
pub const CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT = @as(c_int, 0x02);
pub const CU_LAUNCH_PARAM_BUFFER_SIZE = @import("std").zig.c_translation.cast(?*anyopaque, CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT);
pub const CU_PARAM_TR_DEFAULT = -@as(c_int, 1);
pub const CU_DEVICE_CPU = @import("std").zig.c_translation.cast(CUdevice, -@as(c_int, 1));
pub const CU_DEVICE_INVALID = @import("std").zig.c_translation.cast(CUdevice, -@as(c_int, 2));
pub const CUDAAPI = @compileError("unable to translate C expr: unexpected token '__stdcall'");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/cuda.h:4945:9
pub const RESOURCE_ABI_VERSION = @as(c_int, 1);
pub const RESOURCE_ABI_EXTERNAL_BYTES = @as(c_int, 48);
pub const _CONCAT_INNER = @compileError("unable to translate C expr: unexpected token '##'");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/cuda.h:25215:9
pub inline fn _CONCAT_OUTER(x: anytype, y: anytype) @TypeOf(_CONCAT_INNER(x, y)) {
    _ = &x;
    _ = &y;
    return _CONCAT_INNER(x, y);
}
pub const __CUDA_RUNTIME_H__ = "";
pub const __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__ = "";
pub const __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_CUDA_RUNTIME_H__ = "";
pub const EXCLUDE_FROM_RTC = "";
pub const __HOST_CONFIG_H__ = "";
pub const __DEVICE_TYPES_H__ = "";
pub const __HOST_DEFINES_H__ = "";
pub const __no_return__ = @compileError("unable to translate macro: undefined identifier `noreturn`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/crt/host_defines.h:78:9
pub const __forceinline__ = @compileError("unable to translate macro: undefined identifier `always_inline`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/crt/host_defines.h:95:9
pub const __inline_hint__ = @compileError("unable to translate macro: undefined identifier `nv_inline_hint`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/crt/host_defines.h:97:9
pub const __align__ = @compileError("unable to translate macro: undefined identifier `aligned`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/crt/host_defines.h:99:9
pub const __maxnreg__ = @compileError("unable to translate macro: undefined identifier `maxnreg`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/crt/host_defines.h:101:9
pub const __thread__ = @compileError("unable to translate macro: undefined identifier `__thread`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/crt/host_defines.h:103:9
pub const __import__ = "";
pub const __export__ = "";
pub const __annotate__ = @compileError("unable to translate C expr: unexpected token '__attribute__'");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/crt/host_defines.h:108:9
pub inline fn __location__(a: anytype) @TypeOf(__annotate__(a)) {
    _ = &a;
    return __annotate__(a);
}
pub const CUDARTAPI = "";
pub const CUDARTAPI_CDECL = "";
pub const __specialization_static = "";
pub inline fn __builtin_align__(a: anytype) @TypeOf(__align__(a)) {
    _ = &a;
    return __align__(a);
}
pub const __grid_constant__ = @compileError("unable to translate macro: undefined identifier `grid_constant`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/crt/host_defines.h:217:9
pub const __host__ = @compileError("unable to translate macro: undefined identifier `host`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/crt/host_defines.h:222:9
pub const __device__ = @compileError("unable to translate macro: undefined identifier `device`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/crt/host_defines.h:226:9
pub const __global__ = @compileError("unable to translate macro: undefined identifier `global`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/crt/host_defines.h:230:9
pub const __shared__ = @compileError("unable to translate macro: undefined identifier `shared`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/crt/host_defines.h:234:9
pub const __constant__ = @compileError("unable to translate macro: undefined identifier `constant`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/crt/host_defines.h:238:9
pub const __managed__ = @compileError("unable to translate macro: undefined identifier `managed`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/crt/host_defines.h:242:9
pub const __nv_pure__ = @compileError("unable to translate macro: undefined identifier `nv_pure`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/crt/host_defines.h:246:9
pub const __device_builtin__ = "";
pub const __device_builtin_texture_type__ = "";
pub const __device_builtin_surface_type__ = "";
pub const __cudart_builtin__ = "";
pub const __cluster_dims__ = @compileError("unable to translate C expr: expected ')' instead got '...'");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/crt/host_defines.h:271:9
pub const __CUDA_ARCH_HAS_FEATURE__ = @compileError("unable to translate macro: undefined identifier `__CUDA_ARCH_FEAT_`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/crt/host_defines.h:276:9
pub const __DRIVER_TYPES_H__ = "";
pub const __VECTOR_TYPES_H__ = "";
pub inline fn __cuda_builtin_vector_align8(tag: anytype, members: anytype) @TypeOf(__device_builtin__ ++ __align__(@as(c_int, 8)) ++ @import("std").mem.zeroInit(tag, .{members})) {
    _ = &tag;
    _ = &members;
    return __device_builtin__ ++ __align__(@as(c_int, 8)) ++ @import("std").mem.zeroInit(tag, .{members});
}
pub const __need_ptrdiff_t = "";
pub const __need_size_t = "";
pub const __need_NULL = "";
pub const __need_max_align_t = "";
pub const __need_offsetof = "";
pub const __STDDEF_H = "";
pub const _PTRDIFF_T = "";
pub const _SIZE_T = "";
pub const __CLANG_MAX_ALIGN_T_DEFINED = "";
pub const offsetof = @compileError("unable to translate C expr: unexpected token 'an identifier'");
// C:\zig\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\zig-windows-x86_64-0.15.0-dev.64+2a4e06bcb\lib\include/__stddef_offsetof.h:16:9
pub const cudaHostAllocDefault = @as(c_int, 0x00);
pub const cudaHostAllocPortable = @as(c_int, 0x01);
pub const cudaHostAllocMapped = @as(c_int, 0x02);
pub const cudaHostAllocWriteCombined = @as(c_int, 0x04);
pub const cudaHostRegisterDefault = @as(c_int, 0x00);
pub const cudaHostRegisterPortable = @as(c_int, 0x01);
pub const cudaHostRegisterMapped = @as(c_int, 0x02);
pub const cudaHostRegisterIoMemory = @as(c_int, 0x04);
pub const cudaHostRegisterReadOnly = @as(c_int, 0x08);
pub const cudaPeerAccessDefault = @as(c_int, 0x00);
pub const cudaStreamDefault = @as(c_int, 0x00);
pub const cudaStreamNonBlocking = @as(c_int, 0x01);
pub const cudaStreamLegacy = @import("std").zig.c_translation.cast(cudaStream_t, @as(c_int, 0x1));
pub const cudaStreamPerThread = @import("std").zig.c_translation.cast(cudaStream_t, @as(c_int, 0x2));
pub const cudaEventDefault = @as(c_int, 0x00);
pub const cudaEventBlockingSync = @as(c_int, 0x01);
pub const cudaEventDisableTiming = @as(c_int, 0x02);
pub const cudaEventInterprocess = @as(c_int, 0x04);
pub const cudaEventRecordDefault = @as(c_int, 0x00);
pub const cudaEventRecordExternal = @as(c_int, 0x01);
pub const cudaEventWaitDefault = @as(c_int, 0x00);
pub const cudaEventWaitExternal = @as(c_int, 0x01);
pub const cudaDeviceScheduleAuto = @as(c_int, 0x00);
pub const cudaDeviceScheduleSpin = @as(c_int, 0x01);
pub const cudaDeviceScheduleYield = @as(c_int, 0x02);
pub const cudaDeviceScheduleBlockingSync = @as(c_int, 0x04);
pub const cudaDeviceBlockingSync = @as(c_int, 0x04);
pub const cudaDeviceScheduleMask = @as(c_int, 0x07);
pub const cudaDeviceMapHost = @as(c_int, 0x08);
pub const cudaDeviceLmemResizeToMax = @as(c_int, 0x10);
pub const cudaDeviceSyncMemops = @as(c_int, 0x80);
pub const cudaDeviceMask = @as(c_int, 0xff);
pub const cudaArrayDefault = @as(c_int, 0x00);
pub const cudaArrayLayered = @as(c_int, 0x01);
pub const cudaArraySurfaceLoadStore = @as(c_int, 0x02);
pub const cudaArrayCubemap = @as(c_int, 0x04);
pub const cudaArrayTextureGather = @as(c_int, 0x08);
pub const cudaArrayColorAttachment = @as(c_int, 0x20);
pub const cudaArraySparse = @as(c_int, 0x40);
pub const cudaArrayDeferredMapping = @as(c_int, 0x80);
pub const cudaIpcMemLazyEnablePeerAccess = @as(c_int, 0x01);
pub const cudaMemAttachGlobal = @as(c_int, 0x01);
pub const cudaMemAttachHost = @as(c_int, 0x02);
pub const cudaMemAttachSingle = @as(c_int, 0x04);
pub const cudaOccupancyDefault = @as(c_int, 0x00);
pub const cudaOccupancyDisableCachingOverride = @as(c_int, 0x01);
pub const cudaCpuDeviceId = @import("std").zig.c_translation.cast(c_int, -@as(c_int, 1));
pub const cudaInvalidDeviceId = @import("std").zig.c_translation.cast(c_int, -@as(c_int, 2));
pub const cudaInitDeviceFlagsAreValid = @as(c_int, 0x01);
pub const cudaCooperativeLaunchMultiDeviceNoPreSync = @as(c_int, 0x01);
pub const cudaCooperativeLaunchMultiDeviceNoPostSync = @as(c_int, 0x02);
pub const cudaArraySparsePropertiesSingleMipTail = @as(c_int, 0x1);
pub const CUDART_CB = @compileError("unable to translate C expr: unexpected token '__stdcall'");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/driver_types.h:1401:9
pub const cudaMemPoolCreateUsageHwDecompress = @as(c_int, 0x2);
pub const CUDA_IPC_HANDLE_SIZE = @as(c_int, 64);
pub const cudaExternalMemoryDedicated = @as(c_int, 0x1);
pub const cudaExternalSemaphoreSignalSkipNvSciBufMemSync = @as(c_int, 0x01);
pub const cudaExternalSemaphoreWaitSkipNvSciBufMemSync = @as(c_int, 0x02);
pub const cudaNvSciSyncAttrSignal = @as(c_int, 0x1);
pub const cudaNvSciSyncAttrWait = @as(c_int, 0x2);
pub const cudaGraphKernelNodePortDefault = @as(c_int, 0);
pub const cudaGraphKernelNodePortProgrammatic = @as(c_int, 1);
pub const cudaGraphKernelNodePortLaunchCompletion = @as(c_int, 2);
pub const cudaStreamAttrID = cudaLaunchAttributeID;
pub const cudaStreamAttributeAccessPolicyWindow = cudaLaunchAttributeAccessPolicyWindow;
pub const cudaStreamAttributeSynchronizationPolicy = cudaLaunchAttributeSynchronizationPolicy;
pub const cudaStreamAttributeMemSyncDomainMap = cudaLaunchAttributeMemSyncDomainMap;
pub const cudaStreamAttributeMemSyncDomain = cudaLaunchAttributeMemSyncDomain;
pub const cudaStreamAttributePriority = cudaLaunchAttributePriority;
pub const cudaStreamAttrValue = cudaLaunchAttributeValue;
pub const cudaKernelNodeAttrID = cudaLaunchAttributeID;
pub const cudaKernelNodeAttributeAccessPolicyWindow = cudaLaunchAttributeAccessPolicyWindow;
pub const cudaKernelNodeAttributeCooperative = cudaLaunchAttributeCooperative;
pub const cudaKernelNodeAttributePriority = cudaLaunchAttributePriority;
pub const cudaKernelNodeAttributeClusterDimension = cudaLaunchAttributeClusterDimension;
pub const cudaKernelNodeAttributeClusterSchedulingPolicyPreference = cudaLaunchAttributeClusterSchedulingPolicyPreference;
pub const cudaKernelNodeAttributeMemSyncDomainMap = cudaLaunchAttributeMemSyncDomainMap;
pub const cudaKernelNodeAttributeMemSyncDomain = cudaLaunchAttributeMemSyncDomain;
pub const cudaKernelNodeAttributePreferredSharedMemoryCarveout = cudaLaunchAttributePreferredSharedMemoryCarveout;
pub const cudaKernelNodeAttributeDeviceUpdatableKernelNode = cudaLaunchAttributeDeviceUpdatableKernelNode;
pub const cudaKernelNodeAttrValue = cudaLaunchAttributeValue;
pub const __SURFACE_TYPES_H__ = "";
pub const cudaSurfaceType1D = @as(c_int, 0x01);
pub const cudaSurfaceType2D = @as(c_int, 0x02);
pub const cudaSurfaceType3D = @as(c_int, 0x03);
pub const cudaSurfaceTypeCubemap = @as(c_int, 0x0C);
pub const cudaSurfaceType1DLayered = @as(c_int, 0xF1);
pub const cudaSurfaceType2DLayered = @as(c_int, 0xF2);
pub const cudaSurfaceTypeCubemapLayered = @as(c_int, 0xFC);
pub const __TEXTURE_TYPES_H__ = "";
pub const cudaTextureType1D = @as(c_int, 0x01);
pub const cudaTextureType2D = @as(c_int, 0x02);
pub const cudaTextureType3D = @as(c_int, 0x03);
pub const cudaTextureTypeCubemap = @as(c_int, 0x0C);
pub const cudaTextureType1DLayered = @as(c_int, 0xF1);
pub const cudaTextureType2DLayered = @as(c_int, 0xF2);
pub const cudaTextureTypeCubemapLayered = @as(c_int, 0xFC);
pub const __LIBRARY_TYPES_H__ = "";
pub const __CHANNEL_DESCRIPTOR_H__ = "";
pub const __CUDA_RUNTIME_API_H__ = "";
pub const CUDART_VERSION = @as(c_int, 12080);
pub const __CUDART_API_VERSION = CUDART_VERSION;
pub const __CUDA_DEVICE_RUNTIME_API_H__ = "";
pub const __CUDA_INTERNAL_USE_CDP2 = "";
pub const __DEPRECATED__ = @compileError("unable to translate macro: undefined identifier `deprecated`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/cuda_device_runtime_api.h:156:10
pub const __CDPRT_DEPRECATED = @compileError("unable to translate C expr: unexpected token ''");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/cuda_device_runtime_api.h:166:10
pub inline fn __CUDART_API_PTDS(api: anytype) @TypeOf(api) {
    _ = &api;
    return api;
}
pub inline fn __CUDART_API_PTSZ(api: anytype) @TypeOf(api) {
    _ = &api;
    return api;
}
pub const cudaSignalExternalSemaphoresAsync = __CUDART_API_PTSZ(cudaSignalExternalSemaphoresAsync_v2);
pub const cudaWaitExternalSemaphoresAsync = __CUDART_API_PTSZ(cudaWaitExternalSemaphoresAsync_v2);
pub const cudaStreamGetCaptureInfo = __CUDART_API_PTSZ(cudaStreamGetCaptureInfo_v2);
pub const cudaGetDeviceProperties = cudaGetDeviceProperties_v2;
pub const __dv = @compileError("unable to translate C expr: unexpected token ''");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/cuda_runtime_api.h:254:9
pub const CUDART_DEVICE = __device__;
pub const __DRIVER_FUNCTIONS_H__ = "";
pub const __VECTOR_FUNCTIONS_H__ = "";
pub const __VECTOR_FUNCTIONS_DECL__ = @compileError("unable to translate C expr: unexpected token 'static'");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/vector_functions.h:68:9
pub const __VECTOR_FUNCTIONS_HPP__ = "";
pub const __DEVICE_LAUNCH_PARAMETERS_H__ = "";
pub const __STORAGE__ = @compileError("unable to translate C expr: unexpected token 'extern'");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/device_launch_parameters.h:61:9
pub inline fn __cudaGet_threadIdx() @TypeOf(threadIdx) {
    return threadIdx;
}
pub inline fn __cudaGet_blockIdx() @TypeOf(blockIdx) {
    return blockIdx;
}
pub inline fn __cudaGet_blockDim() @TypeOf(blockDim) {
    return blockDim;
}
pub inline fn __cudaGet_gridDim() @TypeOf(gridDim) {
    return gridDim;
}
pub inline fn __cudaGet_warpSize() @TypeOf(warpSize) {
    return warpSize;
}
pub const CURAND_H_ = "";
pub const CURANDAPI = @compileError("unable to translate C expr: unexpected token '__stdcall'");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/curand.h:64:9
pub const CURAND_VER_MAJOR = @as(c_int, 10);
pub const CURAND_VER_MINOR = @as(c_int, 3);
pub const CURAND_VER_PATCH = @as(c_int, 9);
pub const CURAND_VER_BUILD = @as(c_int, 55);
pub const CURAND_VERSION = ((CURAND_VER_MAJOR * @as(c_int, 1000)) + (CURAND_VER_MINOR * @as(c_int, 100))) + CURAND_VER_PATCH;
pub const __CUDA_FP16_H__ = "";
pub const ___CUDA_FP16_STRINGIFY_INNERMOST = @compileError("unable to translate C expr: unexpected token '#'");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/cuda_fp16.h:136:9
pub inline fn __CUDA_FP16_STRINGIFY(x: anytype) @TypeOf(___CUDA_FP16_STRINGIFY_INNERMOST(x)) {
    _ = &x;
    return ___CUDA_FP16_STRINGIFY_INNERMOST(x);
}
pub const __NV_TARGET_H = "";
pub const _NV__TARGET_MACROS = "";
pub inline fn _NV_EVAL1(x: anytype) @TypeOf(x) {
    _ = &x;
    return x;
}
pub inline fn _NV_EVAL(x: anytype) @TypeOf(_NV_EVAL1(x)) {
    _ = &x;
    return _NV_EVAL1(x);
}
pub const _NV_CONCAT_EVAL1 = @compileError("unable to translate C expr: expected ',' or ')' instead got '##'");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__preprocessor:33:9
pub inline fn _NV_CONCAT_EVAL(l: anytype, r: anytype) @TypeOf(_NV_CONCAT_EVAL1(l, r)) {
    _ = &l;
    _ = &r;
    return _NV_CONCAT_EVAL1(l, r);
}
pub inline fn _NV_IF_0(t: anytype, f: anytype) @TypeOf(f) {
    _ = &t;
    _ = &f;
    return f;
}
pub inline fn _NV_IF_1(t: anytype, f: anytype) @TypeOf(t) {
    _ = &t;
    _ = &f;
    return t;
}
pub const _NV_IF_BIT = @compileError("unable to translate macro: undefined identifier `_NV_IF_`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__preprocessor:39:9
pub inline fn _NV_IF__EVAL(@"fn": anytype, t: anytype, f: anytype) @TypeOf(_NV_EVAL(@"fn"(t, f))) {
    _ = &@"fn";
    _ = &t;
    _ = &f;
    return _NV_EVAL(@"fn"(t, f));
}
pub inline fn _NV_IF_EVAL(cond: anytype, t: anytype, f: anytype) @TypeOf(_NV_IF__EVAL(_NV_IF_BIT(cond), t, f)) {
    _ = &cond;
    _ = &t;
    _ = &f;
    return _NV_IF__EVAL(_NV_IF_BIT(cond), t, f);
}
pub inline fn _NV_IF1(cond: anytype, t: anytype, f: anytype) @TypeOf(_NV_IF_EVAL(cond, t, f)) {
    _ = &cond;
    _ = &t;
    _ = &f;
    return _NV_IF_EVAL(cond, t, f);
}
pub inline fn _NV_IF(cond: anytype, t: anytype, f: anytype) @TypeOf(_NV_IF1(_NV_EVAL(cond), _NV_EVAL(t), _NV_EVAL(f))) {
    _ = &cond;
    _ = &t;
    _ = &f;
    return _NV_IF1(_NV_EVAL(cond), _NV_EVAL(t), _NV_EVAL(f));
}
pub const _NV_TARGET_ARCH_TO_SELECTOR_350 = @compileError("unable to translate macro: undefined identifier `nv`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:20:9
pub const _NV_TARGET_ARCH_TO_SELECTOR_370 = @compileError("unable to translate macro: undefined identifier `nv`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:21:9
pub const _NV_TARGET_ARCH_TO_SELECTOR_500 = @compileError("unable to translate macro: undefined identifier `nv`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:22:9
pub const _NV_TARGET_ARCH_TO_SELECTOR_520 = @compileError("unable to translate macro: undefined identifier `nv`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:23:9
pub const _NV_TARGET_ARCH_TO_SELECTOR_530 = @compileError("unable to translate macro: undefined identifier `nv`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:24:9
pub const _NV_TARGET_ARCH_TO_SELECTOR_600 = @compileError("unable to translate macro: undefined identifier `nv`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:25:9
pub const _NV_TARGET_ARCH_TO_SELECTOR_610 = @compileError("unable to translate macro: undefined identifier `nv`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:26:9
pub const _NV_TARGET_ARCH_TO_SELECTOR_620 = @compileError("unable to translate macro: undefined identifier `nv`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:27:9
pub const _NV_TARGET_ARCH_TO_SELECTOR_700 = @compileError("unable to translate macro: undefined identifier `nv`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:28:9
pub const _NV_TARGET_ARCH_TO_SELECTOR_720 = @compileError("unable to translate macro: undefined identifier `nv`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:29:9
pub const _NV_TARGET_ARCH_TO_SELECTOR_750 = @compileError("unable to translate macro: undefined identifier `nv`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:30:9
pub const _NV_TARGET_ARCH_TO_SELECTOR_800 = @compileError("unable to translate macro: undefined identifier `nv`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:31:9
pub const _NV_TARGET_ARCH_TO_SELECTOR_860 = @compileError("unable to translate macro: undefined identifier `nv`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:32:9
pub const _NV_TARGET_ARCH_TO_SELECTOR_870 = @compileError("unable to translate macro: undefined identifier `nv`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:33:9
pub const _NV_TARGET_ARCH_TO_SELECTOR_890 = @compileError("unable to translate macro: undefined identifier `nv`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:34:9
pub const _NV_TARGET_ARCH_TO_SELECTOR_900 = @compileError("unable to translate macro: undefined identifier `nv`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:35:9
pub const _NV_TARGET_ARCH_TO_SELECTOR_1000 = @compileError("unable to translate macro: undefined identifier `nv`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:36:9
pub const _NV_TARGET_ARCH_TO_SM_350 = @as(c_int, 35);
pub const _NV_TARGET_ARCH_TO_SM_370 = @as(c_int, 37);
pub const _NV_TARGET_ARCH_TO_SM_500 = @as(c_int, 50);
pub const _NV_TARGET_ARCH_TO_SM_520 = @as(c_int, 52);
pub const _NV_TARGET_ARCH_TO_SM_530 = @as(c_int, 53);
pub const _NV_TARGET_ARCH_TO_SM_600 = @as(c_int, 60);
pub const _NV_TARGET_ARCH_TO_SM_610 = @as(c_int, 61);
pub const _NV_TARGET_ARCH_TO_SM_620 = @as(c_int, 62);
pub const _NV_TARGET_ARCH_TO_SM_700 = @as(c_int, 70);
pub const _NV_TARGET_ARCH_TO_SM_720 = @as(c_int, 72);
pub const _NV_TARGET_ARCH_TO_SM_750 = @as(c_int, 75);
pub const _NV_TARGET_ARCH_TO_SM_800 = @as(c_int, 80);
pub const _NV_TARGET_ARCH_TO_SM_860 = @as(c_int, 86);
pub const _NV_TARGET_ARCH_TO_SM_870 = @as(c_int, 87);
pub const _NV_TARGET_ARCH_TO_SM_890 = @as(c_int, 89);
pub const _NV_TARGET_ARCH_TO_SM_900 = @as(c_int, 90);
pub const _NV_TARGET_ARCH_TO_SM_1000 = @as(c_int, 100);
pub const _NV_COMPILER_NVCC = "";
pub const _NV_TARGET_VAL_SM_35 = @as(c_int, 350);
pub const _NV_TARGET_VAL_SM_37 = @as(c_int, 370);
pub const _NV_TARGET_VAL_SM_50 = @as(c_int, 500);
pub const _NV_TARGET_VAL_SM_52 = @as(c_int, 520);
pub const _NV_TARGET_VAL_SM_53 = @as(c_int, 530);
pub const _NV_TARGET_VAL_SM_60 = @as(c_int, 600);
pub const _NV_TARGET_VAL_SM_61 = @as(c_int, 610);
pub const _NV_TARGET_VAL_SM_62 = @as(c_int, 620);
pub const _NV_TARGET_VAL_SM_70 = @as(c_int, 700);
pub const _NV_TARGET_VAL_SM_72 = @as(c_int, 720);
pub const _NV_TARGET_VAL_SM_75 = @as(c_int, 750);
pub const _NV_TARGET_VAL_SM_80 = @as(c_int, 800);
pub const _NV_TARGET_VAL_SM_86 = @as(c_int, 860);
pub const _NV_TARGET_VAL_SM_87 = @as(c_int, 870);
pub const _NV_TARGET_VAL_SM_89 = @as(c_int, 890);
pub const _NV_TARGET_VAL_SM_90 = @as(c_int, 900);
pub const _NV_TARGET_VAL_SM_100 = @as(c_int, 1000);
pub const _NV_TARGET_VAL = @as(c_int, 0);
pub const _NV_TARGET_IS_HOST = @as(c_int, 1);
pub const _NV_TARGET_IS_DEVICE = @as(c_int, 0);
pub inline fn _NV_DEVICE_CHECK(q: anytype) @TypeOf(@"false") {
    _ = &q;
    return @"false";
}
pub inline fn _NV_TARGET_PROVIDES(q: anytype) @TypeOf(_NV_DEVICE_CHECK(_NV_TARGET_VAL >= q)) {
    _ = &q;
    return _NV_DEVICE_CHECK(_NV_TARGET_VAL >= q);
}
pub inline fn _NV_TARGET_IS_EXACTLY(q: anytype) @TypeOf(_NV_DEVICE_CHECK(_NV_TARGET_VAL == q)) {
    _ = &q;
    return _NV_DEVICE_CHECK(_NV_TARGET_VAL == q);
}
pub const _NV_TARGET___NV_PROVIDES_SM_35 = _NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_35);
pub const _NV_TARGET___NV_PROVIDES_SM_37 = _NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_37);
pub const _NV_TARGET___NV_PROVIDES_SM_50 = _NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_50);
pub const _NV_TARGET___NV_PROVIDES_SM_52 = _NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_52);
pub const _NV_TARGET___NV_PROVIDES_SM_53 = _NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_53);
pub const _NV_TARGET___NV_PROVIDES_SM_60 = _NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_60);
pub const _NV_TARGET___NV_PROVIDES_SM_61 = _NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_61);
pub const _NV_TARGET___NV_PROVIDES_SM_62 = _NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_62);
pub const _NV_TARGET___NV_PROVIDES_SM_70 = _NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_70);
pub const _NV_TARGET___NV_PROVIDES_SM_72 = _NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_72);
pub const _NV_TARGET___NV_PROVIDES_SM_75 = _NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_75);
pub const _NV_TARGET___NV_PROVIDES_SM_80 = _NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_80);
pub const _NV_TARGET___NV_PROVIDES_SM_86 = _NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_86);
pub const _NV_TARGET___NV_PROVIDES_SM_87 = _NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_87);
pub const _NV_TARGET___NV_PROVIDES_SM_89 = _NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_89);
pub const _NV_TARGET___NV_PROVIDES_SM_90 = _NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_90);
pub const _NV_TARGET___NV_PROVIDES_SM_100 = _NV_TARGET_PROVIDES(_NV_TARGET_VAL_SM_100);
pub const _NV_TARGET___NV_IS_EXACTLY_SM_35 = _NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_35);
pub const _NV_TARGET___NV_IS_EXACTLY_SM_37 = _NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_37);
pub const _NV_TARGET___NV_IS_EXACTLY_SM_50 = _NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_50);
pub const _NV_TARGET___NV_IS_EXACTLY_SM_52 = _NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_52);
pub const _NV_TARGET___NV_IS_EXACTLY_SM_53 = _NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_53);
pub const _NV_TARGET___NV_IS_EXACTLY_SM_60 = _NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_60);
pub const _NV_TARGET___NV_IS_EXACTLY_SM_61 = _NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_61);
pub const _NV_TARGET___NV_IS_EXACTLY_SM_62 = _NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_62);
pub const _NV_TARGET___NV_IS_EXACTLY_SM_70 = _NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_70);
pub const _NV_TARGET___NV_IS_EXACTLY_SM_72 = _NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_72);
pub const _NV_TARGET___NV_IS_EXACTLY_SM_75 = _NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_75);
pub const _NV_TARGET___NV_IS_EXACTLY_SM_80 = _NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_80);
pub const _NV_TARGET___NV_IS_EXACTLY_SM_86 = _NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_86);
pub const _NV_TARGET___NV_IS_EXACTLY_SM_87 = _NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_87);
pub const _NV_TARGET___NV_IS_EXACTLY_SM_89 = _NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_89);
pub const _NV_TARGET___NV_IS_EXACTLY_SM_90 = _NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_90);
pub const _NV_TARGET___NV_IS_EXACTLY_SM_100 = _NV_TARGET_IS_EXACTLY(_NV_TARGET_VAL_SM_100);
pub const NV_PROVIDES_SM_35 = @compileError("unable to translate macro: undefined identifier `__NV_PROVIDES_SM_35`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:207:9
pub const NV_PROVIDES_SM_37 = @compileError("unable to translate macro: undefined identifier `__NV_PROVIDES_SM_37`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:208:9
pub const NV_PROVIDES_SM_50 = @compileError("unable to translate macro: undefined identifier `__NV_PROVIDES_SM_50`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:209:9
pub const NV_PROVIDES_SM_52 = @compileError("unable to translate macro: undefined identifier `__NV_PROVIDES_SM_52`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:210:9
pub const NV_PROVIDES_SM_53 = @compileError("unable to translate macro: undefined identifier `__NV_PROVIDES_SM_53`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:211:9
pub const NV_PROVIDES_SM_60 = @compileError("unable to translate macro: undefined identifier `__NV_PROVIDES_SM_60`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:212:9
pub const NV_PROVIDES_SM_61 = @compileError("unable to translate macro: undefined identifier `__NV_PROVIDES_SM_61`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:213:9
pub const NV_PROVIDES_SM_62 = @compileError("unable to translate macro: undefined identifier `__NV_PROVIDES_SM_62`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:214:9
pub const NV_PROVIDES_SM_70 = @compileError("unable to translate macro: undefined identifier `__NV_PROVIDES_SM_70`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:215:9
pub const NV_PROVIDES_SM_72 = @compileError("unable to translate macro: undefined identifier `__NV_PROVIDES_SM_72`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:216:9
pub const NV_PROVIDES_SM_75 = @compileError("unable to translate macro: undefined identifier `__NV_PROVIDES_SM_75`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:217:9
pub const NV_PROVIDES_SM_80 = @compileError("unable to translate macro: undefined identifier `__NV_PROVIDES_SM_80`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:218:9
pub const NV_PROVIDES_SM_86 = @compileError("unable to translate macro: undefined identifier `__NV_PROVIDES_SM_86`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:219:9
pub const NV_PROVIDES_SM_87 = @compileError("unable to translate macro: undefined identifier `__NV_PROVIDES_SM_87`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:220:9
pub const NV_PROVIDES_SM_89 = @compileError("unable to translate macro: undefined identifier `__NV_PROVIDES_SM_89`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:221:9
pub const NV_PROVIDES_SM_90 = @compileError("unable to translate macro: undefined identifier `__NV_PROVIDES_SM_90`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:222:9
pub const NV_PROVIDES_SM_100 = @compileError("unable to translate macro: undefined identifier `__NV_PROVIDES_SM_100`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:223:9
pub const NV_IS_EXACTLY_SM_35 = @compileError("unable to translate macro: undefined identifier `__NV_IS_EXACTLY_SM_35`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:225:9
pub const NV_IS_EXACTLY_SM_37 = @compileError("unable to translate macro: undefined identifier `__NV_IS_EXACTLY_SM_37`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:226:9
pub const NV_IS_EXACTLY_SM_50 = @compileError("unable to translate macro: undefined identifier `__NV_IS_EXACTLY_SM_50`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:227:9
pub const NV_IS_EXACTLY_SM_52 = @compileError("unable to translate macro: undefined identifier `__NV_IS_EXACTLY_SM_52`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:228:9
pub const NV_IS_EXACTLY_SM_53 = @compileError("unable to translate macro: undefined identifier `__NV_IS_EXACTLY_SM_53`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:229:9
pub const NV_IS_EXACTLY_SM_60 = @compileError("unable to translate macro: undefined identifier `__NV_IS_EXACTLY_SM_60`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:230:9
pub const NV_IS_EXACTLY_SM_61 = @compileError("unable to translate macro: undefined identifier `__NV_IS_EXACTLY_SM_61`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:231:9
pub const NV_IS_EXACTLY_SM_62 = @compileError("unable to translate macro: undefined identifier `__NV_IS_EXACTLY_SM_62`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:232:9
pub const NV_IS_EXACTLY_SM_70 = @compileError("unable to translate macro: undefined identifier `__NV_IS_EXACTLY_SM_70`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:233:9
pub const NV_IS_EXACTLY_SM_72 = @compileError("unable to translate macro: undefined identifier `__NV_IS_EXACTLY_SM_72`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:234:9
pub const NV_IS_EXACTLY_SM_75 = @compileError("unable to translate macro: undefined identifier `__NV_IS_EXACTLY_SM_75`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:235:9
pub const NV_IS_EXACTLY_SM_80 = @compileError("unable to translate macro: undefined identifier `__NV_IS_EXACTLY_SM_80`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:236:9
pub const NV_IS_EXACTLY_SM_86 = @compileError("unable to translate macro: undefined identifier `__NV_IS_EXACTLY_SM_86`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:237:9
pub const NV_IS_EXACTLY_SM_87 = @compileError("unable to translate macro: undefined identifier `__NV_IS_EXACTLY_SM_87`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:238:9
pub const NV_IS_EXACTLY_SM_89 = @compileError("unable to translate macro: undefined identifier `__NV_IS_EXACTLY_SM_89`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:239:9
pub const NV_IS_EXACTLY_SM_90 = @compileError("unable to translate macro: undefined identifier `__NV_IS_EXACTLY_SM_90`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:240:9
pub const NV_IS_EXACTLY_SM_100 = @compileError("unable to translate macro: undefined identifier `__NV_IS_EXACTLY_SM_100`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:241:9
pub const NV_HAS_FEATURE_SM_90a = NV_NO_TARGET;
pub const NV_HAS_FEATURE_SM_100a = NV_NO_TARGET;
pub const NV_IS_HOST = @compileError("unable to translate macro: undefined identifier `__NV_IS_HOST`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:248:9
pub const NV_IS_DEVICE = @compileError("unable to translate macro: undefined identifier `__NV_IS_DEVICE`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:249:9
pub const NV_ANY_TARGET = @compileError("unable to translate macro: undefined identifier `__NV_ANY_TARGET`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:251:9
pub const NV_NO_TARGET = @compileError("unable to translate macro: undefined identifier `__NV_NO_TARGET`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:252:9
pub const _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_35 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_37 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_50 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_52 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_53 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_60 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_61 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_62 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_70 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_72 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_75 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_80 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_86 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_87 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_89 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_90 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_IS_EXACTLY_SM_100 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_90a = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_HAS_FEATURE_SM_100a = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_IS_HOST = @as(c_int, 1);
pub const _NV_TARGET_BOOL___NV_IS_DEVICE = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_ANY_TARGET = @as(c_int, 1);
pub const _NV_TARGET_BOOL___NV_NO_TARGET = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_PROVIDES_SM_35 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_PROVIDES_SM_37 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_PROVIDES_SM_50 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_PROVIDES_SM_52 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_PROVIDES_SM_53 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_PROVIDES_SM_60 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_PROVIDES_SM_61 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_PROVIDES_SM_62 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_PROVIDES_SM_70 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_PROVIDES_SM_72 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_PROVIDES_SM_75 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_PROVIDES_SM_80 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_PROVIDES_SM_86 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_PROVIDES_SM_87 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_PROVIDES_SM_89 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_PROVIDES_SM_90 = @as(c_int, 0);
pub const _NV_TARGET_BOOL___NV_PROVIDES_SM_100 = @as(c_int, 0);
pub const _NV_ARCH_COND_CAT1 = @compileError("unable to translate macro: undefined identifier `_NV_TARGET_BOOL_`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:500:11
pub inline fn _NV_ARCH_COND_CAT(cond: anytype) @TypeOf(_NV_EVAL(_NV_ARCH_COND_CAT1(cond))) {
    _ = &cond;
    return _NV_EVAL(_NV_ARCH_COND_CAT1(cond));
}
pub const _NV_TARGET_EMPTY_PARAM = @compileError("unable to translate C expr: unexpected token ';'");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:503:11
pub const _NV_BLOCK_EXPAND = @compileError("unable to translate C expr: unexpected token '{'");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/nv/detail/__target_macros:512:13
pub inline fn _NV_TARGET_IF(cond: anytype, t: anytype) @TypeOf(_NV_IF(_NV_ARCH_COND_CAT(cond), t, _NV_TARGET_EMPTY_PARAM)) {
    _ = &cond;
    _ = &t;
    return _NV_IF(_NV_ARCH_COND_CAT(cond), t, _NV_TARGET_EMPTY_PARAM);
}
pub inline fn _NV_TARGET_IF_ELSE(cond: anytype, t: anytype, f: anytype) @TypeOf(_NV_IF(_NV_ARCH_COND_CAT(cond), t, f)) {
    _ = &cond;
    _ = &t;
    _ = &f;
    return _NV_IF(_NV_ARCH_COND_CAT(cond), t, f);
}
pub inline fn NV_IF_TARGET(cond: anytype, t: anytype) @TypeOf(_NV_BLOCK_EXPAND(_NV_TARGET_IF(cond, t))) {
    _ = &cond;
    _ = &t;
    return _NV_BLOCK_EXPAND(_NV_TARGET_IF(cond, t));
}
pub inline fn NV_IF_ELSE_TARGET(cond: anytype, t: anytype, f: anytype) @TypeOf(_NV_BLOCK_EXPAND(_NV_TARGET_IF_ELSE(cond, t, f))) {
    _ = &cond;
    _ = &t;
    _ = &f;
    return _NV_BLOCK_EXPAND(_NV_TARGET_IF_ELSE(cond, t, f));
}
pub const __CUDA_FP16_INLINE__ = @compileError("unable to translate C expr: unexpected token 'inline'");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/cuda_fp16.h:4514:9
pub const __CUDA_FP16_FORCEINLINE__ = __forceinline__;
pub const __CUDA_ALIGN__ = @compileError("unable to translate macro: undefined identifier `aligned`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/cuda_fp16.h:4527:9
pub const __CUDA_FP16_CONSTEXPR__ = "";
pub const __CUDA_FP16_HPP__ = "";
pub inline fn IF_DEVICE_OR_CUDACC(d: anytype, c: anytype, f: anytype) @TypeOf(NV_IF_ELSE_TARGET(NV_IS_DEVICE, d, f)) {
    _ = &d;
    _ = &c;
    _ = &f;
    return NV_IF_ELSE_TARGET(NV_IS_DEVICE, d, f);
}
pub const __BINARY_OP_HALF_MACRO = @compileError("unable to translate macro: undefined identifier `__half`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/cuda_fp16.hpp:66:9
pub const __BINARY_OP_HALF2_MACRO = @compileError("unable to translate macro: undefined identifier `__half2`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/cuda_fp16.hpp:72:9
pub const __TERNARY_OP_HALF_MACRO = @compileError("unable to translate macro: undefined identifier `__half`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/cuda_fp16.hpp:78:9
pub const __TERNARY_OP_HALF2_MACRO = @compileError("unable to translate macro: undefined identifier `__half2`");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/cuda_fp16.hpp:84:9
pub const __CUDA_BF16_H__ = "";
pub const ___CUDA_BF16_STRINGIFY_INNERMOST = @compileError("unable to translate C expr: unexpected token '#'");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/cuda_bf16.h:143:9
pub inline fn __CUDA_BF16_STRINGIFY(x: anytype) @TypeOf(___CUDA_BF16_STRINGIFY_INNERMOST(x)) {
    _ = &x;
    return ___CUDA_BF16_STRINGIFY_INNERMOST(x);
}
pub const __CUDA_BF16_INLINE__ = @compileError("unable to translate C expr: unexpected token 'inline'");
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include/cuda_bf16.h:4329:9
pub const __CUDA_BF16_FORCEINLINE__ = __forceinline__;
pub const __CUDA_BF16_CONSTEXPR__ = "";
pub const __CUDA_BF16_HPP__ = "";
pub const _LIBCPP_STDBOOL_H = "";
pub const __STDBOOL_H = "";
pub const __bool_true_false_are_defined = @as(c_int, 1);
pub const @"bool" = bool;
pub const @"true" = @as(c_int, 1);
pub const @"false" = @as(c_int, 0);
pub const TOMO_OPS_API = @compileError("unable to translate macro: undefined identifier `dllimport`");
// C:\Users\haeryu\Desktop\workspace\zig\study\cuda\tomo\src\kernel/tomo_dll.h:6:9
pub const TOMO_EXTERN_C = @compileError("unable to translate C expr: unexpected token 'extern'");
// C:\Users\haeryu\Desktop\workspace\zig\study\cuda\tomo\src\kernel/tomo_dll.h:12:9
pub const threadlocaleinfostruct = struct_threadlocaleinfostruct;
pub const threadmbcinfostruct = struct_threadmbcinfostruct;
pub const __lc_time_data = struct___lc_time_data;
pub const localeinfo_struct = struct_localeinfo_struct;
pub const tagLC_ID = struct_tagLC_ID;
pub const _div_t = struct__div_t;
pub const _ldiv_t = struct__ldiv_t;
pub const _heapinfo = struct__heapinfo;
pub const CUctx_st = struct_CUctx_st;
pub const CUmod_st = struct_CUmod_st;
pub const CUfunc_st = struct_CUfunc_st;
pub const CUlib_st = struct_CUlib_st;
pub const CUkern_st = struct_CUkern_st;
pub const CUarray_st = struct_CUarray_st;
pub const CUmipmappedArray_st = struct_CUmipmappedArray_st;
pub const CUtexref_st = struct_CUtexref_st;
pub const CUsurfref_st = struct_CUsurfref_st;
pub const CUevent_st = struct_CUevent_st;
pub const CUstream_st = struct_CUstream_st;
pub const CUgraphicsResource_st = struct_CUgraphicsResource_st;
pub const CUextMemory_st = struct_CUextMemory_st;
pub const CUextSemaphore_st = struct_CUextSemaphore_st;
pub const CUgraph_st = struct_CUgraph_st;
pub const CUgraphNode_st = struct_CUgraphNode_st;
pub const CUgraphExec_st = struct_CUgraphExec_st;
pub const CUmemPoolHandle_st = struct_CUmemPoolHandle_st;
pub const CUuserObject_st = struct_CUuserObject_st;
pub const CUgraphDeviceUpdatableNode_st = struct_CUgraphDeviceUpdatableNode_st;
pub const CUasyncCallbackEntry_st = struct_CUasyncCallbackEntry_st;
pub const CUgreenCtx_st = struct_CUgreenCtx_st;
pub const CUuuid_st = struct_CUuuid_st;
pub const CUmemFabricHandle_st = struct_CUmemFabricHandle_st;
pub const CUipcEventHandle_st = struct_CUipcEventHandle_st;
pub const CUipcMemHandle_st = struct_CUipcMemHandle_st;
pub const CUipcMem_flags_enum = enum_CUipcMem_flags_enum;
pub const CUmemAttach_flags_enum = enum_CUmemAttach_flags_enum;
pub const CUctx_flags_enum = enum_CUctx_flags_enum;
pub const CUevent_sched_flags_enum = enum_CUevent_sched_flags_enum;
pub const cl_event_flags_enum = enum_cl_event_flags_enum;
pub const cl_context_flags_enum = enum_cl_context_flags_enum;
pub const CUstream_flags_enum = enum_CUstream_flags_enum;
pub const CUevent_flags_enum = enum_CUevent_flags_enum;
pub const CUevent_record_flags_enum = enum_CUevent_record_flags_enum;
pub const CUevent_wait_flags_enum = enum_CUevent_wait_flags_enum;
pub const CUstreamWaitValue_flags_enum = enum_CUstreamWaitValue_flags_enum;
pub const CUstreamWriteValue_flags_enum = enum_CUstreamWriteValue_flags_enum;
pub const CUstreamBatchMemOpType_enum = enum_CUstreamBatchMemOpType_enum;
pub const CUstreamMemoryBarrier_flags_enum = enum_CUstreamMemoryBarrier_flags_enum;
pub const CUstreamBatchMemOpParams_union = union_CUstreamBatchMemOpParams_union;
pub const CUDA_BATCH_MEM_OP_NODE_PARAMS_v1_st = struct_CUDA_BATCH_MEM_OP_NODE_PARAMS_v1_st;
pub const CUDA_BATCH_MEM_OP_NODE_PARAMS_v2_st = struct_CUDA_BATCH_MEM_OP_NODE_PARAMS_v2_st;
pub const CUoccupancy_flags_enum = enum_CUoccupancy_flags_enum;
pub const CUstreamUpdateCaptureDependencies_flags_enum = enum_CUstreamUpdateCaptureDependencies_flags_enum;
pub const CUasyncNotificationType_enum = enum_CUasyncNotificationType_enum;
pub const CUasyncNotificationInfo_st = struct_CUasyncNotificationInfo_st;
pub const CUarray_format_enum = enum_CUarray_format_enum;
pub const CUaddress_mode_enum = enum_CUaddress_mode_enum;
pub const CUfilter_mode_enum = enum_CUfilter_mode_enum;
pub const CUdevice_attribute_enum = enum_CUdevice_attribute_enum;
pub const CUdevprop_st = struct_CUdevprop_st;
pub const CUpointer_attribute_enum = enum_CUpointer_attribute_enum;
pub const CUfunction_attribute_enum = enum_CUfunction_attribute_enum;
pub const CUfunc_cache_enum = enum_CUfunc_cache_enum;
pub const CUsharedconfig_enum = enum_CUsharedconfig_enum;
pub const CUshared_carveout_enum = enum_CUshared_carveout_enum;
pub const CUmemorytype_enum = enum_CUmemorytype_enum;
pub const CUcomputemode_enum = enum_CUcomputemode_enum;
pub const CUmem_advise_enum = enum_CUmem_advise_enum;
pub const CUmem_range_attribute_enum = enum_CUmem_range_attribute_enum;
pub const CUjit_option_enum = enum_CUjit_option_enum;
pub const CUjit_target_enum = enum_CUjit_target_enum;
pub const CUjit_fallback_enum = enum_CUjit_fallback_enum;
pub const CUjit_cacheMode_enum = enum_CUjit_cacheMode_enum;
pub const CUjitInputType_enum = enum_CUjitInputType_enum;
pub const CUlinkState_st = struct_CUlinkState_st;
pub const CUgraphicsRegisterFlags_enum = enum_CUgraphicsRegisterFlags_enum;
pub const CUgraphicsMapResourceFlags_enum = enum_CUgraphicsMapResourceFlags_enum;
pub const CUarray_cubemap_face_enum = enum_CUarray_cubemap_face_enum;
pub const CUlimit_enum = enum_CUlimit_enum;
pub const CUresourcetype_enum = enum_CUresourcetype_enum;
pub const CUaccessProperty_enum = enum_CUaccessProperty_enum;
pub const CUaccessPolicyWindow_st = struct_CUaccessPolicyWindow_st;
pub const CUDA_KERNEL_NODE_PARAMS_st = struct_CUDA_KERNEL_NODE_PARAMS_st;
pub const CUDA_KERNEL_NODE_PARAMS_v2_st = struct_CUDA_KERNEL_NODE_PARAMS_v2_st;
pub const CUDA_KERNEL_NODE_PARAMS_v3_st = struct_CUDA_KERNEL_NODE_PARAMS_v3_st;
pub const CUDA_MEMSET_NODE_PARAMS_st = struct_CUDA_MEMSET_NODE_PARAMS_st;
pub const CUDA_MEMSET_NODE_PARAMS_v2_st = struct_CUDA_MEMSET_NODE_PARAMS_v2_st;
pub const CUDA_HOST_NODE_PARAMS_st = struct_CUDA_HOST_NODE_PARAMS_st;
pub const CUDA_HOST_NODE_PARAMS_v2_st = struct_CUDA_HOST_NODE_PARAMS_v2_st;
pub const CUgraphConditionalNodeType_enum = enum_CUgraphConditionalNodeType_enum;
pub const CUgraphNodeType_enum = enum_CUgraphNodeType_enum;
pub const CUgraphDependencyType_enum = enum_CUgraphDependencyType_enum;
pub const CUgraphEdgeData_st = struct_CUgraphEdgeData_st;
pub const CUgraphInstantiateResult_enum = enum_CUgraphInstantiateResult_enum;
pub const CUDA_GRAPH_INSTANTIATE_PARAMS_st = struct_CUDA_GRAPH_INSTANTIATE_PARAMS_st;
pub const CUsynchronizationPolicy_enum = enum_CUsynchronizationPolicy_enum;
pub const CUclusterSchedulingPolicy_enum = enum_CUclusterSchedulingPolicy_enum;
pub const CUlaunchMemSyncDomain_enum = enum_CUlaunchMemSyncDomain_enum;
pub const CUlaunchMemSyncDomainMap_st = struct_CUlaunchMemSyncDomainMap_st;
pub const CUlaunchAttributeID_enum = enum_CUlaunchAttributeID_enum;
pub const CUlaunchAttributeValue_union = union_CUlaunchAttributeValue_union;
pub const CUlaunchAttribute_st = struct_CUlaunchAttribute_st;
pub const CUlaunchConfig_st = struct_CUlaunchConfig_st;
pub const CUstreamCaptureStatus_enum = enum_CUstreamCaptureStatus_enum;
pub const CUstreamCaptureMode_enum = enum_CUstreamCaptureMode_enum;
pub const CUdriverProcAddress_flags_enum = enum_CUdriverProcAddress_flags_enum;
pub const CUdriverProcAddressQueryResult_enum = enum_CUdriverProcAddressQueryResult_enum;
pub const CUexecAffinityType_enum = enum_CUexecAffinityType_enum;
pub const CUexecAffinitySmCount_st = struct_CUexecAffinitySmCount_st;
pub const CUexecAffinityParam_st = struct_CUexecAffinityParam_st;
pub const CUcigDataType_enum = enum_CUcigDataType_enum;
pub const CUctxCigParam_st = struct_CUctxCigParam_st;
pub const CUctxCreateParams_st = struct_CUctxCreateParams_st;
pub const CUlibraryOption_enum = enum_CUlibraryOption_enum;
pub const CUlibraryHostUniversalFunctionAndDataTable_st = struct_CUlibraryHostUniversalFunctionAndDataTable_st;
pub const cudaError_enum = enum_cudaError_enum;
pub const CUdevice_P2PAttribute_enum = enum_CUdevice_P2PAttribute_enum;
pub const CUDA_MEMCPY2D_st = struct_CUDA_MEMCPY2D_st;
pub const CUDA_MEMCPY3D_st = struct_CUDA_MEMCPY3D_st;
pub const CUDA_MEMCPY3D_PEER_st = struct_CUDA_MEMCPY3D_PEER_st;
pub const CUDA_MEMCPY_NODE_PARAMS_st = struct_CUDA_MEMCPY_NODE_PARAMS_st;
pub const CUDA_ARRAY_DESCRIPTOR_st = struct_CUDA_ARRAY_DESCRIPTOR_st;
pub const CUDA_ARRAY3D_DESCRIPTOR_st = struct_CUDA_ARRAY3D_DESCRIPTOR_st;
pub const CUDA_ARRAY_SPARSE_PROPERTIES_st = struct_CUDA_ARRAY_SPARSE_PROPERTIES_st;
pub const CUDA_ARRAY_MEMORY_REQUIREMENTS_st = struct_CUDA_ARRAY_MEMORY_REQUIREMENTS_st;
pub const CUDA_RESOURCE_DESC_st = struct_CUDA_RESOURCE_DESC_st;
pub const CUDA_TEXTURE_DESC_st = struct_CUDA_TEXTURE_DESC_st;
pub const CUresourceViewFormat_enum = enum_CUresourceViewFormat_enum;
pub const CUDA_RESOURCE_VIEW_DESC_st = struct_CUDA_RESOURCE_VIEW_DESC_st;
pub const CUtensorMap_st = struct_CUtensorMap_st;
pub const CUtensorMapDataType_enum = enum_CUtensorMapDataType_enum;
pub const CUtensorMapInterleave_enum = enum_CUtensorMapInterleave_enum;
pub const CUtensorMapSwizzle_enum = enum_CUtensorMapSwizzle_enum;
pub const CUtensorMapL2promotion_enum = enum_CUtensorMapL2promotion_enum;
pub const CUtensorMapFloatOOBfill_enum = enum_CUtensorMapFloatOOBfill_enum;
pub const CUtensorMapIm2ColWideMode_enum = enum_CUtensorMapIm2ColWideMode_enum;
pub const CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st = struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st;
pub const CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum = enum_CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum;
pub const CUDA_LAUNCH_PARAMS_st = struct_CUDA_LAUNCH_PARAMS_st;
pub const CUexternalMemoryHandleType_enum = enum_CUexternalMemoryHandleType_enum;
pub const CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st = struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st;
pub const CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st = struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st;
pub const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st = struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st;
pub const CUexternalSemaphoreHandleType_enum = enum_CUexternalSemaphoreHandleType_enum;
pub const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st = struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st;
pub const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st = struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st;
pub const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st = struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st;
pub const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st = struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st;
pub const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2_st = struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2_st;
pub const CUDA_EXT_SEM_WAIT_NODE_PARAMS_st = struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st;
pub const CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2_st = struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2_st;
pub const CUmemAllocationHandleType_enum = enum_CUmemAllocationHandleType_enum;
pub const CUmemAccess_flags_enum = enum_CUmemAccess_flags_enum;
pub const CUmemLocationType_enum = enum_CUmemLocationType_enum;
pub const CUmemAllocationType_enum = enum_CUmemAllocationType_enum;
pub const CUmemAllocationGranularity_flags_enum = enum_CUmemAllocationGranularity_flags_enum;
pub const CUmemRangeHandleType_enum = enum_CUmemRangeHandleType_enum;
pub const CUmemRangeFlags_enum = enum_CUmemRangeFlags_enum;
pub const CUarraySparseSubresourceType_enum = enum_CUarraySparseSubresourceType_enum;
pub const CUmemOperationType_enum = enum_CUmemOperationType_enum;
pub const CUmemHandleType_enum = enum_CUmemHandleType_enum;
pub const CUarrayMapInfo_st = struct_CUarrayMapInfo_st;
pub const CUmemLocation_st = struct_CUmemLocation_st;
pub const CUmemAllocationCompType_enum = enum_CUmemAllocationCompType_enum;
pub const CUmemAllocationProp_st = struct_CUmemAllocationProp_st;
pub const CUmulticastGranularity_flags_enum = enum_CUmulticastGranularity_flags_enum;
pub const CUmulticastObjectProp_st = struct_CUmulticastObjectProp_st;
pub const CUmemAccessDesc_st = struct_CUmemAccessDesc_st;
pub const CUgraphExecUpdateResult_enum = enum_CUgraphExecUpdateResult_enum;
pub const CUgraphExecUpdateResultInfo_st = struct_CUgraphExecUpdateResultInfo_st;
pub const CUmemPool_attribute_enum = enum_CUmemPool_attribute_enum;
pub const CUmemPoolProps_st = struct_CUmemPoolProps_st;
pub const CUmemPoolPtrExportData_st = struct_CUmemPoolPtrExportData_st;
pub const CUDA_MEM_ALLOC_NODE_PARAMS_v1_st = struct_CUDA_MEM_ALLOC_NODE_PARAMS_v1_st;
pub const CUDA_MEM_ALLOC_NODE_PARAMS_v2_st = struct_CUDA_MEM_ALLOC_NODE_PARAMS_v2_st;
pub const CUDA_MEM_FREE_NODE_PARAMS_st = struct_CUDA_MEM_FREE_NODE_PARAMS_st;
pub const CUgraphMem_attribute_enum = enum_CUgraphMem_attribute_enum;
pub const CUDA_CHILD_GRAPH_NODE_PARAMS_st = struct_CUDA_CHILD_GRAPH_NODE_PARAMS_st;
pub const CUDA_EVENT_RECORD_NODE_PARAMS_st = struct_CUDA_EVENT_RECORD_NODE_PARAMS_st;
pub const CUDA_EVENT_WAIT_NODE_PARAMS_st = struct_CUDA_EVENT_WAIT_NODE_PARAMS_st;
pub const CUgraphNodeParams_st = struct_CUgraphNodeParams_st;
pub const CUflushGPUDirectRDMAWritesOptions_enum = enum_CUflushGPUDirectRDMAWritesOptions_enum;
pub const CUGPUDirectRDMAWritesOrdering_enum = enum_CUGPUDirectRDMAWritesOrdering_enum;
pub const CUflushGPUDirectRDMAWritesScope_enum = enum_CUflushGPUDirectRDMAWritesScope_enum;
pub const CUflushGPUDirectRDMAWritesTarget_enum = enum_CUflushGPUDirectRDMAWritesTarget_enum;
pub const CUgraphDebugDot_flags_enum = enum_CUgraphDebugDot_flags_enum;
pub const CUuserObject_flags_enum = enum_CUuserObject_flags_enum;
pub const CUuserObjectRetain_flags_enum = enum_CUuserObjectRetain_flags_enum;
pub const CUgraphInstantiate_flags_enum = enum_CUgraphInstantiate_flags_enum;
pub const CUdeviceNumaConfig_enum = enum_CUdeviceNumaConfig_enum;
pub const CUprocessState_enum = enum_CUprocessState_enum;
pub const CUcheckpointLockArgs_st = struct_CUcheckpointLockArgs_st;
pub const CUcheckpointCheckpointArgs_st = struct_CUcheckpointCheckpointArgs_st;
pub const CUcheckpointRestoreArgs_st = struct_CUcheckpointRestoreArgs_st;
pub const CUcheckpointUnlockArgs_st = struct_CUcheckpointUnlockArgs_st;
pub const CUmemcpyFlags_enum = enum_CUmemcpyFlags_enum;
pub const CUmemcpySrcAccessOrder_enum = enum_CUmemcpySrcAccessOrder_enum;
pub const CUmemcpyAttributes_st = struct_CUmemcpyAttributes_st;
pub const CUmemcpy3DOperandType_enum = enum_CUmemcpy3DOperandType_enum;
pub const CUoffset3D_st = struct_CUoffset3D_st;
pub const CUextent3D_st = struct_CUextent3D_st;
pub const CUmemcpy3DOperand_st = struct_CUmemcpy3DOperand_st;
pub const CUDA_MEMCPY3D_BATCH_OP_st = struct_CUDA_MEMCPY3D_BATCH_OP_st;
pub const CUmoduleLoadingMode_enum = enum_CUmoduleLoadingMode_enum;
pub const CUmemDecompressAlgorithm_enum = enum_CUmemDecompressAlgorithm_enum;
pub const CUmemDecompressParams_st = struct_CUmemDecompressParams_st;
pub const CUfunctionLoadingState_enum = enum_CUfunctionLoadingState_enum;
pub const CUcoredumpSettings_enum = enum_CUcoredumpSettings_enum;
pub const CUdevResourceDesc_st = struct_CUdevResourceDesc_st;
pub const CUdevSmResource_st = struct_CUdevSmResource_st;
pub const CUdevResource_st = struct_CUdevResource_st;
pub const cudaRoundMode = enum_cudaRoundMode;
pub const cudaError = enum_cudaError;
pub const cudaChannelFormatKind = enum_cudaChannelFormatKind;
pub const cudaChannelFormatDesc = struct_cudaChannelFormatDesc;
pub const cudaArray = struct_cudaArray;
pub const cudaMipmappedArray = struct_cudaMipmappedArray;
pub const cudaArraySparseProperties = struct_cudaArraySparseProperties;
pub const cudaArrayMemoryRequirements = struct_cudaArrayMemoryRequirements;
pub const cudaMemoryType = enum_cudaMemoryType;
pub const cudaMemcpyKind = enum_cudaMemcpyKind;
pub const cudaPitchedPtr = struct_cudaPitchedPtr;
pub const cudaExtent = struct_cudaExtent;
pub const cudaPos = struct_cudaPos;
pub const cudaMemcpy3DParms = struct_cudaMemcpy3DParms;
pub const cudaMemcpyNodeParams = struct_cudaMemcpyNodeParams;
pub const cudaMemcpy3DPeerParms = struct_cudaMemcpy3DPeerParms;
pub const cudaMemsetParams = struct_cudaMemsetParams;
pub const cudaMemsetParamsV2 = struct_cudaMemsetParamsV2;
pub const cudaAccessProperty = enum_cudaAccessProperty;
pub const cudaAccessPolicyWindow = struct_cudaAccessPolicyWindow;
pub const cudaHostNodeParams = struct_cudaHostNodeParams;
pub const cudaHostNodeParamsV2 = struct_cudaHostNodeParamsV2;
pub const cudaStreamCaptureStatus = enum_cudaStreamCaptureStatus;
pub const cudaStreamCaptureMode = enum_cudaStreamCaptureMode;
pub const cudaSynchronizationPolicy = enum_cudaSynchronizationPolicy;
pub const cudaClusterSchedulingPolicy = enum_cudaClusterSchedulingPolicy;
pub const cudaStreamUpdateCaptureDependenciesFlags = enum_cudaStreamUpdateCaptureDependenciesFlags;
pub const cudaUserObjectFlags = enum_cudaUserObjectFlags;
pub const cudaUserObjectRetainFlags = enum_cudaUserObjectRetainFlags;
pub const cudaGraphicsResource = struct_cudaGraphicsResource;
pub const cudaGraphicsRegisterFlags = enum_cudaGraphicsRegisterFlags;
pub const cudaGraphicsMapFlags = enum_cudaGraphicsMapFlags;
pub const cudaGraphicsCubeFace = enum_cudaGraphicsCubeFace;
pub const cudaResourceType = enum_cudaResourceType;
pub const cudaResourceViewFormat = enum_cudaResourceViewFormat;
pub const cudaResourceDesc = struct_cudaResourceDesc;
pub const cudaResourceViewDesc = struct_cudaResourceViewDesc;
pub const cudaPointerAttributes = struct_cudaPointerAttributes;
pub const cudaFuncAttributes = struct_cudaFuncAttributes;
pub const cudaFuncAttribute = enum_cudaFuncAttribute;
pub const cudaFuncCache = enum_cudaFuncCache;
pub const cudaSharedMemConfig = enum_cudaSharedMemConfig;
pub const cudaSharedCarveout = enum_cudaSharedCarveout;
pub const cudaComputeMode = enum_cudaComputeMode;
pub const cudaLimit = enum_cudaLimit;
pub const cudaMemoryAdvise = enum_cudaMemoryAdvise;
pub const cudaMemRangeAttribute = enum_cudaMemRangeAttribute;
pub const cudaFlushGPUDirectRDMAWritesOptions = enum_cudaFlushGPUDirectRDMAWritesOptions;
pub const cudaGPUDirectRDMAWritesOrdering = enum_cudaGPUDirectRDMAWritesOrdering;
pub const cudaFlushGPUDirectRDMAWritesScope = enum_cudaFlushGPUDirectRDMAWritesScope;
pub const cudaFlushGPUDirectRDMAWritesTarget = enum_cudaFlushGPUDirectRDMAWritesTarget;
pub const cudaDeviceAttr = enum_cudaDeviceAttr;
pub const cudaMemPoolAttr = enum_cudaMemPoolAttr;
pub const cudaMemLocationType = enum_cudaMemLocationType;
pub const cudaMemLocation = struct_cudaMemLocation;
pub const cudaMemAccessFlags = enum_cudaMemAccessFlags;
pub const cudaMemAccessDesc = struct_cudaMemAccessDesc;
pub const cudaMemAllocationType = enum_cudaMemAllocationType;
pub const cudaMemAllocationHandleType = enum_cudaMemAllocationHandleType;
pub const cudaMemPoolProps = struct_cudaMemPoolProps;
pub const cudaMemPoolPtrExportData = struct_cudaMemPoolPtrExportData;
pub const cudaMemAllocNodeParams = struct_cudaMemAllocNodeParams;
pub const cudaMemAllocNodeParamsV2 = struct_cudaMemAllocNodeParamsV2;
pub const cudaMemFreeNodeParams = struct_cudaMemFreeNodeParams;
pub const cudaGraphMemAttributeType = enum_cudaGraphMemAttributeType;
pub const cudaMemcpyFlags = enum_cudaMemcpyFlags;
pub const cudaMemcpySrcAccessOrder = enum_cudaMemcpySrcAccessOrder;
pub const cudaMemcpyAttributes = struct_cudaMemcpyAttributes;
pub const cudaMemcpy3DOperandType = enum_cudaMemcpy3DOperandType;
pub const cudaOffset3D = struct_cudaOffset3D;
pub const cudaMemcpy3DOperand = struct_cudaMemcpy3DOperand;
pub const cudaMemcpy3DBatchOp = struct_cudaMemcpy3DBatchOp;
pub const cudaDeviceP2PAttr = enum_cudaDeviceP2PAttr;
pub const cudaDeviceProp = struct_cudaDeviceProp;
pub const cudaIpcEventHandle_st = struct_cudaIpcEventHandle_st;
pub const cudaIpcMemHandle_st = struct_cudaIpcMemHandle_st;
pub const cudaMemFabricHandle_st = struct_cudaMemFabricHandle_st;
pub const cudaExternalMemoryHandleType = enum_cudaExternalMemoryHandleType;
pub const cudaExternalMemoryHandleDesc = struct_cudaExternalMemoryHandleDesc;
pub const cudaExternalMemoryBufferDesc = struct_cudaExternalMemoryBufferDesc;
pub const cudaExternalMemoryMipmappedArrayDesc = struct_cudaExternalMemoryMipmappedArrayDesc;
pub const cudaExternalSemaphoreHandleType = enum_cudaExternalSemaphoreHandleType;
pub const cudaExternalSemaphoreHandleDesc = struct_cudaExternalSemaphoreHandleDesc;
pub const cudaExternalSemaphoreSignalParams_v1 = struct_cudaExternalSemaphoreSignalParams_v1;
pub const cudaExternalSemaphoreWaitParams_v1 = struct_cudaExternalSemaphoreWaitParams_v1;
pub const cudaExternalSemaphoreSignalParams = struct_cudaExternalSemaphoreSignalParams;
pub const cudaExternalSemaphoreWaitParams = struct_cudaExternalSemaphoreWaitParams;
pub const CUexternalMemory_st = struct_CUexternalMemory_st;
pub const CUexternalSemaphore_st = struct_CUexternalSemaphore_st;
pub const cudaJitOption = enum_cudaJitOption;
pub const cudaLibraryOption = enum_cudaLibraryOption;
pub const cudalibraryHostUniversalFunctionAndDataTable = struct_cudalibraryHostUniversalFunctionAndDataTable;
pub const cudaJit_CacheMode = enum_cudaJit_CacheMode;
pub const cudaJit_Fallback = enum_cudaJit_Fallback;
pub const cudaCGScope = enum_cudaCGScope;
pub const cudaLaunchParams = struct_cudaLaunchParams;
pub const cudaKernelNodeParams = struct_cudaKernelNodeParams;
pub const cudaKernelNodeParamsV2 = struct_cudaKernelNodeParamsV2;
pub const cudaExternalSemaphoreSignalNodeParams = struct_cudaExternalSemaphoreSignalNodeParams;
pub const cudaExternalSemaphoreSignalNodeParamsV2 = struct_cudaExternalSemaphoreSignalNodeParamsV2;
pub const cudaExternalSemaphoreWaitNodeParams = struct_cudaExternalSemaphoreWaitNodeParams;
pub const cudaExternalSemaphoreWaitNodeParamsV2 = struct_cudaExternalSemaphoreWaitNodeParamsV2;
pub const cudaGraphConditionalHandleFlags = enum_cudaGraphConditionalHandleFlags;
pub const cudaGraphConditionalNodeType = enum_cudaGraphConditionalNodeType;
pub const cudaConditionalNodeParams = struct_cudaConditionalNodeParams;
pub const cudaGraphNodeType = enum_cudaGraphNodeType;
pub const cudaChildGraphNodeParams = struct_cudaChildGraphNodeParams;
pub const cudaEventRecordNodeParams = struct_cudaEventRecordNodeParams;
pub const cudaEventWaitNodeParams = struct_cudaEventWaitNodeParams;
pub const cudaGraphNodeParams = struct_cudaGraphNodeParams;
pub const cudaGraphDependencyType_enum = enum_cudaGraphDependencyType_enum;
pub const cudaGraphEdgeData_st = struct_cudaGraphEdgeData_st;
pub const cudaGraphExecUpdateResult = enum_cudaGraphExecUpdateResult;
pub const cudaGraphInstantiateParams_st = struct_cudaGraphInstantiateParams_st;
pub const cudaGraphExecUpdateResultInfo_st = struct_cudaGraphExecUpdateResultInfo_st;
pub const cudaGraphKernelNodeField = enum_cudaGraphKernelNodeField;
pub const cudaGraphKernelNodeUpdate = struct_cudaGraphKernelNodeUpdate;
pub const cudaGetDriverEntryPointFlags = enum_cudaGetDriverEntryPointFlags;
pub const cudaDriverEntryPointQueryResult = enum_cudaDriverEntryPointQueryResult;
pub const cudaGraphDebugDotFlags = enum_cudaGraphDebugDotFlags;
pub const cudaGraphInstantiateFlags = enum_cudaGraphInstantiateFlags;
pub const cudaLaunchMemSyncDomainMap_st = struct_cudaLaunchMemSyncDomainMap_st;
pub const cudaLaunchAttribute_st = struct_cudaLaunchAttribute_st;
pub const cudaLaunchConfig_st = struct_cudaLaunchConfig_st;
pub const cudaDeviceNumaConfig = enum_cudaDeviceNumaConfig;
pub const cudaAsyncCallbackEntry = struct_cudaAsyncCallbackEntry;
pub const cudaAsyncNotificationType_enum = enum_cudaAsyncNotificationType_enum;
pub const cudaAsyncNotificationInfo = struct_cudaAsyncNotificationInfo;
pub const cudaSurfaceBoundaryMode = enum_cudaSurfaceBoundaryMode;
pub const cudaSurfaceFormatMode = enum_cudaSurfaceFormatMode;
pub const cudaTextureAddressMode = enum_cudaTextureAddressMode;
pub const cudaTextureFilterMode = enum_cudaTextureFilterMode;
pub const cudaTextureReadMode = enum_cudaTextureReadMode;
pub const cudaTextureDesc = struct_cudaTextureDesc;
pub const curandStatus = enum_curandStatus;
pub const curandRngType = enum_curandRngType;
pub const curandOrdering = enum_curandOrdering;
pub const curandDirectionVectorSet = enum_curandDirectionVectorSet;
pub const curandGenerator_st = struct_curandGenerator_st;
pub const curandDistributionShift_st = struct_curandDistributionShift_st;
pub const curandDistributionM2Shift_st = struct_curandDistributionM2Shift_st;
pub const curandHistogramM2_st = struct_curandHistogramM2_st;
pub const curandDiscreteDistribution_st = struct_curandDiscreteDistribution_st;
pub const curandMethod = enum_curandMethod;
