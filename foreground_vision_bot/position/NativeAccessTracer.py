from __future__ import annotations

import ctypes
import os
import struct
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from time import monotonic
from typing import Any, Literal, Protocol

from ctypes import wintypes


EXCEPTION_DEBUG_EVENT = 1
CREATE_THREAD_DEBUG_EVENT = 2
CREATE_PROCESS_DEBUG_EVENT = 3
EXIT_THREAD_DEBUG_EVENT = 4
EXIT_PROCESS_DEBUG_EVENT = 5
LOAD_DLL_DEBUG_EVENT = 6
UNLOAD_DLL_DEBUG_EVENT = 7
OUTPUT_DEBUG_STRING_EVENT = 8
RIP_EVENT = 9

EXCEPTION_BREAKPOINT = 0x80000003
EXCEPTION_SINGLE_STEP = 0x80000004
DBG_CONTINUE = 0x00010002
DBG_EXCEPTION_NOT_HANDLED = 0x80010001

TH32CS_SNAPTHREAD = 0x00000004
THREAD_SUSPEND_RESUME = 0x0002
THREAD_GET_CONTEXT = 0x0008
THREAD_SET_CONTEXT = 0x0010
THREAD_QUERY_INFORMATION = 0x0040
PROCESS_QUERY_INFORMATION = 0x0400

WOW64_CONTEXT_i386 = 0x00010000
WOW64_CONTEXT_CONTROL = WOW64_CONTEXT_i386 | 0x00000001
WOW64_CONTEXT_INTEGER = WOW64_CONTEXT_i386 | 0x00000002
WOW64_CONTEXT_SEGMENTS = WOW64_CONTEXT_i386 | 0x00000004
WOW64_CONTEXT_DEBUG_REGISTERS = WOW64_CONTEXT_i386 | 0x00000010
WOW64_CONTEXT_FULL = (
    WOW64_CONTEXT_CONTROL | WOW64_CONTEXT_INTEGER | WOW64_CONTEXT_SEGMENTS
)
TRACE_CONTEXT_FLAGS = WOW64_CONTEXT_FULL | WOW64_CONTEXT_DEBUG_REGISTERS

RF_EFLAGS = 0x00010000
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


DWORD = ctypes.c_uint32
LONG = ctypes.c_int32
WORD = ctypes.c_uint16

WatchAccess = Literal["execute", "write", "access"]


class TraceMemory(Protocol):
    def read(self, address: int, size: int) -> bytes: ...


class NativeAccessTraceError(RuntimeError):
    """Debugger-assisted native access tracing failed."""


class WOW64_FLOATING_SAVE_AREA(ctypes.Structure):
    _fields_ = (
        ("ControlWord", DWORD),
        ("StatusWord", DWORD),
        ("TagWord", DWORD),
        ("ErrorOffset", DWORD),
        ("ErrorSelector", DWORD),
        ("DataOffset", DWORD),
        ("DataSelector", DWORD),
        ("RegisterArea", ctypes.c_ubyte * 80),
        ("Cr0NpxState", DWORD),
    )


class WOW64_CONTEXT(ctypes.Structure):
    _fields_ = (
        ("ContextFlags", DWORD),
        ("Dr0", DWORD),
        ("Dr1", DWORD),
        ("Dr2", DWORD),
        ("Dr3", DWORD),
        ("Dr6", DWORD),
        ("Dr7", DWORD),
        ("FloatSave", WOW64_FLOATING_SAVE_AREA),
        ("SegGs", DWORD),
        ("SegFs", DWORD),
        ("SegEs", DWORD),
        ("SegDs", DWORD),
        ("Edi", DWORD),
        ("Esi", DWORD),
        ("Ebx", DWORD),
        ("Edx", DWORD),
        ("Ecx", DWORD),
        ("Eax", DWORD),
        ("Ebp", DWORD),
        ("Eip", DWORD),
        ("SegCs", DWORD),
        ("EFlags", DWORD),
        ("Esp", DWORD),
        ("SegSs", DWORD),
        ("ExtendedRegisters", ctypes.c_ubyte * 512),
    )


class EXCEPTION_RECORD(ctypes.Structure):
    _fields_ = (
        ("ExceptionCode", DWORD),
        ("ExceptionFlags", DWORD),
        ("ExceptionRecord", ctypes.c_void_p),
        ("ExceptionAddress", ctypes.c_void_p),
        ("NumberParameters", DWORD),
        ("ExceptionInformation", ctypes.c_size_t * 15),
    )


class EXCEPTION_DEBUG_INFO(ctypes.Structure):
    _fields_ = (
        ("ExceptionRecord", EXCEPTION_RECORD),
        ("dwFirstChance", DWORD),
    )


class CREATE_THREAD_DEBUG_INFO(ctypes.Structure):
    _fields_ = (
        ("hThread", wintypes.HANDLE),
        ("lpThreadLocalBase", ctypes.c_void_p),
        ("lpStartAddress", ctypes.c_void_p),
    )


class CREATE_PROCESS_DEBUG_INFO(ctypes.Structure):
    _fields_ = (
        ("hFile", wintypes.HANDLE),
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("lpBaseOfImage", ctypes.c_void_p),
        ("dwDebugInfoFileOffset", DWORD),
        ("nDebugInfoSize", DWORD),
        ("lpThreadLocalBase", ctypes.c_void_p),
        ("lpStartAddress", ctypes.c_void_p),
        ("lpImageName", ctypes.c_void_p),
        ("fUnicode", WORD),
    )


class EXIT_THREAD_DEBUG_INFO(ctypes.Structure):
    _fields_ = (("dwExitCode", DWORD),)


class EXIT_PROCESS_DEBUG_INFO(ctypes.Structure):
    _fields_ = (("dwExitCode", DWORD),)


class LOAD_DLL_DEBUG_INFO(ctypes.Structure):
    _fields_ = (
        ("hFile", wintypes.HANDLE),
        ("lpBaseOfDll", ctypes.c_void_p),
        ("dwDebugInfoFileOffset", DWORD),
        ("nDebugInfoSize", DWORD),
        ("lpImageName", ctypes.c_void_p),
        ("fUnicode", WORD),
    )


class UNLOAD_DLL_DEBUG_INFO(ctypes.Structure):
    _fields_ = (("lpBaseOfDll", ctypes.c_void_p),)


class OUTPUT_DEBUG_STRING_INFO(ctypes.Structure):
    _fields_ = (
        ("lpDebugStringData", ctypes.c_void_p),
        ("fUnicode", WORD),
        ("nDebugStringLength", WORD),
    )


class RIP_INFO(ctypes.Structure):
    _fields_ = (
        ("dwError", DWORD),
        ("dwType", DWORD),
    )


class DEBUG_EVENT_UNION(ctypes.Union):
    _fields_ = (
        ("Exception", EXCEPTION_DEBUG_INFO),
        ("CreateThread", CREATE_THREAD_DEBUG_INFO),
        ("CreateProcessInfo", CREATE_PROCESS_DEBUG_INFO),
        ("ExitThread", EXIT_THREAD_DEBUG_INFO),
        ("ExitProcess", EXIT_PROCESS_DEBUG_INFO),
        ("LoadDll", LOAD_DLL_DEBUG_INFO),
        ("UnloadDll", UNLOAD_DLL_DEBUG_INFO),
        ("DebugString", OUTPUT_DEBUG_STRING_INFO),
        ("RipInfo", RIP_INFO),
    )


class DEBUG_EVENT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = (
        ("dwDebugEventCode", DWORD),
        ("dwProcessId", DWORD),
        ("dwThreadId", DWORD),
        ("u", DEBUG_EVENT_UNION),
    )


class THREADENTRY32(ctypes.Structure):
    _fields_ = (
        ("dwSize", DWORD),
        ("cntUsage", DWORD),
        ("th32ThreadID", DWORD),
        ("th32OwnerProcessID", DWORD),
        ("tpBasePri", LONG),
        ("tpDeltaPri", LONG),
        ("dwFlags", DWORD),
    )


@dataclass(frozen=True, slots=True)
class AccessWatchpoint:
    label: str
    address: int
    access: WatchAccess = "access"
    size: int = 4

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("watchpoint label cannot be empty")
        if self.address <= 0:
            raise ValueError("watchpoint address must be positive")
        if self.access not in ("execute", "write", "access"):
            raise ValueError(f"unsupported watchpoint access {self.access!r}")
        if self.access == "execute" and self.size != 1:
            raise ValueError("execute watchpoints must use size 1")
        if self.size not in (1, 2, 4):
            raise ValueError("x86 hardware watchpoint size must be 1, 2, or 4")
        if self.address % self.size:
            raise ValueError(
                f"watchpoint 0x{self.address:X} is not aligned to {self.size} bytes"
            )


@dataclass(frozen=True, slots=True)
class DecodedInstruction:
    address: int
    size: int
    mnemonic: str
    operands: str
    bytes_hex: str
    effective_addresses: tuple[int, ...] = ()
    matched_watch_address: bool = False

    @property
    def text(self) -> str:
        return f"{self.mnemonic} {self.operands}".strip()


@dataclass(frozen=True, slots=True)
class AccessTraceHit:
    phase: str
    watch_label: str
    watch_address: int
    thread_id: int
    event_eip: int
    instruction: DecodedInstruction | None
    registers: Mapping[str, int]
    actor_registers: Mapping[str, str]
    stack_words: tuple[int, ...]
    frame_returns: tuple[int, ...]
    timestamp_seconds: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["registers"] = dict(self.registers)
        payload["actor_registers"] = dict(self.actor_registers)
        return payload


@dataclass(frozen=True, slots=True)
class AccessTracePhaseResult:
    phase: str
    watchpoints: tuple[AccessWatchpoint, ...]
    hits: tuple[AccessTraceHit, ...]
    ignored_outside_module: int
    process_exited: bool
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "watchpoints": [asdict(item) for item in self.watchpoints],
            "hits": [item.to_dict() for item in self.hits],
            "ignored_outside_module": self.ignored_outside_module,
            "process_exited": self.process_exited,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _length_encoding(size: int) -> int:
    return {1: 0b00, 2: 0b01, 4: 0b11}[size]


def _access_encoding(access: WatchAccess) -> int:
    return {"execute": 0b00, "write": 0b01, "access": 0b11}[access]


def encode_dr7(watchpoints: Sequence[AccessWatchpoint]) -> int:
    """Build an x86 DR7 value for up to four local hardware watchpoints."""

    if len(watchpoints) > 4:
        raise ValueError("x86 exposes only four hardware breakpoint slots")
    dr7 = 0
    for slot, watchpoint in enumerate(watchpoints):
        dr7 |= 1 << (slot * 2)  # local enable Ln
        shift = 16 + slot * 4
        dr7 |= _access_encoding(watchpoint.access) << shift
        dr7 |= _length_encoding(watchpoint.size) << (shift + 2)
    return dr7


def chunk_watchpoints(
    watchpoints: Sequence[AccessWatchpoint],
    *,
    size: int = 4,
) -> tuple[tuple[AccessWatchpoint, ...], ...]:
    if size <= 0 or size > 4:
        raise ValueError("hardware watchpoint chunks must contain one to four items")
    return tuple(
        tuple(watchpoints[index : index + size])
        for index in range(0, len(watchpoints), size)
    )


def context_registers(context: WOW64_CONTEXT) -> dict[str, int]:
    return {
        "eax": int(context.Eax),
        "ebx": int(context.Ebx),
        "ecx": int(context.Ecx),
        "edx": int(context.Edx),
        "esi": int(context.Esi),
        "edi": int(context.Edi),
        "ebp": int(context.Ebp),
        "esp": int(context.Esp),
        "eip": int(context.Eip),
        "eflags": int(context.EFlags),
    }


def actor_register_matches(
    registers: Mapping[str, int],
    *,
    player_base: int | None,
    monster_bases: Iterable[int],
) -> dict[str, str]:
    monsters = {int(value) for value in monster_bases}
    result: dict[str, str] = {}
    for name, value in registers.items():
        if name in ("eip", "eflags"):
            continue
        if player_base is not None and value == player_base:
            result[name] = "player"
        elif value in monsters:
            result[name] = "monster"
    return result


def instruction_hit_ranking(
    phases: Iterable[AccessTracePhaseResult],
    *,
    module_base: int,
    module_size: int,
    limit: int = 4,
) -> tuple[int, ...]:
    """Greedily select instructions covering the most watched targets."""

    stop = module_base + module_size
    labels_by_address: dict[int, set[str]] = defaultdict(set)
    hits_by_address: Counter[int] = Counter()
    for phase in phases:
        for hit in phase.hits:
            instruction = hit.instruction
            if instruction is None:
                continue
            address = int(instruction.address)
            if not module_base <= address < stop:
                continue
            labels_by_address[address].add(hit.watch_label)
            hits_by_address[address] += 1

    selected: list[int] = []
    covered: set[str] = set()
    candidates = set(labels_by_address)
    while candidates and len(selected) < max(0, int(limit)):
        best = max(
            candidates,
            key=lambda address: (
                len(labels_by_address[address] - covered),
                len(labels_by_address[address]),
                hits_by_address[address],
                -address,
            ),
        )
        selected.append(best)
        covered.update(labels_by_address[best])
        candidates.remove(best)
    return tuple(selected)


class _CapstoneDecoder:
    def __init__(self) -> None:
        try:
            from capstone import CS_ARCH_X86, CS_MODE_32, Cs  # type: ignore[import-not-found]
            from capstone.x86 import X86_OP_MEM  # type: ignore[import-not-found]
        except ImportError as error:
            raise NativeAccessTraceError(
                "The access tracer requires Capstone. Install it with "
                "'python -m pip install -r requirements_pointer_trace.txt'."
            ) from error
        self._x86_op_mem = int(X86_OP_MEM)
        self._decoder = Cs(CS_ARCH_X86, CS_MODE_32)
        self._decoder.detail = True

    @staticmethod
    def _signed32(value: int) -> int:
        return value & 0xFFFFFFFF

    def _effective_addresses(
        self,
        instruction: Any,
        registers: Mapping[str, int],
    ) -> tuple[int, ...]:
        result: list[int] = []
        for operand in instruction.operands:
            if int(operand.type) != self._x86_op_mem:
                continue
            memory = operand.mem
            base = 0
            index = 0
            if int(memory.base):
                base = int(registers.get(instruction.reg_name(memory.base), 0))
            if int(memory.index):
                index = int(registers.get(instruction.reg_name(memory.index), 0))
            value = base + index * int(memory.scale) + int(memory.disp)
            result.append(self._signed32(value))
        return tuple(result)

    def decode_at(
        self,
        memory: TraceMemory,
        address: int,
        registers: Mapping[str, int],
    ) -> DecodedInstruction | None:
        try:
            code = memory.read(address, 16)
        except Exception:
            return None
        instruction = next(iter(self._decoder.disasm(code, address, count=1)), None)
        if instruction is None:
            return None
        effective = self._effective_addresses(instruction, registers)
        return DecodedInstruction(
            address=int(instruction.address),
            size=int(instruction.size),
            mnemonic=str(instruction.mnemonic),
            operands=str(instruction.op_str),
            bytes_hex=bytes(instruction.bytes).hex(" "),
            effective_addresses=effective,
        )

    def decode_previous_access(
        self,
        memory: TraceMemory,
        event_eip: int,
        registers: Mapping[str, int],
        watch_address: int,
        watch_size: int,
    ) -> DecodedInstruction | None:
        start = max(1, int(event_eip) - 15)
        try:
            code = memory.read(start, int(event_eip) - start)
        except Exception:
            return None
        candidates: list[tuple[tuple[int, int, int], DecodedInstruction]] = []
        watch_stop = watch_address + watch_size
        for relative in range(len(code)):
            candidate_address = start + relative
            sequence = tuple(self._decoder.disasm(code[relative:], candidate_address))
            if not sequence:
                continue
            consumed: list[Any] = []
            for instruction in sequence:
                end = int(instruction.address) + int(instruction.size)
                if end > event_eip:
                    break
                consumed.append(instruction)
                if end == event_eip:
                    last = consumed[-1]
                    effective = self._effective_addresses(last, registers)
                    matched = any(
                        address < watch_stop and address + max(1, watch_size) > watch_address
                        for address in effective
                    )
                    decoded = DecodedInstruction(
                        address=int(last.address),
                        size=int(last.size),
                        mnemonic=str(last.mnemonic),
                        operands=str(last.op_str),
                        bytes_hex=bytes(last.bytes).hex(" "),
                        effective_addresses=effective,
                        matched_watch_address=matched,
                    )
                    score = (
                        int(matched),
                        int(bool(effective)),
                        int(last.size),
                    )
                    candidates.append((score, decoded))
                    break
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]


class _WindowsDebugApi:
    def __init__(self, pid: int) -> None:
        if os.name != "nt":
            raise OSError("Native access tracing is only available on Windows")
        self.pid = int(pid)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure()
        self.use_wow64_context = ctypes.sizeof(ctypes.c_void_p) == 8
        self._verify_32bit_target()

    def _configure(self) -> None:
        k32 = self.kernel32
        k32.DebugActiveProcess.argtypes = (DWORD,)
        k32.DebugActiveProcess.restype = wintypes.BOOL
        k32.DebugActiveProcessStop.argtypes = (DWORD,)
        k32.DebugActiveProcessStop.restype = wintypes.BOOL
        k32.DebugSetProcessKillOnExit.argtypes = (wintypes.BOOL,)
        k32.DebugSetProcessKillOnExit.restype = wintypes.BOOL
        k32.WaitForDebugEvent.argtypes = (
            ctypes.POINTER(DEBUG_EVENT),
            DWORD,
        )
        k32.WaitForDebugEvent.restype = wintypes.BOOL
        k32.ContinueDebugEvent.argtypes = (
            DWORD,
            DWORD,
            DWORD,
        )
        k32.ContinueDebugEvent.restype = wintypes.BOOL
        k32.CreateToolhelp32Snapshot.argtypes = (DWORD, DWORD)
        k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        k32.Thread32First.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(THREADENTRY32),
        )
        k32.Thread32First.restype = wintypes.BOOL
        k32.Thread32Next.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(THREADENTRY32),
        )
        k32.Thread32Next.restype = wintypes.BOOL
        k32.OpenThread.argtypes = (DWORD, wintypes.BOOL, DWORD)
        k32.OpenThread.restype = wintypes.HANDLE
        k32.SuspendThread.argtypes = (wintypes.HANDLE,)
        k32.SuspendThread.restype = DWORD
        k32.ResumeThread.argtypes = (wintypes.HANDLE,)
        k32.ResumeThread.restype = DWORD
        k32.GetThreadContext.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(WOW64_CONTEXT),
        )
        k32.GetThreadContext.restype = wintypes.BOOL
        k32.SetThreadContext.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(WOW64_CONTEXT),
        )
        k32.SetThreadContext.restype = wintypes.BOOL
        if hasattr(k32, "Wow64GetThreadContext"):
            k32.Wow64GetThreadContext.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(WOW64_CONTEXT),
            )
            k32.Wow64GetThreadContext.restype = wintypes.BOOL
            k32.Wow64SetThreadContext.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(WOW64_CONTEXT),
            )
            k32.Wow64SetThreadContext.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = (wintypes.HANDLE,)
        k32.CloseHandle.restype = wintypes.BOOL
        k32.OpenProcess.argtypes = (DWORD, wintypes.BOOL, DWORD)
        k32.OpenProcess.restype = wintypes.HANDLE
        if hasattr(k32, "IsWow64Process"):
            k32.IsWow64Process.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(wintypes.BOOL),
            )
            k32.IsWow64Process.restype = wintypes.BOOL

    def _error(self, prefix: str) -> NativeAccessTraceError:
        code = ctypes.get_last_error()
        return NativeAccessTraceError(f"{prefix}: {ctypes.WinError(code)}")

    def _verify_32bit_target(self) -> None:
        handle = self.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, self.pid)
        if not handle:
            raise self._error(f"OpenProcess failed for PID {self.pid}")
        try:
            if ctypes.sizeof(ctypes.c_void_p) == 8:
                is_wow64 = wintypes.BOOL()
                if not self.kernel32.IsWow64Process(handle, ctypes.byref(is_wow64)):
                    raise self._error("IsWow64Process failed")
                if not bool(is_wow64.value):
                    raise NativeAccessTraceError(
                        "The tracer currently supports a 32-bit target under WOW64; "
                        "the selected process was not reported as 32-bit."
                    )
        finally:
            self.kernel32.CloseHandle(handle)

    def attach(self) -> None:
        if not self.kernel32.DebugActiveProcess(self.pid):
            raise self._error(
                f"DebugActiveProcess failed for PID {self.pid}. Close Cheat Engine "
                "or another debugger first; run this terminal as Administrator only "
                "if Windows reports access denied"
            )
        if not self.kernel32.DebugSetProcessKillOnExit(False):
            try:
                self.kernel32.DebugActiveProcessStop(self.pid)
            finally:
                raise self._error("DebugSetProcessKillOnExit(FALSE) failed")

    def detach(self) -> None:
        if not self.kernel32.DebugActiveProcessStop(self.pid):
            code = ctypes.get_last_error()
            if code not in (0, 87, 1168):
                raise self._error("DebugActiveProcessStop failed")

    def wait(self, timeout_ms: int) -> DEBUG_EVENT | None:
        event = DEBUG_EVENT()
        if self.kernel32.WaitForDebugEvent(ctypes.byref(event), int(timeout_ms)):
            return event
        code = ctypes.get_last_error()
        if code in (0, 121):  # timeout
            return None
        raise self._error("WaitForDebugEvent failed")

    def continue_event(self, event: DEBUG_EVENT, status: int) -> None:
        if not self.kernel32.ContinueDebugEvent(
            int(event.dwProcessId),
            int(event.dwThreadId),
            int(status),
        ):
            raise self._error("ContinueDebugEvent failed")

    def close_handle_if_valid(self, handle: object) -> None:
        value = getattr(handle, "value", handle)
        if value:
            self.kernel32.CloseHandle(wintypes.HANDLE(value))

    def enumerate_threads(self) -> tuple[int, ...]:
        snapshot = self.kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
        value = getattr(snapshot, "value", snapshot)
        if value is None or int(value) == INVALID_HANDLE_VALUE:
            raise self._error("CreateToolhelp32Snapshot(threads) failed")
        try:
            entry = THREADENTRY32()
            entry.dwSize = ctypes.sizeof(THREADENTRY32)
            success = self.kernel32.Thread32First(snapshot, ctypes.byref(entry))
            result: list[int] = []
            while success:
                if int(entry.th32OwnerProcessID) == self.pid:
                    result.append(int(entry.th32ThreadID))
                entry.dwSize = ctypes.sizeof(THREADENTRY32)
                success = self.kernel32.Thread32Next(snapshot, ctypes.byref(entry))
            return tuple(result)
        finally:
            self.kernel32.CloseHandle(snapshot)

    def open_thread(self, thread_id: int) -> wintypes.HANDLE:
        rights = (
            THREAD_SUSPEND_RESUME
            | THREAD_GET_CONTEXT
            | THREAD_SET_CONTEXT
            | THREAD_QUERY_INFORMATION
        )
        handle = self.kernel32.OpenThread(rights, False, int(thread_id))
        if not handle:
            raise self._error(f"OpenThread failed for TID {thread_id}")
        return handle

    def get_context(self, handle: wintypes.HANDLE) -> WOW64_CONTEXT:
        context = WOW64_CONTEXT()
        context.ContextFlags = TRACE_CONTEXT_FLAGS
        if self.use_wow64_context:
            success = self.kernel32.Wow64GetThreadContext(
                handle,
                ctypes.byref(context),
            )
        else:
            success = self.kernel32.GetThreadContext(handle, ctypes.byref(context))
        if not success:
            raise self._error("GetThreadContext failed")
        return context

    def set_context(self, handle: wintypes.HANDLE, context: WOW64_CONTEXT) -> None:
        context.ContextFlags = TRACE_CONTEXT_FLAGS
        if self.use_wow64_context:
            success = self.kernel32.Wow64SetThreadContext(
                handle,
                ctypes.byref(context),
            )
        else:
            success = self.kernel32.SetThreadContext(handle, ctypes.byref(context))
        if not success:
            raise self._error("SetThreadContext failed")

    def program_thread(
        self,
        thread_id: int,
        watchpoints: Sequence[AccessWatchpoint],
        *,
        suspend: bool,
    ) -> None:
        handle = self.open_thread(thread_id)
        suspended = False
        try:
            if suspend:
                previous = self.kernel32.SuspendThread(handle)
                if previous == 0xFFFFFFFF:
                    raise self._error(f"SuspendThread failed for TID {thread_id}")
                suspended = True
            context = self.get_context(handle)
            addresses = [item.address for item in watchpoints]
            context.Dr0 = addresses[0] if len(addresses) > 0 else 0
            context.Dr1 = addresses[1] if len(addresses) > 1 else 0
            context.Dr2 = addresses[2] if len(addresses) > 2 else 0
            context.Dr3 = addresses[3] if len(addresses) > 3 else 0
            context.Dr6 = 0
            context.Dr7 = encode_dr7(watchpoints)
            self.set_context(handle, context)
        finally:
            if suspended:
                self.kernel32.ResumeThread(handle)
            self.kernel32.CloseHandle(handle)

    def read_thread_context(self, thread_id: int) -> tuple[wintypes.HANDLE, WOW64_CONTEXT]:
        handle = self.open_thread(thread_id)
        try:
            return handle, self.get_context(handle)
        except Exception:
            self.kernel32.CloseHandle(handle)
            raise


class NativeHardwareAccessTracer:
    """Attach as a debugger and emulate CE's exact access watchpoints.

    The target process is never modified except for the CPU debug registers in
    its threads. Those registers are cleared before every detach.
    """

    def __init__(
        self,
        memory: TraceMemory,
        *,
        pid: int,
        module_base: int,
        module_size: int,
        player_base: int | None = None,
        monster_bases: Iterable[int] = (),
        status: Callable[[str], None] | None = None,
    ) -> None:
        if pid <= 0:
            raise ValueError("pid must be positive")
        if module_base <= 0 or module_size <= 0:
            raise ValueError("module extent must be positive")
        self.memory = memory
        self.pid = int(pid)
        self.module_base = int(module_base)
        self.module_size = int(module_size)
        self.player_base = None if player_base is None else int(player_base)
        self.monster_bases = tuple(int(value) for value in monster_bases)
        self.status = status or (lambda _message: None)
        self.decoder = _CapstoneDecoder()

    def _stack_words(self, esp: int, count: int = 24) -> tuple[int, ...]:
        try:
            data = self.memory.read(int(esp), int(count) * 4)
        except Exception:
            return ()
        return tuple(struct.unpack(f"<{len(data) // 4}I", data))

    def _frame_returns(self, ebp: int, maximum: int = 16) -> tuple[int, ...]:
        result: list[int] = []
        current = int(ebp)
        seen: set[int] = set()
        for _ in range(maximum):
            if current <= 0x10000 or current in seen:
                break
            seen.add(current)
            try:
                next_frame, return_address = struct.unpack(
                    "<II",
                    self.memory.read(current, 8),
                )
            except Exception:
                break
            if return_address:
                result.append(int(return_address))
            if next_frame <= current or next_frame - current > 0x100000:
                break
            current = int(next_frame)
        return tuple(result)

    def _program_all_threads(
        self,
        api: _WindowsDebugApi,
        watchpoints: Sequence[AccessWatchpoint],
        *,
        suspend: bool,
    ) -> None:
        failures: list[str] = []
        for thread_id in api.enumerate_threads():
            try:
                api.program_thread(thread_id, watchpoints, suspend=suspend)
            except Exception as error:
                failures.append(f"TID {thread_id}: {error}")
        if failures:
            self.status(
                "Hardware watchpoints could not be installed on every thread: "
                + "; ".join(failures[:8])
            )

    def _clear_and_detach(self, api: _WindowsDebugApi) -> None:
        """Freeze every thread, clear DR state, detach, then resume safely."""

        suspended: list[wintypes.HANDLE] = []
        for thread_id in api.enumerate_threads():
            try:
                handle = api.open_thread(thread_id)
                previous = api.kernel32.SuspendThread(handle)
                if previous == 0xFFFFFFFF:
                    api.kernel32.CloseHandle(handle)
                    continue
                suspended.append(handle)
            except Exception:
                continue
        try:
            for handle in suspended:
                try:
                    context = api.get_context(handle)
                    context.Dr0 = 0
                    context.Dr1 = 0
                    context.Dr2 = 0
                    context.Dr3 = 0
                    context.Dr6 = 0
                    context.Dr7 = 0
                    api.set_context(handle, context)
                except Exception:
                    continue

            # A watchpoint may have fired while the threads were being frozen.
            # Drain queued events while the explicit suspend counts keep every
            # thread stopped, so no cleared watchpoint can execute again before
            # detachment.
            while True:
                event = api.wait(0)
                if event is None:
                    break
                code = int(event.dwDebugEventCode)
                status = DBG_CONTINUE
                if code == CREATE_PROCESS_DEBUG_EVENT:
                    api.close_handle_if_valid(event.CreateProcessInfo.hFile)
                    api.close_handle_if_valid(event.CreateProcessInfo.hProcess)
                    api.close_handle_if_valid(event.CreateProcessInfo.hThread)
                elif code == CREATE_THREAD_DEBUG_EVENT:
                    api.close_handle_if_valid(event.CreateThread.hThread)
                elif code == LOAD_DLL_DEBUG_EVENT:
                    api.close_handle_if_valid(event.LoadDll.hFile)
                elif code == EXCEPTION_DEBUG_EVENT:
                    exception_code = int(
                        event.Exception.ExceptionRecord.ExceptionCode
                    )
                    if exception_code not in (
                        EXCEPTION_SINGLE_STEP,
                        EXCEPTION_BREAKPOINT,
                    ):
                        status = DBG_EXCEPTION_NOT_HANDLED
                api.continue_event(event, status)
            api.detach()
        finally:
            for handle in suspended:
                try:
                    api.kernel32.ResumeThread(handle)
                finally:
                    api.kernel32.CloseHandle(handle)

    def trace_phase(
        self,
        watchpoints: Sequence[AccessWatchpoint],
        *,
        phase: str,
        duration_seconds: float,
        maximum_hits: int,
        maximum_hits_per_label: int = 100,
        module_only: bool = True,
    ) -> AccessTracePhaseResult:
        plan = tuple(watchpoints)
        if not plan or len(plan) > 4:
            raise ValueError("trace phase needs one to four watchpoints")
        if duration_seconds <= 0.0:
            raise ValueError("duration_seconds must be positive")
        if maximum_hits <= 0 or maximum_hits_per_label <= 0:
            raise ValueError("hit limits must be positive")

        api = _WindowsDebugApi(self.pid)
        start = monotonic()
        hits: list[AccessTraceHit] = []
        label_hits: Counter[str] = Counter()
        ignored_outside_module = 0
        process_exited = False
        attached = False
        initialized = False
        module_stop = self.module_base + self.module_size

        try:
            api.attach()
            attached = True
            self.status(
                f"Debugger attached for {phase}: "
                + ", ".join(
                    f"{item.label}=0x{item.address:X}/{item.access}" for item in plan
                )
            )
            while monotonic() - start < duration_seconds and len(hits) < maximum_hits:
                event = api.wait(100)
                if event is None:
                    continue
                continue_status = DBG_CONTINUE
                try:
                    if not initialized:
                        self._program_all_threads(api, plan, suspend=False)
                        initialized = True
                    code = int(event.dwDebugEventCode)
                    if code == CREATE_PROCESS_DEBUG_EVENT:
                        api.close_handle_if_valid(event.CreateProcessInfo.hFile)
                        api.close_handle_if_valid(event.CreateProcessInfo.hProcess)
                        api.close_handle_if_valid(event.CreateProcessInfo.hThread)
                    elif code == CREATE_THREAD_DEBUG_EVENT:
                        try:
                            api.program_thread(
                                int(event.dwThreadId),
                                plan,
                                suspend=False,
                            )
                        finally:
                            api.close_handle_if_valid(event.CreateThread.hThread)
                    elif code == LOAD_DLL_DEBUG_EVENT:
                        api.close_handle_if_valid(event.LoadDll.hFile)
                    elif code == EXIT_PROCESS_DEBUG_EVENT:
                        process_exited = True
                    elif code == EXCEPTION_DEBUG_EVENT:
                        exception = event.Exception.ExceptionRecord
                        exception_code = int(exception.ExceptionCode)
                        if exception_code == EXCEPTION_SINGLE_STEP:
                            handle, context = api.read_thread_context(
                                int(event.dwThreadId)
                            )
                            try:
                                triggered = tuple(
                                    slot
                                    for slot in range(len(plan))
                                    if int(context.Dr6) & (1 << slot)
                                )
                                if not triggered:
                                    continue_status = DBG_EXCEPTION_NOT_HANDLED
                                    continue
                                registers = context_registers(context)
                                for slot in triggered:
                                    watchpoint = plan[slot]
                                    if label_hits[watchpoint.label] >= maximum_hits_per_label:
                                        continue
                                    if watchpoint.access == "execute":
                                        instruction = self.decoder.decode_at(
                                            self.memory,
                                            int(context.Eip),
                                            registers,
                                        )
                                    else:
                                        instruction = self.decoder.decode_previous_access(
                                            self.memory,
                                            int(context.Eip),
                                            registers,
                                            watchpoint.address,
                                            watchpoint.size,
                                        )
                                    instruction_address = (
                                        int(instruction.address)
                                        if instruction is not None
                                        else int(context.Eip)
                                    )
                                    if module_only and not (
                                        self.module_base
                                        <= instruction_address
                                        < module_stop
                                    ):
                                        ignored_outside_module += 1
                                        continue
                                    hit = AccessTraceHit(
                                        phase=phase,
                                        watch_label=watchpoint.label,
                                        watch_address=watchpoint.address,
                                        thread_id=int(event.dwThreadId),
                                        event_eip=int(context.Eip),
                                        instruction=instruction,
                                        registers=registers,
                                        actor_registers=actor_register_matches(
                                            registers,
                                            player_base=self.player_base,
                                            monster_bases=self.monster_bases,
                                        ),
                                        stack_words=self._stack_words(int(context.Esp)),
                                        frame_returns=self._frame_returns(int(context.Ebp)),
                                        timestamp_seconds=monotonic() - start,
                                    )
                                    hits.append(hit)
                                    label_hits[watchpoint.label] += 1
                                    if len(hits) >= maximum_hits:
                                        break
                                context.Dr6 = 0
                                context.EFlags = int(context.EFlags) | RF_EFLAGS
                                api.set_context(handle, context)
                            finally:
                                api.kernel32.CloseHandle(handle)
                        elif exception_code == EXCEPTION_BREAKPOINT:
                            continue_status = DBG_CONTINUE
                        else:
                            continue_status = DBG_EXCEPTION_NOT_HANDLED
                    if process_exited:
                        continue_status = DBG_CONTINUE
                finally:
                    api.continue_event(event, continue_status)
                if process_exited:
                    break
        finally:
            if attached and not process_exited:
                self._clear_and_detach(api)

        return AccessTracePhaseResult(
            phase=phase,
            watchpoints=plan,
            hits=tuple(hits),
            ignored_outside_module=ignored_outside_module,
            process_exited=process_exited,
            elapsed_seconds=monotonic() - start,
        )
