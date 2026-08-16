from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from dataclasses import dataclass
from time import monotonic
from typing import Callable, Protocol

from .PositionProvider import PositionProviderError

PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
TH32CS_SNAPMODULE = 0x00000008
TH32CS_SNAPMODULE32 = 0x00000010
ERROR_NO_MORE_FILES = 18
MAX_MODULE_NAME32 = 255
MAX_PATH = 260
MEM_COMMIT = 0x1000
MEM_PRIVATE = 0x20000
PAGE_NOACCESS = 0x01
PAGE_GUARD = 0x100


@dataclass(frozen=True, slots=True)
class MemoryRegion:
    """One committed readable process-memory region."""

    base_address: int
    size: int
    protection: int
    region_type: int


@dataclass(frozen=True, slots=True)
class ModuleInfo:
    """Identity and mapped image extent for one process module."""

    name: str
    path: str
    base_address: int
    size: int


@dataclass(frozen=True, slots=True)
class MemorySearchDiagnostics:
    regions_considered: int = 0
    regions_read: int = 0
    bytes_read: int = 0
    read_failures: int = 0
    matches: int = 0


@dataclass(frozen=True, slots=True)
class MemorySearchProgress:
    regions_considered: int
    regions_read: int
    bytes_read: int
    read_failures: int
    matches: int
    elapsed_seconds: float
    complete: bool = False


class ProcessMemoryError(PositionProviderError):
    """A Win32 process-memory operation failed."""


class MemorySearchCancelled(ProcessMemoryError):
    """A cooperative process-memory search was cancelled."""


class MemorySearchDeadline(ProcessMemoryError):
    """A cooperative process-memory search exceeded its deadline."""


def _cancellation_requested(cancellation: object | None) -> bool:
    if cancellation is None:
        return False
    cancelled = getattr(cancellation, "cancelled", None)
    if cancelled is not None:
        return bool(cancelled() if callable(cancelled) else cancelled)
    is_set = getattr(cancellation, "is_set", None)
    return bool(is_set()) if callable(is_set) else False


def _raise_if_search_stopped(
    cancellation: object | None,
    deadline: float | None,
) -> None:
    if _cancellation_requested(cancellation):
        raise MemorySearchCancelled("Process-memory search was cancelled")
    if deadline is not None and monotonic() >= float(deadline):
        raise MemorySearchDeadline("Process-memory search exceeded its deadline")


class Win32MemoryBackend(Protocol):
    def get_window_process_id(self, window_handle: int) -> int: ...

    def open_process(self, pid: int, access: int) -> int: ...

    def read_process_memory(self, handle: int, address: int, size: int) -> bytes: ...

    def get_module_base(self, pid: int, module_name: str) -> int: ...

    def iter_readable_regions(
        self,
        handle: int,
        *,
        maximum_address: int,
    ) -> tuple[MemoryRegion, ...]: ...

    def close_handle(self, handle: int) -> None: ...


class MODULEENTRY32W(ctypes.Structure):
    _fields_ = (
        ("dwSize", wintypes.DWORD),
        ("th32ModuleID", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("GlblcntUsage", wintypes.DWORD),
        ("ProccntUsage", wintypes.DWORD),
        ("modBaseAddr", ctypes.POINTER(ctypes.c_byte)),
        ("modBaseSize", wintypes.DWORD),
        ("hModule", wintypes.HMODULE),
        ("szModule", wintypes.WCHAR * (MAX_MODULE_NAME32 + 1)),
        ("szExePath", wintypes.WCHAR * MAX_PATH),
    )


if ctypes.sizeof(ctypes.c_void_p) == 8:

    class MEMORY_BASIC_INFORMATION(ctypes.Structure):
        _fields_ = (
            ("BaseAddress", ctypes.c_void_p),
            ("AllocationBase", ctypes.c_void_p),
            ("AllocationProtect", wintypes.DWORD),
            ("PartitionId", wintypes.WORD),
            ("RegionSize", ctypes.c_size_t),
            ("State", wintypes.DWORD),
            ("Protect", wintypes.DWORD),
            ("Type", wintypes.DWORD),
        )

else:

    class MEMORY_BASIC_INFORMATION(ctypes.Structure):
        _fields_ = (
            ("BaseAddress", ctypes.c_void_p),
            ("AllocationBase", ctypes.c_void_p),
            ("AllocationProtect", wintypes.DWORD),
            ("RegionSize", ctypes.c_size_t),
            ("State", wintypes.DWORD),
            ("Protect", wintypes.DWORD),
            ("Type", wintypes.DWORD),
        )


class CtypesWin32MemoryBackend:
    """Small ctypes wrapper around the Win32 calls required by the reader."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Native process memory is only available on Windows")

        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)

        self._user32.GetWindowThreadProcessId.argtypes = (
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        )
        self._user32.GetWindowThreadProcessId.restype = wintypes.DWORD

        self._kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        self._kernel32.OpenProcess.restype = wintypes.HANDLE

        self._kernel32.ReadProcessMemory.argtypes = (
            wintypes.HANDLE,
            wintypes.LPCVOID,
            wintypes.LPVOID,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        )
        self._kernel32.ReadProcessMemory.restype = wintypes.BOOL

        self._kernel32.CreateToolhelp32Snapshot.argtypes = (
            wintypes.DWORD,
            wintypes.DWORD,
        )
        self._kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE

        self._kernel32.Module32FirstW.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(MODULEENTRY32W),
        )
        self._kernel32.Module32FirstW.restype = wintypes.BOOL

        self._kernel32.Module32NextW.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(MODULEENTRY32W),
        )
        self._kernel32.Module32NextW.restype = wintypes.BOOL

        self._kernel32.VirtualQueryEx.argtypes = (
            wintypes.HANDLE,
            wintypes.LPCVOID,
            ctypes.POINTER(MEMORY_BASIC_INFORMATION),
            ctypes.c_size_t,
        )
        self._kernel32.VirtualQueryEx.restype = ctypes.c_size_t

        self._kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        self._kernel32.CloseHandle.restype = wintypes.BOOL

    @staticmethod
    def _last_error(prefix: str) -> ProcessMemoryError:
        error_code = ctypes.get_last_error()
        return ProcessMemoryError(f"{prefix}: {ctypes.WinError(error_code)}")

    def get_window_process_id(self, window_handle: int) -> int:
        pid = wintypes.DWORD()
        thread_id = self._user32.GetWindowThreadProcessId(
            wintypes.HWND(window_handle),
            ctypes.byref(pid),
        )
        if thread_id == 0 or pid.value == 0:
            raise self._last_error(
                f"GetWindowThreadProcessId failed for HWND 0x{window_handle:X}"
            )
        return int(pid.value)

    def open_process(self, pid: int, access: int) -> int:
        handle = self._kernel32.OpenProcess(access, False, pid)
        if not handle:
            raise self._last_error(f"OpenProcess failed for PID {pid}")
        handle_value = getattr(handle, "value", handle)
        if handle_value is None:
            raise self._last_error(f"OpenProcess failed for PID {pid}")
        return int(handle_value)

    def read_process_memory(self, handle: int, address: int, size: int) -> bytes:
        buffer = ctypes.create_string_buffer(size)
        bytes_read = ctypes.c_size_t()
        success = self._kernel32.ReadProcessMemory(
            wintypes.HANDLE(handle),
            ctypes.c_void_p(address),
            buffer,
            size,
            ctypes.byref(bytes_read),
        )
        if not success:
            raise self._last_error(f"ReadProcessMemory failed at 0x{address:X}")
        if bytes_read.value != size:
            raise ProcessMemoryError(
                f"ReadProcessMemory returned {bytes_read.value} of {size} bytes "
                f"at 0x{address:X}"
            )
        return buffer.raw

    def get_module_info(self, pid: int, module_name: str) -> ModuleInfo:
        flags = TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32
        snapshot = self._kernel32.CreateToolhelp32Snapshot(flags, pid)
        snapshot_value = getattr(snapshot, "value", snapshot)
        invalid_handle = ctypes.c_void_p(-1).value
        if snapshot_value is None or int(snapshot_value) == invalid_handle:
            raise self._last_error(f"CreateToolhelp32Snapshot failed for PID {pid}")

        snapshot_int = int(snapshot_value)
        try:
            entry = MODULEENTRY32W()
            entry.dwSize = ctypes.sizeof(MODULEENTRY32W)
            success = self._kernel32.Module32FirstW(
                wintypes.HANDLE(snapshot_int),
                ctypes.byref(entry),
            )
            wanted = module_name.casefold()
            while success:
                if str(entry.szModule).casefold() == wanted:
                    address = ctypes.cast(entry.modBaseAddr, ctypes.c_void_p).value
                    if address is None or address <= 0:
                        raise ProcessMemoryError(
                            f"Module {module_name!r} had an invalid base address"
                        )
                    size = int(entry.modBaseSize)
                    if size <= 0:
                        raise ProcessMemoryError(
                            f"Module {module_name!r} had an invalid image size"
                        )
                    return ModuleInfo(
                        name=str(entry.szModule),
                        path=str(entry.szExePath),
                        base_address=int(address),
                        size=size,
                    )
                entry.dwSize = ctypes.sizeof(MODULEENTRY32W)
                success = self._kernel32.Module32NextW(
                    wintypes.HANDLE(snapshot_int),
                    ctypes.byref(entry),
                )

            error_code = ctypes.get_last_error()
            if error_code not in (0, ERROR_NO_MORE_FILES):
                raise ProcessMemoryError(
                    f"Module enumeration failed: {ctypes.WinError(error_code)}"
                )
            raise ProcessMemoryError(
                f"Module {module_name!r} was not found in PID {pid}"
            )
        finally:
            if not self._kernel32.CloseHandle(wintypes.HANDLE(snapshot_int)):
                raise self._last_error("CloseHandle failed for module snapshot")

    def get_module_base(self, pid: int, module_name: str) -> int:
        return self.get_module_info(pid, module_name).base_address

    def iter_readable_regions(
        self,
        handle: int,
        *,
        maximum_address: int,
    ) -> tuple[MemoryRegion, ...]:
        if maximum_address <= 0:
            raise ValueError("maximum_address must be positive")

        regions: list[MemoryRegion] = []
        address = 0
        info_size = ctypes.sizeof(MEMORY_BASIC_INFORMATION)

        while address <= maximum_address:
            info = MEMORY_BASIC_INFORMATION()
            result = self._kernel32.VirtualQueryEx(
                wintypes.HANDLE(handle),
                ctypes.c_void_p(address),
                ctypes.byref(info),
                info_size,
            )
            if result == 0:
                break

            base_value = getattr(info.BaseAddress, "value", info.BaseAddress)
            base = int(base_value or 0)
            size = int(info.RegionSize)
            if size <= 0:
                break

            protection = int(info.Protect)
            is_readable = (
                int(info.State) == MEM_COMMIT
                and protection != 0
                and protection & PAGE_NOACCESS == 0
                and protection & PAGE_GUARD == 0
            )
            if is_readable and base <= maximum_address:
                clipped_size = min(size, maximum_address - base + 1)
                if clipped_size > 0:
                    regions.append(
                        MemoryRegion(
                            base_address=base,
                            size=clipped_size,
                            protection=protection,
                            region_type=int(info.Type),
                        )
                    )

            next_address = base + size
            if next_address <= address:
                break
            address = next_address

        return tuple(regions)

    def close_handle(self, handle: int) -> None:
        if not self._kernel32.CloseHandle(wintypes.HANDLE(handle)):
            raise self._last_error("CloseHandle failed")


class Win32ProcessMemory:
    """Read-only owner of one Windows process handle."""

    def __init__(
        self,
        pid: int,
        *,
        backend: Win32MemoryBackend | None = None,
    ) -> None:
        if pid <= 0:
            raise ValueError("pid must be positive")
        self.pid = int(pid)
        self._backend = backend or CtypesWin32MemoryBackend()
        access = PROCESS_QUERY_INFORMATION | PROCESS_VM_READ
        self._handle: int | None = self._backend.open_process(self.pid, access)
        self.last_search_diagnostics = MemorySearchDiagnostics()

    @classmethod
    def from_window_handle(
        cls,
        window_handle: int,
        *,
        backend: Win32MemoryBackend | None = None,
    ) -> Win32ProcessMemory:
        if window_handle <= 0:
            raise ValueError("window_handle must be positive")
        selected_backend = backend or CtypesWin32MemoryBackend()
        pid = selected_backend.get_window_process_id(int(window_handle))
        return cls(pid, backend=selected_backend)

    @property
    def closed(self) -> bool:
        return self._handle is None

    def read(self, address: int, size: int) -> bytes:
        if self._handle is None:
            raise ProcessMemoryError("Process memory handle is closed")
        if address <= 0:
            raise ValueError("address must be positive")
        if size <= 0:
            raise ValueError("size must be positive")
        return self._backend.read_process_memory(
            self._handle,
            int(address),
            int(size),
        )

    def module_base(self, module_name: str) -> int:
        if self._handle is None:
            raise ProcessMemoryError("Process memory handle is closed")
        if not module_name or not module_name.strip():
            raise ValueError("module_name cannot be empty")
        return self._backend.get_module_base(self.pid, module_name.strip())

    def module_info(self, module_name: str) -> ModuleInfo:
        if self._handle is None:
            raise ProcessMemoryError("Process memory handle is closed")
        if not module_name or not module_name.strip():
            raise ValueError("module_name cannot be empty")
        selected = module_name.strip()
        get_info = getattr(self._backend, "get_module_info", None)
        if callable(get_info):
            info = get_info(self.pid, selected)
            if not isinstance(info, ModuleInfo):
                raise ProcessMemoryError(
                    "The process-memory backend returned invalid module metadata"
                )
            return info
        # Compatibility for injected test/alternate backends. A zero size is
        # explicit: recovery will retain its bounded configured-slot fallback.
        return ModuleInfo(
            name=selected,
            path="",
            base_address=self._backend.get_module_base(self.pid, selected),
            size=0,
        )

    def readable_regions(
        self,
        *,
        maximum_address: int = 0x7FFFFFFF,
        private_only: bool = True,
    ) -> tuple[MemoryRegion, ...]:
        if self._handle is None:
            raise ProcessMemoryError("Process memory handle is closed")
        regions = self._backend.iter_readable_regions(
            self._handle,
            maximum_address=int(maximum_address),
        )
        if private_only:
            regions = tuple(
                region for region in regions if region.region_type == MEM_PRIVATE
            )
        return regions

    def find_u32(
        self,
        value: int,
        *,
        maximum_address: int = 0x7FFFFFFF,
        private_only: bool = True,
        chunk_size: int = 1 << 20,
        cancellation: object | None = None,
        deadline: float | None = None,
        progress_callback: Callable[[MemorySearchProgress], None] | None = None,
    ) -> tuple[int, ...]:
        """Find every little-endian 32-bit value in readable process memory.

        The scan is used only for infrequent actor-slot discovery. Normal
        frame polling reads the cached actor addresses directly.
        """
        if self._handle is None:
            raise ProcessMemoryError("Process memory handle is closed")
        if isinstance(value, bool) or not 0 <= int(value) <= 0xFFFFFFFF:
            raise ValueError("value must fit in an unsigned 32-bit integer")
        if chunk_size < 4096:
            raise ValueError("chunk_size must be at least 4096 bytes")

        needle = int(value).to_bytes(4, "little", signed=False)
        matches: list[int] = []
        regions_considered = 0
        regions_read = 0
        bytes_read = 0
        read_failures = 0
        started_at = monotonic()
        last_progress_at = started_at
        last_progress_bytes = 0

        def publish_progress(*, complete: bool = False) -> None:
            nonlocal last_progress_at, last_progress_bytes
            if progress_callback is None:
                return
            now = monotonic()
            if not complete:
                if (
                    bytes_read - last_progress_bytes < 64 * 1024 * 1024
                    and now - last_progress_at < 2.0
                ):
                    return
            progress = MemorySearchProgress(
                regions_considered=regions_considered,
                regions_read=regions_read,
                bytes_read=bytes_read,
                read_failures=read_failures,
                matches=len(matches),
                elapsed_seconds=max(0.0, now - started_at),
                complete=bool(complete),
            )
            try:
                progress_callback(progress)
            except Exception:
                # Status reporting must never break a process-memory scan.
                pass
            last_progress_at = now
            last_progress_bytes = bytes_read

        _raise_if_search_stopped(cancellation, deadline)
        regions = self.readable_regions(
            maximum_address=maximum_address,
            private_only=private_only,
        )
        _raise_if_search_stopped(cancellation, deadline)

        for region in regions:
            _raise_if_search_stopped(cancellation, deadline)
            regions_considered += 1
            region_read = False
            offset = 0
            carry = b""
            while offset < region.size:
                _raise_if_search_stopped(cancellation, deadline)
                amount = min(int(chunk_size), region.size - offset)
                try:
                    data = self.read(region.base_address + offset, amount)
                except ProcessMemoryError:
                    read_failures += 1
                    carry = b""
                    offset += amount
                    continue

                region_read = True
                bytes_read += len(data)
                haystack = carry + data
                haystack_base = region.base_address + offset - len(carry)
                start = 0
                while True:
                    found = haystack.find(needle, start)
                    if found < 0:
                        break
                    matches.append(haystack_base + found)
                    start = found + 1
                carry = haystack[-3:] if len(haystack) >= 3 else haystack
                offset += amount
                publish_progress()
            _raise_if_search_stopped(cancellation, deadline)
            if region_read:
                regions_read += 1

        unique = tuple(sorted(set(matches)))
        self.last_search_diagnostics = MemorySearchDiagnostics(
            regions_considered=regions_considered,
            regions_read=regions_read,
            bytes_read=bytes_read,
            read_failures=read_failures,
            matches=len(unique),
        )
        publish_progress(complete=True)
        return unique

    def close(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._backend.close_handle(handle)
        self._handle = None

    def __enter__(self) -> Win32ProcessMemory:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
