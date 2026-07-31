from __future__ import annotations

import pytest
from position.Win32ProcessMemory import (
    MEM_PRIVATE,
    PROCESS_QUERY_INFORMATION,
    PROCESS_VM_READ,
    MemoryRegion,
    MemorySearchCancelled,
    MemorySearchDeadline,
    ProcessMemoryError,
    Win32ProcessMemory,
)


class FakeBackend:
    def __init__(self) -> None:
        self.pid = 2468
        self.handle = 1357
        self.opened: list[tuple[int, int]] = []
        self.reads: list[tuple[int, int, int]] = []
        self.closed: list[int] = []

    def get_window_process_id(self, window_handle: int) -> int:
        assert window_handle == 0xBEEF
        return self.pid

    def open_process(self, pid: int, access: int) -> int:
        self.opened.append((pid, access))
        return self.handle

    def read_process_memory(self, handle: int, address: int, size: int) -> bytes:
        self.reads.append((handle, address, size))
        return bytes(range(size))

    def get_module_base(self, pid: int, module_name: str) -> int:
        assert pid == self.pid
        assert module_name == "Neuz.exe"
        return 0x00E10000

    def iter_readable_regions(
        self,
        handle: int,
        *,
        maximum_address: int,
    ) -> tuple[MemoryRegion, ...]:
        assert handle == self.handle
        assert maximum_address == 0x7FFFFFFF
        return (
            MemoryRegion(
                base_address=0x1000,
                size=0x2000,
                protection=0x04,
                region_type=MEM_PRIVATE,
            ),
        )

    def close_handle(self, handle: int) -> None:
        self.closed.append(handle)


def test_process_memory_resolves_pid_from_window_and_reads() -> None:
    backend = FakeBackend()

    memory = Win32ProcessMemory.from_window_handle(0xBEEF, backend=backend)
    result = memory.read(0x1000, 4)

    assert memory.pid == 2468
    assert backend.opened == [(2468, PROCESS_QUERY_INFORMATION | PROCESS_VM_READ)]
    assert backend.reads == [(1357, 0x1000, 4)]
    assert result == b"\x00\x01\x02\x03"


def test_process_memory_close_is_idempotent() -> None:
    backend = FakeBackend()
    memory = Win32ProcessMemory(backend.pid, backend=backend)

    memory.close()
    memory.close()

    assert memory.closed
    assert backend.closed == [backend.handle]


def test_process_memory_rejects_reads_after_close() -> None:
    backend = FakeBackend()
    memory = Win32ProcessMemory(backend.pid, backend=backend)
    memory.close()

    with pytest.raises(ProcessMemoryError, match="closed"):
        memory.read(0x1000, 4)


def test_process_memory_resolves_module_base() -> None:
    backend = FakeBackend()
    memory = Win32ProcessMemory(backend.pid, backend=backend)

    assert memory.module_base("Neuz.exe") == 0x00E10000


def test_process_memory_opens_direct_process_id() -> None:
    backend = FakeBackend()

    memory = Win32ProcessMemory(backend.pid, backend=backend)

    assert memory.pid == backend.pid
    assert backend.opened == [
        (backend.pid, PROCESS_QUERY_INFORMATION | PROCESS_VM_READ)
    ]


def test_process_memory_finds_u32_across_chunk_boundary() -> None:
    class SearchBackend(FakeBackend):
        def read_process_memory(
            self,
            handle: int,
            address: int,
            size: int,
        ) -> bytes:
            self.reads.append((handle, address, size))
            data = bytearray(size)
            needle_address = 0x1FFF
            needle = (0x3EDB1008).to_bytes(4, "little")
            for index, value in enumerate(needle):
                absolute = needle_address + index
                if address <= absolute < address + size:
                    data[absolute - address] = value
            return bytes(data)

    backend = SearchBackend()
    memory = Win32ProcessMemory(backend.pid, backend=backend)

    matches = memory.find_u32(0x3EDB1008, chunk_size=4096)

    assert matches == (0x1FFF,)
    assert memory.last_search_diagnostics.matches == 1
    assert memory.last_search_diagnostics.bytes_read == 0x2000


def test_process_memory_search_stops_before_enumeration_when_cancelled() -> None:
    backend = FakeBackend()
    memory = Win32ProcessMemory(backend.pid, backend=backend)

    class Cancelled:
        cancelled = True

    with pytest.raises(MemorySearchCancelled):
        memory.find_u32(123, chunk_size=4096, cancellation=Cancelled())

    assert backend.reads == []


def test_process_memory_search_checks_cancellation_between_chunks() -> None:
    class Token:
        cancelled = False

    token = Token()

    class CancellingBackend(FakeBackend):
        def read_process_memory(
            self,
            handle: int,
            address: int,
            size: int,
        ) -> bytes:
            self.reads.append((handle, address, size))
            data = bytes(size)
            token.cancelled = True
            return data

    backend = CancellingBackend()
    memory = Win32ProcessMemory(backend.pid, backend=backend)

    with pytest.raises(MemorySearchCancelled):
        memory.find_u32(123, chunk_size=4096, cancellation=token)

    assert len(backend.reads) == 1


def test_process_memory_search_honours_expired_deadline() -> None:
    backend = FakeBackend()
    memory = Win32ProcessMemory(backend.pid, backend=backend)

    with pytest.raises(MemorySearchDeadline):
        memory.find_u32(123, chunk_size=4096, deadline=0.0)

    assert backend.reads == []
