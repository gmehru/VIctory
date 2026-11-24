//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{VIEW_TAG, View, view_at};
use crate::host::host;

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    /// The shader-visible start, for descriptor tables of dynamic slots.
    gpu: u64,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more, and
    /// sampler heaps point samplers up to 2048 slots.
    count: u32,
    /// Dynamic slots past `count` (none in sampler heaps): the first of
    /// [`DYNAMIC`] the host takes, else [`MIN_DYNAMIC`].
    dynamic: u32,
    sampler: bool,
    /// The record (view index and generation) each slot was last made from.
    made: Mutex<Vec<u64>>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    kept: Mutex<Vec<Option<ID3D12Resource>>>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Vec<bool>)>,
    /// Per-slot element byte stride (`u32`) for structured buffers whose
    /// stride shader analysis could not recover statically.
    strides: Option<StrideTable>,
}

struct StrideTable {
    _resource: ID3D12Resource,
    ptr: usize,
    gpu: u64,
    len: u32,
}

impl StrideTable {
    fn new(device: &ID3D12Device, len: u32) -> Option<Self> {
        let bytes = (u64::from(len) * 4).max(4);
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut resource: Option<ID3D12Resource> = None;
        let mut p = std::ptr::null_mut();
        // SAFETY: plain committed upload buffer creation and persistent CPU map.
        let gpu = unsafe {
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_UPLOAD,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut resource,
                )
                .ok()?;
            let res = resource.as_ref()?;
            res.Map(0, None, Some(&mut p)).ok()?;
            std::ptr::write_bytes(p.cast::<u8>(), 0, bytes as usize);
            res.GetGPUVirtualAddress()
        };
        Some(Self {
            _resource: resource?,
            ptr: p as usize,
            gpu,
            len,
        })
    }
}

impl Drop for HostHeap {
    fn drop(&mut self) {
        if let Ok(mut kept) = self.kept.lock() {
            for res in kept.drain(..).flatten() {
                crate::host::retire(res);
            }
        }
    }
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!(
                "{:#x} past its end (slot {})",
                va - start - bytes,
                (va - start) / step
            )
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw), tried in order: the first
/// the host takes on top of every title slot. A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out seconds into the world (every world run on an RX 9070 XT,
/// each a device removal shortly after; #59); 2^18 did not, with constant
/// buffer tables alone. Bounded root shader views
/// (`draw::BOUNDED_ROOT_SRVS`, a slot per distinct view a list binds: up
/// to about 10,800 made in 5 s of the world) and tables made afresh for
/// copied views (`draw::COPIED_VIEWS`) take theirs from the same range, on
/// top of what 2^18 was measured with; the world ran with them at 2^19
/// without a table or view going unmade (RX 9070).
/// shortcut: what lists hold at once is not measured, so neither size is
/// known to be the least that suffices.
const DYNAMIC: [u32; 2] = [1 << 19, 1 << 18];

/// The dynamic slots of a heap whose host would take none of [`DYNAMIC`]
/// more (a capped heap, NVIDIA): the title's slots keep their room first.
const MIN_DYNAMIC: u32 = 1 << 16;

/// The title slots a view heap keeps and its dynamic slots, for a title
/// heap of `count` slots, where `fits(n)` makes a host heap of `n` slots
/// (true if the host took it): the first of [`DYNAMIC`] the host takes on
/// top of every title slot, else [`MIN_DYNAMIC`]; else, past
/// [`PORTABLE_VIEWS`], the title's slots cut to leave [`MIN_DYNAMIC`]
/// within it. `None`: no heap.
fn view_sizes(count: u32, mut fits: impl FnMut(u32) -> bool) -> Option<(u32, u32)> {
    DYNAMIC
        .iter()
        .chain(&[MIN_DYNAMIC])
        .find(|&&extra| fits(count.saturating_add(extra)))
        .map(|&extra| (count, extra))
        .or_else(|| {
            // shortcut: descriptors past the host's limit are left out; a
            // title that reaches them needs remapping.
            (count.saturating_add(MIN_DYNAMIC) > PORTABLE_VIEWS && fits(PORTABLE_VIEWS))
                .then_some((PORTABLE_VIEWS - MIN_DYNAMIC, MIN_DYNAMIC))
        })
}

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        // Sampler heaps get every slot a shader-visible one may have, those
        // past the title's point samplers (as `fill_null` makes them): where
        // translation cannot know a sampler heap index, it bounds it only by
        // that size (`victory-shader`'s samplers held in VGPRs, from a
        // table), and an index past the heap can fault the host GPU.
        let pad = if sampler {
            D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE.saturating_sub(count)
        } else {
            0
        };
        // The larger dynamic ranges only where the host takes them on top of
        // every title slot; else the smallest, and the cap after that
        // ([`view_sizes`]).
        let (heap, count, extra) = if sampler {
            match make(count + pad) {
                Ok(heap) => (heap, count, 0),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else {
            let mut made = None;
            let mut refused = None;
            let sizes = view_sizes(count, |n| match make(n) {
                Ok(heap) => {
                    made = Some(heap);
                    true
                }
                Err(e) => {
                    refused = Some(e);
                    false
                }
            });
            match (sizes, made) {
                (Some((kept, extra)), Some(heap)) => {
                    if kept < count {
                        let e = refused.map(|e| e.to_string()).unwrap_or_default();
                        tracing::warn!(error = %e, count, "host view heap capped at {PORTABLE_VIEWS}");
                    }
                    (heap, kept, extra)
                }
                _ => {
                    let e = refused.map(|e| e.to_string()).unwrap_or_default();
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        // Every slot of the heap (`pad` is 0 for view heaps, capped or not).
        let total = count + extra + pad;
        tracing::debug!(count, extra, pad, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(total) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let strides = if sampler {
            None
        } else {
            StrideTable::new(&host.device, total)
        };
        let h = HostHeap {
            heap,
            cpu,
            gpu,
            step,
            count,
            dynamic: extra,
            sampler,
            made: Mutex::new(vec![0; count as usize]),
            kept: Mutex::new(vec![None; total as usize]),
            next: Mutex::new((0, vec![false; extra as usize])),
            strides,
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, total) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(
            ms = started.elapsed().as_millis(),
            "host descriptor heap made null"
        );
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: BLOCK,
                        ..Default::default()
                    })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device
                    .CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device
                    .CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// Tables not made for want of free dynamic slots (their draws dropped),
/// since the frame statistics last took them ([`take_not_made`]): tables
/// made afresh for copied views take more of them. Bounded root shader
/// views (`draw::BOUNDED_ROOT_SRVS`) take theirs here too and are counted
/// with them.
static NOT_MADE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// [`NOT_MADE`] since the last call.
#[cfg(any(test, feature = "diagnostics"))]
pub(crate) fn take_not_made() -> u64 {
    NOT_MADE.swap(0, std::sync::atomic::Ordering::Relaxed)
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let cap = self.dynamic as usize;
        let n = n as usize;
        // Search from `next` on, then from the start.
        let starts = (*next as usize..cap).chain(0..*next as usize);
        let mut run = 0;
        let mut found = None;
        let mut last = usize::MAX;
        for at in starts {
            // A run cannot wrap past the end.
            if at != last.wrapping_add(1) {
                run = 0;
            }
            last = at;
            run = if taken[at] { 0 } else { run + 1 };
            if run == n {
                found = Some(at + 1 - n);
                break;
            }
        }
        let Some(first) = found else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    n,
                    cap,
                    "no free dynamic descriptor slots: a table (or a root shader view) is not made"
                )
            });
            NOT_MADE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return None;
        };
        taken[first..first + n].fill(true);
        *next = ((first + n) % cap) as u32;
        Some(self.count + first as u32)
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// The dynamic slots no list holds (tests).
    #[cfg(test)]
    pub(crate) fn free_dynamic(&self) -> usize {
        let guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        guard.1.iter().filter(|&&taken| !taken).count()
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = (at + n).min(self.dynamic) as usize;
        if let Some(slots) = guard.1.get_mut(at as usize..end) {
            slots.fill(false);
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let kept = make(view, self.cpu(index));
        let ok = kept.is_some();
        if !ok {
            null_for(view, self.cpu(index));
        }
        self.keep(index, kept.flatten());
        self.set_stride(index, if ok { view_stride(view) } else { 0 });
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
        self.set_stride(index, 0);
    }

    /// Makes slot `index` a null view matching the resource kind a shader
    /// expects there (CBV, buffer SRV/UAV, or 2D texture).
    pub(crate) fn make_null_kind(&self, index: u32, kind: victory_shader::ResourceType, dwords: u8) {
        if self.sampler {
            self.null_sampler(index);
            self.keep(index, None);
            self.set_stride(index, 0);
            return;
        }
        let at = self.cpu(index);
        match kind {
            victory_shader::ResourceType::Cbv => {
                if let Some(host) = host() {
                    let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
                    // SAFETY: a slot of our heap; zero location makes a null CBV.
                    unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
                }
            }
            victory_shader::ResourceType::Srv if dwords == 4 => null_buffer_view(at),
            victory_shader::ResourceType::Uav => {
                if let Some(host) = host() {
                    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                        Format: DXGI_FORMAT_R32_UINT,
                        ViewDimension: if dwords == 4 {
                            D3D12_UAV_DIMENSION_BUFFER
                        } else {
                            D3D12_UAV_DIMENSION_TEXTURE2D
                        },
                        ..Default::default()
                    };
                    // SAFETY: a slot of our heap; null resource makes a null UAV.
                    unsafe {
                        host.device.CreateUnorderedAccessView(
                            None::<&ID3D12Resource>,
                            None::<&ID3D12Resource>,
                            Some(&desc),
                            at,
                        )
                    };
                }
            }
            _ => null_view(at),
        }
        self.keep(index, None);
        self.set_stride(index, 0);
    }

    /// Slot `index` now views `resource` (or nothing of the host's). A
    /// previously viewed resource is retired rather than dropped inline so an
    /// in-flight GPU submission reading that slot cannot fault on freed memory.
    fn keep(&self, index: u32, resource: Option<ID3D12Resource>) {
        let old = {
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            kept.get_mut(index as usize)
                .and_then(|slot| std::mem::replace(slot, resource))
        };
        if let Some(old) = old {
            crate::host::retire(old);
        }
    }

    fn set_stride(&self, index: u32, stride: u32) {
        if let Some(t) = &self.strides
            && index < t.len
        {
            // SAFETY: `t.ptr` points to a live mapped UPLOAD buffer of `t.len` dwords.
            unsafe { *(t.ptr as *mut u32).add(index as usize) = stride };
        }
    }

    /// GPU virtual address of the per-slot element byte stride table.
    pub(crate) fn stride_table_gpu(&self) -> Option<u64> {
        self.strides.as_ref().map(|t| t.gpu)
    }

    /// Makes slot `index` (a dynamic one) the buffer shader view `view`
    /// describes, of a copy of the data it reads at byte `offset` (a
    /// multiple of [`CopiedView::align`]) in `buffer`, the list's scratch
    /// memory ([`crate::host::HostList::scratch_view`]). The slot holds
    /// `buffer` while it views it, so a list released while the slot's
    /// descriptor still names its memory leaves nothing freed behind it.
    /// shortcut: once the list is reset, its scratch memory takes other
    /// data while the slot (released too, its descriptor left in place
    /// until the slot is reused) still names it; a shader indexing past its
    /// own table into the slot reads that data (as for [`constant_at`]).
    ///
    /// [`constant_at`]: HostHeap::constant_at
    pub(crate) fn copy_at(&self, index: u32, view: &CopiedView, buffer: &ID3D12Resource, offset: u64) {
        let mut desc = view.desc;
        // SAFETY: a buffer view's union member.
        let first = unsafe { &mut desc.Anonymous.Buffer };
        // The copy holds the view's elements from its first on.
        first.FirstElement = offset / view.element;
        if let Some(host) = host() {
            // SAFETY: a live buffer and a slot of our heap; the view lies in
            // the copy (`align` covers its start, the caller its length).
            unsafe {
                host.device
                    .CreateShaderResourceView(buffer, Some(&desc), self.cpu(index))
            };
        }
        self.keep(index, Some(buffer.clone()));
        self.set_stride(index, u32::try_from(view.element).unwrap_or(0));
        count_unplaced(COPIED, view.element, view.address, view.offset, view.len);
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory (the list's own).
    /// shortcut: the slot does not hold that memory. Once the list is
    /// reset, other data takes it, and once the list is released, its
    /// scratch chunks are freed, while the slot (released too, its
    /// descriptor left in place until the slot is reused) still names them:
    /// a shader indexing past its own table into the slot would read that
    /// data, or fault.
    pub(crate) fn constant_at(&self, index: u32, gpu: u64, size: u32) {
        self.keep(index, None);
        self.set_stride(index, 0);
        let Some(host) = host() else { return };
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: size.next_multiple_of(256),
        };
        // SAFETY: a slot of our heap; the memory is the caller's.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.cpu(index))
        };
    }

    /// Makes slot `index` (a dynamic one) a raw shader view of `buffer`,
    /// `range` as [`raw_range`] gives it (first dword, dwords): a root
    /// shader view, bounded (`draw::BOUNDED_ROOT_SRVS`). The slot holds
    /// `buffer` while it views it, so a list released while the slot's
    /// descriptor still names the copy leaves nothing freed behind it.
    pub(crate) fn raw_at(&self, index: u32, buffer: &ID3D12Resource, (first, count): (u64, u32)) {
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: first,
                    NumElements: count,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        if let Some(host) = host() {
            // SAFETY: a live buffer and a slot of our heap; the range lies
            // in the buffer and starts on 16 bytes ([`raw_range`]).
            unsafe {
                host.device
                    .CreateShaderResourceView(buffer, Some(&desc), self.cpu(index))
            };
        }
        self.keep(index, Some(buffer.clone()));
        // As [`view_stride`] has it for raw views.
        self.set_stride(index, 4);
    }

    /// Makes slot `index` (a dynamic one) shader view `desc` of `texture`
    /// (a depth twin, [`crate::twin`]); the slot holds the texture while it
    /// views it.
    pub(crate) fn texture_srv_at(
        &self,
        index: u32,
        texture: &ID3D12Resource,
        desc: &D3D12_SHADER_RESOURCE_VIEW_DESC,
    ) {
        if let Some(host) = host() {
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                host.device
                    .CreateShaderResourceView(texture, Some(desc), self.cpu(index))
            };
        }
        self.keep(index, Some(texture.clone()));
        self.set_stride(index, 0);
    }

    /// As [`HostHeap::texture_srv_at`], for unordered access view `desc`.
    pub(crate) fn texture_uav_at(
        &self,
        index: u32,
        texture: &ID3D12Resource,
        desc: &D3D12_UNORDERED_ACCESS_VIEW_DESC,
    ) {
        if let Some(host) = host() {
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                host.device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    Some(desc),
                    self.cpu(index),
                )
            };
        }
        self.keep(index, Some(texture.clone()));
        self.set_stride(index, 0);
    }

    /// The shader-visible handle of slot `index`, for a descriptor table
    /// starting there.
    pub(crate) fn gpu(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: self.gpu + u64::from(index) * u64::from(self.step),
        }
    }

    pub(crate) fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            if made[index as usize] != 0 {
                made[index as usize] = 0;
                self.make_null(index);
            }
            return None;
        };
        if made[index as usize] != key {
            let later = !for_draw && view.is_buffer();
            made[index as usize] = if later { 0 } else { key };
            let kept = if later {
                None
            } else {
                make(&view, self.cpu(index))
            };
            let ok = kept.is_some();
            if !ok && !self.sampler {
                null_for(&view, self.cpu(index));
            }
            self.keep(index, kept.flatten());
            self.set_stride(index, if ok { view_stride(&view) } else { 0 });
        }
        Some(view)
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = *self
            .made
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)?;
        if key == 0 {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

fn uav_element_size(desc: &D3D12_UNORDERED_ACCESS_VIEW_DESC) -> Option<u64> {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        Some(u64::from(buffer.StructureByteStride))
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        Some(4)
    } else {
        crate::resource::element(desc.Format.0 as u32).map(|(bytes, _)| bytes)
    }
}

/// Element byte stride of `view` for the runtime stride table (`0` for
/// non-buffer views or descriptors without a known element size).
fn view_stride(view: &View) -> u32 {
    let bytes = match view {
        View::Shader(_, Some(desc)) => element_size(&srv_desc(desc)),
        View::Placed(false, resource, desc) if resource[0] == 1 => desc[..10]
            .try_into()
            .ok()
            .map_or(0, |w: &[u32; 10]| element_size(&srv_desc(w))),
        View::Unordered(_, _, Some(desc)) => uav_element_size(&uav_desc(desc)).unwrap_or(0),
        View::Placed(true, resource, desc) if resource[0] == 1 => desc[..10]
            .try_into()
            .ok()
            .and_then(|w: &[u32; 10]| uav_element_size(&uav_desc(w)))
            .unwrap_or(0),
        _ => 0,
    };
    u32::try_from(bytes).unwrap_or(0)
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = uav_element_size(&desc)?;
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host resource a slot made by [`make`] views, if any.
type Kept = Option<ID3D12Resource>;

/// [`count_unplaced`]'s kind of a shader view read from a copy.
pub(crate) const COPIED: &str = "srv copy";

/// Buffer views whose buffer does not start on one of their elements in
/// its host copy, by kind ("srv" and "uav": left null; [`COPIED`]) and
/// element size: whether one was logged, and how many were made since the
/// frame statistics last took them ([`take_unplaced`]).
static UNPLACED: Mutex<Unplaced> = Mutex::new(std::collections::BTreeMap::new());
type Unplaced = std::collections::BTreeMap<(&'static str, u64), (bool, u64)>;

/// Counts a buffer view of `kind` and `element` bytes per element whose
/// buffer (`address`, `len` bytes) is at `offset` in its host copy, off its
/// elements: the first of each kind and element size is logged, the rest
/// only counted. Runs only when such a view is made, not per draw.
fn count_unplaced(kind: &'static str, element: u64, address: u64, offset: u64, len: u64) {
    let first = {
        let mut views = UNPLACED.lock().unwrap_or_else(|e| e.into_inner());
        let (logged, n) = views.entry((kind, element)).or_default();
        *n += 1;
        !std::mem::replace(logged, true)
    };
    if !first {
        return;
    }
    let address = format_args!("{address:#x}");
    if kind == COPIED {
        tracing::info!(
            element,
            %address,
            offset,
            len,
            "buffer view not on an element of its host copy read from a copy (the first of this \
             element size; the frame statistics count the rest)"
        );
    } else {
        tracing::warn!(
            kind,
            element,
            %address,
            offset,
            len,
            "buffer view not on an element of its host copy left null: it reads zeros (the first \
             of this kind and element size; the frame statistics count the rest)"
        );
    }
}

/// The counts of [`count_unplaced`] since the last call: kind, element
/// size, views made.
#[cfg(any(test, feature = "diagnostics"))]
pub(crate) fn take_unplaced() -> Vec<(&'static str, u64, u64)> {
    let mut views = UNPLACED.lock().unwrap_or_else(|e| e.into_inner());
    views
        .iter_mut()
        .filter(|(_, (_, n))| *n > 0)
        .map(|(&(kind, element), (_, n))| (kind, element, std::mem::take(n)))
        .collect()
}

/// A buffer shader view [`buffer_srv`] leaves null because its buffer does
/// not start on one of its elements in its host copy, small enough for a
/// table to read it from a copy instead ([`HostHeap::copy_at`]).
pub(crate) struct CopiedView {
    /// The title memory the view reads.
    pub(crate) address: u64,
    pub(crate) len: u64,
    /// Bytes per element.
    element: u64,
    /// Where the view's buffer is in its host copy (for the log).
    offset: u64,
    desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
}

impl CopiedView {
    /// What the copy's offset in its buffer is a multiple of: an element,
    /// and the 16 bytes raw views must start on (the least multiple of
    /// both).
    pub(crate) fn align(&self) -> u64 {
        let raw = u64::from(D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT);
        let (mut a, mut b) = (self.element, raw);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        self.element / a * raw
    }
}

/// `view` as a [`CopiedView`], if it is a buffer shader view the host
/// cannot make on its buffer's host copy and it reads at most `limit`
/// bytes, all inside its buffer.
pub(crate) fn copied_view(view: &View, limit: u64) -> Option<CopiedView> {
    const BUFFER: u32 = 1;
    let (base, len, words): (u64, u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => {
            let (data, len) = crate::resource::buffer_bytes(*r as crate::object::This)?;
            (data as u64, len, desc)
        }
        View::Placed(false, resource, desc) if resource[0] == BUFFER => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            u64::from(resource[4]) | u64::from(resource[5]) << 32,
            desc[..10].try_into().ok()?,
        ),
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    if desc.ViewDimension != D3D12_SRV_DIMENSION_BUFFER || element == 0 {
        return None;
    }
    // As [`buffer_srv`] places it.
    let offset = crate::shadow::find(base, len)?.offset(base);
    if offset % element == 0 {
        return None;
    }
    let (address, bytes) = buffer_range(view)?;
    let inside = address >= base && address.checked_add(bytes)? <= base.checked_add(len)?;
    (bytes > 0 && bytes <= limit && inside).then_some(CopiedView {
        address,
        len: bytes,
        element,
        offset,
        desc,
    })
}

/// The raw view of a host copy from byte `offset` in its buffer (`width`
/// bytes) to the end of the data it holds there (`left` bytes after
/// `offset`), for [`HostHeap::raw_at`]: its first dword and its dwords, the
/// last one partly past the data if `left` is not a multiple of 4 (still in
/// the buffer). `None` if `offset` is not on 16 bytes (raw views start
/// there) or nothing is left.
/// shortcut: at most 2^27 dwords (512 MiB), the host's limit for buffer
/// views; reads past that read zeros.
pub(crate) fn raw_range(offset: u64, left: u64, width: u64) -> Option<(u64, u32)> {
    const MOST: u64 = 1 << D3D12_REQ_BUFFER_RESOURCE_TEXEL_COUNT_2_TO_EXP;
    if !offset.is_multiple_of(u64::from(D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT)) {
        return None;
    }
    let room = width.checked_sub(offset)? / 4;
    let count = left.div_ceil(4).min(room).min(MOST);
    (count > 0).then_some((offset / 4, count as u32))
}

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let offset = shadow.offset(address);
    if element == 0 || offset % element != 0 {
        // A null view, which reads zeros. A table holding a small one reads
        // it from a copy instead (`draw::State::tables`, [`copied_view`]).
        // A view a shader reaches by index into the heap, through no table,
        // cannot be served so: buffer views are made only when a draw's
        // table names them, so such a slot stays null (counted here when a
        // table names it too).
        count_unplaced("srv", element, address, offset, len);
        return None;
    }
    buffer.FirstElement += offset / element;
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let offset = shadow.offset(address);
    if element == 0 || offset % element != 0 {
        // Left null (writes dropped, reads zeros): a copy would hide what
        // shaders write from other views of the same memory.
        count_unplaced("uav", element, address, offset, len);
        return None;
    }
    buffer.FirstElement += offset / element;
    buffer.CounterOffsetInBytes = 0;
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device.CreateUnorderedAccessView(
            &shadow.buffer,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
    Some(Some(shadow.buffer.clone()))
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in another format than a
/// typed resource's. A view of a typed texture in another format of its
/// family takes the texture's format.
/// shortcut: such a view (sRGB of UNORM, say) reads without its
/// conversion; cast with castable formats if colors come out wrong.
fn fits(texture: &ID3D12Resource, mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        // Each kind of refused view once (diagnostics builds): which
        // textures the title reads or writes in a way the host cannot.
        #[cfg(feature = "diagnostics")]
        {
            static SEEN: std::sync::Mutex<Vec<(bool, u32, i32, u64, u32)>> =
                std::sync::Mutex::new(Vec::new());
            let key = (unordered, own, d.Flags.0, d.Width, d.Height);
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            if !seen.contains(&key) && seen.len() < 256 {
                seen.push(key);
                tracing::warn!(
                    unordered,
                    format = own,
                    flags = d.Flags.0,
                    width = d.Width,
                    height = d.Height,
                    "view the host does not allow: first of this kind and size"
                );
            }
        }
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = family(own) == own && family(own + 1) == own;
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    if view == 0 {
        return !typeless;
    }
    if family(view) != family(own) {
        once!(view, own, "view format of another family");
        return false;
    }
    if let Some(format) = format
        && !typeless
    {
        *format = d.Format;
    }
    true
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let gpu = shadow.gpu(address);
            if gpu % 256 != 0 {
                once!(
                    address = format_args!("{address:#x}"),
                    "constant buffer not 256-byte aligned on the host"
                );
                return None;
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: size.next_multiple_of(256),
            };
            // SAFETY: a slot of our heap; the range lies in the host copy.
            unsafe { device.CreateConstantBufferView(Some(&desc), at) };
            Some(Some(shadow.buffer.clone()))
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(texture) = crate::resource::host_resource(resource) {
                let mut desc = desc.as_ref().map(srv_desc);
                if !fits(texture, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some(texture.clone()))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(texture) = crate::resource::host_resource(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            let mut desc = desc.as_ref().map(uav_desc);
            // Depth textures the host allows no unordered access to are
            // written through their colour twin.
            if let Some(twin) = crate::twin::unordered_view(resource, desc.as_ref(), at) {
                return twin.map(Some);
            }
            if !fits(texture, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some(texture.clone()))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // shortcut: unordered access to a texture the title placed
                // itself is not mirrored; give it a resource as shader
                // views of such textures get when a dispatch writes one.
                once!(
                    dimension = resource[0],
                    "placed unordered texture view not mirrored"
                );
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// A shader view of a texture the title placed in memory itself, as a view
/// of the resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(false, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource)?;
    Some(View::Shader(texture as usize, desc[..10].try_into().ok()))
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::Constant(..) => {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of our heap; a zero location makes a null view.
            return unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
        }
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A view heap keeps every title slot and takes the largest dynamic
    /// range the host takes on top of them; only past the host's limit
    /// does it give up title slots, and then only for the smallest range.
    #[test]
    fn view_heaps_keep_the_title_slots_first() {
        let capped = |n| n <= PORTABLE_VIEWS;
        assert_eq!(view_sizes(4, |_| true), Some((4, 1 << 19)));
        assert_eq!(view_sizes(400_000, capped), Some((400_000, 1 << 19)));
        assert_eq!(view_sizes(600_000, capped), Some((600_000, 1 << 18)));
        assert_eq!(view_sizes(900_000, capped), Some((900_000, 1 << 16)));
        assert_eq!(view_sizes(950_000, capped), Some((934_464, 1 << 16)));
        // The sizes asked for, in order.
        let mut asked = Vec::new();
        let sizes = view_sizes(950_000, |n| {
            asked.push(n);
            capped(n)
        });
        assert_eq!(sizes, Some((934_464, 1 << 16)));
        let tried = [19, 18, 16].map(|bits| 950_000 + (1 << bits));
        assert_eq!(asked, [&tried[..], &[PORTABLE_VIEWS]].concat());
        // No heap at all, or none within the limit.
        assert_eq!(view_sizes(4, |_| false), None);
        assert_eq!(view_sizes(950_000, |n| n < PORTABLE_VIEWS), None);
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back,
        // and that is counted for the frame statistics.
        assert_eq!(heap.dynamic(50), None);
        assert!(take_not_made() >= 1);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }

    /// A root shader view's raw view runs from its address to the end of
    /// the data its copy holds (the last dword partly past it), starts only
    /// on 16 bytes, and stays inside the buffer and the host's limit.
    #[test]
    fn raw_views_cover_the_rest_of_the_copy() {
        // In the middle of a 64 KiB buffer, and of a 4 MiB window.
        assert_eq!(
            raw_range(0x1230, 0x10000 - 0x1230, 0x10000),
            Some((0x48c, 0x3b74))
        );
        assert_eq!(
            raw_range(0x2_3450, 0x3d_cbb0, 4 << 20),
            Some((0x8d14, 0xf_72ec))
        );
        // A buffer 10 bytes from its end, its copy rounded up: 3 dwords.
        assert_eq!(raw_range(16, 10, 1 << 16), Some((4, 3)));
        // Never past the buffer, even if the data is said to run on.
        assert_eq!(raw_range(16, 100, 32), Some((4, 4)));
        // Not on 16 bytes, or nothing left: no view.
        assert_eq!(raw_range(0x1234, 0x100, 0x10000), None);
        assert_eq!(raw_range(8, 0x100, 0x10000), None);
        assert_eq!(raw_range(0x10000, 0, 0x10000), None);
        assert_eq!(raw_range(0x10010, 16, 0x10000), None);
        // At most 2^27 dwords.
        assert_eq!(raw_range(0, 1 << 30, 1 << 30), Some((0, 1 << 27)));
    }

    /// A sampler heap has every slot a shader-visible one may have, however
    /// few the title's: translated shaders bound a sampler index they cannot
    /// know by that size only. The title's slots stay its own, and none is
    /// dynamic.
    #[test]
    fn sampler_heaps_have_every_slot_a_shader_may_index() {
        let Some(heap) = HostHeap::new(true, 4) else {
            return; // no host GPU
        };
        // SAFETY: a live heap.
        let slots = unsafe { heap.heap.GetDesc() }.NumDescriptors;
        assert_eq!(slots, D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE);
        assert_eq!(heap.count(), 4);
        assert_eq!(heap.dynamic(1), None);
        assert_eq!(heap.kept.lock().unwrap().len(), slots as usize);
    }

    /// A buffer view's description words: structured, `stride` bytes per
    /// element, `count` elements from the first.
    fn structured(stride: u32, count: u32) -> [u32; 10] {
        const BUFFER: u32 = 1;
        let mapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        [0, BUFFER, mapping, 0, 0, 0, count, stride, 0, 0]
    }

    /// A shader view of memory with no resource object at `address`: a
    /// buffer `len` bytes long, viewed as `words` describe.
    fn placed(address: u64, len: u64, words: [u32; 10]) -> View {
        let mut resource = [0; 14];
        resource[0] = 1;
        (resource[4], resource[5]) = (len as u32, (len >> 32) as u32);
        let mut desc = [0; 16];
        desc[..10].copy_from_slice(&words);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        View::Placed(false, resource, desc)
    }

    /// A 2044-byte structure the title placed in its big buffer at an
    /// offset that is not a multiple of 2044 (a run's numbers: the lit
    /// passes' shadow cascades) gets a null view, which is counted; one on
    /// an element gets its view. A table reads the first from a copy.
    #[test]
    fn a_view_off_its_elements_is_left_null_and_counted() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        // No resource has this key.
        const RESOURCE: usize = 0x5eed_0001;
        const START: u64 = 0x14b2_0000;
        crate::shadow::register(RESOURCE, START, 182_452_224);
        // 114,423,168 bytes in: 55,980 elements and 48 bytes.
        let misplaced = 0x1b83_f580;
        let on_element = misplaced - 48;
        let desc = srv_desc(&structured(2044, 1));
        assert!(buffer_srv(misplaced, 2044, desc, heap.cpu(0)).is_none());
        assert!(
            take_unplaced()
                .iter()
                .any(|&(kind, element, n)| (kind, element) == ("srv", 2044) && n >= 1)
        );
        assert!(buffer_srv(on_element, 2044, desc, heap.cpu(1)).is_some_and(|kept| kept.is_some()));
        let copy = copied_view(&placed(misplaced, 2044, structured(2044, 1)), 64 << 10).unwrap();
        assert_eq!(
            (copy.address, copy.len, copy.offset, copy.align()),
            (misplaced, 2044, 114_423_168, 8176)
        );
        assert!(copied_view(&placed(on_element, 2044, structured(2044, 1)), 64 << 10).is_none());
        // Past the limit, or reading past its buffer: left null.
        assert!(copied_view(&placed(misplaced, 2044, structured(2044, 1)), 2043).is_none());
        assert!(copied_view(&placed(misplaced, 2044, structured(2044, 2)), 64 << 10).is_none());
        crate::shadow::unregister(RESOURCE, START);
    }

    /// A view read from a copy reads the title's data: the copy starts on
    /// one of its elements in the list's scratch memory, and the view's
    /// first element is there.
    #[test]
    fn a_copied_view_reads_the_title_data() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let (Some(host), Some(heap)) = (host(), HostHeap::new(false, 4)) else {
            return; // no host GPU
        };
        // Two structures of 511 words, as title memory.
        let data: Vec<u32> = (0..1022).map(|i| 0x1000_0000 + i).collect();
        let view = CopiedView {
            address: data.as_ptr() as u64,
            len: 4088,
            element: 2044,
            offset: 48,
            desc: srv_desc(&structured(2044, 2)),
        };
        let list = host.new_list().unwrap();
        // The list's next scratch memory is off an element.
        list.scratch(100).unwrap();
        let (buffer, offset, to) = list.scratch_view(view.len, view.align()).unwrap();
        assert!(offset > 0 && offset % view.align() == 0);
        list.defer_fill(to, view.address as *const u8, view.len as usize);
        let index = heap.dynamic(1).unwrap();
        heap.copy_at(index, &view, &buffer, offset);
        const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0)), UAV(u0)"
struct Cascades { uint v[511]; };
StructuredBuffer<Cascades> s : register(t0);
RWByteAddressBuffer o : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint n, stride;
    s.GetDimensions(n, stride);
    o.Store4(0, uint4(s[0].v[0], s[0].v[459], s[1].v[0], s[1].v[510]));
    o.Store2(16, uint2(n, stride));
}
"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call.
        unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                windows::core::s!("test"),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
        }
        .unwrap();
        let code = code.unwrap();
        // SAFETY: the blob's own bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
        };
        let (out, written) = host.shared_buffer(256).unwrap();
        // SAFETY: plain creation; an open list, our heap, pipeline and
        // buffers, which outlive the submission.
        unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            let l = &list.list;
            l.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
            l.SetComputeRootSignature(&root);
            l.SetPipelineState(&pso);
            let table = heap.heap.GetGPUDescriptorHandleForHeapStart().ptr
                + u64::from(index) * u64::from(heap.step);
            l.SetComputeRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: table });
            l.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
            l.Dispatch(1, 1, 1);
            list.close();
            host.execute(&[&list]);
            assert_eq!(
                written.cast::<[u32; 6]>().read_volatile(),
                [data[0], data[459], data[511], data[1021], 2, 2044]
            );
        }
    }
}
