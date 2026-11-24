//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use crate::shadow::Shadow;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: usize,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: (table, what it holds) → first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u64), u32>,
    /// The dynamic slots the list holds: title heap, first, count; given
    /// back when the list is reset or released.
    dynamic: Vec<(usize, u32, u32)>,
    /// A fresh table of the stage being bound could not be made.
    table_failed: bool,
    /// Root shader views made in this list ([`BOUNDED_ROOT_SRVS`]): (title
    /// view heap, title address, host address) → their dynamic slot, so
    /// draws binding the same root view share one slot instead of taking
    /// one each.
    root_views: std::collections::HashMap<(usize, u64, u64), u32>,
    /// The viewports, scissor rectangles and stencil reference last set on
    /// the list (logged with the indirect executions of listed frames in
    /// diagnostics builds; set again after a depth twin's copy pass; the
    /// host list gets them as the title sets them).
    viewports: Vec<D3D12_VIEWPORT>,
    scissors: Vec<RECT>,
    #[cfg(feature = "diagnostics")]
    stencil: u32,
    /// The title's render target and depth view handles last set, set
    /// again after a depth twin's copy pass ([`crate::twin`]).
    target_handles: Vec<u64>,
    depth_handle: u64,
}

impl Drop for State {
    fn drop(&mut self) {
        for &(heap, first, n) in &self.dynamic {
            crate::descriptor::release_dynamic(heap, first, n);
        }
    }
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Rectangle draws through the geometry shader ([`crate::rects`]) since the
/// frame statistics last took them: recorded, dropped.
static RECTS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
const RAN: usize = 0;
const NOT_RUN: usize = 1;

/// Why those were dropped, by reason (diagnostics).
#[cfg(feature = "diagnostics")]
static RECTS_WHY: std::sync::Mutex<Vec<(&str, u64)>> = std::sync::Mutex::new(Vec::new());

/// A rectangle draw of title pipeline `pipeline` for the geometry shader
/// (`what`: draw, indexed draw, indirect kind) dropped for `why`; in
/// diagnostics builds also among the drop reasons, with the pipeline's name
/// (its stages' hashes, as the host pipeline lines log them) so the log
/// says which pipelines stay undrawn: one reason per pipeline and reason,
/// no more than there are pipelines. Not for want of the pipeline's own
/// host pipeline, which is noted as for any draw. The rectangle line of the
/// frame statistics counts by reason alone.
fn rect_dropped(what: &str, why: &'static str, pipeline: usize) {
    RECTS[NOT_RUN].fetch_add(1, Ordering::Relaxed);
    #[cfg(feature = "diagnostics")]
    {
        {
            let mut whys = RECTS_WHY.lock().unwrap_or_else(|e| e.into_inner());
            match whys.iter_mut().find(|w| w.0 == why) {
                Some(w) => w.1 += 1,
                None => whys.push((why, 1)),
            }
        }
        if why != crate::rects::NO_HOST_PIPELINE {
            // Made already (the rectangle pipeline is asked for after it),
            // so nothing is waited for here.
            let why = match crate::pipeline::host_pipeline(pipeline as This) {
                Some(p) => format!("{what}: {why} ({})", p.name),
                None => format!("{what}: {why}"),
            };
            crate::pipeline::note_why(why);
        }
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = (what, why, pipeline);
}

/// [`RECTS`] since the last call, and why the dropped ones were, most
/// first (diagnostics: the frame statistics).
#[cfg(feature = "diagnostics")]
pub(crate) fn take_rects() -> ([u64; 2], Vec<(&'static str, u64)>) {
    let counts = RECTS.each_ref().map(|n| n.swap(0, Ordering::Relaxed));
    let mut whys = std::mem::take(&mut *RECTS_WHY.lock().unwrap_or_else(|e| e.into_inner()));
    whys.sort_by_key(|w| std::cmp::Reverse(w.1));
    (counts, whys)
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// Whether root shader views are bounded: each is bound as a table of one
/// raw view, from its address to the end of its buffer's host copy, made
/// in a dynamic slot of the bound view heap ([`State::root_srv_table`]);
/// else as a root descriptor, a bare host address nothing bounds. The
/// title's mesh programs read vertex streams through a root shader view at
/// offsets up to about 36 MiB past it (a stream offset from a header plus
/// an index times a stride, with no record limit), while title memory
/// outside buffers is copied in separate 4 MiB windows
/// ([`crate::shadow`]): unbounded, such a read runs off the copy into
/// whatever the host has after it, and where that was freed the device is
/// removed ("page fault ... inside freed texture", just past a window).
/// A draw whose view cannot be made (no view heap bound, no free dynamic
/// slot, an address not on 16 bytes in its copy) is dropped. The shader
/// crate's host harness (`victory_shader::emit::host`, its executed tests
/// and `translate --pipelines`) always builds root shader views as such
/// tables: with this off, it no longer binds them as this crate does.
/// shortcut: a read past the copy reads zeros, so a stream that crosses
/// into the next window gives wrong vertices, not a fault. The real fix
/// makes title-contiguous memory contiguous on the host (one reserved
/// range with the windows mapped into it), so the copy reaches as far as
/// the title's memory does.
pub(crate) const BOUNDED_ROOT_SRVS: bool = true;

/// A root view's host copy: the copy, the host address bound (the copy's,
/// or for a constant buffer not 256-byte aligned there, as the console
/// needs only 16, a copy in upload memory) and the bytes of its buffer
/// from the address on.
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB,
/// shader views as 1 MiB (both cut at their buffer's end) for what is
/// brought up to date: past that the copy keeps what an earlier draw
/// copied.
fn root_view(
    list: &HostList,
    address: u64,
    srv: bool,
    pipeline: &str,
) -> Option<(Arc<Shadow>, u64, u64)> {
    let shadow = crate::shadow::find(address, 1)?;
    let window = if srv { 1 << 20 } else { 1 << 16 };
    let left = shadow.left(address);
    let len = left.min(window);
    let gpu = shadow.gpu(address);
    let bound = if srv || gpu % 256 == 0 {
        shadow.refresh(list, address, len);
        gpu
    } else {
        let (gpu, to) = list.scratch(len)?;
        list.defer_fill(to, address as *const u8, len as usize);
        gpu
    };
    if left < window {
        short_root(srv, address, bound, left, pipeline);
    }
    Some((shadow, bound, left))
}

/// Root views bound with less than their window left in their buffer
/// (constant buffers, shader resources), since the frame statistics last
/// took them.
static SHORT_ROOTS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// A root view bound at title `address` (host `host`) with only `left`
/// bytes of its buffer after it. Root constant buffers are read unbounded
/// (the translated shader declares 64 KiB), so a read past the buffer's
/// host copy (its length rounded up to 64 KiB) is a GPU page fault, which
/// the device-removed report places just past a copy; a root shader view
/// reads zeros there when bounded ([`BOUNDED_ROOT_SRVS`]). Logged the first
/// 16 times, then counted for the frame statistics.
fn short_root(srv: bool, address: u64, host: u64, left: u64, pipeline: &str) {
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    SHORT_ROOTS[usize::from(srv)].fetch_add(1, Ordering::Relaxed);
    if LOGGED.fetch_add(1, Ordering::Relaxed) < 16 {
        tracing::warn!(
            kind = if srv { "shader resource" } else { "constant buffer" },
            address = format_args!("{address:#x}"),
            host = format_args!("{host:#x}"),
            left,
            pipeline,
            bounded = srv && BOUNDED_ROOT_SRVS,
            "root view with less than 64 KiB (constant buffer) or 1 MiB (shader resource) left \
             in its buffer's host copy: reads past the copy fault, or read zeros if bounded (the \
             first 16 are logged; the frame statistics count the rest)"
        );
    }
}

/// Root shader views ([`BOUNDED_ROOT_SRVS`]) since the frame statistics
/// last took them: views made (not those a list made earlier), draws
/// dropped for want of a bound view heap, for want of a free dynamic slot
/// (which the heap also counts among its tables not made), and because
/// the view's address is not on 16 bytes in its copy.
static ROOT_SRVS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
const MADE: usize = 0;
const NO_HEAP: usize = 1;
const NO_SLOT: usize = 2;
const MISALIGNED: usize = 3;

/// [`ROOT_SRVS`] since the last call: made, dropped without a heap,
/// without a slot, off 16 bytes (diagnostics: the frame statistics).
#[cfg(any(test, feature = "diagnostics"))]
pub(crate) fn take_root_srvs() -> [u64; 4] {
    ROOT_SRVS.each_ref().map(|n| n.swap(0, Ordering::Relaxed))
}

/// Root views bound short of their window since the last call: constant
/// buffers, shader resources (diagnostics: the frame statistics).
#[cfg(feature = "diagnostics")]
pub(crate) fn take_short_roots() -> [u64; 2] {
    SHORT_ROOTS.each_ref().map(|n| n.swap(0, Ordering::Relaxed))
}

/// The most a buffer shader view in a table may read to be read from a
/// copy when the host cannot make it on its buffer's host copy (it does
/// not start on one of its elements there; [`State::tables`]).
/// shortcut: larger ones stay null views, counted in the frame statistics.
/// The copy is of title memory when the list is submitted, as for constant
/// buffers in tables: what shaders of the same submission write there
/// before the draw runs is not seen.
const COPIED_VIEWS: u64 = 64 << 10;

/// Whether table entry `u` may have its table made afresh for a copy of
/// its view ([`COPIED_VIEWS`]): a buffer shader view, in a table the shader
/// does not index per lane (`per_lane`, [`State::tables`]).
fn copies_table(u: &UsedResource, per_lane: &[u8]) -> bool {
    u.kind == ResourceType::Srv && u.dwords == 4 && !per_lane.contains(&u.slot)
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// Makes the raw view of the root shader view at title `address` in
/// `shadow` in a new dynamic slot of `host` (taken until the caller gives
/// it back): the slot's index. `None`, counted, without a free dynamic
/// slot or when the address is not on 16 bytes in its copy.
fn root_srv_view(
    host: &crate::descriptor::host::HostHeap,
    shadow: &Shadow,
    address: u64,
) -> Option<u32> {
    // SAFETY: a live buffer.
    let width = unsafe { shadow.buffer.GetDesc() }.Width;
    let range =
        crate::descriptor::host::raw_range(shadow.offset(address), shadow.left(address), width);
    let Some(range) = range else {
        once!(
            address = format_args!("{address:#x}"),
            offset = shadow.offset(address),
            "a root shader view not on 16 bytes in its host copy: the draw is dropped"
        );
        ROOT_SRVS[MISALIGNED].fetch_add(1, Ordering::Relaxed);
        return None;
    };
    let Some(index) = host.dynamic(1) else {
        ROOT_SRVS[NO_SLOT].fetch_add(1, Ordering::Relaxed);
        return None;
    };
    host.raw_at(index, &shadow.buffer, range);
    ROOT_SRVS[MADE].fetch_add(1, Ordering::Relaxed);
    Some(index)
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    /// Keeps the viewports the title set, to log them (diagnostics) and set
    /// them again after a depth twin's copy pass (the host list gets them
    /// through [`viewports`]).
    pub(crate) fn note_viewports(&mut self, viewports: &[D3D12_VIEWPORT]) {
        self.viewports = viewports.to_vec();
    }

    /// As [`State::note_viewports`], for scissor rectangles.
    pub(crate) fn note_scissors(&mut self, rects: &[RECT]) {
        self.scissors = rects.to_vec();
    }

    /// As [`State::note_viewports`], for the stencil reference.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_stencil_ref(&mut self, reference: u32) {
        self.stencil = reference;
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self.vertex.iter_mut().skip(start as usize).take(count as usize) {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = crate::host::host() else {
            return;
        };
        self.targets.clear();
        self.target_handles = handles.to_vec();
        self.depth_handle = depth;
        let mut rtvs = Vec::new();
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                continue;
            };
            self.targets.push(record.resource as usize);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            rtvs.push(host.rtv(texture, record.desc().as_ref()));
        }
        self.depth = 0;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            Some(host.dsv(texture, record.depth_desc().as_ref()))
        });
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        self.prepare_with(list, indexed, mesh, p)
    }

    /// As [`State::prepare_for`], with host pipeline `p`.
    fn prepare_with(&mut self, list: &HostList, indexed: bool, mesh: bool, p: &HostPipeline) -> bool {
        if p.compute || p.mesh != mesh {
            once!(
                mesh,
                "a draw or mesh dispatch with another kind of pipeline"
            );
            return false;
        }
        // A depth target a shader wrote through its twin gets the twin's
        // values first.
        if self.depth != 0 {
            self.sync_twin(list, self.depth as This);
        }
        if !self.bind(list, p, false) {
            return false;
        }
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                return false;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        const DEPTH_WRITE: u32 = 0x10;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            crate::list::ensure_state(list, self.depth as This, DEPTH_WRITE);
            crate::twin::depth_written(self.depth as This);
        }
        true
    }

    /// As [`State::prepare`], for a rectangle draw through the geometry
    /// shader ([`crate::rects`]): the pipeline's rectangle pipeline, made
    /// by a worker on first use (dropped until it is there, as any pipeline
    /// being made). False, counted with why (`what`: the kind of draw), if
    /// the draw cannot run.
    fn prepare_rects(&mut self, list: &HostList, indexed: bool, what: &str) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let why = match crate::pipeline::rect_pipeline(self.pipeline as This) {
            Ok(p) if self.prepare_with(list, indexed, false, p) => {
                RECTS[RAN].fetch_add(1, Ordering::Relaxed);
                return true;
            }
            Ok(_) => "rectangles: not bound (tables, root views or buffers; see the warnings)",
            Err(why) => why,
        };
        rect_dropped(what, why, self.pipeline);
        false
    }

    /// Whether a direct rectangle draw of the bound pipeline goes through
    /// the geometry shader ([`crate::rects`]): one whose corners the CPU
    /// expansion ([`State::rectangles`]) cannot complete, as it reads no
    /// vertex buffer or reads the vertex id (the expansion runs other ids
    /// than the console). Others keep the expansion, which the menus and
    /// the interface draw with.
    fn rects_on_gpu(&self) -> bool {
        crate::rects::on()
            && crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| {
                p.inputs.iter().all(|e| e.per_instance)
                    || p.stages.iter().any(|s| reads_vertex_id(&s.translated().host.dxil))
            })
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a table it reads could
    /// not be made, or a root view it reads could not be bound.
    fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        if crate::twin::any_twins() {
            self.sync_twins(list, p, compute);
        }
        let l = &list.list;
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them.
        if self.host_heaps != self.heaps {
            let heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { l.SetDescriptorHeaps(&heaps) };
            self.host_heaps = self.heaps;
            self.host_roots = [0; 2];
        }
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(&p.pso);
            list.keep(&p.root);
        }
        for stage in &p.stages {
            let t = stage.translated();
            self.table_failed = false;
            let made = self.tables(list, &t.used, &t.host.per_lane_tables, compute);
            if self.table_failed {
                static DROPPED_TABLES: AtomicU64 = AtomicU64::new(0);
                let n = DROPPED_TABLES.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::warn!(
                        dropped = n,
                        "a draw or dispatch dropped: no dynamic slots for its tables"
                    );
                }
                return false;
            }
            if let Some((param, dwords)) = stage.user_data {
                let mut values = vec![0u32; dwords as usize];
                for u in &t.host.user_data {
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None => self.user_value(compute, u.slot, u.root, u.dword),
                    };
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            for (views, srv) in [(&stage.root_cbvs, false), (&stage.root_srvs, true)] {
                for &(slot, param) in views {
                    let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot))
                    else {
                        once!(slot, srv, "a shader reads a root view the title did not set");
                        return false;
                    };
                    let Some((shadow, gpu, _)) = root_view(list, address, srv, &p.name) else {
                        once!(
                            slot,
                            srv,
                            address = format_args!("{address:#x}"),
                            "a shader reads a root view outside title buffers"
                        );
                        return false;
                    };
                    if srv && BOUNDED_ROOT_SRVS {
                        let Some(table) = self.root_srv_table(&shadow, address) else {
                            return false;
                        };
                        // SAFETY: an open list; the table is a slot of the
                        // bound view heap, holding its buffer.
                        unsafe {
                            if compute {
                                l.SetComputeRootDescriptorTable(param, table);
                            } else {
                                l.SetGraphicsRootDescriptorTable(param, table);
                            }
                        }
                        continue;
                    }
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, srv) {
                            (false, false) => l.SetGraphicsRootConstantBufferView(param, gpu),
                            (false, true) => l.SetGraphicsRootShaderResourceView(param, gpu),
                            (true, false) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, true) => l.SetComputeRootShaderResourceView(param, gpu),
                        }
                    }
                }
            }
            if let Some(param) = stage.stride_table {
                let gpu = (self.heaps[0] != 0)
                    .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
                    .flatten()
                    .and_then(|h| h.stride_table_gpu());
                if let Some(gpu) = gpu {
                    // SAFETY: an open list; the stride table buffer lives with the host heap.
                    unsafe {
                        if compute {
                            l.SetComputeRootShaderResourceView(param, gpu);
                        } else {
                            l.SetGraphicsRootShaderResourceView(param, gpu);
                        }
                    }
                } else {
                    // The shader would read a root view nothing bound: the
                    // draw or dispatch is left out (review of the stride
                    // table's change).
                    once!("a shader reads the runtime stride table without a host view heap");
                    return false;
                }
            }
        }
        true
    }

    /// Writes the twin of title depth texture `resource` into it if the
    /// twin is newer ([`crate::twin::sync`]), then sets the title's state
    /// again: the heaps stay the title's, the pipeline and root signature
    /// are bound again by the next draw or dispatch, targets, viewports,
    /// scissors and topology are set again here (the stencil reference is
    /// left alone: the pass has no stencil). Nothing without a bound view
    /// heap (logged once; the depth texture is left as it was).
    pub(crate) fn sync_twin(&mut self, list: &HostList, resource: This) {
        let Some(copy) = crate::twin::newer(resource) else {
            return;
        };
        let Some((title, host)) = self.twin_heap(list) else {
            once!("a depth twin's copy pass without a host view heap bound: not run");
            return;
        };
        let Some(index) = crate::twin::sync(list, host, &copy) else {
            return;
        };
        self.dynamic.push((title, index, 1));
        // What the pass replaced.
        self.host_pipeline = 0;
        self.host_roots[0] = 0;
        let (handles, depth) = (std::mem::take(&mut self.target_handles), self.depth_handle);
        self.set_targets(list, &handles, depth);
        if !self.viewports.is_empty() {
            viewports(list, &self.viewports);
        }
        if !self.scissors.is_empty() {
            scissors(list, &self.scissors);
        }
        if self.topology != 0 {
            self.set_topology(list, self.topology);
        }
    }

    /// Copies the depth plane of title depth texture `resource` into its
    /// twin if the twin is stale ([`crate::twin::refresh`]); the next
    /// dispatch binds its compute root signature and pipeline again.
    fn refresh_twin(&mut self, list: &HostList, resource: This) {
        let Some(copy) = crate::twin::stale(resource) else {
            return;
        };
        let Some((title, host)) = self.twin_heap(list) else {
            once!("a depth twin's refresh without a host view heap bound: not run");
            return;
        };
        let Some((first, n)) = crate::twin::refresh(list, host, &copy) else {
            return;
        };
        self.dynamic.push((title, first, n));
        self.host_pipeline = 0;
        self.host_roots[1] = 0;
    }

    /// The bound view heap (title heap and its host heap), set on the list
    /// if it is not yet, for a depth twin's pass.
    fn twin_heap(
        &mut self,
        list: &HostList,
    ) -> Option<(usize, &'static crate::descriptor::host::HostHeap)> {
        let title = self.heaps[0];
        let host = (title != 0)
            .then(|| crate::descriptor::heap(title as This).host())
            .flatten()?;
        if self.host_heaps != self.heaps {
            let heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { list.list.SetDescriptorHeaps(&heaps) };
            self.host_heaps = self.heaps;
            self.host_roots = [0; 2];
        }
        Some((title, host))
    }

    /// Before a draw or dispatch with `p`: the depth textures its stages
    /// read through shader views in tables whose twins are newer get the
    /// twins' values ([`State::sync_twin`]); those they write through
    /// unordered views whose twins are stale get their depth planes copied
    /// into the twins ([`State::refresh_twin`]).
    fn sync_twins(&mut self, list: &HostList, p: &HostPipeline, compute: bool) {
        let newer = crate::twin::any_newer();
        let mut read = Vec::new();
        let mut written = Vec::new();
        for stage in &p.stages {
            for u in &stage.translated().used {
                let wanted = match u.kind {
                    ResourceType::Srv => newer,
                    ResourceType::Uav => true,
                    _ => false,
                };
                if u.root != RootType::Table || !wanted {
                    continue;
                }
                let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot))
                else {
                    continue;
                };
                let handle = table + u64::from(u.table_offset_dwords) * 4;
                if self.heap_of(handle).is_none() {
                    continue;
                }
                match crate::descriptor::view(handle) {
                    Some(View::Shader(resource, _)) => read.push(resource),
                    Some(View::Unordered(resource, ..)) => written.push(resource),
                    _ => {}
                }
            }
        }
        read.sort_unstable();
        read.dedup();
        for resource in read {
            self.sync_twin(list, resource as This);
        }
        written.sort_unstable();
        written.dedup();
        for resource in written {
            self.refresh_twin(list, resource as This);
        }
    }

    /// The table of the root shader view at title `address` in `shadow`
    /// ([`BOUNDED_ROOT_SRVS`]): a raw view from the address to the end of
    /// its host copy, in a dynamic slot of the bound view heap that the list
    /// holds until it is reset, made once per list. `None` (the draw is
    /// dropped, and counted) without a bound view heap or a free dynamic
    /// slot, or when the address is not on 16 bytes in its copy, where raw
    /// views must start (a view starting before it would read the wrong
    /// bytes).
    fn root_srv_table(
        &mut self,
        shadow: &Shadow,
        address: u64,
    ) -> Option<D3D12_GPU_DESCRIPTOR_HANDLE> {
        let title = self.heaps[0];
        let host = (title != 0)
            .then(|| crate::descriptor::heap(title as This).host())
            .flatten();
        let Some(host) = host else {
            once!(
                "a shader reads a root shader view without a host view heap: the draw is dropped"
            );
            ROOT_SRVS[NO_HEAP].fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let key = (title, address, shadow.gpu(address));
        if let Some(&index) = self.root_views.get(&key) {
            return Some(host.gpu(index));
        }
        let index = root_srv_view(host, shadow, address)?;
        self.dynamic.push((title, index, 1));
        self.root_views.insert(key, index);
        Some(host.gpu(index))
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return;
        }
        if !self.bind(list, p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(x, y, z) };
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = self.user_value(compute, u.slot, u.root, u.dword);
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                }
            }
            for &(slot, _) in stage.root_cbvs.iter().chain(&stage.root_srvs) {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        text
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let ran = self.prepare_for(list, false, true)
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x, 1, 1) };
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some(view) = host.mirror(index, handle, true) {
            readable(list, u, &view);
        }
    }

    /// The descriptors a shader reads through tables. Tables holding a
    /// constant buffer are made afresh for the draw in dynamic slots, with
    /// the constant data copied to upload memory when the list is
    /// submitted: the host wants constant buffers 256-byte aligned, the
    /// console only 16. So are tables holding a buffer shader view whose
    /// buffer does not start on one of its elements in the buffer's host
    /// copy (the host has no view for it there), its data copied the same
    /// way to where it does ([`COPIED_VIEWS`]). Returns those tables' new
    /// byte offsets by root parameter; the other tables' slots are made
    /// from the title's.
    /// shortcut: a table made afresh holds the entries up to the last one
    /// the shader is known to read; a shader indexing past them reads other
    /// dynamic slots. Tables the shader indexes per lane (`per_lane`,
    /// [`victory_shader::emit::HostShader::per_lane_tables`]) are not made
    /// afresh for copied views: their misaligned views stay null.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once: its constant data is copied at submission either way, and the
    /// big tables the world binds would otherwise take hundreds of slots per
    /// draw, wrapping the dynamic slots onto lists not yet submitted.
    fn tables(
        &mut self,
        list: &HostList,
        used: &[UsedResource],
        per_lane: &[u8],
        compute: bool,
    ) -> Vec<(u8, u32)> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = used
            .iter()
            .filter(tables)
            .filter(|u| u.kind == ResourceType::Cbv)
            .map(|u| u.slot)
            .collect();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let end = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(u.slot);
            }
        }
        // Tables holding a buffer shader view the host cannot make on its
        // buffer's host copy: a null view there reads zeros (the lit passes'
        // shadow cascades, 2044-byte structures the title places anywhere,
        // then loop to the iteration limit and discard every pixel).
        // Unordered access views stay null: a copy would hide what shaders
        // write from other views of the same memory. So do views in tables
        // the shader indexes per lane: an index past the entries a fresh
        // table holds would read another list's dynamic slot, of any kind
        // (a descriptor of another kind read as a buffer gives a wild
        // address on AMD, #59), where the title's table holds the title's
        // descriptors.
        // shortcut: a table indexed by a value every lane shares is still
        // made afresh, with the same reach past its entries, as it was
        // before per-lane descriptors were translated.
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) || !copies_table(u, per_lane) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let handle = table + u64::from(u.table_offset_dwords) * 4;
            // In a bound heap, as `table_entry` reads it.
            if self.heap_of(handle).is_some()
                && crate::descriptor::view(handle).is_some_and(|v| {
                    crate::descriptor::host::copied_view(&v, COPIED_VIEWS).is_some()
                })
            {
                fresh.push(u.slot);
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let Some(host) = heap.host() else {
                continue;
            };
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8 + 1)
                .max()
                .unwrap_or(1);
            // The entries this stage reads from copies: buffer shader views
            // off their elements in their buffer's host copy, which it reads
            // as such (no unordered access view at the same position).
            // Entries it does not read are made as before (left null).
            let mut copies: Vec<(u32, crate::descriptor::host::CopiedView)> = Vec::new();
            for u in &entries {
                let k = u.table_offset_dwords / 8;
                let written = entries
                    .iter()
                    .any(|w| w.table_offset_dwords / 8 == k && w.kind == ResourceType::Uav);
                if u.kind != ResourceType::Srv
                    || u.dwords != 4
                    || written
                    || copies.iter().any(|&(c, _)| c == k)
                {
                    continue;
                }
                if let Some(copy) = crate::descriptor::view(table + u64::from(k) * 32)
                    .and_then(|v| crate::descriptor::host::copied_view(&v, COPIED_VIEWS))
                {
                    copies.push((k, copy));
                }
            }
            // The table and what its entries hold (record words), and which
            // are copies: a stage never gets a table another stage made
            // without a copy it reads.
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    // SAFETY: a descriptor slot in title memory.
                    unsafe { (at as *const [u32; 3]).read_unaligned() }.hash(&mut h);
                }
                for &(k, _) in &copies {
                    k.hash(&mut h);
                }
                (table, h.finish())
            };
            if let Some(&first) = self.fresh_tables.get(&key) {
                for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle) {
                        readable(list, u, &view);
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            let Some(first) = host.dynamic(n) else {
                // Without its fresh table the shader would read the title's
                // slots, which for constant buffer tables hold no constant
                // view: a texture or null descriptor read as a buffer gives
                // a wild address on AMD and the device is removed (#59).
                // Tables made afresh for copied views would read null views.
                // The draw or dispatch is dropped instead.
                self.table_failed = true;
                continue;
            };
            self.fresh_tables.insert(key, first);
            self.dynamic
                .push((std::ptr::from_ref(heap) as usize, first, n));
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    Some(View::Constant(address, size)) => {
                        let Some((gpu, to)) = list.scratch(u64::from(size)) else {
                            host.make_null_kind(index, ResourceType::Cbv, 4);
                            continue;
                        };
                        list.defer_fill(to, address as *const u8, size as usize);
                        host.constant_at(index, gpu, size);
                    }
                    Some(view) => {
                        // A buffer shader view off its elements: a copy of
                        // what it reads, made at submission as constant
                        // data is, where the view starts on one.
                        if let Some((_, copy)) = copies.iter().find(|&&(c, _)| c == k)
                            && let Some((buffer, offset, to)) =
                                list.scratch_view(copy.len, copy.align())
                        {
                            list.defer_fill(to, copy.address as *const u8, copy.len as usize);
                            host.copy_at(index, copy, &buffer, offset);
                            continue;
                        }
                        host.make_at(index, &view);
                        if let Some(u) = used {
                            readable(list, u, &view);
                        }
                    }
                    None => {
                        if let Some(u) = used {
                            host.make_null_kind(index, u.kind, u.dwords);
                            once!(slot, "a table entry without a view");
                        } else {
                            host.make_null(index);
                        }
                    }
                }
            }
            made.push((slot, first * 32));
        }
        made
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !kind.starts_with("indirect") {
            self.log_draw(kind, count);
        }
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_resource(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let rects = self.topology == RECTANGLES;
        // Rectangles the CPU expansion cannot complete: the title's
        // vertices as triangles, through the geometry shader.
        if rects && self.rects_on_gpu() {
            let ran = self.prepare_rects(list, false, "draw");
            if ran {
                #[cfg(feature = "diagnostics")]
                list.note_op(|| self.op_text(format!("draw {vertices}x{instances} rectangles"), false));
                // SAFETY: an open list with everything set.
                unsafe {
                    list.list
                        .DrawInstanced(vertices, instances, first, first_instance)
                };
                DRAWN.fetch_add(1, Ordering::Relaxed);
            } else {
                DROPPED.fetch_add(1, Ordering::Relaxed);
                #[cfg(feature = "diagnostics")]
                dropped(self.pipeline);
            }
            return;
        }
        let ran = self.prepare(list, false) && (!rects || self.rectangles(list, vertices, first));
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// The vertices an indirect rectangle draw of pipeline `p` expands: the
    /// fewest that any buffer it reads per vertex holds, in whole
    /// rectangles. Refused, with the reason, if a buffer it reads is not
    /// bound or holds no rectangle, and for pipelines whose rectangles the
    /// expansion cannot complete: those reading no vertex buffer (their
    /// corners come from the vertex id, the instance or memory, which the
    /// expansion leaves alone) and those reading the vertex id (the
    /// expanded draw runs ids 0 to 6n where the console ran 0 to 3n: the
    /// title's tile shaders would draw half of each tile twice).
    /// shortcut: at most 4 MiB of expanded vertices per buffer, as the
    /// expansion copies the bound buffers whole on the CPU at submission;
    /// records past that draw nothing.
    fn rectangle_vertices(&self, p: &HostPipeline) -> Result<u32, &'static str> {
        const EXPANDED: u32 = 4 << 20;
        if p.inputs.iter().all(|e| e.per_instance) {
            return Err("rectangles: no vertex input");
        }
        if p.stages.iter().any(|s| reads_vertex_id(&s.translated().host.dxil)) {
            return Err("rectangles: vertex-id pipeline");
        }
        let mut vertices = 1 << 24;
        for e in p.inputs.iter().filter(|e| !e.per_instance) {
            let v = self.vertex[e.slot as usize];
            if v.address == 0 || v.stride == 0 {
                return Err("rectangles not expanded");
            }
            vertices = (v.size / v.stride)
                .min(EXPANDED / 2 / v.stride)
                .min(vertices);
        }
        if vertices < 3 {
            return Err("rectangles not expanded");
        }
        Ok(vertices / 3 * 3)
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || u64::from(first + count) * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: (rects * 6 * stride as u64) as u32,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        // Indexed rectangles: three indices a rectangle, as three vertices
        // are, through the geometry shader; else not drawn.
        let ran = match self.topology {
            RECTANGLES if crate::rects::on() => self.prepare_rects(list, true, "indexed draw"),
            RECTANGLES => {
                once!("indexed rectangles not drawn");
                false
            }
            _ => self.prepare(list, true),
        };
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host. Diagnostics builds
    /// trace the ones that ran (as draws are, [`State::trace`]) and log each
    /// one of a listed frame ([`State::log_indirect`]).
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        records: (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        let ran = self.run_indirect(list, records, arguments, count);
        #[cfg(feature = "diagnostics")]
        {
            const DISPATCH: u32 = 2;
            // After the run: the host pipeline is made if it ran, so the
            // trace's lookup of it neither starts nor waits for one.
            if ran && records.0 != DISPATCH {
                self.trace(&format!("indirect kind {}", records.0), records.2, 0);
            }
            self.log_indirect(records, arguments, count, ran);
        }
        ran
    }

    /// For a frame [`crate::capture::listed`] (the one being recorded):
    /// one line per indirect execution, whether it ran or not, with its
    /// pipeline (its stages' hashes, as the host pipeline lines name it;
    /// looked up without waiting for or starting its host pipeline), whether
    /// its rectangles went through the geometry shader, the bound render
    /// targets and depth by title resource (as the target dump files name
    /// them, with size and format), the stencil reference, viewports and
    /// scissor rectangles last set, and the records' max count and count
    /// buffer.
    #[cfg(feature = "diagnostics")]
    fn log_indirect(&self, (kind, stride, max): (u32, u32, u32), arguments: u64, count: Option<u64>, ran: bool) {
        let frame = crate::queue::presented() + 1;
        if !crate::capture::listed(frame) {
            return;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        let rectangles = matches!(kind, DRAW | INDEXED) && self.topology == RECTANGLES;
        let gs = rectangles && crate::rects::on();
        let expanded = rectangles && !gs && kind == DRAW && indirect_rectangles();
        let resource = |r: usize| {
            let r = r as This;
            match crate::resource::host_resource(r) {
                Some(t) => {
                    // SAFETY: a live resource.
                    let d = unsafe { t.GetDesc() };
                    format!("{r:?} {}x{} f{}", d.Width, d.Height, d.Format.0)
                }
                None => format!("{r:?} (no host copy)"),
            }
        };
        let targets: Vec<String> = self.targets.iter().map(|&t| resource(t)).collect();
        let depth = if self.depth == 0 {
            "none".to_owned()
        } else {
            resource(self.depth)
        };
        let viewports: Vec<String> = self
            .viewports
            .iter()
            .map(|v| {
                format!(
                    "{},{} {}x{} z{}..{}",
                    v.TopLeftX, v.TopLeftY, v.Width, v.Height, v.MinDepth, v.MaxDepth
                )
            })
            .collect();
        let scissors: Vec<String> = self
            .scissors
            .iter()
            .map(|r| format!("{},{}..{},{}", r.left, r.top, r.right, r.bottom))
            .collect();
        let (pipeline, host) = crate::pipeline::describe(self.pipeline as This);
        // The first record as title memory holds it now (what the GPU wrote
        // there comes back after each submission): counts of zero say the
        // pass that fills the records did not.
        let record: Vec<u32> = if crate::shadow::find(arguments, 16).is_some() {
            (0..4u64)
                // SAFETY: sixteen bytes of a title buffer the shadow table knows.
                .map(|i| unsafe { std::ptr::read_unaligned((arguments + i * 4) as *const u32) })
                .collect()
        } else {
            Vec::new()
        };
        tracing::info!(
            frame,
            kind,
            ran,
            record = ?record,
            pipeline = %pipeline,
            host_pipeline = %host,
            topology = self.topology,
            geometry_shader = gs,
            expanded,
            targets = ?targets,
            depth = %depth,
            stencil_ref = self.stencil,
            viewports = ?viewports,
            scissors = ?scissors,
            max,
            stride,
            arguments = format_args!("{arguments:#x}"),
            count_buffer = count.is_some(),
            count_at = ?count.map(|c| format!("{c:#x}")),
            "listed frame: indirect execution"
        );
    }

    /// One line per direct draw of a frame listed for capture, before it is
    /// prepared: its pipeline and the resources it draws into, as
    /// [`State::log_indirect`] prints them. Says which draws write a target
    /// that a later pass tests against.
    #[cfg(feature = "diagnostics")]
    fn log_draw(&self, kind: &str, count: u32) {
        let frame = crate::queue::presented() + 1;
        if !crate::capture::listed(frame) {
            return;
        }
        let resource = |r: usize| {
            let r = r as This;
            match crate::resource::host_resource(r) {
                Some(t) => {
                    // SAFETY: a live resource.
                    let d = unsafe { t.GetDesc() };
                    format!("{r:?} {}x{} f{}", d.Width, d.Height, d.Format.0)
                }
                None => format!("{r:?} (no host copy)"),
            }
        };
        let targets: Vec<String> = self.targets.iter().map(|&t| resource(t)).collect();
        let depth = if self.depth == 0 {
            "none".to_owned()
        } else {
            resource(self.depth)
        };
        let scissors: Vec<String> = self
            .scissors
            .iter()
            .map(|r| format!("{},{}..{},{}", r.left, r.top, r.right, r.bottom))
            .collect();
        let (pipeline, host) = crate::pipeline::describe(self.pipeline as This);
        tracing::info!(
            frame,
            kind,
            count,
            pipeline = %pipeline,
            host_pipeline = %host,
            topology = self.topology,
            targets = ?targets,
            depth = %depth,
            stencil_ref = self.stencil,
            scissors = ?scissors,
            "listed frame: draw"
        );
    }

    /// [`State::execute_indirect`] itself.
    fn run_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(kind, stride))
            .flatten()
        else {
            why("no host signature");
            return false;
        };
        // Rectangles through the geometry shader ([`crate::rects`]): the
        // title's records as they are, drawn with the pipeline's rectangle
        // pipeline, which is looked for before anything is recorded (no
        // clamp, nothing charged, for a draw that does not run).
        let gs = matches!(kind, DRAW | INDEXED) && self.topology == RECTANGLES && crate::rects::on();
        let what = if kind == INDEXED {
            "indirect kind 1"
        } else {
            "indirect kind 0"
        };
        if gs && let Err(refused) = crate::pipeline::rect_pipeline(self.pipeline as This) {
            rect_dropped(what, refused, self.pipeline);
            #[cfg(feature = "diagnostics")]
            if refused == crate::rects::NO_HOST_PIPELINE {
                crate::pipeline::note_drop(self.pipeline as This);
            }
            return false;
        }
        args.refresh(list, arguments, len);
        if let Some((s, at)) = &counter {
            s.refresh(list, *at, 4);
        }
        // Bounded first: the clamp binds a compute pipeline of its own. The
        // geometry shader runs two triangles for each rectangle's three
        // vertices: charged twice.
        let weight = if gs { 2 } else { 1 };
        let records = (kind, stride, max);
        if !crate::indirect::bound(list, &args.buffer, args.gpu(arguments), records, weight) {
            why("not bounded (no clamp, or arguments not 4-byte aligned)");
            return false;
        }
        // Whatever follows, the list's pipeline and compute root signature
        // are the clamp's now: the next draw or dispatch binds its own.
        self.host_pipeline = 0;
        self.host_roots[1] = 0;
        // Rectangles without the geometry shader (with
        // `VICTORY_INDIRECT_RECTS`, else dropped below): the draws of their
        // expanded vertices, from records of our own; the title's stay as
        // they are (and hold the count, if the title keeps it beside them).
        let rects = !gs && kind == DRAW && self.topology == RECTANGLES && indirect_rectangles();
        let (mut from, mut at, mut signature) =
            (args.buffer.clone(), args.offset(arguments), signature);
        let mut expanded = 0;
        if rects {
            let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
                // Being made, refused or not translated: counted as such.
                crate::indirect::done(list, &args.buffer);
                #[cfg(feature = "diagnostics")]
                crate::pipeline::note_drop(self.pipeline as This);
                return false;
            };
            let made = self.rectangle_vertices(p).and_then(|vertices| {
                let failed = "rectangles not expanded";
                let triangles = host_signature(DRAW, 16).ok_or(failed)?;
                let records = (stride, max, vertices);
                let gpu = args.gpu(arguments);
                let (buffer, offset) =
                    crate::indirect::rectangles(list, &args.buffer, gpu, records).ok_or(failed)?;
                Ok((buffer, offset, triangles, vertices))
            });
            match made {
                Ok(made) => (from, at, signature, expanded) = made,
                Err(refused) => {
                    crate::indirect::done(list, &args.buffer);
                    why(refused);
                    return false;
                }
            }
        }
        let ran = match kind {
            DRAW | INDEXED if gs => self.prepare_rects(list, kind == INDEXED, what),
            DRAW if rects => self.prepare(list, false) && self.rectangles(list, expanded, 0),
            DRAW | INDEXED => self.topology != RECTANGLES && self.prepare(list, kind == INDEXED),
            MESH => self.prepare_for(list, false, true),
            DISPATCH => match crate::pipeline::host_pipeline(self.pipeline as This) {
                Some(p) if p.compute => self.bind(list, p, true),
                _ => false,
            },
            _ => false,
        };
        if !ran {
            crate::indirect::done(list, &from);
            if rects {
                crate::indirect::done(list, &args.buffer);
            }
            // Which pipeline, by its name (its stages' hashes, as the host
            // pipeline lines log them), and the title's topology: one
            // reason per pipeline and topology, so there are no more
            // reasons than pipelines that drop.
            #[cfg(feature = "diagnostics")]
            match crate::pipeline::host_pipeline(self.pipeline as This) {
                None => crate::pipeline::note_drop(self.pipeline as This),
                // Noted with its reason (`rect_dropped`).
                Some(_) if gs => {}
                Some(p) => why(&format!(
                    "prepare failed (compute {}, mesh {}, topology {}, {})",
                    p.compute, p.mesh, self.topology, p.name
                )),
            }
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(
                format!("indirect kind {kind} max {max} at {arguments:#x}"),
                kind == DISPATCH,
            )
        });
        // SAFETY: an open list with everything set; the buffers are host
        // copies the list keeps alive (refresh), the signature is cached.
        unsafe {
            list.list.ExecuteIndirect(
                &signature,
                max,
                &from,
                at,
                counter.as_ref().map(|(s, _)| &s.buffer),
                counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
            )
        };
        crate::indirect::done(list, &from);
        if rects {
            crate::indirect::done(list, &args.buffer);
        }
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need.
fn readable(list: &HostList, u: &UsedResource, view: &View) {
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((address, len)) = crate::descriptor::host::buffer_range(view)
        && len > 0
        && let Some(shadow) = crate::shadow::find(address, len)
    {
        shadow.refresh(list, address, len);
        if u.kind == ResourceType::Uav {
            shadow.gpu_writes(list, address, len);
        }
    }
    match (view, u.kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            if let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
            {
                crate::texture::upload(list, *resource as This, t);
            }
            crate::list::ensure_state(list, *resource as This, SHADER_RESOURCE);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(*resource as This);
            crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
            crate::twin::written(list, *resource as This);
        }
        _ => {}
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_resource(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!(
                "{}x{} f{} flags {:#x}",
                d.Width, d.Height, d.Format.0, d.Flags.0
            )
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        // A buffer has no host texture: its title range, as for a shader view.
        Some(v @ View::Unordered(r, ..)) => texture(*r as This).map_or_else(
            || format!("uav buffer {:?}", crate::descriptor::host::buffer_range(v)),
            |t| format!("uav {t}"),
        ),
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        Some(View::Constant(address, size))
            if *size >= 16 && crate::texture::committed(*address as *const u8, 16) =>
        {
            // SAFETY: 16 committed bytes of title memory.
            let v = unsafe { (*address as *const [f32; 4]).read_unaligned() };
            format!("Constant({address:#x}, {size}) {v:?}")
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// Whether indirect rectangle-list draws run, expanded into triangles
/// (`VICTORY_INDIRECT_RECTS` set, any build), when the geometry shader does
/// not take them ([`crate::rects::on`], which comes first). Off, they are
/// dropped: the expansion leaves vertex ids as they are, which only some
/// pipelines survive ([`State::rectangle_vertices`] refuses the others),
/// and the argument clamp's budget is charged half the vertices it runs.
fn indirect_rectangles() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_RECTS").is_some())
}

/// Whether a translated shader reads the vertex id: its container's input
/// signature (`ISG1`) has an element of that system value, or names it.
/// True when the container or the signature does not read (refused rather
/// than drawn wrong); false without an input signature (no inputs).
fn reads_vertex_id(dxil: &[u8]) -> bool {
    const VERTEX_ID: u32 = 6;
    let Ok(container) = victory_shader::Container::parse(dxil) else {
        return true;
    };
    let Some(inputs) = container.part(victory_shader::FourCC::ISG1) else {
        return false;
    };
    let s = inputs.data;
    if s.windows(11).any(|w| w.eq_ignore_ascii_case(b"SV_VertexID")) {
        return true;
    }
    let word = |at: usize| {
        let bytes = s.get(at..at.checked_add(4)?)?;
        Some(u32::from_le_bytes(bytes.try_into().ok()?) as usize)
    };
    let (Some(count), Some(first)) = (word(0), word(4)) else {
        return true;
    };
    // 32-byte elements; the system value is the fourth word.
    if count > s.len() / 32 {
        return true;
    }
    (0..count).any(|i| {
        first
            .checked_add(32 * i + 12)
            .and_then(word)
            .is_none_or(|v| v == VERTEX_ID as usize)
    })
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

#[cfg(test)]
mod tests {
    use super::reads_vertex_id;

    /// Only buffer shader views take a fresh table for a copy, and not in a
    /// table the shader indexes per lane.
    #[test]
    fn tables_indexed_per_lane_take_no_copies() {
        use victory_shader::{ResourceType, RootType, UsedResource};
        let entry = |kind, dwords| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 8,
            lower_bound: 0,
            stride: 2044,
            dwords,
            user_data: 2,
            slot: 3,
        };
        let srv = entry(ResourceType::Srv, 4);
        assert!(super::copies_table(&srv, &[]));
        assert!(super::copies_table(&srv, &[1, 5]));
        assert!(!super::copies_table(&srv, &[3]));
        assert!(!super::copies_table(&entry(ResourceType::Srv, 8), &[]));
        assert!(!super::copies_table(&entry(ResourceType::Uav, 4), &[]));
        assert!(!super::copies_table(&entry(ResourceType::Cbv, 4), &[]));
    }

    /// A root shader view is a raw view from its address to the end of its
    /// buffer's host copy, read through a table of the shape pipelines get:
    /// for an address in the middle of a title buffer (the view ends at the
    /// buffer's end) and in title memory outside buffers (at the end of its
    /// 4 MiB window). On the host GPU, its first dword and its last are the
    /// title's, its size is what is left, and reads past its end, as far as
    /// a byte offset goes, read zeros: the device stays.
    /// Not covered: the list's memo and the giving back of the slots
    /// (`State::root_srv_table`, which needs a title heap).
    #[test]
    fn root_shader_views_are_bounded() {
        use super::Shadow;
        use crate::descriptor::host::{HostHeap, raw_range};
        use crate::pipeline::host::{root_srv_range, root_srv_table};
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Direct3D12::*;
        // The table pipelines get: one raw view at t3 in the root view space,
        // volatile, so the driver cannot make it a root descriptor.
        let range = root_srv_range(3);
        assert_eq!(range.RangeType, D3D12_DESCRIPTOR_RANGE_TYPE_SRV);
        assert_eq!((range.NumDescriptors, range.BaseShaderRegister), (1, 3));
        assert_eq!(range.RegisterSpace, victory_shader::emit::ROOT_CBV_SPACE);
        assert_eq!(range.RegisterSpace, 1, "the shader below binds space1");
        assert_eq!(
            range.Flags,
            D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE
                | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE
        );
        let (Some(host), Some(heap)) = (crate::host::host(), HostHeap::new(false, 4)) else {
            return; // no host GPU
        };
        const WINDOW: u64 = 4 << 20;
        // A 64 KiB title buffer, as title memory (no resource has this key).
        const RESOURCE: usize = 0x5eed_0002;
        let buffer: Vec<u32> = (0..0x4000).map(|i| 0xb000_0000 | i).collect();
        let start = buffer.as_ptr() as u64;
        crate::shadow::register(RESOURCE, start, 0x10000);
        let in_buffer = start + 0x1230;
        // Memory outside buffers holding a whole window.
        let loose: Vec<u32> = (0..3 << 20).map(|i| 0x1000_0000 | i).collect();
        let base = loose.as_ptr() as u64;
        let window = base.next_multiple_of(WINDOW);
        let in_window = window + 0x2_3450;
        let word = |at: u64| loose[((at - base) / 4) as usize];
        let a = crate::shadow::find(in_buffer, 1).unwrap();
        let b = crate::shadow::find(in_window, 1).unwrap();
        // SAFETY: live buffers.
        let width = |s: &Shadow| unsafe { s.buffer.GetDesc() }.Width;
        assert_eq!(
            (a.offset(in_buffer), a.left(in_buffer), width(&a)),
            (0x1230, 0x10000 - 0x1230, 0x10000)
        );
        assert_eq!(
            (b.offset(in_window), b.left(in_window), width(&b)),
            (0x2_3450, WINDOW - 0x2_3450, WINDOW)
        );
        // First dword, dwords.
        let range_a = raw_range(a.offset(in_buffer), a.left(in_buffer), width(&a));
        let range_b = raw_range(b.offset(in_window), b.left(in_window), width(&b));
        assert_eq!(range_a, Some((0x1230 / 4, (0x10000 - 0x1230) / 4)));
        let dwords_b = ((WINDOW - 0x2_3450) / 4) as u32;
        assert_eq!(range_b, Some((0x2_3450 / 4, dwords_b)));
        let slot_a = super::root_srv_view(&heap, &a, in_buffer).unwrap();
        let slot_b = super::root_srv_view(&heap, &b, in_window).unwrap();
        assert!(super::take_root_srvs()[super::MADE] >= 2);
        // Both copies brought up to date to their ends, as draws do for
        // their first MiB.
        let list = host.new_list().unwrap();
        a.refresh(&list, in_buffer, a.left(in_buffer));
        b.refresh(&list, in_window, b.left(in_window));
        // Per view: the first dword, the last, the size, the fifth, then
        // dwords at its end, 64 KiB and 36 MiB past it (the mesh programs'
        // reach), and near the last byte offset there is (not at it: the
        // host's compiler may merge a load at 0xfffffffc with the one at 0,
        // the offset wrapping, and the merged load is out of bounds whole).
        const SOURCE: &str = r#"
ByteAddressBuffer a : register(t3, space1);
ByteAddressBuffer b : register(t4, space1);
RWByteAddressBuffer o : register(u0);
#define PROBE(r, at) { \
    uint n; \
    r.GetDimensions(n); \
    o.Store4(at, uint4(r.Load(0), r.Load(n - 4), n, r.Load(16))); \
    o.Store4(at + 16, uint4(r.Load(n), r.Load(n + 0x10000), r.Load(0x2400000), r.Load(0xfffffff0))); \
}
[numthreads(1, 1, 1)]
void main() {
    PROBE(a, 0)
    PROBE(b, 32)
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
        // The two views as pipelines bind them, and a root UAV for the
        // results.
        let ranges = [root_srv_range(3), root_srv_range(4)];
        let params = [
            root_srv_table(&ranges[0], D3D12_SHADER_VISIBILITY_ALL),
            root_srv_table(&ranges[1], D3D12_SHADER_VISIBILITY_ALL),
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1::default(),
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: params.len() as u32,
                    pParameters: params.as_ptr(),
                    ..Default::default()
                },
            },
        };
        let (out, written) = host.shared_buffer(256).unwrap();
        // SAFETY: the description points at live arrays; plain creation;
        // an open list, our heap, pipeline and buffers, which outlive the
        // submission.
        let got = unsafe {
            let mut blob: Option<ID3DBlob> = None;
            D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).unwrap();
            let blob = blob.unwrap();
            let root: ID3D12RootSignature = host
                .device
                .CreateRootSignature(
                    0,
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer().cast::<u8>(),
                        blob.GetBufferSize(),
                    ),
                )
                .unwrap();
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
            l.SetComputeRootDescriptorTable(0, heap.gpu(slot_a));
            l.SetComputeRootDescriptorTable(1, heap.gpu(slot_b));
            l.SetComputeRootUnorderedAccessView(2, out.GetGPUVirtualAddress());
            l.Dispatch(1, 1, 1);
            list.close();
            host.execute(&[&list]);
            written.cast::<[u32; 16]>().read_volatile()
        };
        // SAFETY: a plain query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        // First, last, size, fifth; then four reads past the end.
        let (first_a, size_a) = (0x1230 / 4, 0x10000 - 0x1230);
        let inside_a = [buffer[first_a], buffer[0x3fff], size_a, buffer[first_a + 4]];
        assert_eq!(got[..8], [inside_a, [0; 4]].concat());
        let last_b = word(window + WINDOW - 4);
        let size_b = (WINDOW - 0x2_3450) as u32;
        let inside_b = [word(in_window), last_b, size_b, word(in_window + 16)];
        assert_eq!(got[8..], [inside_b, [0; 4]].concat());
        crate::shadow::unregister(RESOURCE, start);
    }

    /// A list makes a root shader view's table once in a dynamic slot of the
    /// title heap it has bound, binds that slot again for the same view,
    /// takes another for another address, and gives the slots back when it
    /// is reset or released. Without a heap, or off 16 bytes in its copy,
    /// there is no table (the draw is dropped).
    #[test]
    fn root_shader_views_are_made_once_per_list_and_given_back() {
        use super::State;
        if crate::host::host().is_none() {
            return; // no host GPU
        }
        let title = crate::descriptor::test_view_heap(4);
        let Some(host) = crate::descriptor::heap(title).host() else {
            return;
        };
        // A 64 KiB title buffer, as title memory (no resource has this key).
        const RESOURCE: usize = 0x5eed_0003;
        let buffer = vec![0u32; 0x4000];
        let start = buffer.as_ptr() as u64;
        crate::shadow::register(RESOURCE, start, 0x10000);
        let shadow = crate::shadow::find(start, 1).unwrap();
        assert_eq!(shadow.offset(start + 0x100), 0x100);
        let free = host.free_dynamic();
        let mut list = State::default();
        // No view heap bound: no table.
        assert!(list.root_srv_table(&shadow, start + 0x100).is_none());
        list.heaps[0] = title as usize;
        let a = list.root_srv_table(&shadow, start + 0x100).unwrap();
        let again = list.root_srv_table(&shadow, start + 0x100).unwrap();
        assert_eq!(again.ptr, a.ptr);
        let b = list.root_srv_table(&shadow, start + 0x200).unwrap();
        assert_ne!(a.ptr, b.ptr);
        assert!(list.root_srv_table(&shadow, start + 0x104).is_none());
        assert_eq!((list.dynamic.len(), host.free_dynamic()), (2, free - 2));
        // A reset is a fresh state: the old one's slots are given back.
        list = State::default();
        assert_eq!(host.free_dynamic(), free);
        assert!(list.root_views.is_empty() && list.dynamic.is_empty());
        crate::shadow::unregister(RESOURCE, start);
    }

    /// A container of `parts`, laid out as DXBC is.
    fn container(parts: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
        let table = 0x20 + 4 * parts.len();
        let total = table + parts.iter().map(|(_, d)| 8 + d.len()).sum::<usize>();
        let mut out = b"DXBC".to_vec();
        out.extend([0; 16]);
        out.extend(1u16.to_le_bytes());
        out.extend(0u16.to_le_bytes());
        out.extend((total as u32).to_le_bytes());
        out.extend((parts.len() as u32).to_le_bytes());
        let mut offset = table;
        for (_, d) in parts {
            out.extend((offset as u32).to_le_bytes());
            offset += 8 + d.len();
        }
        for (fourcc, d) in parts {
            out.extend(*fourcc);
            out.extend((d.len() as u32).to_le_bytes());
            out.extend(*d);
        }
        out
    }

    /// An input signature of one row per (system value, name).
    fn signature(rows: &[(u32, &str)]) -> Vec<u8> {
        let mut s = (rows.len() as u32).to_le_bytes().to_vec();
        s.extend(8u32.to_le_bytes());
        let mut name_at = 8 + 32 * rows.len() as u32;
        let mut names = Vec::new();
        for (row, &(value, name)) in rows.iter().enumerate() {
            for word in [0, name_at, 0, value, 1, row as u32] {
                s.extend(word.to_le_bytes());
            }
            s.extend([1, 1, 0, 0]);
            s.extend(0u32.to_le_bytes());
            name_at += name.len() as u32 + 1;
            names.extend(name.as_bytes());
            names.push(0);
        }
        s.extend(names);
        s
    }

    /// Indirect rectangle draws through a vertex shader that reads the
    /// vertex id are refused: the expansion hands it ids the console never
    /// ran.
    #[test]
    fn vertex_id_readers_are_found() {
        let with = |rows: &[(u32, &str)]| container(&[(b"SFI0", &[0; 8]), (b"ISG1", &signature(rows))]);
        assert!(reads_vertex_id(&with(&[(0, "TEXCOORD"), (6, "SV_VertexID")])));
        // By system value, or by name.
        assert!(reads_vertex_id(&with(&[(6, "ID")])));
        assert!(reads_vertex_id(&with(&[(0, "sv_vertexid")])));
        // Attributes and the instance id only, or nothing.
        assert!(!reads_vertex_id(&with(&[(0, "TEXCOORD"), (8, "SV_InstanceID")])));
        assert!(!reads_vertex_id(&with(&[])));
        assert!(!reads_vertex_id(&container(&[(b"SFI0", &[0; 8])])));
        // Unreadable: taken as reading it.
        let mut short = signature(&[(0, "TEXCOORD")]);
        short[0] = 2;
        assert!(reads_vertex_id(&container(&[(b"ISG1", &short)])));
        assert!(reads_vertex_id(b"not a container"));
    }
}
