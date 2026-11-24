//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128, rectangles through the
//! geometry shader at twice their vertices: each rectangle's three as its
//! two triangles' six): records past it run empty, so a list of thousands
//! of garbage records cannot run for seconds either. Records past it are
//! not charged, so the count cannot wrap round and let later ones run.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1)"
// weight: what a draw's vertex instance costs (1 to 4).
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint weight; };
RWByteAddressBuffer args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= records) return;
    uint at = id.x * stride;
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args.Load(at), 1u << 24);
        uint instances = min(args.Load(at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances * weight;
        args.Store2(at, uint2(count, instances));
    } else if (kind == 2) {
        uint3 g = min(args.Load3(at), uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        args.Store3(at, g);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = args.Load3(at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        args.Store3(at, g);
    }
    // Charged only while budget is left, at most the budget at a time, so
    // the count stays under twice the budget: an add that wrapped it
    // would let the records after it run again.
    uint before;
    budget.InterlockedAdd(0, 0, before);
    [allow_uav_condition] while (before < BUDGET) {
        uint seen;
        budget.InterlockedCompareExchange(0, before, before + min(cost, BUDGET), seen);
        if (seen == before) {
            break;
        }
        before = seen;
    }
    if (before >= BUDGET || before + cost > BUDGET) {
        // Draws keep their layout with no vertices; groups become zero.
        args.Store(at, 0);
    }
}
"#;

/// Rectangle-list draws, rewritten as the triangle-list draws of their
/// expanded vertices (`draw::State::rectangles`: six per rectangle, from
/// the first vertex of the bound buffers on) into records of our own, so
/// the title's records stay as they are. A record must start on a
/// rectangle and stay within the `vertices` expanded; others draw nothing.
/// The clamp's bound of 2^24 vertices holds for the triangles too.
/// shortcut: the budget is charged the title's count, half the vertices
/// the triangles run.
const RECTANGLES: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) { uint stride; uint records; uint vertices; uint unused; };
RWByteAddressBuffer args : register(u0);
RWByteAddressBuffer triangles : register(u1);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= records) return;
    // Vertex count, instances, first vertex, first instance.
    uint4 a = args.Load4(id.x * stride);
    uint rects = min(a.x / 3, (1u << 24) / 6);
    bool whole = a.z % 3 == 0 && a.z <= vertices && rects * 3 <= vertices - a.z;
    triangles.Store4(id.x * 16, whole ? uint4(rects * 6, a.y, a.z / 3 * 6, a.w) : uint4(0, 0, 0, 0));
}
"#;

/// One of our compute passes: its root signature (inside the shader) and
/// pipeline.
struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

fn rectangle_pass() -> Option<&'static Pass> {
    static PASS: std::sync::OnceLock<Option<Pass>> = std::sync::OnceLock::new();
    PASS.get_or_init(|| match make(RECTANGLES, "rectangle pass") {
        Ok(c) => Some(c),
        Err(e) => {
            tracing::warn!(error = %e, "no rectangle pass: indirect rectangles dropped");
            None
        }
    })
    .as_ref()
}

fn clamp() -> Option<&'static Pass> {
    static CLAMP: std::sync::OnceLock<Option<Pass>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make(SOURCE, "argument clamp") {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make(source: &str, what: &str) -> windows::core::Result<Pass> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("victory"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("{what}: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Pass { root, pso })
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` in
/// `buffer`, and leaves the buffer ready to be read as arguments. A draw
/// record is charged `weight` (1 to 4) times its vertex instances: what
/// runs per vertex the title counts (2 for rectangles through the geometry
/// shader, two triangles per rectangle: [`crate::rects`]). Replaces the
/// list's compute root signature and pipeline (callers bind theirs after).
/// False if the clamp is not available.
pub(crate) fn bound(
    list: &HostList,
    buffer: &ID3D12Resource,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
    weight: u32,
) -> bool {
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return false;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return false;
    }
    let l = &list.list;
    // At most 2^26 vertex instances a record: 4 times that still fits.
    let constants = [kind, stride, records, weight.clamp(1, 4)];
    // SAFETY: an open list; our root signature, pipeline and a live buffer
    // the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.Dispatch(records.div_ceil(64), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
    true
}

/// Records, in `list`, the rewrite of `records` rectangle-list draw records
/// `stride` bytes apart at host address `gpu` in `buffer` (bounded: read as
/// arguments, and left so) into triangle-list records over the first
/// `vertices` vertices expanded (see [`RECTANGLES`]): 16-byte records in
/// the list's argument memory, ready to be read as arguments ([`done`] for
/// both buffers after the execution). Replaces the list's compute root
/// signature and pipeline. None, with nothing recorded, if the pass or the
/// memory is not available.
/// shortcut: at most 4,096 records (the list's argument memory comes in
/// 64 KiB pieces); more are dropped.
pub(crate) fn rectangles(
    list: &HostList,
    buffer: &ID3D12Resource,
    gpu: u64,
    (stride, records, vertices): (u32, u32, u32),
) -> Option<(ID3D12Resource, u64)> {
    let p = rectangle_pass()?;
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) || stride < 16 {
        return None;
    }
    let (out, offset) = list.argument_space(u64::from(records) * 16)?;
    const READ: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    const WRITE: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    let l = &list.list;
    let constants = [stride, records, vertices, 0];
    // SAFETY: an open list; our root signature, pipeline, the caller's live
    // buffer and the list's own argument memory.
    unsafe {
        let to = out.GetGPUVirtualAddress() + offset;
        l.ResourceBarrier(&[transition(buffer, READ, WRITE)]);
        l.SetComputeRootSignature(&p.root);
        l.SetPipelineState(&p.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, to);
        l.Dispatch(records.div_ceil(64), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(buffer, WRITE, READ),
            transition(&out, WRITE, READ),
        ]);
    }
    Some((out, offset))
}

/// After the execution: the buffer back to unordered access, where the
/// title's passes keep using it.
pub(crate) fn done(list: &HostList, buffer: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            buffer,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )])
    };
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

#[cfg(test)]
mod tests {
    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    /// A draw record's vertex instances are charged `weight` times: once for
    /// plain draws, twice for rectangles through the geometry shader, which
    /// runs two triangles for each three vertices. The records themselves
    /// come out clamped only.
    #[test]
    fn the_budget_is_charged_what_runs() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        for weight in [1, 2] {
            // 12 vertices of 3 instances, then 6 of 1.
            let records: [[u32; 4]; 2] = [[12, 3, 0, 0], [6, 1, 3, 0]];
            let (input, data) = host.shared_buffer(32).unwrap();
            // SAFETY: the buffer's mapping holds 32 bytes.
            unsafe { data.cast::<[[u32; 4]; 2]>().write(records) };
            let list = host.new_list().unwrap();
            // SAFETY: a live buffer.
            let gpu = unsafe { input.GetGPUVirtualAddress() };
            assert!(super::bound(&list, &input, gpu, (0, 16, 2), weight));
            super::done(&list, &input);
            list.close();
            host.execute(&[&list]);
            assert_eq!(list.budget_spent(), Some(42 * weight));
            // SAFETY: as above.
            assert_eq!(unsafe { data.cast::<[[u32; 4]; 2]>().read() }, records);
        }
    }

    /// Records past the budget run empty and are not charged: a hundred
    /// records costing half the budget, or all of it, let two or one run
    /// and the count stop at the budget. (Charged, the count wrapped round
    /// after 64 or 32 of them and let the next ones run again.)
    #[test]
    fn the_budget_does_not_wrap() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        const RECORDS: usize = 100;
        for weight in [1, 2] {
            // 2^24 vertices of 4 instances: 2^26 vertex instances each.
            let records = [[1u32 << 24, 4, 0, 0]; RECORDS];
            let (input, data) = host.shared_buffer(16 * RECORDS as u64).unwrap();
            // SAFETY: the buffer's mapping holds the records.
            unsafe { data.cast::<[[u32; 4]; RECORDS]>().write(records) };
            let list = host.new_list().unwrap();
            // SAFETY: a live buffer.
            let gpu = unsafe { input.GetGPUVirtualAddress() };
            let records_at = (0, 16, RECORDS as u32);
            assert!(super::bound(&list, &input, gpu, records_at, weight));
            super::done(&list, &input);
            list.close();
            host.execute(&[&list]);
            assert_eq!(list.budget_spent(), Some(1 << 27), "weight {weight}");
            // SAFETY: as above.
            let out = unsafe { data.cast::<[[u32; 4]; RECORDS]>().read() };
            let ran = out.iter().filter(|r| r[0] != 0).count();
            assert_eq!(ran, 2 / weight as usize, "weight {weight}");
            assert!(out.iter().all(|r| r[0] == 0 || *r == records[0]));
        }
    }

    /// Rectangle records become records of their expanded triangles: six
    /// vertices per rectangle, from the rectangle's own start. Records that
    /// start inside a rectangle, or reach past the expanded vertices, draw
    /// nothing; the title's records are left as they were.
    #[test]
    fn rectangle_records_become_triangle_records() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        // 20-byte records (4 bytes past the draw), 9 expanded vertices.
        let records: [[u32; 5]; 4] = [
            [6, 2, 0, 1, 0xdead],
            [7, 1, 3, 0, 0],
            [3, 1, 1, 0, 0],
            [3, 1, 9, 0, 0],
        ];
        let (input, data) = host.shared_buffer(80).unwrap();
        // SAFETY: the buffer's mapping holds 80 bytes.
        unsafe { data.cast::<[[u32; 5]; 4]>().write(records) };
        let list = host.new_list().unwrap();
        // SAFETY: a live buffer.
        let gpu = unsafe { input.GetGPUVirtualAddress() };
        assert!(super::bound(&list, &input, gpu, (0, 20, 4), 1));
        let (out, offset) = super::rectangles(&list, &input, gpu, (20, 4, 9)).unwrap();
        super::done(&list, &out);
        super::done(&list, &input);
        list.close();
        host.execute(&[&list]);
        let read = |buffer: &super::ID3D12Resource, at: u64| {
            let mut p = std::ptr::null_mut();
            // SAFETY: a mapped CPU-visible buffer holding the records read.
            unsafe {
                buffer.Map(0, None, Some(&mut p)).unwrap();
                let v = p
                    .cast::<u8>()
                    .add(at as usize)
                    .cast::<[[u32; 4]; 4]>()
                    .read_unaligned();
                buffer.Unmap(0, None);
                v
            }
        };
        assert_eq!(
            read(&out, offset),
            [[12, 2, 0, 1], [12, 1, 6, 0], [0; 4], [0; 4]]
        );
        // SAFETY: as above.
        assert_eq!(unsafe { data.cast::<[[u32; 5]; 4]>().read() }, records);
    }
}
