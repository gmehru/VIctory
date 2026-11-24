//! Colour twins of depth textures that shaders write through unordered
//! access views. The console lets a title make unordered access views of
//! depth textures and write them from compute (the intro's tiled lighting
//! reads a material depth a compute pass writes so); the host allows no
//! unordered access to a depth target, so the host copy of such a texture
//! has none, and its views were left null: the writes were lost and the
//! depth tests against the texture failed everywhere.
//!
//! Each such texture gets a twin, a texture of the same size, mips and
//! array slices in a colour format the host writes through unordered
//! access (R32_FLOAT for the 32- and 24-bit depth families, R16_UNORM for
//! the 16-bit one; R32_UINT or R16_UINT when the title's view is an
//! integer one). Its unordered access views are made on the twin; a draw
//! or dispatch running with such a view marks the twin newer, and before
//! the depth texture is next used otherwise (a depth target, read through
//! a shader view, cleared, copied) a full-screen pass writes the twin's
//! values into its depth plane ([`sync`]).

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::host::HostList;
use crate::object::This;
use crate::resource::HostTexture;

/// Whether depth textures the title writes through unordered access views
/// get colour twins (this module); off, such views are left null, as the
/// host refuses them, and what shaders write there is lost.
/// The other direction: a depth texture bound as a depth target, cleared
/// or copied into since its twin was last synced has its depth plane
/// copied into the twin before a draw or dispatch next runs with an
/// unordered view of it ([`refresh`]).
/// shortcut: the stencil plane is not served (an unordered view of plane 1
/// stays null, counted); a depth texture the host cannot read through a
/// shader view (the title denied it) is not copied into its twin (a later
/// unordered view reads the twin's older values; counted as stale uses),
/// nor is one written through an unordered view outside a table (none
/// seen); every depth target draw counts as a write (the pipeline's depth
/// write mask is not looked at), so the copy into the twin may be
/// needless; multisampled depth textures get no twin (the host has no
/// multisampled unordered access); a twin is float or integer after the
/// first view made of it, and views of the other kind are left null
/// (counted); the marks (newer, stale) are the texture's, not a list's:
/// lists recorded on several threads that write and use one texture are
/// taken in recording order, not submission order.
pub(crate) const DEPTH_TWINS: bool = true;

/// A depth texture's colour twin.
pub(crate) struct Twin {
    /// The twin texture, with the host state the list tracker keeps.
    pub(crate) copy: Arc<HostTexture>,
    format: DXGI_FORMAT,
    family: Family,
    integer: bool,
    /// Whether the depth texture's host copy is typeless and allows shader
    /// views (its depth plane can be copied into the twin).
    readable: bool,
    /// Shaders may have written the twin since its values last went into
    /// the depth texture.
    newer: AtomicBool,
    /// The depth texture may have been written since the twin was synced.
    stale: AtomicBool,
}

impl Drop for Twin {
    fn drop(&mut self) {
        TWINS.fetch_sub(1, Ordering::Relaxed);
        if self.newer.load(Ordering::Relaxed) {
            NEWER.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// A depth family: its typed depth format, the twin's float and integer
/// formats, and how the pass reads an integer twin (0 the float's bits, 1
/// 24-bit UNORM, 2 16-bit UNORM).
#[derive(Clone, Copy)]
struct Family {
    /// The typeless format and the depth plane's shader view format.
    typeless: u32,
    read: DXGI_FORMAT,
    depth: DXGI_FORMAT,
    float: DXGI_FORMAT,
    integer: DXGI_FORMAT,
    kind: u32,
}

/// The depth family of a host texture that may get a twin: a single-sample
/// 2D depth target without unordered access.
fn family_of(d: &D3D12_RESOURCE_DESC) -> Option<Family> {
    if d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 == 0
        || d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
        || d.SampleDesc.Count > 1
        || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
    {
        return None;
    }
    let typeless = crate::resource::family(d.Format.0 as u32);
    let (read, depth, float, integer, kind) = match typeless {
        19 => (
            DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
            DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
            DXGI_FORMAT_R32_FLOAT,
            DXGI_FORMAT_R32_UINT,
            0,
        ),
        39 => (
            DXGI_FORMAT_R32_FLOAT,
            DXGI_FORMAT_D32_FLOAT,
            DXGI_FORMAT_R32_FLOAT,
            DXGI_FORMAT_R32_UINT,
            0,
        ),
        44 => (
            DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
            DXGI_FORMAT_D24_UNORM_S8_UINT,
            DXGI_FORMAT_R32_FLOAT,
            DXGI_FORMAT_R32_UINT,
            1,
        ),
        53 => (
            DXGI_FORMAT_R16_UNORM,
            DXGI_FORMAT_D16_UNORM,
            DXGI_FORMAT_R16_UNORM,
            DXGI_FORMAT_R16_UINT,
            2,
        ),
        _ => return None,
    };
    Some(Family {
        typeless,
        read,
        depth,
        float,
        integer,
        kind,
    })
}

/// Integer view formats (the twin is then an integer one).
const INTEGER_VIEWS: [u32; 4] = [42, 43, 57, 59];
/// View formats of the stencil plane.
const STENCIL_VIEWS: [u32; 2] = [22, 47];

/// Twins made, unordered views served from a twin, copy passes run,
/// stencil-plane views left null, stale-twin uses, views of a kind (float
/// or integer) other than their twin's left null, copies of a depth plane
/// into its twin; since the last statistics line.
static STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
const MADE: usize = 0;
const SERVED: usize = 1;
const PASSES: usize = 2;
const STENCIL: usize = 3;
const STALE: usize = 4;
const MISMATCH: usize = 5;
const REFRESHES: usize = 6;

/// Twins marked newer: none, and draws skip looking for them.
static NEWER: AtomicUsize = AtomicUsize::new(0);

/// Twins alive: none, and draws skip looking for stale ones.
static TWINS: AtomicUsize = AtomicUsize::new(0);

/// Whether any twin is alive.
pub(crate) fn any_twins() -> bool {
    DEPTH_TWINS && TWINS.load(Ordering::Relaxed) > 0
}

/// The statistics since the last call ([`STATS`]).
#[cfg(feature = "diagnostics")]
pub(crate) fn take_stats() -> [u64; 7] {
    std::array::from_fn(|i| STATS[i].swap(0, Ordering::Relaxed))
}

/// Whether any twin is newer than its depth texture.
pub(crate) fn any_newer() -> bool {
    DEPTH_TWINS && NEWER.load(Ordering::Relaxed) > 0
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// Makes `copy`'s twin, of `family` (integer formats if `integer`).
fn make_twin(copy: &HostTexture, family: Family, integer: bool) -> Option<Twin> {
    let host = crate::host::host()?;
    // SAFETY: a live resource.
    let d = unsafe { copy.texture.GetDesc() };
    let format = if integer {
        family.integer
    } else {
        family.float
    };
    let desc = D3D12_RESOURCE_DESC {
        Format: format,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        Alignment: 0,
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        ..d
    };
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: a plain committed texture; the out pointer is ours.
    let made = unsafe {
        host.device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            None,
            &mut texture,
        )
    };
    if let Err(e) = made {
        tracing::warn!(error = %e, ?desc, "no depth twin");
        return None;
    }
    let texture = texture?;
    crate::host::name(&texture, || {
        format!("depth twin {}x{} f{}", d.Width, d.Height, format.0)
    });
    STATS[MADE].fetch_add(1, Ordering::Relaxed);
    TWINS.fetch_add(1, Ordering::Relaxed);
    let readable = d.Format.0 as u32 == family.typeless
        && d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0;
    Some(Twin {
        copy: Arc::new(HostTexture {
            texture,
            state: D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                .0
                .cast_unsigned()
                .into(),
            twin: OnceLock::new(),
        }),
        format,
        family,
        integer,
        readable,
        newer: AtomicBool::new(false),
        stale: AtomicBool::new(false),
    })
}

/// The unordered access view `desc` (host words, `None` for the whole
/// texture) of title texture `resource`, made on its twin at `at` when it
/// is a depth texture ([`DEPTH_TWINS`]): `None` if it is not one (the
/// caller goes on as before), else the twin the slot views, or `None` for
/// a view left null (the stencil plane, or a view of another kind than
/// the twin's).
pub(crate) fn unordered_view(
    resource: This,
    desc: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Option<ID3D12Resource>> {
    if !DEPTH_TWINS {
        return None;
    }
    unordered_view_of(&crate::resource::host_copy(resource)?, desc, at)
}

/// As [`unordered_view`], for host copy `copy`.
fn unordered_view_of(
    copy: &Arc<HostTexture>,
    desc: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Option<ID3D12Resource>> {
    // SAFETY: a live resource.
    let d = unsafe { copy.texture.GetDesc() };
    let family = family_of(&d)?;
    let host = crate::host::host()?;
    let view = desc.map_or(0, |v| v.Format.0 as u32);
    // Which view format the title asks for, once per kind and size: a
    // float (UNORM) or an integer write.
    {
        static SEEN: Mutex<Vec<(u32, i32, u64, u32)>> = Mutex::new(Vec::new());
        let key = (view, d.Format.0, d.Width, d.Height);
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if !seen.contains(&key) && seen.len() < 256 {
            seen.push(key);
            tracing::info!(
                view_format = view,
                format = d.Format.0,
                width = d.Width,
                height = d.Height,
                dimension = desc.map_or(0, |v| v.ViewDimension.0),
                "depth texture written through an unordered access view: served from its twin"
            );
        }
    }
    let mut desc = desc.copied();
    // The plane: 2D views and 2D array views name one.
    let plane = desc.as_mut().and_then(|v| {
        // SAFETY: the union member the dimension names.
        unsafe {
            match v.ViewDimension {
                D3D12_UAV_DIMENSION_TEXTURE2D => Some(&mut v.Anonymous.Texture2D.PlaneSlice),
                D3D12_UAV_DIMENSION_TEXTURE2DARRAY => {
                    Some(&mut v.Anonymous.Texture2DArray.PlaneSlice)
                }
                _ => None,
            }
        }
    });
    if STENCIL_VIEWS.contains(&view) || plane.as_ref().is_some_and(|p| **p != 0) {
        STATS[STENCIL].fetch_add(1, Ordering::Relaxed);
        once!(
            view,
            "unordered access to a depth texture's stencil plane: left null"
        );
        return Some(None);
    }
    let integer = INTEGER_VIEWS.contains(&view);
    let twin = copy
        .twin
        .get_or_init(|| make_twin(copy, family, integer))
        .as_ref();
    let Some(twin) = twin else {
        return Some(None);
    };
    if twin.integer != integer {
        STATS[MISMATCH].fetch_add(1, Ordering::Relaxed);
        once!(
            view,
            twin = twin.format.0,
            "an unordered view of a depth twin of another kind (float or integer): left null"
        );
        return Some(None);
    }
    if let Some(v) = desc.as_mut() {
        v.Format = twin.format;
    }
    // SAFETY: a live texture and a slot of the caller's heap.
    unsafe {
        host.device.CreateUnorderedAccessView(
            &twin.copy.texture,
            None::<&ID3D12Resource>,
            desc.as_ref().map(std::ptr::from_ref),
            at,
        )
    };
    STATS[SERVED].fetch_add(1, Ordering::Relaxed);
    Some(Some(twin.copy.texture.clone()))
}

/// Moves our host texture `copy` to `state` in `list`, through the list's
/// tracker.
fn ensure(list: &HostList, copy: &Arc<HostTexture>, state: u32) -> Option<D3D12_RESOURCE_BARRIER> {
    let before = list.transition_copy(copy, state)?;
    let after = crate::host::legal_state(&copy.texture, state);
    Some(crate::host::transition(
        &copy.texture,
        D3D12_RESOURCE_STATES(before as i32),
        D3D12_RESOURCE_STATES(after as i32),
    ))
}

const UNORDERED_ACCESS: u32 = 0x8;
const DEPTH_WRITE: u32 = 0x10;
const SHADER_RESOURCE: u32 = 0x40 | 0x80;

/// A draw or dispatch in `list` runs with an unordered view of title
/// texture `resource`: if it is a depth texture with a twin, the twin is
/// made ready for unordered access and marked newer.
pub(crate) fn written(list: &HostList, resource: This) {
    if !DEPTH_TWINS {
        return;
    }
    if let Some(copy) = crate::resource::host_copy(resource) {
        written_copy(list, &copy);
    }
}

/// As [`written`], for host copy `copy`.
fn written_copy(list: &HostList, copy: &HostTexture) {
    let Some(twin) = copy.twin.get().and_then(Option::as_ref) else {
        return;
    };
    if let Some(barrier) = ensure(list, &twin.copy, UNORDERED_ACCESS) {
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
    if twin.stale.swap(false, Ordering::Relaxed) {
        STATS[STALE].fetch_add(1, Ordering::Relaxed);
    }
    if !twin.newer.swap(true, Ordering::Relaxed) {
        NEWER.fetch_add(1, Ordering::Relaxed);
    }
}

/// Title texture `resource` was written as a depth texture (a depth
/// target, a clear, a copy): its twin, if any, no longer has its values.
pub(crate) fn depth_written(resource: This) {
    if !DEPTH_TWINS {
        return;
    }
    if let Some(copy) = crate::resource::host_copy(resource)
        && let Some(twin) = copy.twin.get().and_then(Option::as_ref)
    {
        twin.stale.store(true, Ordering::Relaxed);
    }
}

/// The twin of title texture `resource`, if it has one: what unordered
/// access barriers on the texture name (its unordered views write the
/// twin).
pub(crate) fn barrier_texture(resource: This) -> Option<This> {
    use windows::core::Interface;
    if !DEPTH_TWINS {
        return None;
    }
    let copy = crate::resource::host_copy(resource)?;
    let twin = copy.twin.get().and_then(Option::as_ref)?;
    Some(twin.copy.texture.as_raw().cast_const().cast())
}

/// Title texture `resource`'s host copy, if its twin may not have the
/// depth plane's values and can get them ([`refresh`]).
pub(crate) fn stale(resource: This) -> Option<Arc<HostTexture>> {
    if !any_twins() {
        return None;
    }
    let copy = crate::resource::host_copy(resource)?;
    let stale = copy.twin.get().and_then(Option::as_ref).is_some_and(|t| {
        t.readable && t.stale.load(Ordering::Relaxed) && !t.newer.load(Ordering::Relaxed)
    });
    stale.then_some(copy)
}

/// Title texture `resource`'s host copy, if it has a twin newer than it.
pub(crate) fn newer(resource: This) -> Option<Arc<HostTexture>> {
    if !any_newer() {
        return None;
    }
    let copy = crate::resource::host_copy(resource)?;
    let newer = copy
        .twin
        .get()
        .and_then(Option::as_ref)
        .is_some_and(|t| t.newer.load(Ordering::Relaxed));
    newer.then_some(copy)
}

/// The copy pass's root signature and pipeline for one depth format and
/// twin kind.
struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

const ROOT: &str = r#"#define RS "RootConstants(num32BitConstants=4, b0), DescriptorTable(SRV(t0))"
"#;

const VERTEX: &str = r#"
[RootSignature(RS)]
float4 main(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
"#;

const PIXEL_FLOAT: &str = r#"
cbuffer Pass : register(b0) { uint mip; uint slice; uint kind; uint unused; };
Texture2DArray<float> twin : register(t0);
[RootSignature(RS)]
float main(float4 position : SV_Position) : SV_Depth {
    return twin.Load(int4(int2(position.xy), slice, mip));
}
"#;

const PIXEL_UINT: &str = r#"
cbuffer Pass : register(b0) { uint mip; uint slice; uint kind; uint unused; };
Texture2DArray<uint> twin : register(t0);
[RootSignature(RS)]
float main(float4 position : SV_Position) : SV_Depth {
    uint v = twin.Load(int4(int2(position.xy), slice, mip));
    if (kind == 1) return (v & 0xffffff) / 16777215.0;
    if (kind == 2) return (v & 0xffff) / 65535.0;
    return asfloat(v);
}
"#;

/// The pass's shaders: vertex, float pixel, integer pixel.
type Shaders = (Vec<u8>, Vec<u8>, Vec<u8>);

/// The pass's shaders, compiled once. `None`, logged once, without the
/// compiler.
fn shaders() -> Option<&'static Shaders> {
    static MADE: OnceLock<Option<Shaders>> = OnceLock::new();
    MADE.get_or_init(|| {
        let compile = |body: &str, profile: &str| crate::rects::compile(&format!("{ROOT}{body}"), profile);
        let made = compile(VERTEX, "vs_6_0").and_then(|vs| {
            Ok((vs, compile(PIXEL_FLOAT, "ps_6_0")?, compile(PIXEL_UINT, "ps_6_0")?))
        });
        match made {
            Ok(made) => Some(made),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "no depth twin copy pass (dxcompiler.dll): depth written through unordered views stays unseen by depth tests"
                );
                None
            }
        }
    })
    .as_ref()
}

/// The pass for depth format `depth` from a twin of kind `integer`, made
/// once.
fn pass(depth: DXGI_FORMAT, integer: bool) -> Option<&'static Pass> {
    type Passes = Mutex<Vec<((i32, bool), Option<&'static Pass>)>>;
    static PASSES: Passes = Mutex::new(Vec::new());
    let mut passes = PASSES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&(_, p)) = passes.iter().find(|(k, _)| *k == (depth.0, integer)) {
        return p;
    }
    let made = make_pass(depth, integer);
    if let Err(e) = &made {
        tracing::warn!(error = %e, format = depth.0, integer, "no depth twin copy pass");
    }
    // Made once and kept for the process, as pipelines are.
    let made = made.ok().map(|p| &*Box::leak(Box::new(p)));
    passes.push(((depth.0, integer), made));
    made
}

fn make_pass(depth: DXGI_FORMAT, integer: bool) -> windows::core::Result<Pass> {
    let fail = windows::core::Error::empty;
    let device = &crate::host::host().ok_or_else(fail)?.device;
    let (vs, ps_float, ps_uint) = shaders().ok_or_else(fail)?;
    let ps = if integer { ps_uint } else { ps_float };
    let code = |b: &[u8]| D3D12_SHADER_BYTECODE {
        pShaderBytecode: b.as_ptr().cast(),
        BytecodeLength: b.len(),
    };
    // SAFETY: the blobs outlive the calls; the root signature is inside
    // the shaders.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, vs)?;
        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            VS: code(vs),
            PS: code(ps),
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                StencilEnable: false.into(),
                ..Default::default()
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 0,
            DSVFormat: depth,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        // The descriptor borrows the root signature (never released
        // through it).
        let pso = device.CreateGraphicsPipelineState(&desc)?;
        Ok(Pass { root, pso })
    }
}

/// Records in `list` the pass that writes the twin of `copy` (a depth
/// texture's host copy) into its depth plane, every mip and array slice,
/// and clears the twin's mark. The twin is read through a new dynamic slot
/// of `heap`, which the caller has bound and gives back when the list is
/// reset: the slot. Replaces the list's graphics root signature and
/// arguments, pipeline, targets, viewport, scissor and topology (the
/// caller restores the title's). `None`, with nothing recorded and the
/// mark kept, without the pass or a free slot.
pub(crate) fn sync(
    list: &HostList,
    heap: &crate::descriptor::host::HostHeap,
    copy: &Arc<HostTexture>,
) -> Option<u32> {
    let twin = copy.twin.get().and_then(Option::as_ref)?;
    if !twin.newer.load(Ordering::Relaxed) {
        return None;
    }
    let host = crate::host::host()?;
    let pass = pass(twin.family.depth, twin.integer)?;
    // SAFETY: a live resource.
    let d = unsafe { twin.copy.texture.GetDesc() };
    let mips = u32::from(d.MipLevels.max(1));
    let slices = u32::from(d.DepthOrArraySize.max(1));
    let Some(slot) = heap.dynamic(1) else {
        once!(
            "no free dynamic slot for a depth twin copy pass: the depth texture is left as it was"
        );
        return None;
    };
    let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: twin.format,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                MostDetailedMip: 0,
                MipLevels: mips,
                FirstArraySlice: 0,
                ArraySize: slices,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    heap.texture_srv_at(slot, &twin.copy.texture, &srv);
    let barriers: Vec<D3D12_RESOURCE_BARRIER> = [
        ensure(list, &twin.copy, SHADER_RESOURCE),
        ensure(list, copy, DEPTH_WRITE),
    ]
    .into_iter()
    .flatten()
    .collect();
    #[cfg(feature = "diagnostics")]
    list.note_op(|| {
        format!(
            "depth twin copy pass {}x{} f{}",
            d.Width, d.Height, d.Format.0
        )
    });
    let l = &list.list;
    // SAFETY: an open list; our root signature and pipeline, a slot of the
    // bound view heap, live textures the list's tracker holds.
    unsafe {
        if !barriers.is_empty() {
            l.ResourceBarrier(&barriers);
        }
        l.SetGraphicsRootSignature(&pass.root);
        l.SetPipelineState(&pass.pso);
        l.SetGraphicsRootDescriptorTable(1, heap.gpu(slot));
        l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        for mip in 0..mips {
            let width = (d.Width >> mip).max(1) as u32;
            let height = (d.Height >> mip).max(1);
            l.RSSetViewports(&[D3D12_VIEWPORT {
                Width: width as f32,
                Height: height as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            l.RSSetScissorRects(&[RECT {
                right: width as i32,
                bottom: height as i32,
                ..Default::default()
            }]);
            for slice in 0..slices {
                let dsv = D3D12_DEPTH_STENCIL_VIEW_DESC {
                    Format: twin.family.depth,
                    ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
                    Flags: D3D12_DSV_FLAG_NONE,
                    Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                        Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                            MipSlice: mip,
                            FirstArraySlice: slice,
                            ArraySize: 1,
                        },
                    },
                };
                let dsv = host.dsv(&copy.texture, Some(&dsv));
                l.OMSetRenderTargets(0, None, false, Some(&dsv));
                let constants = [mip, slice, twin.family.kind, 0];
                l.SetGraphicsRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                l.DrawInstanced(3, 1, 0, 0);
            }
        }
    }
    if twin.newer.swap(false, Ordering::Relaxed) {
        NEWER.fetch_sub(1, Ordering::Relaxed);
    }
    twin.stale.store(false, Ordering::Relaxed);
    STATS[PASSES].fetch_add(1, Ordering::Relaxed);
    Some(slot)
}

const REFRESH_FLOAT: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), DescriptorTable(SRV(t0), UAV(u0))"
cbuffer Pass : register(b0) { uint kind; uint width; uint height; uint unused; };
Texture2DArray<float> depth : register(t0);
RWTexture2DArray<float> twin : register(u0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= width || id.y >= height) return;
    twin[id] = depth.Load(int4(id, 0));
}
"#;

const REFRESH_UINT: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), DescriptorTable(SRV(t0), UAV(u0))"
cbuffer Pass : register(b0) { uint kind; uint width; uint height; uint unused; };
Texture2DArray<float> depth : register(t0);
RWTexture2DArray<uint> twin : register(u0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= width || id.y >= height) return;
    float v = depth.Load(int4(id, 0));
    twin[id] = kind == 1 ? uint(round(v * 16777215.0)) : kind == 2 ? uint(round(v * 65535.0)) : asuint(v);
}
"#;

/// The compute pass that copies a depth plane into a float (or integer)
/// twin, made once.
fn refresh_pass(integer: bool) -> Option<&'static Pass> {
    static PASSES: [OnceLock<Option<Pass>>; 2] = [OnceLock::new(), OnceLock::new()];
    PASSES[usize::from(integer)]
        .get_or_init(|| {
            let source = if integer { REFRESH_UINT } else { REFRESH_FLOAT };
            let made = crate::rects::compile(source, "cs_6_0")
                .map_err(|e| e.to_string())
                .and_then(|cs| make_compute(&cs).map_err(|e| e.to_string()));
            match made {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, integer, "no depth twin refresh pass");
                    None
                }
            }
        })
        .as_ref()
}

fn make_compute(cs: &[u8]) -> windows::core::Result<Pass> {
    let device = &crate::host::host()
        .ok_or_else(windows::core::Error::empty)?
        .device;
    // SAFETY: the blob outlives the calls; the root signature is inside
    // the shader and borrowed by the description, never released there.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, cs)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: cs.as_ptr().cast(),
                BytecodeLength: cs.len(),
            },
            ..Default::default()
        })?;
        Ok(Pass { root, pso })
    }
}

/// Records in `list` the copy of the depth plane of `copy` (a depth
/// texture's host copy, [`stale`]) into its twin, every mip and array
/// slice, through new dynamic slots of `heap` (bound by the caller, which
/// gives them back when the list is reset): the first slot and the count.
/// Replaces the list's compute root signature, arguments and pipeline
/// (the caller binds the title's again). `None`, with nothing recorded and
/// the twin left stale, without the pass or free slots.
pub(crate) fn refresh(
    list: &HostList,
    heap: &crate::descriptor::host::HostHeap,
    copy: &Arc<HostTexture>,
) -> Option<(u32, u32)> {
    let twin = copy.twin.get().and_then(Option::as_ref)?;
    if !twin.readable || !twin.stale.load(Ordering::Relaxed) {
        return None;
    }
    let pass = refresh_pass(twin.integer)?;
    // SAFETY: a live resource.
    let d = unsafe { twin.copy.texture.GetDesc() };
    let mips = u32::from(d.MipLevels.max(1));
    let slices = u32::from(d.DepthOrArraySize.max(1));
    let Some(first) = heap.dynamic(2 * mips) else {
        once!("no free dynamic slots for a depth twin refresh: the twin is left as it was");
        return None;
    };
    for mip in 0..mips {
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: twin.family.read,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                    MostDetailedMip: mip,
                    MipLevels: 1,
                    FirstArraySlice: 0,
                    ArraySize: slices,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: twin.format,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_UAV {
                    MipSlice: mip,
                    FirstArraySlice: 0,
                    ArraySize: slices,
                    PlaneSlice: 0,
                },
            },
        };
        heap.texture_srv_at(first + 2 * mip, &copy.texture, &srv);
        heap.texture_uav_at(first + 2 * mip + 1, &twin.copy.texture, &uav);
    }
    let barriers: Vec<D3D12_RESOURCE_BARRIER> = [
        ensure(list, copy, SHADER_RESOURCE),
        ensure(list, &twin.copy, UNORDERED_ACCESS),
    ]
    .into_iter()
    .flatten()
    .collect();
    #[cfg(feature = "diagnostics")]
    list.note_op(|| {
        format!(
            "depth twin refresh {}x{} f{}",
            d.Width, d.Height, d.Format.0
        )
    });
    let l = &list.list;
    // SAFETY: an open list; our root signature and pipeline, slots of the
    // bound view heap, live textures the list's tracker holds.
    unsafe {
        if !barriers.is_empty() {
            l.ResourceBarrier(&barriers);
        }
        l.SetComputeRootSignature(&pass.root);
        l.SetPipelineState(&pass.pso);
        for mip in 0..mips {
            let width = (d.Width >> mip).max(1) as u32;
            let height = (d.Height >> mip).max(1);
            let constants = [twin.family.kind, width, height, 0];
            l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
            l.SetComputeRootDescriptorTable(1, heap.gpu(first + 2 * mip));
            l.Dispatch(width.div_ceil(8), height.div_ceil(8), slices);
        }
        // The title's writes after it.
        l.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::transmute_copy(&twin.copy.texture),
                }),
            },
        }]);
    }
    twin.stale.store(false, Ordering::Relaxed);
    STATS[REFRESHES].fetch_add(1, Ordering::Relaxed);
    Some((first, 2 * mips))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::ManuallyDrop;

    const SIZE: u32 = 64;
    /// The id of texel `(x, y)`: 8x8 blocks, ids 1 to 4.
    fn id_at(x: u32, y: u32) -> u32 {
        (x / 8 + y / 8) % 4 + 1
    }

    /// Writes ids through a depth texture's unordered view (made as the
    /// descriptor path makes it, on the twin), runs the copy pass, then
    /// draws at the depth of one id with depth func EQUAL: exactly that
    /// id's texels are drawn. `typeless` is the depth texture's host
    /// format, `view` the title's view format. With `cleared`, the depth
    /// texture is cleared to 1 first and the twin marked stale (refreshed
    /// from it), the shader skips id 4's texels and the draw is at depth 1:
    /// exactly id 4's texels are drawn, the clear's value reaching the
    /// texels the shader left.
    fn ids_reach_the_depth_texture(typeless: DXGI_FORMAT, view: DXGI_FORMAT, cleared: bool) {
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        if shaders().is_none() {
            eprintln!("no dxcompiler.dll: not run");
            return;
        }
        let device = &host.device;
        let heap = crate::descriptor::host::HostHeap::new(false, 4).unwrap();
        let Some(family) = family_of(&D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Format: typeless,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        }) else {
            panic!("not a depth family: {typeless:?}");
        };
        let texture = |format, flags, state, clear: Option<&D3D12_CLEAR_VALUE>| {
            let mut t: Option<ID3D12Resource> = None;
            // SAFETY: plain creation.
            unsafe {
                device.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                        Width: u64::from(SIZE),
                        Height: SIZE,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: format,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Flags: flags,
                        ..Default::default()
                    },
                    state,
                    clear.map(std::ptr::from_ref),
                    &mut t,
                )
            }
            .unwrap();
            t.unwrap()
        };
        // The depth texture's host copy, cleared to 1 (no id's depth).
        let depth = Arc::new(HostTexture {
            texture: texture(
                typeless,
                D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                Some(&D3D12_CLEAR_VALUE {
                    Format: family.depth,
                    Anonymous: D3D12_CLEAR_VALUE_0 {
                        DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                            Depth: 1.0,
                            Stencil: 0,
                        },
                    },
                }),
            ),
            state: D3D12_RESOURCE_STATE_DEPTH_WRITE.0.cast_unsigned().into(),
            twin: OnceLock::new(),
        });
        let target = texture(
            DXGI_FORMAT_R32_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            None,
        );
        // The title's view: 2D, plane 0.
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: view,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        let slot = heap.dynamic(1).unwrap();
        let made = unordered_view_of(&depth, Some(&uav), heap.cpu(slot));
        let twin_texture = made.expect("a depth texture").expect("its twin");
        // The stencil plane is not served.
        let mut stencil = uav;
        stencil.Anonymous.Texture2D.PlaneSlice = 1;
        let spare = heap.dynamic(1).unwrap();
        assert!(matches!(
            unordered_view_of(&depth, Some(&stencil), heap.cpu(spare)),
            Some(None)
        ));
        // The ids' depths, as the title's programs compute them.
        let depths: [f32; 4] = std::array::from_fn(|i| (i + 1) as f32 / 65535.0);
        let write = crate::rects::compile(
            r#"
cbuffer Depths : register(b0) { float4 depths; uint skip; };
RWTexture2D<float> o : register(u0);
[RootSignature("RootConstants(num32BitConstants=5, b0), DescriptorTable(UAV(u0))")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = (id.x / 8 + id.y / 8) % 4;
    if (i + 1 == skip) return;
    o[id.xy] = depths[i];
}
"#,
            "cs_6_0",
        )
        .unwrap();
        let draw_vs = crate::rects::compile(
            r#"
cbuffer Z : register(b0) { float z; };
[RootSignature("RootConstants(num32BitConstants=1, b0)")]
float4 main(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), z, 1);
}
"#,
            "vs_6_0",
        )
        .unwrap();
        let draw_ps = crate::rects::compile(
            r#"
[RootSignature("RootConstants(num32BitConstants=1, b0)")]
float main() : SV_Target { return 1.0; }
"#,
            "ps_6_0",
        )
        .unwrap();
        let code = |b: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: b.as_ptr().cast(),
            BytecodeLength: b.len(),
        };
        let (id, z) = if cleared { (4, 1.0f32) } else { (3, depths[2]) };
        let constants: [u32; 5] = [
            depths[0].to_bits(),
            depths[1].to_bits(),
            depths[2].to_bits(),
            depths[3].to_bits(),
            if cleared { 4 } else { 0 },
        ];
        let list = host.new_list().unwrap();
        let (readback, read) = host.shared_buffer(u64::from(SIZE * SIZE * 4)).unwrap();
        // SAFETY: plain creation; an open list, our heap, pipelines and
        // textures, which outlive the submission.
        unsafe {
            let write_root: ID3D12RootSignature = device.CreateRootSignature(0, &write).unwrap();
            let write_pso: ID3D12PipelineState = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&write_root),
                    CS: code(&write),
                    ..Default::default()
                })
                .unwrap();
            let draw_root: ID3D12RootSignature = device.CreateRootSignature(0, &draw_vs).unwrap();
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&draw_root),
                VS: code(&draw_vs),
                PS: code(&draw_ps),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D12_COMPARISON_FUNC_EQUAL,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                DSVFormat: family.depth,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            desc.RTVFormats[0] = DXGI_FORMAT_R32_FLOAT;
            desc.BlendState.RenderTarget[0].RenderTargetWriteMask =
                D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            let draw_pso: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc).unwrap();
            let l = &list.list;
            l.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
            let dsv_desc = D3D12_DEPTH_STENCIL_VIEW_DESC {
                Format: family.depth,
                ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
                ..Default::default()
            };
            let mut refreshed = None;
            if cleared {
                // Through the list's tracker, as clears are.
                assert!(
                    ensure(&list, &depth, DEPTH_WRITE).is_none(),
                    "the list's first use"
                );
                let dsv = host.dsv(&depth.texture, Some(&dsv_desc));
                l.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
                depth
                    .twin
                    .get()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .stale
                    .store(true, Ordering::Relaxed);
                refreshed = Some(refresh(&list, &heap, &depth).expect("the refresh"));
            }
            // The title's dispatch, with the view in a table.
            written_copy(&list, &depth);
            l.SetComputeRootSignature(&write_root);
            l.SetPipelineState(&write_pso);
            l.SetComputeRoot32BitConstants(0, 5, constants.as_ptr().cast(), 0);
            l.SetComputeRootDescriptorTable(1, heap.gpu(slot));
            l.Dispatch(SIZE / 8, SIZE / 8, 1);
            // Before the depth texture is bound as a target.
            let pass_slot = sync(&list, &heap, &depth).expect("the copy pass");
            assert!(
                !depth
                    .twin
                    .get()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .newer
                    .load(Ordering::Relaxed),
                "the mark is cleared"
            );
            // The title's lit draw.
            let rtv = host.rtv(&target, None);
            l.ClearRenderTargetView(rtv, &[0.0; 4], None);
            let dsv = host.dsv(&depth.texture, Some(&dsv_desc));
            l.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
            l.RSSetViewports(&[D3D12_VIEWPORT {
                Width: SIZE as f32,
                Height: SIZE as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            l.RSSetScissorRects(&[RECT {
                right: SIZE as i32,
                bottom: SIZE as i32,
                ..Default::default()
            }]);
            l.SetGraphicsRootSignature(&draw_root);
            l.SetPipelineState(&draw_pso);
            l.SetGraphicsRoot32BitConstants(0, 1, (&raw const z).cast(), 0);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            l.DrawInstanced(3, 1, 0, 0);
            l.ResourceBarrier(&[crate::host::transition(
                &target,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(readback.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R32_FLOAT,
                            Width: SIZE,
                            Height: SIZE,
                            Depth: 1,
                            RowPitch: SIZE * 4,
                        },
                    },
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(target.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            l.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            drop(ManuallyDrop::into_inner(to.pResource));
            drop(ManuallyDrop::into_inner(from.pResource));
            list.keep(&write_pso);
            list.keep(&write_root);
            list.keep(&draw_pso);
            list.keep(&draw_root);
            list.keep(&twin_texture);
            list.close();
            host.execute(&[&list]);
            heap.release(pass_slot, 1);
            if let Some((first, n)) = refreshed {
                heap.release(first, n);
            }
        }
        // SAFETY: a plain query.
        assert!(unsafe { device.GetDeviceRemovedReason() }.is_ok());
        // SAFETY: the readback's mapped bytes, written by the finished copy.
        let texels = unsafe {
            std::slice::from_raw_parts(read.cast::<f32>(), (SIZE * SIZE) as usize).to_vec()
        };
        let wrong: Vec<(u32, u32, f32)> = (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .map(|(x, y)| (x, y, texels[(y * SIZE + x) as usize]))
            .filter(|&(x, y, t)| (t == 1.0) != (id_at(x, y) == id))
            .collect();
        assert!(
            wrong.is_empty(),
            "{typeless:?} (cleared {cleared}): {} texels wrong, first {:?}",
            wrong.len(),
            &wrong[..wrong.len().min(8)]
        );
        heap.release(slot, 1);
        heap.release(spare, 1);
    }

    /// The material depth's family (D24S8) and the shadow maps' (D16).
    #[test]
    fn unordered_writes_reach_the_depth_texture() {
        for cleared in [false, true] {
            ids_reach_the_depth_texture(
                DXGI_FORMAT_R24G8_TYPELESS,
                DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
                cleared,
            );
            ids_reach_the_depth_texture(DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_UNORM, cleared);
        }
    }
}
