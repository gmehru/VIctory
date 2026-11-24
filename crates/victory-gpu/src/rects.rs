//! Rectangle lists through a geometry shader. The console's rectangle-list
//! topology takes three corners per rectangle and the hardware adds the
//! fourth; the host has no such topology. The title's tiled lighting and
//! its other tile passes draw rectangles indirectly, from vertex programs
//! that read no vertex buffer and make each corner from the vertex id, so
//! neither the CPU expansion of vertex buffers ([`crate::primitive`]) nor a
//! rewrite of the indirect records (`indirect::rectangles`) completes them:
//! both change which vertex ids run. Here such a draw runs as a triangle
//! list with the title's own vertices, ids, instances and records, and a
//! geometry shader of ours turns each triangle (one rectangle's three
//! corners as the vertex program made them) into the rectangle: two
//! triangles, the fourth corner made from the other three.
//!
//! Which corner: [`crate::primitive`] takes the second vertex as the corner
//! between the other two (the title's full-screen rectangles come as top
//! left, bottom left, bottom right) and adds `v0 + v2 - v1`. The title's
//! tiled-lighting vertex programs put the corner first (top left, bottom
//! left, top right, from the vertex id), which that rule turns into a
//! sliver outside the tile. So the corner is found from the positions: the
//! vertex opposite the longest side (the diagonal), the second vertex on a
//! tie, and the fourth vertex is the diagonal's two ends minus it. For
//! rectangles given as [`crate::primitive`] expects, both make the same two
//! triangles, `(v0, v1, v2)` and `(v0, v2, v3)`, which turn as the title's
//! triangle does (front or back faces as the console has them). Rectangle
//! pipelines cull nothing unless asked to ([`CULLED`]).
//!
//! The geometry shader passes on exactly the vertex program's outputs (its
//! `OSG1`: names, indices, registers, masks, types). Float elements get
//! `a + b - corner` at the fourth vertex, the position included; integer
//! elements and those the pixel shader reads without interpolation take
//! the first vertex's value (the console's provoking vertex) at all four.
//! Clip and cull distances are extrapolated, render target and viewport
//! indices copied; another system value refuses the pipeline for
//! rectangles. The shader is HLSL compiled to DXIL (`gs_6_0`) by the
//! Windows SDK's `dxcompiler.dll`, loaded at run time and signing with the
//! SDK's `dxil.dll`: the host refuses a pipeline that mixes DXBC and DXIL
//! stages ("mixes shader model 6.0 or above with 5.1 or below"), so the
//! system's compiler cannot make it. The compiled signatures must match the
//! vertex program's row for row (the compiler packs them itself), or the
//! pipeline is refused for rectangles.
//!
//! shortcut: the fourth corner is extrapolated linearly in clip space: right
//! for screen-space rectangles (all corners with the same `w`, as the
//! title's are), only near for others.

use std::collections::HashMap;
use std::fmt::Write;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use victory_shader::{Container, FourCC};
use windows::Win32::Graphics::Direct3D::Dxc::*;
use windows::core::{GUID, HRESULT, HSTRING, Interface, PCWSTR, s, w};

/// Whether rectangle-list draws go through the geometry shader: on unless
/// `VICTORY_RECT_GS=0` (read in diagnostics builds). Off, rectangle draws
/// run as before it: indirect and indexed ones are dropped, direct ones
/// expanded on the CPU ([`crate::primitive`]), vertex ids and all.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let off = cfg!(feature = "diagnostics")
            && std::env::var_os("VICTORY_RECT_GS").is_some_and(|v| v == "0");
        if off {
            tracing::info!("rectangle draws through the geometry shader off (VICTORY_RECT_GS=0)");
        }
        !off
    })
}

/// Whether rectangle pipelines cull as the title's rasterizer state says;
/// off, they cull nothing. Whether the console face-culls rectangle lists
/// at all is not known (the Xbox 360 does not, as Xenia found), and for the
/// passes this path serves culling nothing gives the console's picture
/// either way: their vertex programs make every rectangle's corners in the
/// same order, so all rectangles of a draw turn the same way, and a draw
/// the console culled whole would not be drawn by the title. Culled as the
/// state says, the title's tile programs (counter-clockwise on screen)
/// under a back-culling state with clockwise front faces would lose every
/// tile, the pass black with nothing in the log to say why. The host
/// rectangle pipeline lines log the title's state.
/// shortcut: a draw that relies on the console culling some of its
/// rectangles (turning both ways) draws them all.
pub(crate) const CULLED: bool = false;

/// Why a rectangle draw is dropped when its pipeline has no host pipeline
/// itself (being made, refused, not translated: reasons noted as for any
/// draw).
pub(crate) const NO_HOST_PIPELINE: &str = "rectangles: no host pipeline";

/// One row of an `ISG1`/`OSG1` part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Element {
    pub(crate) name: String,
    pub(crate) index: u32,
    /// The system value (`D3D_NAME`: 0 none, 1 position, 2 clip distance...).
    pub(crate) system: u32,
    /// The component type: 1 uint, 2 int, 3 float (others: 16 or 64 bits).
    pub(crate) kind: u32,
    pub(crate) register: u32,
    pub(crate) mask: u8,
    pub(crate) stream: u32,
    /// Minimum precision (0: none).
    pub(crate) precision: u32,
}

/// The rows of `container`'s `part` signature (none without the part);
/// `None` if the container or the part does not read.
pub(crate) fn elements(container: &[u8], part: FourCC) -> Option<Vec<Element>> {
    let c = Container::parse(container).ok()?;
    let Some(p) = c.part(part) else {
        return Some(Vec::new());
    };
    let s = p.data;
    let word = |at: usize| -> Option<u32> {
        Some(u32::from_le_bytes(s.get(at..at.checked_add(4)?)?.try_into().ok()?))
    };
    let (count, first) = (word(0)? as usize, word(4)? as usize);
    (0..count)
        .map(|i| {
            let at = first.checked_add(32 * i)?;
            let name = s.get(word(at + 4)? as usize..)?.split(|&b| b == 0).next()?;
            Some(Element {
                name: std::str::from_utf8(name).ok()?.to_owned(),
                index: word(at + 8)?,
                system: word(at + 12)?,
                kind: word(at + 16)?,
                register: word(at + 20)?,
                mask: *s.get(at + 24)?,
                stream: word(at)?,
                precision: word(at + 28)?,
            })
        })
        .collect()
}

/// A pixel shader input as its `PSV0` lists it: first row, rows, mask and
/// DXIL interpolation mode (1 constant, 2 linear, 3 centroid, 4
/// noperspective, 5 noperspective centroid, 6 sample, 7 noperspective
/// sample).
type Mode = (u32, u32, u8, u8);

/// The interpolation modes of `ps`'s inputs that have registers; none
/// without a readable `PSV0` (runtime info version 1 on).
pub(crate) fn pixel_modes(ps: &[u8]) -> Vec<Mode> {
    fn read(ps: &[u8]) -> Option<Vec<Mode>> {
        let c = Container::parse(ps).ok()?;
        let d = c.part(FourCC::PSV0)?.data;
        let word = |at: usize| -> Option<usize> {
            Some(u32::from_le_bytes(d.get(at..at.checked_add(4)?)?.try_into().ok()?) as usize)
        };
        // The runtime info, then the resources, the string table, the
        // semantic index table and the signature elements (inputs first).
        let info = word(0)?;
        if info < 36 {
            return None;
        }
        let inputs = usize::from(*d.get(4 + 28)?);
        let mut at = 4 + info;
        let resources = word(at)?;
        at += 4;
        if resources > 0 {
            at = at.checked_add(4 + resources.checked_mul(word(at)?)?)?;
        }
        at = at.checked_add(4 + word(at)?)?;
        at = at.checked_add(4 + 4 * word(at)?)?;
        if inputs == 0 {
            return Some(Vec::new());
        }
        let size = word(at)?;
        at += 4;
        if size < 16 {
            return None;
        }
        let mut modes = Vec::new();
        for i in 0..inputs {
            let start = at.checked_add(i.checked_mul(size)?)?;
            let e = d.get(start..start.checked_add(16)?)?;
            let (rows, row, packed, mode) = (e[8], e[9], e[10], e[13]);
            // Bit 6: the element has a register.
            if packed & 0x40 == 0 {
                continue;
            }
            let (cols, col) = (u32::from(packed & 0xf), u32::from((packed >> 4) & 3));
            let mask = (((1u32 << cols.min(4)) - 1) << col) as u8;
            modes.push((u32::from(row), u32::from(rows), mask, mode));
        }
        Some(modes)
    }
    read(ps).unwrap_or_default()
}

/// The pixel shader's interpolation mode for vertex output `e` (0 if it
/// does not read it).
fn mode(modes: &[Mode], e: &Element) -> u8 {
    modes
        .iter()
        .find(|&&(row, rows, mask, _)| {
            (row..row + rows.max(1)).contains(&e.register) && mask & e.mask != 0
        })
        .map_or(0, |m| m.3)
}

/// The HLSL modifier of a DXIL interpolation mode (the compiler packs
/// elements of different modes into different rows, as the title's did).
fn modifier(mode: u8) -> &'static str {
    match mode {
        1 => "nointerpolation ",
        3 => "centroid ",
        4 => "noperspective ",
        5 => "noperspective centroid ",
        6 => "sample ",
        7 => "noperspective sample ",
        _ => "",
    }
}

/// The geometry shader's source for a vertex program with outputs
/// `outputs` and a pixel shader that interpolates them as `modes` says
/// ([`pixel_modes`]); why not for outputs it cannot carry.
pub(crate) fn source(outputs: &[Element], modes: &[Mode]) -> Result<String, &'static str> {
    let mut rows: Vec<&Element> = outputs.iter().collect();
    // In register order: the compiler packs in declaration order.
    rows.sort_by_key(|e| (e.register, e.mask.trailing_zeros()));
    let mut fields = String::new();
    let (mut extrapolated, mut copied, mut position) = (Vec::new(), Vec::new(), None);
    for (i, e) in rows.iter().enumerate() {
        if e.stream != 0 || e.precision != 0 {
            return Err("rectangles: an output of another stream or of minimum precision");
        }
        let mask = u32::from(e.mask);
        let cols = mask.count_ones();
        let contiguous = (1..=0xf).contains(&mask) && mask >> mask.trailing_zeros() == (1 << cols) - 1;
        if e.register == u32::MAX || !contiguous {
            return Err("rectangles: an output without a register or with a broken mask");
        }
        let ty = match e.kind {
            1 => "uint",
            2 => "int",
            3 => "float",
            _ => return Err("rectangles: a 16-bit or 64-bit output"),
        };
        let float = e.kind == 3;
        // Semantic, whether the fourth vertex extrapolates it, modifier.
        let (name, extrapolate, modifier) = match e.system {
            0 => {
                let first = e.name.bytes().next();
                let named = first.is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
                    && e.name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
                    // A trailing digit would be read as the index.
                    && !e.name.ends_with(|c: char| c.is_ascii_digit())
                    && !e.name.to_ascii_uppercase().starts_with("SV_");
                if !named {
                    return Err("rectangles: an output semantic HLSL cannot name");
                }
                let m = mode(modes, e);
                if float && m != 1 {
                    (e.name.as_str(), true, modifier(m))
                } else {
                    (e.name.as_str(), false, "nointerpolation ")
                }
            }
            1 => {
                if e.mask != 0xf || !float {
                    return Err("rectangles: a position of other than four floats");
                }
                position = Some(i);
                ("SV_Position", true, "")
            }
            2 => ("SV_ClipDistance", true, ""),
            3 => ("SV_CullDistance", true, ""),
            4 => ("SV_RenderTargetArrayIndex", false, ""),
            5 => ("SV_ViewportArrayIndex", false, ""),
            7 => ("SV_PrimitiveID", false, ""),
            _ => return Err("rectangles: an output system value the geometry shader cannot carry"),
        };
        let vector = if cols == 1 {
            ty.to_owned()
        } else {
            format!("{ty}{cols}")
        };
        let _ = writeln!(fields, "    {modifier}{vector} e{i} : {name}{};", e.index);
        if extrapolate {
            extrapolated.push(i);
        } else {
            copied.push(i);
        }
    }
    let Some(p) = position else {
        return Err("rectangles: no position output");
    };
    let mut s = format!(
        "struct V {{\n{fields}}};\n\
         [maxvertexcount(4)]\n\
         void main(triangle V v[3], inout TriangleStream<V> s) {{\n\
         \x20   float2 p0 = v[0].e{p}.xy / v[0].e{p}.w;\n\
         \x20   float2 p1 = v[1].e{p}.xy / v[1].e{p}.w;\n\
         \x20   float2 p2 = v[2].e{p}.xy / v[2].e{p}.w;\n\
         \x20   // Squared sides, each named after the vertex opposite it.\n\
         \x20   float l0 = dot(p2 - p1, p2 - p1);\n\
         \x20   float l1 = dot(p0 - p2, p0 - p2);\n\
         \x20   float l2 = dot(p1 - p0, p1 - p0);\n\
         \x20   // The corner m is opposite the diagonal, the longest side: the\n\
         \x20   // second vertex unless another side is longer. (a, m, b) turns\n\
         \x20   // as (v0, v1, v2) does.\n\
         \x20   V a = v[0], m = v[1], b = v[2];\n\
         \x20   if (l0 > l1 && l0 >= l2) {{ a = v[2]; m = v[0]; b = v[1]; }}\n\
         \x20   else if (l2 > l1 && l2 > l0) {{ a = v[1]; m = v[2]; b = v[0]; }}\n\
         \x20   V d = a;\n"
    );
    for i in extrapolated {
        let _ = writeln!(s, "    d.e{i} = a.e{i} + b.e{i} - m.e{i};");
    }
    // The provoking vertex's value at all four, whichever vertex the host
    // takes as provoking.
    for i in copied {
        let _ = writeln!(
            s,
            "    a.e{i} = v[0].e{i}; m.e{i} = v[0].e{i}; b.e{i} = v[0].e{i}; d.e{i} = v[0].e{i};"
        );
    }
    // A strip: (m, b, a) is the title's triangle, (a, b, d) the other half,
    // turned the same way (odd triangles of a strip turn back).
    s.push_str("    s.Append(m); s.Append(b); s.Append(a); s.Append(d);\n}\n");
    Ok(s)
}

/// The geometry shader for rectangles drawn with vertex program `vs` and
/// pixel shader `ps` (DXIL containers): compiled once per source and
/// vertex output rows, on the caller's thread (a pipeline worker). Why not,
/// as a drop reason, if it cannot be made.
pub(crate) fn geometry_shader(vs: &[u8], ps: Option<&[u8]>) -> Result<Arc<Vec<u8>>, &'static str> {
    type Made = Arc<OnceLock<Result<Arc<Vec<u8>>, &'static str>>>;
    // Source and output rows.
    type Key = (String, Vec<Row>);
    static MADE: Mutex<Option<HashMap<Key, Made>>> = Mutex::new(None);
    let outputs = elements(vs, FourCC::OSG1).ok_or("rectangles: vertex outputs unreadable")?;
    let modes = ps.map(pixel_modes).unwrap_or_default();
    let source = source(&outputs, &modes)?;
    // Keyed by the rows too: the source holds no registers or masks, so a
    // vertex program packed otherwise has the same source, and needs its
    // own check against what the compiler packs.
    let want = rows(&outputs);
    let slot = MADE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .entry((source.clone(), want.clone()))
        .or_default()
        .clone();
    slot.get_or_init(|| make(&source, &want)).clone()
}

/// A signature row as stages link by it: register, mask, system value,
/// semantic index, component type, and the semantic name for those without
/// a system value (names of system values vary).
type Row = (u32, u8, u32, u32, u32, String);

/// `elements`' rows, sorted.
fn rows(elements: &[Element]) -> Vec<Row> {
    let mut rows: Vec<Row> = elements
        .iter()
        .map(|e| {
            let name = if e.system == 0 {
                e.name.to_ascii_uppercase()
            } else {
                String::new()
            };
            (e.register, e.mask, e.system, e.index, e.kind, name)
        })
        .collect();
    rows.sort();
    rows
}

/// Compiles the geometry shader `source` for a vertex program whose output
/// rows are `want` ([`rows`]), checks that it is signed and that its
/// signatures have those rows.
fn make(source: &str, want: &[Row]) -> Result<Arc<Vec<u8>>, &'static str> {
    let code = match compile(source, "gs_6_0") {
        Ok(code) => code,
        Err(e) => {
            tracing::warn!(error = %e, source, "rectangle geometry shader not compiled: its pipelines' rectangle draws are dropped");
            return Err("rectangles: geometry shader not compiled");
        }
    };
    if code.get(4..20).is_none_or(|digest| digest.iter().all(|&b| b == 0)) {
        tracing::warn!("rectangle geometry shader not signed (no dxil.dll): its pipelines' rectangle draws are dropped");
        return Err("rectangles: geometry shader not signed");
    }
    for (part, what) in [(FourCC::ISG1, "inputs"), (FourCC::OSG1, "outputs")] {
        let got = rows(&elements(&code, part).unwrap_or_default());
        if got != want {
            tracing::warn!(
                what,
                ?want,
                ?got,
                source,
                "rectangle geometry shader packed otherwise than the vertex program: its pipelines' rectangle draws are dropped"
            );
            return Err("rectangles: geometry shader signature differs from the vertex program's");
        }
    }
    tracing::info!(bytes = code.len(), outputs = want.len(), "rectangle geometry shader made");
    Ok(Arc::new(code))
}

/// `DxcCreateInstance`.
type CreateInstance =
    unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut std::ffi::c_void) -> HRESULT;

/// `dxcompiler.dll`'s entry point, loaded once from the folder of the
/// `dxil.dll` the shader crate validates with (`VICTORY_DXIL`, else the
/// newest Windows SDK's; else wherever the system finds them). `dxil.dll`
/// is loaded first: the compiler signs with the `dxil.dll` it asks for by
/// name, which is then this one. `None`, logged once, without either.
fn create_instance() -> Option<CreateInstance> {
    static ENTRY: OnceLock<Option<CreateInstance>> = OnceLock::new();
    *ENTRY.get_or_init(|| {
        use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
        let signer = victory_shader::emit::validator::Validator::default_path();
        let compiler = signer.as_deref().and_then(Path::parent).map(|d| d.join("dxcompiler.dll"));
        let load = |path: Option<&Path>, name: &str| {
            let path = path.map_or_else(|| HSTRING::from(name), HSTRING::from);
            // SAFETY: a NUL-terminated name; the library stays loaded.
            unsafe { LoadLibraryW(&path) }.ok()
        };
        let module =
            load(signer.as_deref(), "dxil.dll").and_then(|_| load(compiler.as_deref(), "dxcompiler.dll"));
        // SAFETY: a loaded module and a NUL-terminated name.
        let entry = module.and_then(|m| unsafe { GetProcAddress(m, s!("DxcCreateInstance")) });
        match entry {
            Some(entry) => {
                tracing::info!(?compiler, "rectangle geometry shaders compile with dxcompiler.dll");
                // SAFETY: the export's documented signature.
                Some(unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, CreateInstance>(entry) })
            }
            None => {
                tracing::warn!(
                    ?compiler,
                    "no dxcompiler.dll and dxil.dll (Windows SDK): rectangle-list draws that need the geometry shader stay dropped"
                );
                None
            }
        }
    })
}

/// Compiles HLSL `source` (entry `main`) for `profile` with
/// `dxcompiler.dll`: the container, signed if `dxil.dll` was found, or the
/// compiler's messages.
pub(crate) fn compile(source: &str, profile: &str) -> Result<Vec<u8>, String> {
    let create = create_instance().ok_or("no dxcompiler.dll")?;
    let profile = HSTRING::from(profile);
    let args = [
        w!("-T"),
        PCWSTR(profile.as_ptr()),
        w!("-E"),
        w!("main"),
        w!("-Qstrip_debug"),
        w!("-Qstrip_reflect"),
    ];
    let failed = |e: windows::core::Error| e.to_string();
    // SAFETY: the compiler's own entry point and interfaces; the source and
    // the arguments live through the call; blobs are copied before release.
    unsafe {
        let mut raw = std::ptr::null_mut();
        create(&CLSID_DxcCompiler, &IDxcCompiler3::IID, &mut raw)
            .ok()
            .map_err(failed)?;
        let compiler = IDxcCompiler3::from_raw(raw);
        let buffer = DxcBuffer {
            Ptr: source.as_ptr().cast(),
            Size: source.len(),
            Encoding: DXC_CP_UTF8.0,
        };
        let result: IDxcResult = compiler
            .Compile(&buffer, Some(&args), None::<&IDxcIncludeHandler>)
            .map_err(failed)?;
        let status = result.GetStatus().map_err(failed)?;
        if status.is_err() {
            let text = result.GetErrorBuffer().map_or(String::new(), |b| {
                let bytes =
                    std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize());
                String::from_utf8_lossy(bytes).trim_end_matches('\0').trim().to_owned()
            });
            return Err(format!("{status}: {text}"));
        }
        let blob = result.GetResult().map_err(failed)?;
        Ok(std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::Win32::Graphics::Dxgi::*;

    fn element(name: &str, index: u32, system: u32, kind: u32, register: u32, mask: u8) -> Element {
        Element {
            name: name.into(),
            index,
            system,
            kind,
            register,
            mask,
            stream: 0,
            precision: 0,
        }
    }

    /// Hand-written output signatures: the position alone (as the title's
    /// tiled-lighting programs have it); the position, two float
    /// interpolants sharing a row, one the pixel shader reads
    /// noperspective, and an integer; and outputs it cannot carry.
    #[test]
    fn sources_follow_the_outputs() {
        let position = element("SV_Position", 0, 1, 3, 0, 0xf);
        let only = source(std::slice::from_ref(&position), &[]).unwrap();
        assert!(only.contains("    float4 e0 : SV_Position0;\n"), "{only}");
        assert!(only.contains("d.e0 = a.e0 + b.e0 - m.e0;"), "{only}");
        assert!(only.contains("s.Append(m); s.Append(b); s.Append(a); s.Append(d);"));
        let outputs = [
            element("COLOR", 0, 0, 1, 2, 0x1),
            element("TEXCOORD", 1, 0, 3, 1, 0xc),
            element("TEXCOORD", 0, 0, 3, 1, 0x3),
            position.clone(),
        ];
        // The pixel shader reads TEXCOORD1 (r1.zw) noperspective.
        let s = source(&outputs, &[(1, 1, 0xc, 4), (2, 1, 0x1, 1)]).unwrap();
        let fields = "    float4 e0 : SV_Position0;\n    float2 e1 : TEXCOORD0;\n    \
                      noperspective float2 e2 : TEXCOORD1;\n    nointerpolation uint e3 : COLOR0;\n";
        assert!(s.contains(fields), "{s}");
        for i in 0..3 {
            assert!(s.contains(&format!("d.e{i} = a.e{i} + b.e{i} - m.e{i};")), "{s}");
        }
        assert!(s.contains("d.e3 = v[0].e3;"), "{s}");
        // A float the pixel shader reads flat is copied too.
        let flat = source(&outputs, &[(1, 1, 0x3, 1)]).unwrap();
        assert!(flat.contains("nointerpolation float2 e1 : TEXCOORD0;"), "{flat}");
        assert!(flat.contains("d.e1 = v[0].e1;") && !flat.contains("d.e1 = a.e1"));
        let refused = |e: Element| source(&[position.clone(), e], &[]).unwrap_err();
        assert_eq!(refused(element("TEXCOORD", 0, 0, 6, 1, 0xf)), "rectangles: a 16-bit or 64-bit output");
        assert_eq!(
            refused(element("SV_IsFrontFace", 0, 9, 1, 1, 0x1)),
            "rectangles: an output system value the geometry shader cannot carry"
        );
        assert_eq!(refused(element("TEX2", 0, 0, 3, 1, 0xf)), "rectangles: an output semantic HLSL cannot name");
        assert_eq!(
            refused(element("TEXCOORD", 0, 0, 3, 1, 0x5)),
            "rectangles: an output without a register or with a broken mask"
        );
        assert_eq!(
            source(&[element("TEXCOORD", 0, 0, 3, 1, 0xf)], &[]).unwrap_err(),
            "rectangles: no position output"
        );
    }

    /// A container holding only `outputs` as its output signature: what
    /// [`geometry_shader`] reads of a vertex program.
    fn with_outputs(outputs: &[Element]) -> Vec<u8> {
        let mut rows = Vec::new();
        let mut names = Vec::new();
        for e in outputs {
            let name_at = 8 + 32 * outputs.len() + names.len();
            for word in [e.stream, name_at as u32, e.index, e.system, e.kind, e.register] {
                rows.extend(word.to_le_bytes());
            }
            rows.extend([e.mask, e.mask, 0, 0]);
            rows.extend(e.precision.to_le_bytes());
            names.extend(e.name.as_bytes());
            names.push(0);
        }
        let mut part = (outputs.len() as u32).to_le_bytes().to_vec();
        part.extend(8u32.to_le_bytes());
        part.extend(rows);
        part.extend(names);
        // The container: header, one part's offset, the part.
        let total = 0x24 + 8 + part.len();
        let mut out = b"DXBC".to_vec();
        out.extend([0; 16]);
        out.extend(1u16.to_le_bytes());
        out.extend(0u16.to_le_bytes());
        out.extend((total as u32).to_le_bytes());
        out.extend(1u32.to_le_bytes());
        out.extend(0x24u32.to_le_bytes());
        out.extend(b"OSG1");
        out.extend((part.len() as u32).to_le_bytes());
        out.extend(part);
        out
    }

    /// Geometry shaders are made per vertex output rows, registers and
    /// masks included, which their source does not hold: a vertex program
    /// with a float2 in the upper half of its row (where the compiler does
    /// not put it) is refused, and that refusal is not taken for another
    /// with the same semantics in the lower half, which the compiler packs
    /// alike.
    #[test]
    fn geometry_shaders_follow_the_packing() {
        let position = element("SV_Position", 0, 1, 3, 0, 0xf);
        let vs = |mask| with_outputs(&[position.clone(), element("PACKING", 0, 0, 3, 1, mask)]);
        assert_eq!(elements(&vs(0xc), FourCC::OSG1).unwrap()[1].mask, 0xc);
        if compile("float4 main() : SV_Target { return 0; }", "ps_6_0").is_err() {
            eprintln!("no dxcompiler.dll: not run");
            return;
        }
        assert_eq!(
            geometry_shader(&vs(0xc), None).unwrap_err(),
            "rectangles: geometry shader signature differs from the vertex program's"
        );
        assert!(geometry_shader(&vs(0x3), None).is_ok());
    }

    /// Four rectangles in pixels of a 32x32 target, three corners each:
    /// corner first (as the title's tile programs give them), second (as
    /// its full-screen ones and [`crate::primitive`] do), third, and first
    /// again turned the other way. Each vertex carries its pixel position,
    /// a float and an integer the pixel shader reads flat.
    const RECTS: [[u32; 4]; 4] = [[2, 3, 10, 9], [14, 4, 30, 12], [5, 16, 20, 28], [22, 18, 30, 30]];
    const VS: &str = r#"
static const float2 P[12] = {
    float2(2, 3), float2(2, 9), float2(10, 3),
    float2(14, 4), float2(14, 12), float2(30, 12),
    float2(5, 28), float2(20, 16), float2(5, 16),
    float2(30, 18), float2(30, 30), float2(22, 18),
};
struct V {
    float4 pos : SV_Position;
    float2 at : TEXCOORD0;
    nointerpolation float f : TEXCOORD1;
    nointerpolation uint id : TEXCOORD2;
};
V main(uint vid : SV_VertexID) {
    V o;
    float2 p = P[vid];
    o.pos = float4(p.x / 16 - 1, 1 - p.y / 16, 0.5, 1);
    o.at = p;
    o.f = vid;
    o.id = vid * 7;
    return o;
}
"#;
    const PS: &str = r#"
struct V {
    float4 pos : SV_Position;
    float2 at : TEXCOORD0;
    nointerpolation float f : TEXCOORD1;
    nointerpolation uint id : TEXCOORD2;
};
float4 main(V v) : SV_Target { return float4(v.at, v.f, v.id); }
"#;

    /// Draws `RECTS`' twelve vertices as a triangle list through `vs`, `gs`
    /// (none if empty) and `ps` on `device` into a 32x32 RGBA32F target
    /// cleared to -1, culling `cull`, directly or through `ExecuteIndirect`:
    /// the texels.
    fn draw(
        device: &ID3D12Device,
        (vs, gs, ps): (&[u8], &[u8], &[u8]),
        cull: D3D12_CULL_MODE,
        indirect: bool,
    ) -> Vec<[f32; 4]> {
        use std::mem::ManuallyDrop;
        const SIZE: u32 = 32;
        const FORMAT: DXGI_FORMAT = DXGI_FORMAT_R32G32B32A32_FLOAT;
        let code = |b: &[u8]| {
            if b.is_empty() {
                return D3D12_SHADER_BYTECODE::default();
            }
            D3D12_SHADER_BYTECODE {
                pShaderBytecode: b.as_ptr().cast(),
                BytecodeLength: b.len(),
            }
        };
        let heap = |kind| D3D12_HEAP_PROPERTIES {
            Type: kind,
            ..Default::default()
        };
        let buffer = |kind, size: u64, state| -> ID3D12Resource {
            let mut r: Option<ID3D12Resource> = None;
            // SAFETY: plain creation.
            unsafe {
                device.CreateCommittedResource(
                    &heap(kind),
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: size,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    state,
                    None,
                    &mut r,
                )
            }
            .unwrap();
            r.unwrap()
        };
        // SAFETY: D3D12 calls with descriptions that outlive them; the
        // resources live until the fence says the GPU is done.
        unsafe {
            let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
                Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
                Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                    Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
                        ..Default::default()
                    },
                },
            };
            let mut blob: Option<ID3DBlob> = None;
            D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).unwrap();
            let blob = blob.unwrap();
            let root: ID3D12RootSignature = device
                .CreateRootSignature(
                    0,
                    std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()),
                )
                .unwrap();
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(vs),
                GS: code(gs),
                PS: code(ps),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: cull,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            desc.RTVFormats[0] = FORMAT;
            desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            let pso: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc).unwrap();
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let allocator: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, &pso)
                .unwrap();
            let clear = [-1.0f32; 4];
            let mut target: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(
                    &heap(D3D12_HEAP_TYPE_DEFAULT),
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                        Width: u64::from(SIZE),
                        Height: SIZE,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: FORMAT,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    Some(&D3D12_CLEAR_VALUE {
                        Format: FORMAT,
                        Anonymous: D3D12_CLEAR_VALUE_0 { Color: clear },
                    }),
                    &mut target,
                )
                .unwrap();
            let target = target.unwrap();
            let rtvs: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    ..Default::default()
                })
                .unwrap();
            let rtv = rtvs.GetCPUDescriptorHandleForHeapStart();
            device.CreateRenderTargetView(&target, None, rtv);
            let pitch = SIZE * 16;
            let readback = buffer(
                D3D12_HEAP_TYPE_READBACK,
                u64::from(pitch * SIZE),
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            list.ClearRenderTargetView(rtv, &clear, None);
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.RSSetViewports(&[D3D12_VIEWPORT {
                Width: SIZE as f32,
                Height: SIZE as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            list.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                right: SIZE as i32,
                bottom: SIZE as i32,
                ..Default::default()
            }]);
            list.SetGraphicsRootSignature(&root);
            list.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            // Kept alive past the submission.
            let mut kept = Vec::new();
            if indirect {
                let arguments = buffer(D3D12_HEAP_TYPE_UPLOAD, 16, D3D12_RESOURCE_STATE_GENERIC_READ);
                let mut at = std::ptr::null_mut();
                arguments.Map(0, None, Some(&mut at)).unwrap();
                at.cast::<[u32; 4]>().write([12, 1, 0, 0]);
                arguments.Unmap(0, None);
                let argument = D3D12_INDIRECT_ARGUMENT_DESC {
                    Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW,
                    ..Default::default()
                };
                let mut signature: Option<ID3D12CommandSignature> = None;
                device
                    .CreateCommandSignature(
                        &D3D12_COMMAND_SIGNATURE_DESC {
                            ByteStride: 16,
                            NumArgumentDescs: 1,
                            pArgumentDescs: &argument,
                            NodeMask: 0,
                        },
                        None::<&ID3D12RootSignature>,
                        &mut signature,
                    )
                    .unwrap();
                let signature = signature.unwrap();
                list.ExecuteIndirect(&signature, 1, &arguments, 0, None, 0);
                kept.push(arguments.cast::<windows::core::IUnknown>().unwrap());
                kept.push(signature.cast().unwrap());
            } else {
                list.DrawInstanced(12, 1, 0, 0);
            }
            list.ResourceBarrier(&[crate::host::transition(
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
                            Format: FORMAT,
                            Width: SIZE,
                            Height: SIZE,
                            Depth: 1,
                            RowPitch: pitch,
                        },
                    },
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(target.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            drop(ManuallyDrop::into_inner(to.pResource));
            drop(ManuallyDrop::into_inner(from.pResource));
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            let started = std::time::Instant::now();
            while fence.GetCompletedValue() < 1 {
                assert!(started.elapsed().as_secs() < 30, "the GPU did not finish");
                std::thread::yield_now();
            }
            assert!(device.GetDeviceRemovedReason().is_ok());
            let mut at = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut at)).unwrap();
            let texels =
                std::slice::from_raw_parts(at.cast::<[f32; 4]>(), (SIZE * SIZE) as usize).to_vec();
            readback.Unmap(0, None);
            drop(kept);
            texels
        }
    }

    /// The rectangle holding pixel `(x, y)`, if one does.
    fn rect_at(x: u32, y: u32) -> Option<usize> {
        RECTS
            .iter()
            .position(|&[x0, y0, x1, y1]| (x0..x1).contains(&x) && (y0..y1).contains(&y))
    }

    /// Checks a drawing of `drawn` rectangles: each of their pixels, all
    /// four corners' included, holds its interpolated position (the fourth
    /// corner's made by the geometry shader) and the first vertex's flat
    /// values; nothing else is written.
    fn check(texels: &[[f32; 4]], drawn: &[usize], what: &str) {
        for (i, t) in texels.iter().enumerate() {
            let (x, y) = (i as u32 % 32, i as u32 / 32);
            match rect_at(x, y).filter(|r| drawn.contains(r)) {
                Some(r) => {
                    let first = 3.0 * r as f32;
                    let want = [x as f32 + 0.5, y as f32 + 0.5, first, first * 7.0];
                    let near = t.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-3);
                    assert!(near, "{what}: pixel ({x}, {y}) of rectangle {r} holds {t:?}, not {want:?}");
                }
                None => assert_eq!(*t, [-1.0; 4], "{what}: pixel ({x}, {y}) outside the rectangles written"),
            }
        }
    }

    /// Executed: the rectangles drawn as triangles through the geometry
    /// shader made for the vertex program, directly and through
    /// `ExecuteIndirect`, on the host GPU and on WARP: each rectangle whole
    /// whichever vertex is its corner, nothing outside, the fourth corner's
    /// interpolated position right, flat values the first vertex's; culled
    /// whole or not at all, as the title's own triangle (drawn without the
    /// geometry shader) is culled.
    #[test]
    fn rectangles_are_completed_on_the_gpu() {
        let (Ok(vs), Ok(ps)) = (compile(VS, "vs_6_0"), compile(PS, "ps_6_0")) else {
            eprintln!("no dxcompiler.dll: not run");
            return;
        };
        let gs = geometry_shader(&vs, Some(&ps)).unwrap();
        let source = source(&elements(&vs, FourCC::OSG1).unwrap(), &pixel_modes(&ps)).unwrap();
        // The pixel shader reads the float flat: copied, not extrapolated.
        assert!(source.contains("d.e1 = a.e1 + b.e1 - m.e1;") && source.contains("d.e2 = v[0].e2;"), "{source}");
        let mut devices: Vec<(&str, ID3D12Device)> = Vec::new();
        if let Some(host) = crate::host::host() {
            devices.push(("host GPU", host.device.clone()));
        }
        // SAFETY: plain DXGI and D3D12 creation.
        let warp = unsafe {
            CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                .and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>())
                .and_then(|a| {
                    let mut d: Option<ID3D12Device> = None;
                    windows::Win32::Graphics::Direct3D12::D3D12CreateDevice(
                        &a,
                        windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0,
                        &mut d,
                    )
                    .map(|()| d)
                })
        };
        match warp {
            Ok(Some(d)) => devices.push(("WARP", d)),
            _ => eprintln!("no WARP device: drawn on the host GPU only"),
        }
        let all = [0, 1, 2, 3];
        for (name, device) in &devices {
            let shaders = (vs.as_slice(), gs.as_slice(), ps.as_slice());
            for indirect in [false, true] {
                let what = format!("{name}, indirect {indirect}");
                check(&draw(device, shaders, D3D12_CULL_MODE_NONE, indirect), &all, &what);
            }
            // The rectangles whose own triangle a culling mode keeps, drawn
            // without the geometry shader: the first three turn one way,
            // the fourth the other, so each mode keeps some and drops some.
            let kept = |cull| -> Vec<usize> {
                let texels = draw(device, (vs.as_slice(), &[], ps.as_slice()), cull, false);
                let mut kept: Vec<usize> = (0..texels.len())
                    .filter(|&i| texels[i] != [-1.0; 4])
                    .filter_map(|i| rect_at(i as u32 % 32, i as u32 / 32))
                    .collect();
                kept.sort_unstable();
                kept.dedup();
                kept
            };
            let back = kept(D3D12_CULL_MODE_BACK);
            let front = kept(D3D12_CULL_MODE_FRONT);
            let mut both = [back.clone(), front.clone()].concat();
            both.sort_unstable();
            assert_eq!(both, all, "{name}: each triangle kept by one mode: {back:?} {front:?}");
            assert!(!back.is_empty() && !front.is_empty(), "{name}: {back:?} {front:?}");
            // Through the geometry shader, exactly those rectangles, whole.
            for (cull, title) in [(D3D12_CULL_MODE_BACK, back), (D3D12_CULL_MODE_FRONT, front)] {
                let what = format!("{name}, culling {}", cull.0);
                check(&draw(device, shaders, cull, false), &title, &what);
            }
        }
    }
}
