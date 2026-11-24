//! Host pipelines for title pipelines (decision 0007, step 3): the
//! title's graphics state with the translated shaders, under a root
//! signature of our own that the translated shaders' bindings define
//! (`victory_shader::emit`): per stage one block of root constants for the
//! user data, root views for the title's root constant buffers and tables
//! of one view for its root shader views (bounded:
//! [`crate::draw::BOUNDED_ROOT_SRVS`]), static samplers for the immediate
//! samplers, and directly indexed heaps. A pipeline's rectangle-list draws
//! get a second host pipeline, the same plus a geometry shader of ours
//! ([`HostPipeline::rects`], [`crate::rects`]).

use std::ffi::{CStr, CString};
use std::sync::Arc;

use victory_shader::emit::{ROOT_CBV_SPACE, SAMPLER_SPACE, STRIDE_SPACE, USER_DATA_SPACE};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::PCSTR;

use super::{Shader, Translated};
use crate::host::host;

/// A stream's graphics state, owned (the title's stream lives only for the
/// creation call). Subobjects the stream leaves out keep D3D12's defaults.
pub(crate) struct GraphicsState {
    blend: D3D12_BLEND_DESC,
    sample_mask: u32,
    rasterizer: D3D12_RASTERIZER_DESC,
    depth: D3D12_DEPTH_STENCIL_DESC,
    /// Input elements, their names pointing into `names`.
    inputs: Vec<D3D12_INPUT_ELEMENT_DESC>,
    names: Vec<CString>,
    strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE,
    pub(crate) topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE,
    pub(crate) targets: D3D12_RT_FORMAT_ARRAY,
    depth_format: DXGI_FORMAT,
    samples: DXGI_SAMPLE_DESC,
}

impl Default for GraphicsState {
    fn default() -> Self {
        let target = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let keep = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        GraphicsState {
            blend: D3D12_BLEND_DESC {
                RenderTarget: [target; 8],
                ..Default::default()
            },
            sample_mask: u32::MAX,
            rasterizer: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_BACK,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            depth: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS,
                StencilReadMask: 0xff,
                StencilWriteMask: 0xff,
                FrontFace: keep,
                BackFace: keep,
                ..Default::default()
            },
            inputs: Vec::new(),
            names: Vec::new(),
            strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED,
            targets: D3D12_RT_FORMAT_ARRAY::default(),
            depth_format: DXGI_FORMAT_UNKNOWN,
            samples: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
        }
    }
}

impl GraphicsState {
    /// Takes in one subobject of a stream: its type, then its data at +4
    /// (+8 for those holding pointers).
    ///
    /// # Safety
    /// `at` points at a whole subobject of type `kind`.
    pub(crate) unsafe fn read(&mut self, kind: u32, at: *const u8) {
        // SAFETY: the caller passes a whole subobject.
        unsafe {
            let data = at.add(4);
            match kind {
                8 => self.blend = data.cast::<D3D12_BLEND_DESC>().read_unaligned(),
                9 => self.sample_mask = data.cast::<u32>().read_unaligned(),
                10 => self.rasterizer = data.cast::<D3D12_RASTERIZER_DESC>().read_unaligned(),
                11 => self.depth = data.cast::<D3D12_DEPTH_STENCIL_DESC>().read_unaligned(),
                21 => {
                    // shortcut: the depth bounds test is dropped (the host
                    // pipeline uses the plain description); bring it back
                    // with a stream if a title enables it.
                    let d = data.cast::<D3D12_DEPTH_STENCIL_DESC1>().read_unaligned();
                    self.depth = D3D12_DEPTH_STENCIL_DESC {
                        DepthEnable: d.DepthEnable,
                        DepthWriteMask: d.DepthWriteMask,
                        DepthFunc: d.DepthFunc,
                        StencilEnable: d.StencilEnable,
                        StencilReadMask: d.StencilReadMask,
                        StencilWriteMask: d.StencilWriteMask,
                        FrontFace: d.FrontFace,
                        BackFace: d.BackFace,
                    };
                }
                12 => {
                    let layout = at.add(8).cast::<D3D12_INPUT_LAYOUT_DESC>().read_unaligned();
                    if layout.pInputElementDescs.is_null() {
                        return;
                    }
                    let elements = std::slice::from_raw_parts(
                        layout.pInputElementDescs,
                        layout.NumElements as usize,
                    );
                    self.names = elements
                        .iter()
                        .map(|e| CStr::from_ptr(e.SemanticName.0.cast()).to_owned())
                        .collect();
                    self.inputs = elements
                        .iter()
                        .zip(&self.names)
                        .map(|(e, name)| D3D12_INPUT_ELEMENT_DESC {
                            SemanticName: PCSTR(name.as_ptr().cast()),
                            Format: DXGI_FORMAT(
                                crate::resource::pc_format(e.Format.0 as u32) as i32
                            ),
                            ..*e
                        })
                        .collect();
                }
                13 => {
                    self.strip_cut =
                        D3D12_INDEX_BUFFER_STRIP_CUT_VALUE(data.cast::<i32>().read_unaligned())
                }
                14 => {
                    self.topology =
                        D3D12_PRIMITIVE_TOPOLOGY_TYPE(data.cast::<i32>().read_unaligned())
                }
                15 => {
                    self.targets = data.cast::<D3D12_RT_FORMAT_ARRAY>().read_unaligned();
                    for f in &mut self.targets.RTFormats {
                        f.0 = crate::resource::pc_format(f.0 as u32) as i32;
                    }
                }
                16 => {
                    let f = data.cast::<u32>().read_unaligned();
                    self.depth_format = DXGI_FORMAT(crate::resource::pc_format(f) as i32);
                }
                17 => self.samples = data.cast::<DXGI_SAMPLE_DESC>().read_unaligned(),
                _ => {}
            }
        }
    }

    /// Everything the state holds, as text: part of a pipeline's name in
    /// the library and of the key a title pipeline is started early by.
    pub(crate) fn describe(&self) -> String {
        let inputs: Vec<String> = self
            .inputs
            .iter()
            .zip(&self.names)
            .map(|(e, name)| {
                format!(
                    "{name:?} {} {} {} {} {} {}",
                    e.SemanticIndex,
                    e.Format.0,
                    e.InputSlot,
                    e.AlignedByteOffset,
                    e.InputSlotClass.0,
                    e.InstanceDataStepRate
                )
            })
            .collect();
        format!(
            "{:?} {} {:?} {:?} {inputs:?} {} {} {:?} {} {:?}",
            self.blend,
            self.sample_mask,
            self.rasterizer,
            self.depth,
            self.strip_cut.0,
            self.topology.0,
            self.targets,
            self.depth_format.0,
            self.samples
        )
    }

    /// The vertex buffer slots the input layout reads.
    fn slots(&self) -> u32 {
        self.inputs
            .iter()
            .fold(0, |m, e| m | 1 << e.InputSlot.min(31))
    }

    /// The input elements with their byte offsets worked out.
    fn elements(&self) -> Vec<Input> {
        let mut next = [0u32; 32];
        self.inputs
            .iter()
            .map(|e| {
                let slot = e.InputSlot.min(31) as usize;
                let format = e.Format.0 as u32;
                let offset = if e.AlignedByteOffset == D3D12_APPEND_ALIGNED_ELEMENT {
                    next[slot]
                } else {
                    e.AlignedByteOffset
                };
                let bytes = crate::resource::element(format).map_or(0, |(b, _)| b as u32);
                next[slot] = offset + bytes;
                Input {
                    slot: slot as u32,
                    offset,
                    format,
                    per_instance: e.InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                }
            })
            .collect()
    }
}

/// One input element: vertex buffer slot, byte offset, format.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Input {
    pub(crate) slot: u32,
    pub(crate) offset: u32,
    pub(crate) format: u32,
    pub(crate) per_instance: bool,
}

/// The console's own primitive topology types.
const RECTANGLES: i32 = 19;
const QUADS: i32 = 32;

/// One stage of a host pipeline and where its bindings go.
#[derive(Clone)]
pub(crate) struct Stage {
    pub(crate) shader: Arc<Shader>,
    /// The root parameter of its user data, and its size in dwords.
    pub(crate) user_data: Option<(u32, u32)>,
    /// Root parameters of its root views: (title root parameter, ours).
    pub(crate) root_cbvs: Vec<(u8, u32)>,
    pub(crate) root_srvs: Vec<(u8, u32)>,
    /// Root parameter of the runtime stride table (`t0`, [`STRIDE_SPACE`]).
    pub(crate) stride_table: Option<u32>,
}

impl Stage {
    pub(crate) fn translated(&self) -> &Translated {
        self.shader
            .translated()
            .expect("host pipelines are made from translated shaders")
    }
}

pub(crate) struct HostPipeline {
    /// A compute pipeline (else graphics).
    pub(crate) compute: bool,
    /// A mesh pipeline (graphics without vertex input).
    pub(crate) mesh: bool,
    pub(crate) pso: ID3D12PipelineState,
    pub(crate) root: ID3D12RootSignature,
    /// Vertex, then pixel stage, or the compute stage.
    pub(crate) stages: Vec<Stage>,
    /// The vertex buffer slots it reads (bit per slot).
    pub(crate) slots: u32,
    pub(crate) inputs: Vec<Input>,
    /// Its shaders, for reports of slow submissions.
    pub(crate) name: String,
    /// Its name in the pipeline library (what its description holds), which
    /// its rectangle pipeline's name is made from ([`HostPipeline::rects`]).
    library: String,
}

/// Logs a pipeline that gets no host pipeline, once per reason.
fn refuse(reason: &'static str, shaders: &[Arc<Shader>]) -> Option<HostPipeline> {
    static SEEN: std::sync::Mutex<Vec<&str>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&reason) {
        seen.push(reason);
        tracing::warn!(?shaders, "no host pipeline: {reason} (logged once)");
    }
    None
}

impl HostPipeline {
    /// A host pipeline for vertex and pixel shaders with the graphics
    /// state `g`, or for one compute shader.
    pub(crate) fn new(shaders: &[Arc<Shader>], g: Option<&GraphicsState>) -> Option<HostPipeline> {
        let find = |stage| shaders.iter().find(|s| s.stage == stage);
        let compute = match shaders {
            [cs] if cs.stage == "compute" => Some(cs),
            _ => None,
        };
        if compute.is_none()
            && shaders
                .iter()
                .any(|s| !matches!(s.stage, "vertex" | "pixel" | "mesh"))
        {
            return refuse(
                "only vertex, mesh, pixel and compute stages are translated",
                shaders,
            );
        }
        let mesh = find("mesh");
        let stages = if let Some(cs) = compute {
            vec![(cs, D3D12_SHADER_VISIBILITY_ALL)]
        } else {
            let first = match (mesh, find("vertex")) {
                (Some(ms), _) => (ms, D3D12_SHADER_VISIBILITY_MESH),
                (None, Some(vs)) => (vs, D3D12_SHADER_VISIBILITY_VERTEX),
                (None, None) => return refuse("no vertex or mesh shader", shaders),
            };
            let mut stages = vec![first];
            stages.extend(find("pixel").map(|ps| (ps, D3D12_SHADER_VISIBILITY_PIXEL)));
            stages
        };
        if mesh.is_some() && !host()?.mesh_shaders() {
            return refuse(
                "the host GPU has no mesh shaders (or no shader model 6.6)",
                shaders,
            );
        }
        for (s, _) in &stages {
            s.translated()?;
        }
        let host = host()?;

        // The root signature the translated shaders' bindings define.
        let mut params: Vec<D3D12_ROOT_PARAMETER1> = Vec::new();
        // The ranges of root shader views' tables, by parameter: pointed
        // at once all are made (the vector no longer moves).
        let mut ranges: Vec<(usize, D3D12_DESCRIPTOR_RANGE1)> = Vec::new();
        let mut samplers = Vec::new();
        let mut bound = Vec::new();
        for &(shader, visibility) in &stages {
            let t = &shader.translated()?.host;
            let mut stage = Stage {
                shader: shader.clone(),
                user_data: None,
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                stride_table: None,
            };
            let dwords = t.user_data_dwords();
            if dwords > 0 {
                stage.user_data = Some((params.len() as u32, dwords));
                params.push(D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: USER_DATA_SPACE,
                            Num32BitValues: dwords,
                        },
                    },
                    ShaderVisibility: visibility,
                });
            }
            for (slots, kind, out) in [
                (
                    &t.root_cbvs,
                    D3D12_ROOT_PARAMETER_TYPE_CBV,
                    &mut stage.root_cbvs,
                ),
                (
                    &t.root_srvs,
                    D3D12_ROOT_PARAMETER_TYPE_SRV,
                    &mut stage.root_srvs,
                ),
            ] {
                for &slot in slots {
                    out.push((slot, params.len() as u32));
                    if kind == D3D12_ROOT_PARAMETER_TYPE_SRV && crate::draw::BOUNDED_ROOT_SRVS {
                        ranges.push((params.len(), root_srv_range(slot)));
                        params.push(root_srv_table(std::ptr::null(), visibility));
                        continue;
                    }
                    params.push(D3D12_ROOT_PARAMETER1 {
                        ParameterType: kind,
                        Anonymous: D3D12_ROOT_PARAMETER1_0 {
                            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                                ShaderRegister: u32::from(slot),
                                RegisterSpace: ROOT_CBV_SPACE,
                                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                            },
                        },
                        ShaderVisibility: visibility,
                    });
                }
            }
            if t.stride_table {
                stage.stride_table = Some(params.len() as u32);
                params.push(D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: 0,
                            RegisterSpace: STRIDE_SPACE,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                        },
                    },
                    ShaderVisibility: visibility,
                });
            }
            for (n, s) in t.static_samplers.iter().enumerate() {
                samplers.push(D3D12_STATIC_SAMPLER_DESC {
                    Filter: D3D12_FILTER(s.filter as i32),
                    AddressU: D3D12_TEXTURE_ADDRESS_MODE(s.address[0] as i32),
                    AddressV: D3D12_TEXTURE_ADDRESS_MODE(s.address[1] as i32),
                    AddressW: D3D12_TEXTURE_ADDRESS_MODE(s.address[2] as i32),
                    MipLODBias: s.mip_lod_bias,
                    MaxAnisotropy: s.max_anisotropy,
                    ComparisonFunc: D3D12_COMPARISON_FUNC(s.comparison as i32),
                    BorderColor: D3D12_STATIC_BORDER_COLOR(s.border as i32),
                    MinLOD: s.min_lod,
                    MaxLOD: s.max_lod,
                    ShaderRegister: n as u32,
                    RegisterSpace: SAMPLER_SPACE,
                    ShaderVisibility: visibility,
                });
            }
            bound.push(stage);
        }
        for (param, range) in &ranges {
            params[*param] = root_srv_table(range, params[*param].ShaderVisibility);
        }
        let (root, root_blob) =
            match root_signature(&host.device, &params, &samplers, compute.is_some()) {
                Ok(root) => root,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host root signature refused");
                    return None;
                }
            };
        let code = |i: usize| {
            stages.get(i).and_then(|(s, _)| s.translated()).map_or(
                D3D12_SHADER_BYTECODE::default(),
                |t| D3D12_SHADER_BYTECODE {
                    pShaderBytecode: t.host.dxil.as_ptr().cast(),
                    BytecodeLength: t.host.dxil.len(),
                },
            )
        };
        if compute.is_some() {
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                // SAFETY: a borrowed pointer the description never releases.
                pRootSignature: unsafe { std::mem::transmute_copy(&root) },
                CS: code(0),
                ..Default::default()
            };
            let name = super::library::name("compute", &root_blob, &[&code_bytes(&code(0))], "");
            let made = super::library::pipeline(
                &name,
                // SAFETY: the description points at live data.
                |library, name| unsafe { library.LoadComputePipeline(name, &desc) },
                // SAFETY: as above.
                || unsafe { host.device.CreateComputePipelineState(&desc) },
            );
            let pso = match made {
                Ok(pso) => pso,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: true,
                mesh: false,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                slots: 0,
                inputs: Vec::new(),
                library: name,
            });
        }
        let Some(g) = g else {
            return refuse("graphics state not kept (plain description)", shaders);
        };
        let state = g.describe();
        if mesh.is_some() {
            let name = super::library::name(
                "mesh",
                &root_blob,
                &[&code_bytes(&code(0)), &code_bytes(&code(1))],
                &state,
            );
            let pso = match mesh_pipeline(&host.device, &root, code(0), code(1), g, &name) {
                Ok(pso) => pso,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host mesh pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(
                ?shaders,
                cull = g.rasterizer.CullMode.0,
                front_ccw = g.rasterizer.FrontCounterClockwise.as_bool(),
                output = %OutputState::of(&graphics_desc(&root, [D3D12_SHADER_BYTECODE::default(); 3], g)),
                "host mesh pipeline"
            );
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: false,
                mesh: true,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                slots: 0,
                inputs: Vec::new(),
                library: name,
            });
        }

        let none = D3D12_SHADER_BYTECODE::default();
        let desc = graphics_desc(&root, [code(0), none, code(1)], g);
        let name = super::library::name(
            "graphics",
            &root_blob,
            &[&code_bytes(&code(0)), &code_bytes(&code(1))],
            &state,
        );
        let made = super::library::pipeline(
            &name,
            // SAFETY: the description points at live data.
            |library, name| unsafe { library.LoadGraphicsPipeline(name, &desc) },
            // SAFETY: as above.
            || unsafe { host.device.CreateGraphicsPipelineState(&desc) },
        );
        let pso = match made {
            Ok(pso) => pso,
            Err(e) => {
                tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                host.log_messages();
                return None;
            }
        };
        tracing::info!(
            ?shaders,
            cull = g.rasterizer.CullMode.0,
            front_ccw = g.rasterizer.FrontCounterClockwise.as_bool(),
            output = %OutputState::of(&desc),
            "host pipeline"
        );
        crate::host::name(&pso, || format!("{shaders:?}"));
        tracing::debug!(
            ?shaders,
            depth = ?g.depth,
            dsv = g.depth_format.0,
            targets = ?&g.targets.RTFormats[..g.targets.NumRenderTargets.min(8) as usize],
            blend = ?g.blend.RenderTarget[0],
            cull = g.rasterizer.CullMode.0,
            "host graphics state"
        );
        Some(HostPipeline {
            compute: false,
            mesh: false,
            pso,
            root,
            stages: bound,
            name: format!("{shaders:?}"),
            slots: g.slots(),
            inputs: g.elements(),
            library: name,
        })
    }

    /// The pipeline for this one's rectangle-list draws ([`crate::rects`]):
    /// the same root signature, shaders and state `g` (the title's, culling
    /// no faces unless [`crate::rects::CULLED`]) plus the geometry shader
    /// that completes each rectangle, drawing triangles. Why not, as a drop reason, if it cannot be made. Made by a
    /// pipeline worker on the first rectangle draw (`make::ready_rects`), so
    /// the non-rectangle pipeline is untouched; kept in the library under a
    /// name of its own, made from this one's and the geometry shader's
    /// bytes.
    pub(crate) fn rects(&self, g: Option<&GraphicsState>) -> Result<HostPipeline, &'static str> {
        if self.compute || self.mesh {
            return Err("rectangles: not a vertex pipeline");
        }
        let g = g.ok_or("rectangles: graphics state not kept")?;
        // Triangles in, for the geometry shader.
        if g.topology != D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE && g.topology.0 != RECTANGLES {
            return Err("rectangles: not a triangle or rectangle pipeline");
        }
        let code = |b: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: b.as_ptr().cast(),
            BytecodeLength: b.len(),
        };
        let vs = &self.stages[0].translated().host.dxil;
        let ps = self
            .stages
            .get(1)
            .map(|s| s.translated().host.dxil.as_slice());
        let gs = crate::rects::geometry_shader(vs, ps)?;
        let host = host().ok_or("rectangles: no host device")?;
        let ps = ps.map_or_else(D3D12_SHADER_BYTECODE::default, code);
        let mut desc = graphics_desc(&self.root, [code(vs), code(&gs), ps], g);
        // Both faces unless asked otherwise (`rects::CULLED`); the library
        // name says which, as the description differs.
        let culled = crate::rects::CULLED;
        if !culled {
            desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        }
        let mut faces = if culled { "" } else { "cull none" }.to_owned();
        // An experiment (diagnostics builds, VICTORY_RECT_DEPTH_ALWAYS=1):
        // rectangle pipelines that test depth for equality pass everywhere
        // instead. Wrong pictures on purpose: it says whether such a pass
        // (the tiled lighting) writes nothing because of its depth test or
        // because of what its shaders read.
        #[cfg(feature = "diagnostics")]
        if std::env::var_os("VICTORY_RECT_DEPTH_ALWAYS").is_some_and(|v| v == "1")
            && desc.DepthStencilState.DepthFunc == D3D12_COMPARISON_FUNC_EQUAL
        {
            desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
            faces.push_str(" depth always");
        }
        let faces = faces.as_str();
        let library = super::library::name("rectangles", self.library.as_bytes(), &[&gs], faces);
        let made = super::library::pipeline(
            &library,
            // SAFETY: the description points at live data.
            |library, name| unsafe { library.LoadGraphicsPipeline(name, &desc) },
            // SAFETY: as above.
            || unsafe { host.device.CreateGraphicsPipelineState(&desc) },
        );
        let pso = match made {
            Ok(pso) => pso,
            Err(e) => {
                tracing::warn!(error = %e, pipeline = self.name, "host rectangle pipeline refused");
                host.log_messages();
                return Err("rectangles: host pipeline refused");
            }
        };
        let name = format!("{} with rectangles", self.name);
        // The title's face culling (1 none, 2 front, 3 back), whatever the
        // host does with it: what settles whether the console culls
        // rectangle lists, if a pass stays black. Then what the host
        // pipeline gets ([`OutputState`]): whether its pixels can land.
        tracing::info!(
            pipeline = self.name,
            cull = g.rasterizer.CullMode.0,
            front_ccw = g.rasterizer.FrontCounterClockwise.as_bool(),
            culled,
            host_cull = desc.RasterizerState.CullMode.0,
            output = %OutputState::of(&desc),
            "host rectangle pipeline"
        );
        crate::host::name(&pso, || name.clone());
        Ok(HostPipeline {
            compute: false,
            mesh: false,
            pso,
            root: self.root.clone(),
            stages: self.stages.clone(),
            slots: self.slots,
            inputs: self.inputs.clone(),
            name,
            library,
        })
    }
}

/// What a host graphics pipeline's description says about where its pixels
/// go, for the pipeline lines: depth and stencil test, sample mask, depth
/// format, render target formats, render target 0's blend, samples and
/// depth clip. Written as
/// `depth=on write=ALL func=LESS stencil=off read=0xff write=0xff
/// front=ALWAYS/KEEP/KEEP/KEEP back=… sample_mask=0xffffffff dsv=f40
/// rtvs=[f10, f34] blend0=[on ONE+ONE ADD alpha ONE+ZERO ADD mask=0xf]
/// samples=1 depth_clip=true` (stencil faces: func/fail/depth fail/pass;
/// formats by DXGI number, as the dump file names give them).
pub(crate) struct OutputState<'a>(&'a D3D12_GRAPHICS_PIPELINE_STATE_DESC);

impl<'a> OutputState<'a> {
    pub(crate) fn of(desc: &'a D3D12_GRAPHICS_PIPELINE_STATE_DESC) -> Self {
        OutputState(desc)
    }
}

/// A D3D12 enumeration value by name, or its number if not in `names`
/// (indexed by value).
fn named(names: &[&'static str], value: i32) -> String {
    match usize::try_from(value).ok().and_then(|i| names.get(i)) {
        Some(name) if !name.is_empty() => (*name).to_owned(),
        _ => format!("{value}"),
    }
}

const COMPARISONS: [&str; 9] = [
    "NONE",
    "NEVER",
    "LESS",
    "EQUAL",
    "LESS_EQUAL",
    "GREATER",
    "NOT_EQUAL",
    "GREATER_EQUAL",
    "ALWAYS",
];
const STENCIL_OPS: [&str; 9] = [
    "", "KEEP", "ZERO", "REPLACE", "INCR_SAT", "DECR_SAT", "INVERT", "INCR", "DECR",
];
const BLENDS: [&str; 22] = [
    "",
    "ZERO",
    "ONE",
    "SRC_COLOR",
    "INV_SRC_COLOR",
    "SRC_ALPHA",
    "INV_SRC_ALPHA",
    "DEST_ALPHA",
    "INV_DEST_ALPHA",
    "DEST_COLOR",
    "INV_DEST_COLOR",
    "SRC_ALPHA_SAT",
    "",
    "",
    "BLEND_FACTOR",
    "INV_BLEND_FACTOR",
    "SRC1_COLOR",
    "INV_SRC1_COLOR",
    "SRC1_ALPHA",
    "INV_SRC1_ALPHA",
    "ALPHA_FACTOR",
    "INV_ALPHA_FACTOR",
];
const BLEND_OPS: [&str; 6] = ["", "ADD", "SUBTRACT", "REV_SUBTRACT", "MIN", "MAX"];

impl std::fmt::Display for OutputState<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let d = self.0;
        let ds = &d.DepthStencilState;
        let on = |b: windows::core::BOOL| if b.as_bool() { "on" } else { "off" };
        let face = |s: &D3D12_DEPTH_STENCILOP_DESC| {
            format!(
                "{}/{}/{}/{}",
                named(&COMPARISONS, s.StencilFunc.0),
                named(&STENCIL_OPS, s.StencilFailOp.0),
                named(&STENCIL_OPS, s.StencilDepthFailOp.0),
                named(&STENCIL_OPS, s.StencilPassOp.0)
            )
        };
        write!(
            f,
            "depth={} write={} func={} stencil={} read={:#04x} write={:#04x} front={} back={}",
            on(ds.DepthEnable),
            named(&["ZERO", "ALL"], ds.DepthWriteMask.0),
            named(&COMPARISONS, ds.DepthFunc.0),
            on(ds.StencilEnable),
            ds.StencilReadMask,
            ds.StencilWriteMask,
            face(&ds.FrontFace),
            face(&ds.BackFace)
        )?;
        let n = d.NumRenderTargets.min(8) as usize;
        let rtvs: Vec<String> = d.RTVFormats[..n].iter().map(|r| format!("f{}", r.0)).collect();
        let b = &d.BlendState.RenderTarget[0];
        write!(
            f,
            " sample_mask={:#x} dsv=f{} rtvs=[{}] blend0=[{} {}+{} {} alpha {}+{} {} logic={} mask={:#x}] \
             alpha_to_coverage={} independent_blend={} samples={} depth_clip={}",
            d.SampleMask,
            d.DSVFormat.0,
            rtvs.join(", "),
            on(b.BlendEnable),
            named(&BLENDS, b.SrcBlend.0),
            named(&BLENDS, b.DestBlend.0),
            named(&BLEND_OPS, b.BlendOp.0),
            named(&BLENDS, b.SrcBlendAlpha.0),
            named(&BLENDS, b.DestBlendAlpha.0),
            named(&BLEND_OPS, b.BlendOpAlpha.0),
            on(b.LogicOpEnable),
            b.RenderTargetWriteMask,
            d.BlendState.AlphaToCoverageEnable.as_bool(),
            d.BlendState.IndependentBlendEnable.as_bool(),
            d.SampleDesc.Count,
            d.RasterizerState.DepthClipEnable.as_bool()
        )
    }
}

/// The host description of a graphics pipeline: `root` (borrowed), the
/// vertex, geometry and pixel stages, and the title's state `g`, which it
/// points into.
fn graphics_desc(
    root: &ID3D12RootSignature,
    [vs, gs, ps]: [D3D12_SHADER_BYTECODE; 3],
    g: &GraphicsState,
) -> D3D12_GRAPHICS_PIPELINE_STATE_DESC {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        // SAFETY: a borrowed pointer the description never releases.
        pRootSignature: unsafe { std::mem::transmute_copy(root) },
        VS: vs,
        GS: gs,
        PS: ps,
        BlendState: g.blend,
        SampleMask: g.sample_mask,
        RasterizerState: g.rasterizer,
        DepthStencilState: g.depth,
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: g.inputs.as_ptr(),
            NumElements: g.inputs.len() as u32,
        },
        IBStripCutValue: g.strip_cut,
        // The console's rectangle and quad types draw as triangles on the
        // host (see `draw`).
        PrimitiveTopologyType: match g.topology.0 {
            RECTANGLES | QUADS => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            _ => g.topology,
        },
        NumRenderTargets: g.targets.NumRenderTargets,
        RTVFormats: g.targets.RTFormats,
        DSVFormat: g.depth_format,
        SampleDesc: g.samples,
        ..Default::default()
    }
}

/// A mesh pipeline from the title's graphics state: the pipeline-state
/// stream form (no input layout or topology).
fn mesh_pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    ms: D3D12_SHADER_BYTECODE,
    ps: D3D12_SHADER_BYTECODE,
    g: &GraphicsState,
    name: &str,
) -> windows::core::Result<ID3D12PipelineState> {
    use windows::core::Interface;
    /// A stream subobject: its type, then its description, pointer-aligned.
    #[repr(C, align(8))]
    struct Sub<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);
    #[repr(C)]
    struct Stream {
        root: Sub<*mut std::ffi::c_void>,
        ms: Sub<D3D12_SHADER_BYTECODE>,
        ps: Sub<D3D12_SHADER_BYTECODE>,
        blend: Sub<D3D12_BLEND_DESC>,
        mask: Sub<u32>,
        raster: Sub<D3D12_RASTERIZER_DESC>,
        depth: Sub<D3D12_DEPTH_STENCIL_DESC>,
        targets: Sub<D3D12_RT_FORMAT_ARRAY>,
        dsv: Sub<DXGI_FORMAT>,
        samples: Sub<DXGI_SAMPLE_DESC>,
    }
    let mut stream = Stream {
        root: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
            root.as_raw(),
        ),
        ms: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms),
        ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps),
        blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, g.blend),
        mask: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK,
            g.sample_mask,
        ),
        raster: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, g.rasterizer),
        depth: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, g.depth),
        targets: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
            g.targets,
        ),
        dsv: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,
            g.depth_format,
        ),
        samples: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, g.samples),
    };
    let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: size_of::<Stream>(),
        pPipelineStateSubobjectStream: (&raw mut stream).cast(),
    };
    let device = device.cast::<ID3D12Device2>()?;
    super::library::pipeline(
        name,
        // SAFETY: the stream holds whole subobjects pointing at live data.
        |library, name| unsafe { library.LoadPipeline(name, &desc) },
        // SAFETY: as above.
        || unsafe { device.CreatePipelineState(&desc) },
    )
}

/// The range of a root shader view's table ([`root_srv_table`]): one raw
/// view at the register the translated shader binds (`t<slot>` in
/// [`ROOT_CBV_SPACE`]). Volatile descriptors and data: with static ones
/// (root signature 1.1's default for tables) the driver may make the view
/// a root descriptor, which nothing bounds.
pub(crate) fn root_srv_range(slot: u8) -> D3D12_DESCRIPTOR_RANGE1 {
    D3D12_DESCRIPTOR_RANGE1 {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: u32::from(slot),
        RegisterSpace: ROOT_CBV_SPACE,
        Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE
            | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
        OffsetInDescriptorsFromTableStart: 0,
    }
}

/// The root parameter of a root shader view, bounded
/// ([`crate::draw::BOUNDED_ROOT_SRVS`]): a table of `range` (live until the
/// root signature is serialized), set per draw to a raw view of the
/// buffer's host copy (`draw::State::root_srv_table`).
pub(crate) fn root_srv_table(
    range: *const D3D12_DESCRIPTOR_RANGE1,
    visibility: D3D12_SHADER_VISIBILITY,
) -> D3D12_ROOT_PARAMETER1 {
    D3D12_ROOT_PARAMETER1 {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                NumDescriptorRanges: 1,
                pDescriptorRanges: range,
            },
        },
        ShaderVisibility: visibility,
    }
}

/// The bytes of shader code (none for an absent stage).
fn code_bytes(code: &D3D12_SHADER_BYTECODE) -> Vec<u8> {
    if code.pShaderBytecode.is_null() {
        return Vec::new();
    }
    // SAFETY: a translation's own bytes, alive while its shader is.
    unsafe { std::slice::from_raw_parts(code.pShaderBytecode.cast::<u8>(), code.BytecodeLength) }
        .to_vec()
}

fn root_signature(
    device: &ID3D12Device,
    params: &[D3D12_ROOT_PARAMETER1],
    samplers: &[D3D12_STATIC_SAMPLER_DESC],
    compute: bool,
) -> windows::core::Result<(ID3D12RootSignature, Vec<u8>)> {
    let input = if compute {
        D3D12_ROOT_SIGNATURE_FLAG_NONE
    } else {
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
    };
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: windows::Win32::Graphics::Direct3D12::D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: samplers.len() as u32,
                pStaticSamplers: samplers.as_ptr(),
                Flags: input
                    | D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                    | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED,
            },
        },
    };
    // SAFETY: the description points at live arrays; the blob is read in
    // full before it is released.
    unsafe {
        let mut blob: Option<ID3DBlob> = None;
        let mut error: Option<ID3DBlob> = None;
        if let Err(e) = D3D12SerializeVersionedRootSignature(&desc, &mut blob, Some(&mut error)) {
            if let Some(error) = error {
                let text = std::slice::from_raw_parts(
                    error.GetBufferPointer().cast::<u8>(),
                    error.GetBufferSize(),
                );
                tracing::warn!("root signature: {}", String::from_utf8_lossy(text));
            }
            return Err(e);
        }
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        Ok((device.CreateRootSignature(0, bytes)?, bytes.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_shader::emit::HostShader;

    /// The pipeline lines' output state: a stencil-tested, equal-depth,
    /// additive pass as a host description holds it.
    #[test]
    fn output_state_reads_as_words() {
        let mut g = GraphicsState::default();
        g.depth.DepthFunc = D3D12_COMPARISON_FUNC_EQUAL;
        g.depth.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        g.depth.StencilEnable = true.into();
        g.depth.StencilWriteMask = 0;
        g.depth.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_NOT_EQUAL;
        g.depth.FrontFace.StencilPassOp = D3D12_STENCIL_OP_REPLACE;
        g.blend.RenderTarget[0].BlendEnable = true.into();
        g.blend.RenderTarget[0].DestBlend = D3D12_BLEND_ONE;
        g.targets.NumRenderTargets = 2;
        g.targets.RTFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
        g.targets.RTFormats[1] = DXGI_FORMAT_R16G16_FLOAT;
        g.depth_format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
        // As `graphics_desc` fills it, without a root signature.
        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            BlendState: g.blend,
            SampleMask: g.sample_mask,
            RasterizerState: g.rasterizer,
            DepthStencilState: g.depth,
            NumRenderTargets: g.targets.NumRenderTargets,
            RTVFormats: g.targets.RTFormats,
            DSVFormat: g.depth_format,
            SampleDesc: g.samples,
            ..Default::default()
        };
        let text = OutputState::of(&desc).to_string();
        assert_eq!(
            text,
            "depth=on write=ZERO func=EQUAL stencil=on read=0xff write=0x00 \
             front=NOT_EQUAL/KEEP/KEEP/REPLACE back=ALWAYS/KEEP/KEEP/KEEP \
             sample_mask=0xffffffff dsv=f20 rtvs=[f10, f34] \
             blend0=[on ONE+ONE ADD alpha ONE+ZERO ADD logic=off mask=0xf] \
             alpha_to_coverage=false independent_blend=false samples=1 depth_clip=true"
        );
    }

    /// Rectangle pipelines as [`HostPipeline::rects`] describes them, for
    /// hand-written output signatures: a vertex shader as translations emit
    /// it (the shader crate's stub, its outputs the pixel shader's inputs,
    /// registers and all), the geometry shader made for it and a DXIL pixel
    /// shader. The position alone (as the title's tiled-lighting programs
    /// have it); with float interpolants sharing rows and a flat integer;
    /// with interpolation modifiers, a flat float and clip distances. The
    /// host takes each.
    #[test]
    fn rectangle_pipelines_take_the_geometry_shader() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        const INPUTS: [&str; 3] = [
            "float4 pos : SV_Position",
            "float4 pos : SV_Position, float2 a : TEXCOORD0, float2 b : TEXCOORD1, \
             nointerpolation uint c : TEXCOORD2",
            "float4 pos : SV_Position, noperspective float2 a : A, nointerpolation float b : B, \
             centroid float4 c : C, float2 clip : SV_ClipDistance0",
        ];
        let code = |b: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: b.as_ptr().cast(),
            BytecodeLength: b.len(),
        };
        for inputs in INPUTS {
            let source = format!("float4 main({inputs}) : SV_Target {{ return pos; }}");
            let Ok(ps) = crate::rects::compile(&source, "ps_6_0") else {
                eprintln!("no dxcompiler.dll: not run");
                return;
            };
            let ps = HostShader {
                stage: victory_shader::HwStage::Ps,
                dxil: ps,
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                stride_table: false,
                per_lane_tables: Vec::new(),
            };
            let vs = victory_shader::emit::host::stub_vs(&ps).unwrap();
            let outputs = crate::rects::elements(&vs.dxil, victory_shader::FourCC::OSG1).unwrap();
            assert_eq!(outputs.len(), inputs.split(',').count(), "{inputs}");
            let gs = crate::rects::geometry_shader(&vs.dxil, Some(&ps.dxil))
                .unwrap_or_else(|e| panic!("{inputs}: {e}"));
            let (root, _) = root_signature(&host.device, &[], &[], false).unwrap();
            let mut g = GraphicsState {
                topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE(RECTANGLES),
                ..Default::default()
            };
            g.targets.NumRenderTargets = 1;
            g.targets.RTFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
            g.depth.DepthEnable = false.into();
            let desc = graphics_desc(&root, [code(&vs.dxil), code(&gs), code(&ps.dxil)], &g);
            assert_eq!(desc.PrimitiveTopologyType, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
            // SAFETY: the description points at live data.
            let made = unsafe { host.device.CreateGraphicsPipelineState::<ID3D12PipelineState>(&desc) };
            assert!(made.is_ok(), "{inputs}: {made:?} {:?}", host.log_messages());
        }
    }
}
