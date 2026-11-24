//! Hand-assembled GFX10 compute programs (and a few pixel programs, drawn
//! after a vertex shader of constants) through the whole translator:
//! validated by `dxil.dll` when there is one, and run on the host devices
//! (hardware and WARP) against results worked out from the instruction
//! set, for what the differential tests cannot reach without the console
//! compiler: spilled SGPRs, descriptor loads read as numbers, derivatives.

use super::container::SigElement;
use super::host::{Dispatch, Draw, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::dxbc::{Container, FourCC};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, PsInput, ResourceType, RootParam, RootType, StageIo, Usage,
    UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

/// The validator and the host devices do not take several tests at once
/// (the process dies in them): each test holds this while it runs.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

const S_ENDPGM: u32 = 0xBF81_0000;
/// A literal follows the instruction.
const LIT: u16 = 255;
const VCC_LO: u16 = 106;
const VCC_HI: u16 = 107;

/// An inline integer constant 0-64.
const fn k(n: u16) -> u16 {
    128 + n
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u16) -> u32 {
    0xBE80_0300 | (sdst as u32) << 16 | src0 as u32
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u32, sdst: u8, src0: u16, src1: u16) -> u32 {
    0x8000_0000 | op << 23 | (sdst as u32) << 16 | (src1 as u32) << 8 | src0 as u32
}

/// SOPC `s_cmp_eq_u32 src0, src1`.
const fn s_cmp_eq_u32(src0: u16, src1: u16) -> u32 {
    0xBF06_0000 | (src1 as u32) << 8 | src0 as u32
}

/// SMEM `s_load_dwordx4` (op 2) or `x8` (op 3) of `s[base:base+1]` plus
/// `offset` into `s[data..]`.
const fn s_load(op: u32, data: u8, base: u8, offset: u32) -> [u32; 2] {
    [
        0xF400_0000 | op << 18 | (data as u32) << 6 | (base as u32) >> 1,
        125 << 25 | offset,
    ]
}

/// VOP1 `op vdst, src0`.
const fn vop1(op: u32, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | op << 9 | src0 as u32
}

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u32, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    op << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | src0 as u32
}

/// VOPC `op vcc, src0, vsrc1`.
const fn vopc(op: u32, src0: u16, vsrc1: u8) -> u32 {
    0x7C00_0000 | op << 17 | (vsrc1 as u32) << 9 | src0 as u32
}

/// VOP3 `op dst, src0, src1, src2`.
const fn vop3(op: u32, dst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0xD400_0000 | op << 16 | dst as u32,
        src0 as u32 | (src1 as u32) << 9 | (src2 as u32) << 18,
    ]
}

/// MUBUF `buffer_store_dword vdata, vaddr, s[rsrc..], 0 offen offset:offset`.
const fn buffer_store_dword(vdata: u8, vaddr: u8, rsrc: u8, offset: u32) -> [u32; 2] {
    [
        0xE070_1000 | offset,
        0x80 << 24 | ((rsrc as u32) >> 2) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// MIMG `op vdata, v[vaddr..], s[rsrc..], s[sampler..]` of a 2D image, one
/// channel.
const fn image_2d(op: u32, vdata: u8, vaddr: u8, rsrc: u8, sampler: u8) -> [u32; 2] {
    [
        0xF000_0000 | op << 18 | 1 << 8 | 1 << 3,
        ((sampler as u32) >> 2) << 21
            | ((rsrc as u32) >> 2) << 16
            | (vdata as u32) << 8
            | vaddr as u32,
    ]
}

const V_MOV_B32: u32 = 0x01;
const V_CVT_F32_U32: u32 = 0x06;
const V_CNDMASK_B32: u32 = 0x01;
const V_LSHRREV_B32: u32 = 0x16;
const V_AND_B32: u32 = 0x1B;
const V_CMP_EQ_U32: u32 = 0xC2;
const V_LSHLREV_B32: u32 = 0x1A;
const V_ADD_NC_U32: u32 = 0x25;
const V_CMP_LT_U32: u32 = 0xC1;
const V_CMP_GT_U32: u32 = 0xC4;
const V3_CNDMASK_B32: u32 = 0x101;
const V_READLANE_B32: u32 = 0x360;
const V_WRITELANE_B32: u32 = 0x361;
const S_AND_B32: u32 = 0x0E;
const S_OR_B32: u32 = 0x10;
const S_LSHL_B32: u32 = 0x1E;
const S_LSHR_B32: u32 = 0x20;
const S_BFE_U32: u32 = 0x27;
const IMAGE_LOAD: u32 = 0x00;
const IMAGE_ATOMIC_UMAX: u32 = 0x17;
const IMAGE_SAMPLE: u32 = 0x20;

fn table(user_data: u8, slot: u8) -> RootParam {
    RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot,
    }
}

fn used(kind: ResourceType, table_offset_dwords: u32, dwords: u8, user_data: u8) -> UsedResource {
    UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 4,
        dwords,
        user_data,
        slot: user_data,
    }
}

/// A compute program of `threads` threads whose user SGPRs follow `layout`
/// (thread ids in v0 and v1).
fn program(
    words: &[u32],
    wave64: bool,
    threads: [u32; 3],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
) -> NativeProgram {
    let user_sgprs: u32 = layout
        .iter()
        .map(|p| u32::from(p.user_data) + p.sgprs)
        .max()
        .unwrap_or(0);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 8,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: user_sgprs << 1 | 1 << 11,
            threads,
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn translate(native: &NativeProgram) -> Result<HostShader, ShaderError> {
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    // As `emit::translate` orders them.
    let st = structure::structure(&p);
    let res = resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
    let io = io::raise(&p, native)?;
    let st = st?;
    let l = lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })?;
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        stride_table: l.stride_table,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation");
        assert_eq!(copy[4..20], ours[..], "container hash");
    } else {
        eprintln!("no dxil.dll; not validated");
    }
    Ok(shader)
}

/// Whether the shader declares wave operations (`SFI0`).
fn wave_ops(shader: &HostShader) -> bool {
    let c = Container::parse(&shader.dxil).unwrap();
    let features = c.part(FourCC::SFI0).unwrap().data;
    u64::from_le_bytes(features.try_into().unwrap()) & 0x4000 != 0
}

/// Runs one group of `shader` on every host device there is, with table
/// parameters at heap index 0 and `constant` in every root constant, and
/// hands the first UAV's dwords to `check`.
fn run(shader: &HostShader, views: &[View], constant: u32, check: impl Fn(&str, &[u32])) {
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let mut data = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            data[u.index as usize] = match u.root {
                RootType::Constants => constant,
                _ => 0,
            };
        }
        let sampler = windows::Win32::Graphics::Direct3D12::D3D12_SAMPLER_DESC {
            Filter: windows::Win32::Graphics::Direct3D12::D3D12_FILTER_MIN_MAG_MIP_POINT,
            AddressU: windows::Win32::Graphics::Direct3D12::D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: windows::Win32::Graphics::Direct3D12::D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: windows::Win32::Graphics::Direct3D12::D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            MaxAnisotropy: 1,
            MaxLOD: f32::MAX,
            ..Default::default()
        };
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(data))],
                    views,
                    samplers: &[sampler],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        check(label, &out[0]);
        ran += 1;
    }
    if ran == 0 {
        eprintln!("no SM 6.6 device; not run");
    }
}

/// A compare's lane mask goes to a lane of v7, VCC is overwritten, and a
/// select reads the mask back: the select sees the first compare, with no
/// lane read across the wave.
#[test]
fn a_spilled_lane_mask_selects_after_its_reload() {
    let _serial = serial();
    for wave64 in [false, true] {
        let mut words = vec![s_mov_b32(1, k(0))];
        words.extend(s_load(2, 8, 0, 0));
        // vcc: thread > 5; kept in lanes 0 (and 1) of v7.
        words.push(vopc(V_CMP_LT_U32, k(5), 0));
        words.extend(vop3(V_WRITELANE_B32, 7, VCC_LO, k(0), 0));
        if wave64 {
            words.extend(vop3(V_WRITELANE_B32, 7, VCC_HI, k(1), 0));
        }
        // vcc: thread < 3.
        words.push(vopc(V_CMP_GT_U32, k(3), 0));
        words.push(vop1(V_MOV_B32, 1, k(11)));
        words.push(vop1(V_MOV_B32, 2, k(29)));
        words.push(vop2(V_CNDMASK_B32, 3, vgpr(1), 2));
        words.extend(vop3(V_READLANE_B32, 12, vgpr(7), k(0), 0));
        if wave64 {
            words.extend(vop3(V_READLANE_B32, 13, vgpr(7), k(1), 0));
        }
        words.extend(vop3(V3_CNDMASK_B32, 4, vgpr(1), vgpr(2), 12));
        // Both selects in one dword, stored at 4 × thread.
        words.push(vop2(V_LSHLREV_B32, 4, k(8), 4));
        words.push(vop2(V_ADD_NC_U32, 3, vgpr(3), 4));
        words.push(vop2(V_LSHLREV_B32, 5, k(2), 0));
        words.extend(buffer_store_dword(3, 5, 8, 0));
        words.push(S_ENDPGM);
        let native = program(
            &words,
            wave64,
            [8, 1, 1],
            vec![table(0, 0)],
            vec![used(ResourceType::Uav, 0, 4, 0)],
        );
        let shader = translate(&native).unwrap();
        assert!(!wave_ops(&shader), "the reload is no lane read");
        run(&shader, &[View::RawUav(8)], 0, |label, out| {
            for (t, &x) in out.iter().enumerate() {
                let low = if t < 3 { 29 } else { 11 };
                let high = if t > 5 { 29 } else { 11 };
                assert_eq!(x, low | high << 8, "{label}, wave64 {wave64}, thread {t}");
            }
        });
    }
}

/// Two paths keep different masks in the same lane of v7: the reload after
/// the join selects by the mask of the path taken.
#[test]
fn spilled_lane_masks_join() {
    let _serial = serial();
    let mut words = vec![s_mov_b32(1, k(0))];
    words.extend(s_load(2, 8, 0, 0));
    words.push(vopc(V_CMP_LT_U32, k(5), 0)); // thread > 5
    words.push(s_cmp_eq_u32(2, k(0)));
    words.push(0xBF85_0003); // s_cbranch_scc1 over the store and the jump
    words.extend(vop3(V_WRITELANE_B32, 7, VCC_LO, k(0), 0));
    words.push(0xBF82_0003); // s_branch to the join
    words.push(vopc(V_CMP_GT_U32, k(3), 0)); // thread < 3
    words.extend(vop3(V_WRITELANE_B32, 7, VCC_LO, k(0), 0));
    words.push(vopc(V_CMP_LT_U32, k(63), 0)); // vcc: no thread
    words.push(vop1(V_MOV_B32, 1, k(11)));
    words.push(vop1(V_MOV_B32, 2, k(29)));
    words.extend(vop3(V_READLANE_B32, 12, vgpr(7), k(0), 0));
    words.extend(vop3(V3_CNDMASK_B32, 3, vgpr(1), vgpr(2), 12));
    words.push(vop2(V_LSHLREV_B32, 5, k(2), 0));
    words.extend(buffer_store_dword(3, 5, 8, 0));
    words.push(S_ENDPGM);
    let constants = RootParam {
        root: RootType::Constants,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 2,
        slot: 1,
    };
    let native = program(
        &words,
        false,
        [8, 1, 1],
        vec![table(0, 0), constants],
        vec![used(ResourceType::Uav, 0, 4, 0)],
    );
    let shader = translate(&native).unwrap();
    assert!(!wave_ops(&shader));
    for constant in [0, 1] {
        run(&shader, &[View::RawUav(8)], constant, |label, out| {
            for (t, &x) in out.iter().enumerate() {
                let set = if constant == 0 { t < 3 } else { t > 5 };
                assert_eq!(
                    x,
                    if set { 29 } else { 11 },
                    "{label}, {constant}, thread {t}"
                );
            }
        });
    }
}

/// The program that reads a texture's size from its descriptor's dwords:
/// with `reload`, it loads the descriptor again for an `image_load`.
fn size_reader(reload: bool) -> NativeProgram {
    let mut words = vec![s_mov_b32(1, k(0))];
    words.extend(s_load(2, 8, 0, 0));
    words.extend(s_load(3, 12, 0, 0x20));
    // Width - 1: dword 1 bits 30-31 and dword 2 bits 0-13. Height - 1:
    // dword 2 bits 14-29.
    words.push(sop2(S_LSHR_B32, 2, 13, k(30)));
    words.push(sop2(S_LSHL_B32, 3, 14, k(2)));
    words.push(sop2(S_OR_B32, 2, 2, 3));
    words.extend([sop2(S_AND_B32, 2, 2, LIT), 0x3FFF]);
    words.extend([sop2(S_BFE_U32, 3, 14, LIT), 0x0010_000E]);
    words.push(vop1(V_MOV_B32, 3, 2));
    words.push(vop1(V_MOV_B32, 4, 3));
    words.push(vop1(V_MOV_B32, 5, k(0)));
    if reload {
        words.extend(s_load(3, 20, 0, 0x20));
        words.push(vop1(V_MOV_B32, 1, k(0)));
        words.extend(image_2d(IMAGE_LOAD, 5, 0, 20, 0));
    }
    words.push(vop2(V_LSHLREV_B32, 6, k(4), 0));
    words.extend(buffer_store_dword(3, 6, 8, 0));
    words.extend(buffer_store_dword(4, 6, 8, 4));
    words.extend(buffer_store_dword(5, 6, 8, 8));
    words.push(S_ENDPGM);
    program(
        &words,
        false,
        [4, 1, 1],
        vec![table(0, 0)],
        vec![
            used(ResourceType::Uav, 0, 4, 0),
            used(ResourceType::Srv, 8, 8, 0),
        ],
    )
}

/// A descriptor load only read as numbers is the view another load of the
/// same table entry is used as.
#[test]
fn a_descriptor_read_as_data_is_the_view_its_reload_uses() {
    let _serial = serial();
    let shader = translate(&size_reader(true)).unwrap();
    let texels: Vec<[f32; 4]> = (0..8).map(|i| [i as f32 + 0.5, 0.0, 0.0, 1.0]).collect();
    let views = [View::RawUav(16), View::Texture(4, 2, &texels)];
    run(&shader, &views, 0, |label, out| {
        for t in 0..4 {
            let got = &out[4 * t..4 * t + 3];
            let want = [3, 1, texels[t][0].to_bits()];
            assert_eq!(got, want, "{label}, thread {t}");
        }
    });
}

/// A structured buffer only read by scalar loads: its descriptor's third
/// dword counts records, as the code that clamps an index to it expects.
#[test]
fn a_structured_buffer_read_by_scalar_loads_counts_its_records() {
    let _serial = serial();
    let mut words = vec![s_mov_b32(1, k(0))];
    words.extend(s_load(2, 8, 0, 0));
    words.extend(s_load(2, 12, 0, 0x20));
    // s_buffer_load_dword s2, s[12:15], 0x14
    words.extend([0xF420_0000 | 2 << 6 | 12 >> 1, 125 << 25 | 0x14]);
    words.push(vop1(V_MOV_B32, 3, 14));
    words.push(vop1(V_MOV_B32, 4, 2));
    words.push(vop2(V_LSHLREV_B32, 6, k(3), 0));
    words.extend(buffer_store_dword(3, 6, 8, 0));
    words.extend(buffer_store_dword(4, 6, 8, 4));
    words.push(S_ENDPGM);
    let records = UsedResource {
        stride: 16,
        ..used(ResourceType::Srv, 8, 4, 0)
    };
    let native = program(
        &words,
        false,
        [2, 1, 1],
        vec![table(0, 0)],
        vec![used(ResourceType::Uav, 0, 4, 0), records],
    );
    let shader = translate(&native).unwrap();
    let data: Vec<u32> = (0..32).map(|i| 0x1000 + i).collect();
    let views = [
        View::RawUav(4),
        View::Buffer {
            data: &data,
            stride: super::host::Stride::Structured(16),
        },
    ];
    run(&shader, &views, 0, |label, out| {
        assert_eq!(out, [8, data[5], 8, data[5]], "{label}");
    });
}

/// Without a use, nothing says what the descriptor is a view of.
#[test]
fn a_descriptor_only_read_as_data_stays_refused() {
    let _serial = serial();
    let native = size_reader(false);
    let p = ir::build(&native.code, false, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let l = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    assert!(
        matches!(
            l,
            Err(ShaderError::Unsupported {
                what: "a descriptor read only as data",
                ..
            })
        ),
        "{:?}",
        l.err()
    );
}

/// A sample that takes its level of detail from the quad (shader model
/// 6.6 in a compute shader) passes the validator, which wants the module to
/// name validator 1.8 for it.
#[test]
fn derivatives_in_compute_validate() {
    let _serial = serial();
    let mut words = vec![s_mov_b32(4, 1), s_mov_b32(5, k(0)), s_mov_b32(1, k(0))];
    words.extend(s_load(2, 8, 0, 0));
    words.extend(s_load(3, 12, 0, 0x20));
    words.extend(s_load(2, 28, 4, 0));
    words.push(vop1(V_CVT_F32_U32, 2, vgpr(0)));
    words.push(vop1(V_CVT_F32_U32, 3, vgpr(1)));
    words.extend(image_2d(IMAGE_SAMPLE, 4, 2, 12, 28));
    words.push(vop2(V_LSHLREV_B32, 5, k(2), 0));
    words.extend(buffer_store_dword(4, 5, 8, 0));
    words.push(S_ENDPGM);
    let native = program(
        &words,
        false,
        [2, 2, 1],
        vec![table(0, 0), table(1, 1)],
        vec![
            used(ResourceType::Uav, 0, 4, 0),
            used(ResourceType::Srv, 8, 8, 0),
            used(ResourceType::Sampler, 0, 4, 1),
        ],
    );
    let shader = translate(&native).unwrap();
    let c = Container::parse(&shader.dxil).unwrap();
    let psv = c.part(FourCC::PSV0).unwrap().data;
    assert_eq!(psv[..4], 52u32.to_le_bytes(), "runtime info version 3");
    // Every texel is the same, so any level and any coordinate reads it.
    let texels = [[0.75f32, 0.0, 0.0, 1.0]; 4];
    let views = [View::RawUav(4), View::Texture(2, 2, &texels)];
    run(&shader, &views, 0, |label, out| {
        assert_eq!(out[..2], [0.75f32.to_bits(); 2], "{label}");
    });
}

/// `image_atomic_umax` of two dwords is one 64-bit maximum, the low dword
/// in the texel's first channel: a value with a larger high dword wins over
/// one with a larger low dword.
#[test]
fn a_two_dword_image_atomic_is_64_bits_wide() {
    let _serial = serial();
    let mut words = vec![s_mov_b32(1, k(0))];
    words.extend(s_load(3, 8, 0, 0));
    // Odd threads: (thread, 1); even ones: (0xfffffff0 + thread, 0).
    words.push(vop2(V_AND_B32, 5, k(1), 0));
    words.push(vopc(V_CMP_EQ_U32, k(1), 5));
    words.extend([vop2(V_ADD_NC_U32, 3, LIT, 0), 0xFFFF_FFF0]);
    words.push(vop2(V_CNDMASK_B32, 4, vgpr(3), 0));
    // Texel (thread / 2, 0).
    words.push(vop2(V_LSHRREV_B32, 6, k(1), 0));
    words.push(vop1(V_MOV_B32, 7, k(0)));
    let [w0, w1] = image_2d(IMAGE_ATOMIC_UMAX, 4, 6, 8, 0);
    words.extend([w0 | 3 << 8, w1]);
    words.push(S_ENDPGM);
    let native = program(
        &words,
        false,
        [4, 1, 1],
        vec![table(0, 0)],
        vec![used(ResourceType::Uav, 0, 8, 0)],
    );
    let shader = translate(&native).unwrap();
    let c = Container::parse(&shader.dxil).unwrap();
    let features = u64::from_le_bytes(c.part(FourCC::SFI0).unwrap().data.try_into().unwrap());
    // 64-bit integers, 64-bit atomics on typed and on heap resources.
    assert_eq!(features & 0x1040_8000, 0x1040_8000);
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        // Not every device has these atomics.
        let (root, pso) = match host.compute_pipeline(&shader) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{label}: no pipeline ({e}); skipped");
                continue;
            }
        };
        let data = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(data))],
                    views: &[View::Uint64TextureUav(2, 1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(out[0], [1, 1, 3, 1], "{label}");
    }
}

// ---- samplers held in VGPRs ([`crate::resource::Source::Held`]) ----

const V_READFIRSTLANE_B32: u32 = 0x02;
const V_CMP_EQ_U64: u32 = 0xE2;
const S_XOR_B32: u32 = 0x12;
const IMAGE_SAMPLE_LZ: u32 = 0x27;
const EXEC_LO: u16 = 126;

/// A compute program of 8 threads that samples a 4×1 texture at u = 1.375
/// through whatever sampler v4-v7 hold, as the console compiler does it:
/// in a loop that takes the first active lane's four dwords back and runs
/// the lanes holding the same. The sampler table has three point samplers
/// whose address mode tells them apart there: clamp reads texel 3, wrap
/// texel 1, mirror texel 2. `k` (a root constant) is in s2.
struct Held(Vec<u32>);

impl Held {
    fn new() -> Self {
        let mut w = vec![s_mov_b32(4, 1), s_mov_b32(5, k(0)), s_mov_b32(1, k(0))];
        w.extend(s_load(2, 8, 0, 0));
        w.extend(s_load(3, 12, 0, 0x20));
        w.extend([vop1(V_MOV_B32, 1, LIT), 1.375f32.to_bits()]);
        w.push(vop1(V_MOV_B32, 2, 240)); // 0.5
        w.push(vop1(V_MOV_B32, 8, k(0)));
        Held(w)
    }

    /// Sampler `n` of the table into v4-v7, in the lanes EXEC has.
    fn copy(&mut self, n: u32) {
        self.0.extend(s_load(2, 20, 4, 16 * n));
        for d in 0..4u8 {
            self.0.push(vop1(V_MOV_B32, 4 + d, u16::from(20 + d)));
        }
    }

    /// Sampler 1 into v4-v7, then sampler 2 in the odd threads: each lane
    /// holds its own, which only holding follows (per-lane tracing refuses
    /// a copy that some lanes skip). Copied in every lane, a sampler is
    /// traced to its load, and the checks below never run.
    fn copy_per_lane(&mut self) {
        self.copy(1);
        self.odd_only();
        self.copy(2);
        self.all();
    }

    /// EXEC narrowed to the odd threads (kept in s24), without a branch.
    fn odd_only(&mut self) {
        self.0.push(vop2(V_AND_B32, 3, k(1), 0));
        self.0.push(vopc(V_CMP_EQ_U32, k(1), 3));
        self.0.push(s_mov_b32(24, EXEC_LO));
        self.0.push(sop2(S_AND_B32, 126, EXEC_LO, VCC_LO));
    }

    fn all(&mut self) {
        self.0.push(s_mov_b32(126, 24));
    }

    /// The sample, into v8.
    fn sample(&mut self) {
        let w = &mut self.0;
        w.push(s_mov_b32(25, EXEC_LO));
        let top = w.len();
        for d in 0..4u8 {
            w.push(vop1(V_READFIRSTLANE_B32, 28 + d, vgpr(4 + d)));
        }
        w.push(vopc(V_CMP_EQ_U64, 28, 4));
        w.extend(vop3(V_CMP_EQ_U64, 6, 30, vgpr(6), 0));
        w.push(sop2(S_AND_B32, 6, VCC_LO, 6));
        w.push(0xBE80_3C00 | 6 << 16 | 6); // s_and_saveexec_b32 s6, s6
        w.extend(image_2d(IMAGE_SAMPLE_LZ, 8, 1, 12, 28));
        w.push(sop2(S_XOR_B32, 126, EXEC_LO, 6));
        let back = top as i32 - (w.len() as i32 + 1);
        w.push(0xBF89_0000 | u32::from(back as u16)); // s_cbranch_execnz top
        w.push(s_mov_b32(126, 25));
    }

    /// v8 to dword `8 * region + thread` of the buffer.
    fn store(&mut self, region: u32) {
        self.0.push(vop2(V_LSHLREV_B32, 9, k(2), 0));
        self.0.extend(buffer_store_dword(8, 9, 8, 32 * region));
    }

    fn program(self) -> NativeProgram {
        self.program_in(RootType::Table)
    }

    /// The program with the samplers at s1 in `root`: their table, or the
    /// sampler heap itself (whose base is 0, so the heap indices are known
    /// at translation).
    fn program_in(mut self, root: RootType) -> NativeProgram {
        self.0.push(S_ENDPGM);
        let constants = RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 2,
            slot: 2,
        };
        let samplers = RootParam {
            root,
            ..table(1, 1)
        };
        let mut entries = vec![
            used(ResourceType::Uav, 0, 4, 0),
            used(ResourceType::Srv, 8, 8, 0),
        ];
        if root == RootType::Table {
            for n in 0..3 {
                entries.push(used(ResourceType::Sampler, 4 * n, 4, 1));
            }
        }
        program(
            &self.0,
            false,
            [8, 1, 1],
            vec![table(0, 0), samplers, constants],
            entries,
        )
    }
}

const CLAMP: u32 = 40;
const WRAP: u32 = 20;
const MIRROR: u32 = 30;

/// Runs a [`Held`] program on every host device, with `k` as the root
/// constant, and hands the buffer's dwords (as the texel values sampled)
/// to `check`.
fn run_held(shader: &HostShader, k: u32, check: impl Fn(&str, &[u32])) {
    run_held_on(&[("hardware", false), ("warp", true)], shader, k, check);
}

/// [`run_held`] on the `devices` given (label, WARP).
fn run_held_on(
    devices: &[(&str, bool)],
    shader: &HostShader,
    k: u32,
    check: impl Fn(&str, &[u32]),
) {
    use windows::Win32::Graphics::Direct3D12::*;
    let sampler = |u| D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: u,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    };
    let samplers = [
        sampler(D3D12_TEXTURE_ADDRESS_MODE_CLAMP),
        sampler(D3D12_TEXTURE_ADDRESS_MODE_WRAP),
        sampler(D3D12_TEXTURE_ADDRESS_MODE_MIRROR),
    ];
    let texels: Vec<[f32; 4]> = (1..=4).map(|i| [10.0 * i as f32, 0.0, 0.0, 1.0]).collect();
    let views = [View::RawUav(32), View::Texture(4, 1, &texels)];
    for &(label, warp) in devices {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let mut data = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if u.root == RootType::Constants {
                data[u.index as usize] = k;
            }
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(data))],
                    views: &views,
                    samplers: &samplers,
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let texel: Vec<u32> = out[0].iter().map(|&x| f32::from_bits(x) as u32).collect();
        check(label, &texel);
    }
}

fn refusal(native: &NativeProgram) -> &'static str {
    match translate(native) {
        Err(ShaderError::Unsupported { what, .. }) => what,
        other => panic!("{:?}", other.map(|s| s.stage)),
    }
}

#[test]
fn a_sampler_held_in_vgprs_samples() {
    let _serial = serial();
    let mut h = Held::new();
    h.copy(1);
    h.sample();
    h.store(0);
    let shader = translate(&h.program()).unwrap();
    run_held(&shader, 0, |label, out| {
        assert_eq!(out[..8], [WRAP; 8], "{label}");
    });
}

#[test]
fn vgprs_reused_by_several_samplers_sample_with_each_in_turn() {
    let _serial = serial();
    let mut h = Held::new();
    for (region, n) in [(0, 1), (1, 2), (2, 0)] {
        h.copy(n);
        h.sample();
        h.store(region);
    }
    let shader = translate(&h.program()).unwrap();
    run_held(&shader, 0, |label, out| {
        assert_eq!(out[..8], [WRAP; 8], "{label}: the first");
        assert_eq!(out[8..16], [MIRROR; 8], "{label}: the second");
        assert_eq!(out[16..24], [CLAMP; 8], "{label}: the third");
    });
}

#[test]
fn a_sampler_held_in_vgprs_joins_a_leftover() {
    let _serial = serial();
    let mut h = Held::new();
    // k != 0: the sampler; k == 0: a sum is left in v4, and nothing samples.
    h.0.push(s_cmp_eq_u32(2, k(0)));
    let branch = h.0.len();
    h.0.push(0);
    h.copy(1);
    let jump = h.0.len();
    h.0.push(0);
    h.0[branch] = 0xBF85_0000 | (h.0.len() - branch - 1) as u32; // s_cbranch_scc1
    h.0.push(vop2(V_ADD_NC_U32, 4, k(1), 0));
    h.0[jump] = 0xBF82_0000 | (h.0.len() - jump - 1) as u32; // s_branch
    h.0.push(s_cmp_eq_u32(2, k(0)));
    let skip = h.0.len();
    h.0.push(0);
    h.sample();
    h.0[skip] = 0xBF85_0000 | (h.0.len() - skip - 1) as u32;
    h.store(0);
    let shader = translate(&h.program()).unwrap();
    for (k, want) in [(1, WRAP), (0, 0)] {
        run_held(&shader, k, |label, out| {
            assert_eq!(out[..8], [want; 8], "{label}, k = {k}");
        });
    }
}

#[test]
fn a_sampler_overwritten_in_some_lanes_samples_with_each_lanes_own() {
    let _serial = serial();
    let mut h = Held::new();
    h.copy(1);
    h.odd_only();
    h.copy(2);
    h.all();
    h.sample();
    h.store(0);
    let shader = translate(&h.program()).unwrap();
    run_held(&shader, 0, |label, out| {
        let want: Vec<u32> = (0..8)
            .map(|t| if t % 2 == 1 { MIRROR } else { WRAP })
            .collect();
        assert_eq!(out[..8], want[..], "{label}");
    });
}

/// Four constants that some lanes may hold in the VGPRs, beside a sampler
/// copied in others, may be an immediate sampler, which nothing stands in
/// for: refused, as is a program that computes with a held sampler's
/// dword. (Copied in every lane, the four constants are an immediate
/// sampler, which per-lane tracing follows.)
#[test]
fn held_samplers_that_cannot_be_followed_are_refused() {
    let _serial = serial();
    let mut immediate = Held::new();
    for d in 0..4u8 {
        immediate.0.push(vop1(V_MOV_B32, 4 + d, k(u16::from(d))));
    }
    immediate.sample();
    immediate.store(0);
    let shader = translate(&immediate.program()).unwrap();
    assert_eq!(shader.samplers, [[0, 1, 2, 3]]);

    // With a sampler copied in some lanes, the others may still hold them.
    let mut either = Held::new();
    for d in 0..4u8 {
        either.0.push(vop1(V_MOV_B32, 4 + d, k(u16::from(d))));
    }
    either.odd_only();
    either.copy(1);
    either.all();
    either.sample();
    either.store(0);
    assert_eq!(
        refusal(&either.program()),
        "a sampler held in VGPRs that may be an immediate one"
    );

    let mut computed = Held::new();
    computed.copy_per_lane();
    computed.0.push(vop2(V_ADD_NC_U32, 10, vgpr(4), 0));
    computed.sample();
    computed.store(0);
    assert_eq!(
        refusal(&computed.program()),
        "a sampler held in VGPRs whose dwords are computed with"
    );
}

const READ: &str = "a sampler held in VGPRs whose dwords are computed with";
const V_FMAC_F32: u32 = 0x2B;
const V_CMP_EQ_U16: u32 = 0xAA;
/// VOP3 opcode of VOP1 `v_mov_b32`.
const V3_MOV_B32: u32 = 0x181;

/// A failed held check refuses the program, not only the sample: here
/// nothing asks for the held sampler's handle (the texture's descriptor,
/// made of constants, fails first) and the sample's result is dead, while
/// a sum with the stand-in for dword 0 is stored.
#[test]
fn a_failed_held_check_refuses_the_program() {
    let _serial = serial();
    let mut h = Held::new();
    h.copy_per_lane();
    h.0.push(vop2(V_ADD_NC_U32, 10, vgpr(4), 0));
    for d in 0..8u8 {
        h.0.push(s_mov_b32(12 + d, k(0)));
    }
    h.sample();
    h.0.push(vop1(V_MOV_B32, 8, vgpr(10)));
    h.store(0);
    assert_eq!(refusal(&h.program()), READ);
}

/// A sampler held in VGPRs read back with no waterfall loop would take the
/// first active host lane's: refused (by the waterfall proof every
/// descriptor read from the first lane needs).
#[test]
fn a_held_sampler_read_back_outside_a_waterfall_is_refused() {
    let _serial = serial();
    let mut h = Held::new();
    h.copy_per_lane();
    for d in 0..4u8 {
        h.0.push(vop1(V_READFIRSTLANE_B32, 28 + d, vgpr(4 + d)));
    }
    h.0.extend(image_2d(IMAGE_SAMPLE_LZ, 8, 1, 12, 28));
    h.store(0);
    assert_eq!(
        refusal(&h.program()),
        "a descriptor read from the first lane outside a waterfall loop"
    );
}

/// What computes with a stand-in by other means than its operands: an
/// accumulator on it, an SDWA word written over it keeping the rest, DPP
/// taking it from a lane that did not overwrite it, a copy that negates
/// it (which flips a heap index's top bit). The first three results are
/// stored (in v10) instead of the sample; the negated dword 0 is sampled
/// with.
#[test]
fn held_dwords_computed_with_in_other_ways_are_refused() {
    let _serial = serial();
    let cases: [(&str, Vec<u32>); 4] = [
        // v_mov_b32 v10, v4; v_fmac_f32 v10, v1, v2
        (
            "accumulator",
            vec![
                vop1(V_MOV_B32, 10, vgpr(4)),
                vop2(V_FMAC_F32, 10, vgpr(1), 2),
            ],
        ),
        // v_mov_b32 v10, v5; v_mov_b32_sdwa v10, v1 dst_sel:WORD_0
        // dst_unused:UNUSED_PRESERVE src0_sel:DWORD
        (
            "preserved bits",
            vec![
                vop1(V_MOV_B32, 10, vgpr(5)),
                vop1(V_MOV_B32, 10, 0xF9),
                1 | 4 << 8 | 2 << 11 | 6 << 16,
            ],
        ),
        // v_mov_b32 v10, v4; odd lanes: v_mov_b32 v10, 0 and
        // v_add_nc_u32_dpp v11, v10, v0 quad_perm:[1,0,3,2] (an even lane's
        // stand-in into each odd lane, under the write's own guard); then
        // v_mov_b32 v10, v11
        ("another lane", {
            let mut odd = Held(vec![vop1(V_MOV_B32, 10, vgpr(4))]);
            odd.odd_only();
            odd.0.push(vop1(V_MOV_B32, 10, k(0)));
            odd.0.push(vop2(V_ADD_NC_U32, 11, 0xFA, 0));
            odd.0.push(10 | 0xB1 << 8 | 0xF << 24 | 0xF << 28);
            odd.all();
            odd.0.push(vop1(V_MOV_B32, 10, vgpr(11)));
            odd.0
        }),
        // v_mov_b32_e64 v4, -v4
        ("negated", {
            let [w0, w1] = vop3(V3_MOV_B32, 4, vgpr(4), 0, 0);
            vec![w0, w1 | 1 << 29]
        }),
    ];
    for (what, code) in cases {
        let mut h = Held::new();
        h.copy_per_lane();
        h.0.extend(code);
        h.sample();
        h.0.push(vop1(V_MOV_B32, 8, vgpr(10)));
        h.store(0);
        assert_eq!(refusal(&h.program()), READ, "{what}");
    }
}

/// A stand-in compared with anything but its own read-back (a null-sampler
/// test, say) compares a heap index where the console compares sampler
/// state: refused, in 32-bit and 16-bit compares.
#[test]
fn held_dwords_compared_outside_the_waterfall_are_refused() {
    let _serial = serial();
    for op in [V_CMP_EQ_U32, V_CMP_EQ_U16] {
        let mut h = Held::new();
        h.copy_per_lane();
        h.0.push(vopc(op, k(0), 4));
        h.0.push(vop2(V_CNDMASK_B32, 10, vgpr(1), 2));
        h.sample();
        h.0.push(vop1(V_MOV_B32, 8, vgpr(10)));
        h.store(0);
        assert_eq!(refusal(&h.program()), READ, "{op:#x}");
    }
}

/// A lane whose dword-1 register still holds the tag while its dword-0
/// register holds another number (here a valid slot, 2) samples with slot
/// 0, not with whatever that number indexes: the index must be one a copy
/// left there (sampler heap: known at translation).
#[test]
fn a_held_index_no_copy_left_samples_slot_0() {
    let _serial = serial();
    let mut h = Held::new();
    h.copy(1);
    h.odd_only();
    h.0.push(vop1(V_MOV_B32, 4, k(2)));
    h.all();
    h.sample();
    h.store(0);
    let shader = translate(&h.program_in(RootType::SamplerHeap)).unwrap();
    run_held(&shader, 0, |label, out| {
        let want: Vec<u32> = (0..8)
            .map(|t| if t % 2 == 1 { CLAMP } else { WRAP })
            .collect();
        assert_eq!(out[..8], want[..], "{label}");
    });
}

/// The same with a sampler table, whose heap indices depend on user data:
/// a dword-0 number past any sampler heap samples slot 0. Run on WARP only
/// (an index past the heap could fault a hardware device if the bound were
/// lost).
#[test]
fn a_held_index_past_the_sampler_heap_samples_slot_0() {
    let _serial = serial();
    let mut h = Held::new();
    h.copy(1);
    h.odd_only();
    h.0.extend([vop1(V_MOV_B32, 4, LIT), 1.0f32.to_bits()]);
    h.all();
    h.sample();
    h.store(0);
    let shader = translate(&h.program()).unwrap();
    run_held_on(&[("warp", true)], &shader, 0, |label, out| {
        let want: Vec<u32> = (0..8)
            .map(|t| if t % 2 == 1 { CLAMP } else { WRAP })
            .collect();
        assert_eq!(out[..8], want[..], "{label}");
    });
}

/// `v_readfirstlane_b32` under a lane guard (EXEC narrowed to the odd
/// threads without a branch) of a value the same in those lanes (7) but
/// not in the others (their thread id): every lane reaches it on the host,
/// but the console reads a lane in EXEC.
#[test]
fn a_guarded_first_lane_read_takes_a_lane_in_exec() {
    let _serial = serial();
    let mut h = Held::new();
    h.0.push(vop1(V_MOV_B32, 12, vgpr(0)));
    h.odd_only();
    h.0.push(vop1(V_MOV_B32, 12, k(7)));
    h.0.push(vop1(V_READFIRSTLANE_B32, 40, vgpr(12)));
    h.all();
    h.0.push(vop1(V_CVT_F32_U32, 8, 40));
    h.store(0);
    let shader = translate(&h.program()).unwrap();
    run_held(&shader, 0, |label, out| {
        assert_eq!(out[..8], [7; 8], "{label}");
    });
}

/// A texture's descriptor copied into v10-v17 and read back from the first
/// lane in a waterfall loop is every lane's only if every lane ran the copy.
#[test]
fn a_descriptor_copied_in_some_lanes_only_is_not_every_lanes() {
    let _serial = serial();
    let build = |masked: bool| {
        let mut h = Held::new();
        if masked {
            h.odd_only();
        }
        for d in 0..8u8 {
            h.0.push(vop1(V_MOV_B32, 10 + d, u16::from(12 + d)));
        }
        if masked {
            h.all();
        }
        h.0.extend(s_load(2, 20, 4, 16));
        // The loop of `Held::sample` over the eight dwords.
        let w = &mut h.0;
        w.push(s_mov_b32(25, EXEC_LO));
        let top = w.len();
        for d in 0..8u8 {
            w.push(vop1(V_READFIRSTLANE_B32, 32 + d, vgpr(10 + d)));
        }
        w.push(vopc(V_CMP_EQ_U64, 32, 10));
        w.extend(vop3(V_CMP_EQ_U64, 6, 34, vgpr(12), 0));
        w.push(sop2(S_AND_B32, 6, VCC_LO, 6));
        for d in [36u16, 38] {
            w.extend(vop3(V_CMP_EQ_U64, 7, d, vgpr(d as u8 - 22), 0));
            w.push(sop2(S_AND_B32, 6, 6, 7));
        }
        w.push(0xBE80_3C00 | 6 << 16 | 6); // s_and_saveexec_b32 s6, s6
        w.extend(image_2d(IMAGE_SAMPLE_LZ, 8, 1, 32, 20));
        w.push(sop2(S_XOR_B32, 126, EXEC_LO, 6));
        let back = top as i32 - (w.len() as i32 + 1);
        w.push(0xBF89_0000 | u32::from(back as u16)); // s_cbranch_execnz top
        w.push(s_mov_b32(126, 25));
        h.store(0);
        h.program()
    };
    assert_eq!(
        refusal(&build(true)),
        "a per-lane descriptor written where some lanes may keep the old one"
    );
    let shader = translate(&build(false)).unwrap();
    run_held(&shader, 0, |label, out| {
        assert_eq!(out[..8], [WRAP; 8], "{label}");
    });
}

// ---- sampling forms: gathers at a level, a16 biases, clamped gradients ----

const IMAGE_SAMPLE_D: u32 = 0x22;
const IMAGE_SAMPLE_D_CL: u32 = 0x23;
const IMAGE_SAMPLE_B: u32 = 0x25;
const IMAGE_GATHER4_L: u32 = 0x44;
const IMAGE_GATHER4_L_O: u32 = 0x54;
const IMAGE_GATHER4_LZ_O: u32 = 0x57;
const V_ADD_F32: u32 = 0x03;
const V_MUL_F32: u32 = 0x08;
const V_CVT_F32_I32: u32 = 0x05;
const V_CVT_PKRTZ_F16_F32: u32 = 0x2F;
const V3_CVT_F32_I32: u32 = 0x185;
/// SOPP `s_cbranch_scc1`, the dwords to skip in the low half.
const S_CBRANCH_SCC1: u32 = 0xBF85_0000;
const DIM_2D: u32 = 1;
const DIM_CUBE: u32 = 3;
const DIM_2D_ARRAY: u32 = 5;

/// MIMG `op vdata, v[vaddr..], s[rsrc..], s[sampler..]` of dimension `dim`
/// with `dmask`, 32-bit addresses or (`a16`) 16-bit ones.
#[allow(clippy::too_many_arguments)]
const fn mimg(
    op: u32,
    vdata: u8,
    vaddr: u8,
    rsrc: u8,
    sampler: u8,
    dmask: u32,
    dim: u32,
    a16: bool,
) -> [u32; 2] {
    [
        0xF000_0000 | op << 18 | dmask << 8 | dim << 3,
        (a16 as u32) << 30
            | ((sampler as u32) >> 2) << 21
            | ((rsrc as u32) >> 2) << 16
            | (vdata as u32) << 8
            | vaddr as u32,
    ]
}

/// MUBUF `buffer_load_dword vdata, vaddr, s[rsrc..], 0 offen offset:offset`.
const fn buffer_load_dword(vdata: u8, vaddr: u8, rsrc: u8, offset: u32) -> [u32; 2] {
    [
        0xE030_1000 | offset,
        0x80 << 24 | ((rsrc as u32) >> 2) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// A 4×4 texture with levels of 2×2 and 1×1: red 1 + 10y + x at level 0,
/// 100 + 10y + x at level 1, 200 at level 2; green 1000 more; alpha 1.
fn level_texels() -> Vec<[f32; 4]> {
    let mut t = Vec::new();
    for (size, base) in [(4, 1.0), (2, 100.0), (1, 200.0)] {
        for y in 0..size {
            for x in 0..size {
                let r = base + 10.0 * y as f32 + x as f32;
                t.push([r, r + 1000.0, 0.0, 1.0]);
            }
        }
    }
    t
}

/// Immediate sampler words: x border (transparent black), y clamped,
/// trilinear, LOD 0 to 13, word 2's bit 28 set as the console sets it.
const GATHER_SAMPLER: [u32; 4] = [0x16, 0x00D0_0000, 0x1850_0000, 0];

/// A compute program of 8 threads. Each reads (offset dword, u, v, level
/// as an integer) from the buffer at 16 × thread into v4-v7, runs `lod` (on
/// v7), and stores at 64 × thread: `image_gather4_l_o` of red and of green
/// with the immediate sampler `sampler` in s[24:27] (or, with `heap`, the
/// heap's at s[28:31]), `image_gather4_lz_o` of red with the heap's (point,
/// clamped), `image_gather4_l` of red without the offsets.
fn gathers(sampler: [u32; 4], lod: &[u32], heap: bool) -> NativeProgram {
    gathers_as((1, DIM_2D, false), sampler, lod, heap)
}

/// [`gathers`] with its first gather's channels, dimension and 16-bit
/// addressing `(dmask, dim, a16)` changed.
fn gathers_as(form: (u32, u32, bool), sampler: [u32; 4], lod: &[u32], heap: bool) -> NativeProgram {
    let (dmask, dim, a16) = form;
    let mut w = vec![s_mov_b32(4, 1), s_mov_b32(5, k(0)), s_mov_b32(1, k(0))];
    w.extend(s_load(2, 8, 0, 0));
    w.extend(s_load(3, 12, 0, 0x20));
    w.extend(s_load(2, 20, 0, 0x40));
    w.extend(s_load(2, 28, 4, 0));
    for (d, &word) in sampler.iter().enumerate() {
        w.extend([s_mov_b32(24 + d as u8, LIT), word]);
    }
    let at = if heap { 28 } else { 24 };
    w.push(vop2(V_LSHLREV_B32, 1, k(4), 0));
    for d in 0..4u8 {
        w.extend(buffer_load_dword(4 + d, 1, 20, 4 * u32::from(d)));
    }
    w.extend_from_slice(lod);
    w.extend(mimg(IMAGE_GATHER4_L_O, 8, 4, 12, at, dmask, dim, a16));
    w.extend(mimg(IMAGE_GATHER4_L_O, 12, 4, 12, at, 2, DIM_2D, false));
    w.extend(mimg(IMAGE_GATHER4_LZ_O, 16, 4, 12, 28, 1, DIM_2D, false));
    w.extend(mimg(IMAGE_GATHER4_L, 20, 5, 12, at, 1, DIM_2D, false));
    w.push(vop2(V_LSHLREV_B32, 2, k(6), 0));
    for r in 0..16u8 {
        w.extend(buffer_store_dword(8 + r, 2, 8, 4 * u32::from(r)));
    }
    w.push(S_ENDPGM);
    program(
        &w,
        false,
        [8, 1, 1],
        vec![table(0, 0), table(1, 1)],
        vec![
            used(ResourceType::Uav, 0, 4, 0),
            used(ResourceType::Srv, 8, 8, 0),
            used(ResourceType::Srv, 16, 4, 0),
            used(ResourceType::Sampler, 0, 4, 1),
        ],
    )
}

/// Per thread: offset dword, u, v, level of detail.
const GATHER_CASES: [(u32, f32, f32, i32); 8] = [
    (0, 0.375, 0.625, 0),
    (0, 0.25, 0.75, 1),
    (0, 0.0, 0.125, 0),
    (0, 0.5, 0.5, 7),
    // x + 1, y - 1.
    (0x3F01, 0.375, 0.375, 0),
    // 0.375 less an ulp: a texel centre through a reciprocal, say.
    (0, f32::from_bits(0x3EBF_FFFF), 0.625, 0),
    (0, 0.625, 0.375, -3),
    (0x0001, 0.75, 0.25, 1),
];

/// Runs a [`gathers`] program over [`level_texels`] with [`GATHER_CASES`]
/// and hands each thread's 16 results (as floats) to `check`.
fn run_gathers(shader: &HostShader, check: impl Fn(&str, usize, &[f32])) {
    let params: Vec<u32> = GATHER_CASES
        .iter()
        .flat_map(|&(o, u, v, l)| [o, u.to_bits(), v.to_bits(), l as u32])
        .collect();
    let texels = level_texels();
    let views = [
        View::RawUav(8 * 16),
        View::Levels(4, 4, 3, &texels),
        View::Buffer {
            data: &params,
            stride: super::host::Stride::Raw,
        },
    ];
    run(shader, &views, 0, |label, out| {
        for t in 0..8 {
            let got: Vec<f32> = out[16 * t..16 * (t + 1)]
                .iter()
                .map(|&x| f32::from_bits(x))
                .collect();
            check(label, t, &got);
        }
    });
}

/// `image_gather4_l(_o)` at a whole level of detail: the footprint at the
/// level picked (clamped to the sampler's bounds and the view's last
/// level), border texels 0 in x, clamped ones in y, offsets in texels of
/// that level, in the order (0, 1), (1, 1), (1, 0), (0, 0); green 1000
/// more where inside. At level 0 the host's own gather (clamped) agrees.
/// The level of detail is converted from the integer, or (as the meshlet
/// culling passes choose theirs) joined from the two sides of a uniform
/// branch, each whole: lane 0's integer is 0, so the side without the
/// added 1 runs.
#[test]
fn a_gather_at_a_level_reads_that_levels_footprint() {
    let _serial = serial();
    let cvt = vop1(V_CVT_F32_I32, 7, vgpr(7));
    let joined = [
        vop1(V_READFIRSTLANE_B32, 40, vgpr(7)),
        s_cmp_eq_u32(40, k(0)),
        cvt,
        S_CBRANCH_SCC1 | 1,
        vop2(V_ADD_F32, 7, 242, 7),
    ];
    for lod in [&[cvt][..], &joined] {
        let shader = translate(&gathers(GATHER_SAMPLER, lod, false)).unwrap();
        check_gathers(&shader, GATHER_LEVEL, GATHER_PLAIN);
    }
}

/// Red of [`GATHER_CASES`] at the level with the offsets, with
/// [`GATHER_SAMPLER`].
const GATHER_LEVEL: [[f32; 4]; 8] = [
    [32., 33., 23., 22.],
    [110., 111., 111., 110.],
    [0., 11., 1., 0.],
    [200., 0., 0., 200.],
    [13., 14., 4., 3.],
    [32., 33., 23., 22.],
    [23., 24., 14., 13.],
    [0., 0., 0., 0.],
];

/// Red of [`GATHER_CASES`] at the level without the offsets, with
/// [`GATHER_SAMPLER`].
const GATHER_PLAIN: [[f32; 4]; 8] = [
    [32., 33., 23., 22.],
    [110., 111., 111., 110.],
    [0., 11., 1., 0.],
    [200., 0., 0., 200.],
    [22., 23., 13., 12.],
    [32., 33., 23., 22.],
    [23., 24., 14., 13.],
    [111., 0., 0., 101.],
];

/// Runs a [`gathers`] program and checks red at the level with the offsets
/// (`level`), green (1000 more where inside), red without the offsets
/// (`plain`) and the host's own level-0 gather (of the heap's sampler).
fn check_gathers(shader: &HostShader, level: [[f32; 4]; 8], plain: [[f32; 4]; 8]) {
    // The host's level-0 gather (clamped); none for the ulp case, where
    // hosts may round otherwise.
    let level0: [Option<[f32; 4]>; 8] = [
        Some([32., 33., 23., 22.]),
        Some([31., 32., 22., 21.]),
        Some([11., 11., 1., 1.]),
        Some([22., 23., 13., 12.]),
        Some([13., 14., 4., 3.]),
        None,
        Some([23., 24., 14., 13.]),
        Some([14., 14., 4., 4.]),
    ];
    run_gathers(shader, |label, t, got| {
        let red = level[t];
        let green = red.map(|r| if r == 0.0 { 0.0 } else { r + 1000.0 });
        assert_eq!(got[..4], red, "{label}, thread {t}: red");
        assert_eq!(got[4..8], green, "{label}, thread {t}: green");
        if let Some(level0) = level0[t] {
            assert_eq!(got[8..12], level0, "{label}, thread {t}: host gather");
        }
        assert_eq!(got[12..], plain[t], "{label}, thread {t}: without offsets");
    });
}

/// The sampler's bounds on the level of detail (4.8 fixed point, word 1):
/// a MAX_LOD of 1 keeps level of detail 7 at level 1; a MIN_LOD of 1 lifts
/// 0 and -3 to level 1 (1 and 7 stay at 1 and the last level, 2).
#[test]
fn a_gather_at_a_level_keeps_to_the_samplers_bounds() {
    let _serial = serial();
    let cvt = vop1(V_CVT_F32_I32, 7, vgpr(7));
    // Level 1's whole texture: (0, 1), (1, 1), (1, 0), (0, 0).
    let quad = [110., 111., 101., 100.];
    let at_most_1 = {
        let (mut level, mut plain) = (GATHER_LEVEL, GATHER_PLAIN);
        level[3] = quad;
        plain[3] = quad;
        (level, plain)
    };
    let at_least_1 = (
        [
            quad,
            [110., 111., 111., 110.],
            [0., 100., 100., 0.],
            [200., 0., 0., 200.],
            [101., 0., 0., 101.],
            quad,
            quad,
            [0., 0., 0., 0.],
        ],
        [
            quad,
            [110., 111., 111., 110.],
            [0., 100., 100., 0.],
            [200., 0., 0., 200.],
            quad,
            quad,
            quad,
            [111., 0., 0., 101.],
        ],
    );
    for (bounds, (level, plain)) in [(0x0010_0000, at_most_1), (0x00D0_0100, at_least_1)] {
        let mut sampler = GATHER_SAMPLER;
        sampler[1] = bounds;
        let shader = translate(&gathers(sampler, &[cvt], false)).unwrap();
        check_gathers(&shader, level, plain);
    }
}

/// Without a mip filter the base level is read, whatever the level of
/// detail (here the integer's bits, not a whole float).
#[test]
fn a_gather_without_a_mip_filter_reads_the_base_level() {
    let _serial = serial();
    let mut sampler = GATHER_SAMPLER;
    sampler[2] = 0x1050_0000;
    let shader = translate(&gathers(sampler, &[], false)).unwrap();
    let want: [[f32; 4]; 8] = [
        [32., 33., 23., 22.],
        [31., 32., 22., 21.],
        [0., 11., 1., 0.],
        [22., 23., 13., 12.],
        [13., 14., 4., 3.],
        [32., 33., 23., 22.],
        [23., 24., 14., 13.],
        [14., 0., 0., 4.],
    ];
    run_gathers(&shader, |label, t, got| {
        assert_eq!(got[..4], want[t], "{label}, thread {t}");
    });
}

/// What a gather at a level cannot do exactly stays refused: a level of
/// detail that may be fractional (loaded, halved by a multiply or by the
/// output modifier, whole only in the lanes a guard lets through or on one
/// side of a branch) or sampler bounds that may be; wrapped addressing,
/// state a gather does not ignore (a LOD bias, unnormalized or truncated
/// coordinates), a border other than 0 in the channel gathered (opaque
/// white; opaque black's alpha), a sampler from the heap (its state unknown
/// here); an array, or 16-bit coordinates.
#[test]
fn gathers_at_a_level_that_may_differ_are_refused() {
    let _serial = serial();
    let cvt = vop1(V_CVT_F32_I32, 7, vgpr(7));
    let not_whole = "a gather at a level of detail that may not be whole";
    assert_eq!(refusal(&gathers(GATHER_SAMPLER, &[], false)), not_whole);
    let halved = [cvt, vop2(V_MUL_F32, 7, 240, 7)];
    assert_eq!(refusal(&gathers(GATHER_SAMPLER, &halved, false)), not_whole);
    // omod 3: the result halved.
    let mut cvt_halved = vop3(V3_CVT_F32_I32, 7, vgpr(7), 0, 0);
    cvt_halved[1] |= 3 << 27;
    assert_eq!(
        refusal(&gathers(GATHER_SAMPLER, &cvt_halved, false)),
        not_whole
    );
    // Converted where lane 0's integer is not 0; loaded as it is where it is.
    let one_side = [
        vop1(V_READFIRSTLANE_B32, 40, vgpr(7)),
        s_cmp_eq_u32(40, k(0)),
        S_CBRANCH_SCC1 | 1,
        cvt,
    ];
    assert_eq!(
        refusal(&gathers(GATHER_SAMPLER, &one_side, false)),
        not_whole
    );
    let odd_lanes = [
        vop2(V_AND_B32, 3, k(1), 0),
        vopc(V_CMP_EQ_U32, k(1), 3),
        s_mov_b32(40, EXEC_LO),
        sop2(S_AND_B32, 126, EXEC_LO, VCC_LO),
        cvt,
        s_mov_b32(126, 40),
    ];
    assert_eq!(
        refusal(&gathers(GATHER_SAMPLER, &odd_lanes, false)),
        not_whole
    );
    let with = |word: usize, value: u32| {
        let mut s = GATHER_SAMPLER;
        s[word] = value;
        refusal(&gathers(s, &[cvt], false))
    };
    // MIN_LOD 0.5, MAX_LOD 12.5.
    assert_eq!(with(1, 0x00D0_0080), not_whole);
    assert_eq!(with(1, 0x00C8_0000), not_whole);
    assert_eq!(
        with(0, 0x10),
        "a gather at a level with wrapped or mirrored addressing"
    );
    let ignored = "a gather at a level with sampler state it does not ignore";
    assert_eq!(with(2, 0x1850_0100), ignored);
    // FORCE_UNNORMALIZED, TRUNC_COORD.
    assert_eq!(with(0, 0x16 | 1 << 15), ignored);
    assert_eq!(with(0, 0x16 | 1 << 27), ignored);
    let border = "a gather at a level with a border colour other than 0";
    assert_eq!(with(3, 0x8000_0000), border);
    // Opaque black: 0 in red and green, 1 in alpha.
    let mut opaque_black = GATHER_SAMPLER;
    opaque_black[3] = 0x4000_0000;
    assert!(translate(&gathers(opaque_black, &[cvt], false)).is_ok());
    assert_eq!(
        refusal(&gathers_as((8, DIM_2D, false), opaque_black, &[cvt], false)),
        border
    );
    assert_eq!(
        refusal(&gathers(GATHER_SAMPLER, &[cvt], true)),
        "a gather at a level with a sampler from memory"
    );
    let other = "a gather at a level of other than 2D normalized coordinates";
    for form in [(1, DIM_2D_ARRAY, false), (1, DIM_2D, true)] {
        assert_eq!(
            refusal(&gathers_as(form, GATHER_SAMPLER, &[cvt], false)),
            other,
            "{form:?}"
        );
    }
}

/// `image_sample_b` with a16: the bias is an f16 in the low half of its
/// own dword (the high half, 2.0 here, unused), then u and v as f16 in the
/// next. In a quad stepping a texel per thread (level of detail 0), a bias
/// of +1 reads level 1, -1 level 0.
#[test]
fn a_biased_sample_of_16_bit_coordinates() {
    let _serial = serial();
    let quarter = 0.25f32.to_bits();
    let mut w = vec![s_mov_b32(4, 1), s_mov_b32(5, k(0)), s_mov_b32(1, k(0))];
    w.extend(s_load(2, 8, 0, 0));
    w.extend(s_load(3, 12, 0, 0x20));
    w.extend(s_load(2, 28, 4, 0));
    // (x + 0.5) / 4 and (y + 0.5) / 4 as f16 in v5, u low.
    for (r, id) in [(2u8, 0u8), (3, 1)] {
        w.push(vop1(V_CVT_F32_U32, r, vgpr(id)));
        w.push(vop2(V_ADD_F32, r, 240, r));
        w.extend([vop2(V_MUL_F32, r, LIT, r), quarter]);
    }
    w.push(vop2(V_CVT_PKRTZ_F16_F32, 5, vgpr(2), 3));
    for (r, bias) in [(6u8, 0x4000_3C00u32), (7, 0x4000_BC00)] {
        w.extend([vop1(V_MOV_B32, 4, LIT), bias]);
        w.extend(mimg(IMAGE_SAMPLE_B, r, 4, 12, 28, 1, DIM_2D, true));
    }
    // Thread x + 2y stores at 8 × that.
    w.push(vop2(V_LSHLREV_B32, 8, k(1), 1));
    w.push(vop2(V_ADD_NC_U32, 8, vgpr(0), 8));
    w.push(vop2(V_LSHLREV_B32, 8, k(3), 8));
    w.extend(buffer_store_dword(6, 8, 8, 0));
    w.extend(buffer_store_dword(7, 8, 8, 4));
    w.push(S_ENDPGM);
    let native = program(
        &w,
        false,
        [2, 2, 1],
        vec![table(0, 0), table(1, 1)],
        vec![
            used(ResourceType::Uav, 0, 4, 0),
            used(ResourceType::Srv, 8, 8, 0),
            used(ResourceType::Sampler, 0, 4, 1),
        ],
    );
    let shader = translate(&native).unwrap();
    let texels = level_texels();
    let views = [View::RawUav(8), View::Levels(4, 4, 3, &texels)];
    run(&shader, &views, 0, |label, out| {
        for t in 0..4 {
            let (x, y) = (t % 2, t / 2);
            let got = [f32::from_bits(out[2 * t]), f32::from_bits(out[2 * t + 1])];
            let level0 = 1.0 + 10.0 * y as f32 + x as f32;
            assert_eq!(got, [100.0, level0], "{label}, thread {t}");
        }
    });
}

/// The program of [`a_gradient_sample_clamps_its_level`]: 4 threads, a
/// texel per thread in x at level 0 (derivatives in v10-v13), s = (x +
/// 0.5) / 4 and t = 0.125 in v14-v15, the clamp in v16; `image_sample_d_cl`
/// with clamps 0, 1 and 2, then `image_sample_d`, stored at 16 × thread.
fn clamped_gradients(dim: u32, a16: bool) -> NativeProgram {
    let quarter = 0.25f32.to_bits();
    let mut w = vec![s_mov_b32(4, 1), s_mov_b32(5, k(0)), s_mov_b32(1, k(0))];
    w.extend(s_load(2, 8, 0, 0));
    w.extend(s_load(3, 12, 0, 0x20));
    w.extend(s_load(2, 28, 4, 0));
    w.extend([vop1(V_MOV_B32, 10, LIT), quarter]);
    w.push(vop1(V_MOV_B32, 11, k(0)));
    w.push(vop1(V_MOV_B32, 12, k(0)));
    w.extend([vop1(V_MOV_B32, 13, LIT), quarter]);
    w.push(vop1(V_CVT_F32_U32, 14, vgpr(0)));
    w.push(vop2(V_ADD_F32, 14, 240, 14));
    w.extend([vop2(V_MUL_F32, 14, LIT, 14), quarter]);
    w.extend([vop1(V_MOV_B32, 15, LIT), 0.125f32.to_bits()]);
    w.push(vop1(V_MOV_B32, 17, k(0)));
    for (r, clamp) in [(20u8, 0.0f32), (21, 1.0), (22, 2.0)] {
        w.extend([vop1(V_MOV_B32, 16, LIT), clamp.to_bits()]);
        w.extend(mimg(IMAGE_SAMPLE_D_CL, r, 10, 12, 28, 1, dim, a16));
    }
    w.extend(mimg(IMAGE_SAMPLE_D, 23, 10, 12, 28, 1, dim, a16));
    w.push(vop2(V_LSHLREV_B32, 8, k(4), 0));
    for r in 0..4u8 {
        w.extend(buffer_store_dword(20 + r, 8, 8, 4 * u32::from(r)));
    }
    w.push(S_ENDPGM);
    program(
        &w,
        false,
        [4, 1, 1],
        vec![table(0, 0), table(1, 1)],
        vec![
            used(ResourceType::Uav, 0, 4, 0),
            used(ResourceType::Srv, 8, 8, 0),
            used(ResourceType::Sampler, 0, 4, 1),
        ],
    )
}

/// `image_sample_d_cl`: derivatives, coordinates, then the least level of
/// detail. Level 0 by the derivatives; clamped to 1 and 2 it reads those
/// levels (level 1's texel floor(2s)).
#[test]
fn a_gradient_sample_clamps_its_level() {
    let _serial = serial();
    let shader = translate(&clamped_gradients(DIM_2D, false)).unwrap();
    let texels = level_texels();
    let views = [View::RawUav(16), View::Levels(4, 4, 3, &texels)];
    run(&shader, &views, 0, |label, out| {
        for x in 0..4 {
            let got: Vec<f32> = out[4 * x..4 * x + 4]
                .iter()
                .map(|&v| f32::from_bits(v))
                .collect();
            let level0 = 1.0 + x as f32;
            let level1 = 100.0 + (x / 2) as f32;
            assert_eq!(got, [level0, level1, 200.0, level0], "{label}, thread {x}");
        }
    });
}

/// Gradients of a cube are the face's, which the host would project from
/// 3D ones in its own way; 16-bit coordinates with gradients have no
/// program to check them against. Both stay refused.
#[test]
fn gradient_samples_that_cannot_be_checked_are_refused() {
    let _serial = serial();
    assert_eq!(
        refusal(&clamped_gradients(DIM_CUBE, false)),
        "a cube sample with texel coordinates, gradients or a level query"
    );
    assert_eq!(
        refusal(&clamped_gradients(DIM_2D, true)),
        "a sample with gradients and 16-bit coordinates"
    );
}

/// The f16 bits of `x`, a normal f16 value exactly.
fn f16_bits(x: f32) -> u32 {
    let b = x.to_bits();
    let exp = (b >> 23 & 0xFF) as i32 - 127 + 15;
    assert!((1..31).contains(&exp) && b & 0x1FFF == 0, "{x} is no f16");
    (b >> 31) << 15 | (exp as u32) << 10 | (b & 0x7F_FFFF) >> 13
}

/// VOP3P `op vdst, src0, src1, src2` with f32 sources (op_sel_hi clear).
const fn vop3p(op: u32, dst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0xCC00_0000 | op << 16 | dst as u32,
        src0 as u32 | (src1 as u32) << 9 | (src2 as u32) << 18,
    ]
}

/// SOPP `s_cbranch_scc1` and `s_branch` over `n` dwords.
const fn s_cbranch_scc1(n: u32) -> u32 {
    0xBF85_0000 | n
}

const fn s_branch(n: u32) -> u32 {
    0xBF82_0000 | n
}

const fn s_cmp_lg_u32(src0: u16, src1: u16) -> u32 {
    0xBF07_0000 | (src1 as u32) << 8 | src0 as u32
}

/// The second dword of an SDWA VOP1 or VOP2: `src0` a VGPR read as `sel`
/// (4 its low word, 5 its high word), the whole dword written.
const fn sdwa_src0(src0: u8, sel: u32) -> u32 {
    src0 as u32 | 6 << 8 | sel << 16
}

const SDWA: u16 = 0xF9;
const V_CVT_F32_F16: u32 = 0x0B;
const V_FMA_MIXLO_F16: u32 = 0x21;
const INLINE_HALF: u16 = 240;
const INLINE_TWO: u16 = 244;

/// Root constants at user data 2 (s2), after the UAV table's s0.
fn constant_at_s2() -> RootParam {
    RootParam {
        root: RootType::Constants,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 2,
        slot: 1,
    }
}

/// `sgpr`, written only where the root constant is not 0 and read only
/// there: 41 then, else 7, stored at 4 × thread. Three user SGPRs, and
/// what `rsrc2` adds to the registers the hardware loads.
fn joined_sgpr(sgpr: u8, rsrc2: u32) -> NativeProgram {
    let mut w = vec![s_mov_b32(1, k(0))];
    w.extend(s_load(2, 8, 0, 0));
    w.push(s_cmp_eq_u32(2, k(0)));
    w.push(s_cbranch_scc1(1));
    w.push(s_mov_b32(sgpr, k(41)));
    w.push(vop1(V_MOV_B32, 3, k(7)));
    w.push(s_cmp_eq_u32(2, k(0)));
    w.push(s_cbranch_scc1(1));
    w.push(vop1(V_MOV_B32, 3, u16::from(sgpr)));
    w.push(vop2(V_LSHLREV_B32, 5, k(2), 0));
    w.extend(buffer_store_dword(3, 5, 8, 0));
    w.push(S_ENDPGM);
    let mut native = program(
        &w,
        false,
        [4, 1, 1],
        vec![table(0, 0), constant_at_s2()],
        vec![used(ResourceType::Uav, 0, 4, 0)],
    );
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= rsrc2;
    }
    native
}

/// An SGPR the hardware leaves alone (s20: three user SGPRs, nothing
/// else) is read only where the path that wrote it joins one that did
/// not: the program sees the written value.
#[test]
fn an_sgpr_the_hardware_leaves_alone_joins_a_written_one() {
    let _serial = serial();
    let shader = translate(&joined_sgpr(20, 0)).unwrap();
    for constant in [0, 1] {
        run(&shader, &[View::RawUav(4)], constant, |label, out| {
            let want = if constant == 0 { 7 } else { 41 };
            assert_eq!(out, [want; 4], "{label}, {constant}");
        });
    }
}

/// The register after the user SGPRs is one the hardware loads with
/// scratch enabled (its offset, s3 here): refused. Past the registers it
/// loads, one read directly holds what an earlier wave left, and reads
/// zero ([`crate::native::ComputeIo::initialized_sgprs`]).
#[test]
fn hardware_sgprs_read_otherwise_stay_refused() {
    let _serial = serial();
    assert_eq!(refusal(&joined_sgpr(3, 1)), "a hardware SGPR input");
    // Past it, the join is the program's again.
    translate(&joined_sgpr(4, 1)).unwrap();
    let mut w = vec![s_mov_b32(1, k(0))];
    w.extend(s_load(2, 8, 0, 0));
    w.push(vop1(V_MOV_B32, 3, 20));
    w.push(vop2(V_LSHLREV_B32, 5, k(2), 0));
    w.extend(buffer_store_dword(3, 5, 8, 0));
    w.push(S_ENDPGM);
    let direct = program(
        &w,
        false,
        [4, 1, 1],
        vec![table(0, 0), constant_at_s2()],
        vec![used(ResourceType::Uav, 0, 4, 0)],
    );
    let shader = translate(&direct).unwrap();
    run(&shader, &[View::RawUav(4)], 0, |label, out| {
        assert_eq!(out, [0; 4], "{label}");
    });
}

/// A compute program of 4 threads: v1 the thread-group info (which no
/// translation has), its low half then f16(2t + 0.5) by
/// `v_fma_mixlo_f16`, which keeps the high half; `read` then reads v1
/// into v4-v7, stored at 16 × thread.
fn kept_half(read: &[u32]) -> NativeProgram {
    let mut w = vec![s_mov_b32(1, k(0))];
    w.extend(s_load(2, 8, 0, 0));
    w.push(vop1(V_MOV_B32, 1, 3));
    w.push(vop1(V_CVT_F32_U32, 2, vgpr(0)));
    w.extend(vop3p(V_FMA_MIXLO_F16, 1, vgpr(2), INLINE_TWO, INLINE_HALF));
    w.extend(read);
    w.push(vop2(V_LSHLREV_B32, 9, k(4), 0));
    for r in 0..4u8 {
        w.extend(buffer_store_dword(4 + r, 9, 8, 4 * u32::from(r)));
    }
    w.push(S_ENDPGM);
    let mut native = program(
        &w,
        false,
        [4, 1, 1],
        vec![table(0, 0), constant_at_s2()],
        vec![used(ResourceType::Uav, 0, 4, 0)],
    );
    // The group info in s3.
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 1 << 10;
    }
    native
}

/// A 16-bit result in a register whose other half no translation has:
/// what reads only the result's half sees it (as an f16, shifted to the
/// top, masked, as SDWA's low word).
#[test]
fn a_mixed_precision_result_beside_a_missing_half_is_read_alone() {
    let _serial = serial();
    let read = [
        vop1(V_CVT_F32_F16, 4, vgpr(1)),
        vop2(V_LSHLREV_B32, 5, k(16), 1),
        vop2(V_AND_B32, 6, LIT, 1),
        0xFFFF,
        vop1(V_MOV_B32, 7, SDWA),
        sdwa_src0(1, 4),
    ];
    let shader = translate(&kept_half(&read)).unwrap();
    run(&shader, &[View::RawUav(16)], 0, |label, out| {
        for t in 0..4 {
            let f = 2.0 * t as f32 + 0.5;
            let h = f16_bits(f);
            assert_eq!(
                out[4 * t..4 * t + 4],
                [f.to_bits(), h << 16, h, h],
                "{label}, thread {t}"
            );
        }
    });
}

/// What reads the missing half, or the whole dword, stays refused for it.
#[test]
fn reads_of_a_missing_half_stay_refused() {
    let _serial = serial();
    let reads: [&[u32]; 4] = [
        &[vop2(V_LSHRREV_B32, 4, k(16), 1)],
        &[vop1(V_MOV_B32, 4, vgpr(1))],
        &[vop1(V_MOV_B32, 4, SDWA), sdwa_src0(1, 5)],
        &[vop2(V_AND_B32, 4, LIT, 1), 0x1_FFFF],
    ];
    for read in reads {
        assert_eq!(
            refusal(&kept_half(read)),
            "the thread-group info SGPR",
            "{read:x?}"
        );
    }
}

/// [`kept_half`]'s mixed-precision write where the root constant is not
/// 0, a whole dword (0x4040_4040, f16 2.125 in each half) where it is,
/// then `read` of v1 into v4 after the join, stored at 4 × thread. With
/// `twice`, a second join first: v1 is overwritten (0x1234_5678, f16 103.5
/// in its low half) where the constant is 0, so only that join takes the
/// first.
fn kept_half_joined(read: u32, twice: bool) -> NativeProgram {
    let mut w = vec![s_mov_b32(1, k(0))];
    w.extend(s_load(2, 8, 0, 0));
    w.push(vop1(V_MOV_B32, 1, 3));
    w.push(vop1(V_CVT_F32_U32, 2, vgpr(0)));
    w.push(s_cmp_eq_u32(2, k(0)));
    w.push(s_cbranch_scc1(3));
    w.extend(vop3p(V_FMA_MIXLO_F16, 1, vgpr(2), INLINE_TWO, INLINE_HALF));
    w.push(s_branch(2));
    w.extend([vop1(V_MOV_B32, 1, LIT), 0x4040_4040]);
    if twice {
        w.push(s_cmp_lg_u32(2, k(0)));
        w.push(s_cbranch_scc1(2));
        w.extend([vop1(V_MOV_B32, 1, LIT), 0x1234_5678]);
    }
    w.push(read);
    w.push(vop2(V_LSHLREV_B32, 9, k(2), 0));
    w.extend(buffer_store_dword(4, 9, 8, 0));
    w.push(S_ENDPGM);
    let mut native = program(
        &w,
        false,
        [4, 1, 1],
        vec![table(0, 0), constant_at_s2()],
        vec![used(ResourceType::Uav, 0, 4, 0)],
    );
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 1 << 10;
    }
    native
}

/// The join of a dword known in its low half and a whole one is known in
/// its low half: an f16 read sees the path's value, a whole read is
/// refused.
#[test]
fn a_join_knows_the_bits_every_path_knows() {
    let _serial = serial();
    let shader = translate(&kept_half_joined(vop1(V_CVT_F32_F16, 4, vgpr(1)), false)).unwrap();
    for constant in [0, 1] {
        run(&shader, &[View::RawUav(4)], constant, |label, out| {
            for (t, &x) in out.iter().enumerate() {
                let want = if constant == 0 {
                    2.125
                } else {
                    2.0 * t as f32 + 0.5
                };
                assert_eq!(f32::from_bits(x), want, "{label}, {constant}, thread {t}");
            }
        });
    }
    assert_eq!(
        refusal(&kept_half_joined(vop1(V_MOV_B32, 4, vgpr(1)), false)),
        "the thread-group info SGPR"
    );
}

/// So does a join only another join takes: through both, an f16 read sees
/// the path's value, and a whole read is refused (it would see the
/// stand-in for the missing high half).
#[test]
fn a_join_only_a_join_takes_knows_the_same_bits() {
    let _serial = serial();
    let shader = translate(&kept_half_joined(vop1(V_CVT_F32_F16, 4, vgpr(1)), true)).unwrap();
    assert_eq!(f16_bits(103.5), 0x5678);
    for constant in [0, 1] {
        run(&shader, &[View::RawUav(4)], constant, |label, out| {
            for (t, &x) in out.iter().enumerate() {
                let want = if constant == 0 {
                    103.5
                } else {
                    2.0 * t as f32 + 0.5
                };
                assert_eq!(f32::from_bits(x), want, "{label}, {constant}, thread {t}");
            }
        });
    }
    assert_eq!(
        refusal(&kept_half_joined(vop1(V_MOV_B32, 4, vgpr(1)), true)),
        "the thread-group info SGPR"
    );
}

/// A pixel program reading the perspective-center barycentrics (v0, v1)
/// and the attributes `inputs` describe, through the whole translator,
/// with `sig` its input signature (by register; 16-bit elements marked as
/// the console's containers mark them) and one f32 render target.
fn translate_pixel(
    words: &[u32],
    inputs: Vec<PsInput>,
    sig: &[SigElement],
) -> Result<HostShader, ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 16,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs,
            input_addr: 1 << 1,
            db_shader_control: 0,
            patches: vec![],
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let mut isg1 = super::container::signature(sig, true);
    for (i, e) in sig.iter().enumerate() {
        if e.half {
            // Component type 6: f16.
            isg1[8 + 32 * i + 16..][..4].copy_from_slice(&6u32.to_le_bytes());
        }
    }
    let target = SigElement {
        name: "SV_Target".into(),
        index: 0,
        kind: 16,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0,
        half: false,
    };
    let osg1 = super::container::signature(&[target], false);
    let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
    let c = Container::parse(&bytes).unwrap();
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    // As `emit::translate` orders them.
    let st = structure::structure(&p);
    let res = resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
    let io = io::raise(&p, &native)?;
    let st = st?;
    let shader = super::emit(&c, &native, &p, &st, &io, &res)?;
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation");
    }
    Ok(shader)
}

/// A vertex shader for `ps`: `SV_Position` from `POSITION`, and channel c
/// of each input of `ps` (element of semantic index i) the constant
/// 4i + c + 0.5.
fn constant_vs(ps: &HostShader) -> HostShader {
    use super::bitcode::{Attr, Term};
    let c = Container::parse(&ps.dxil).unwrap();
    let mut outputs: Vec<SigElement> = super::signature(c.part(FourCC::ISG1).unwrap().data)
        .unwrap()
        .into_iter()
        .filter(|e| e.kind == 0)
        .collect();
    let row = outputs.iter().filter_map(|e| e.row).map(|r| r + 1).max();
    let mut position = SigElement {
        name: "SV_Position".into(),
        index: 0,
        kind: 3,
        comp: 9,
        interp: 4,
        row: Some(row.unwrap_or(0)),
        col: 0,
        cols: 4,
        usage: 0,
        half: false,
    };
    outputs.push(position.clone());
    let mut s = super::shader::Shader::new(false);
    (position.name, position.kind, position.row) = ("POSITION".into(), 0, Some(0));
    (position.interp, position.usage) = (0, 0xF);
    s.inputs.push(position);
    let b = s.entry;
    for (id, e) in outputs.iter().enumerate() {
        for col in 0..e.cols {
            let x = if e.kind == 3 {
                let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), s.m.undef(s.t.i32)];
                s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args)
            } else {
                let x = 4.0 * e.index as f32 + f32::from(col) + 0.5;
                s.m.f32(x.to_bits())
            };
            let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), x];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
    }
    s.outputs = outputs
        .into_iter()
        .map(|mut e| {
            e.usage = (1 << e.cols) - 1;
            e
        })
        .collect();
    s.m.terminate(b, Term::Ret);
    HostShader {
        stage: HwStage::Vs,
        dxil: s.finish(),
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        stride_table: false,
    }
}

/// Draws `ps` over a 4×4 target on every host device there is, after
/// [`constant_vs`], and checks each texel is `want`.
fn draw_pixel(ps: &HostShader, want: [f32; 4]) {
    let vs = constant_vs(ps);
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut vs.dxil.clone()).expect("DXIL validation");
    }
    // A triangle over the whole target.
    let vertices = [
        -1.0, -1.0, 0.5, 1.0, //
        -1.0, 3.0, 0.5, 1.0, //
        3.0, -1.0, 0.5, 1.0,
    ];
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .pipeline(&vs, ps)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let out = host
            .draw(
                &root,
                &pso,
                &Draw {
                    roots: vec![(0, Root::Constants(vec![0])), (1, Root::Constants(vec![0]))],
                    textures: &[],
                    samplers: &[],
                    vertices: &vertices,
                    stride: 16,
                    size: (4, 4),
                    mesh_groups: 0,
                },
            )
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        for (i, texel) in out.iter().enumerate() {
            assert_eq!(*texel, want, "{label}, texel {i}");
        }
        ran += 1;
    }
    if ran == 0 {
        eprintln!("no SM 6.6 device; not drawn");
    }
}

/// An interpolation attribute code: attribute `index`, channel `chan`,
/// the high f16 half with `high`.
const fn attr(index: u16, chan: u16, high: bool) -> u16 {
    index | chan << 6 | (high as u16) << 8
}

const V_INTERP_P1LL_F16: u32 = 0x342;
const V_INTERP_P2_F16: u32 = 0x35A;

/// `v_interp_p1ll_f16` then `v_interp_p2_f16` of `code` into `dst`, the
/// second keeping the first's high half in `dst`.
fn interp_f16(dst: u8, code: u16) -> [u32; 4] {
    let [a, b] = vop3(V_INTERP_P1LL_F16, dst, code, vgpr(0), 0);
    let [c, d] = vop3(V_INTERP_P2_F16, dst, code, vgpr(1), vgpr(dst));
    [a, b, c, d]
}

/// `exp mrt0, v4, v5, v6, v7 done vm`.
const EXP_V4_V7: [u32; 2] = [0xF800_180F, 0x0706_0504];

fn half_input(name: &str, index: u32, row: u32, col: u8, cols: u8) -> SigElement {
    SigElement {
        name: name.into(),
        index,
        kind: 0,
        comp: 9,
        interp: 0,
        row: Some(row),
        col,
        cols,
        usage: 0,
        half: true,
    }
}

fn packed(attr: u8, row: u8, row2: u8) -> PsInput {
    PsInput {
        usage: Usage::Generic,
        row,
        mask: 3,
        attr,
        flat: false,
        fp16: true,
        pack2: true,
        row2,
    }
}

/// Two parameters of 16-bit values packed two per dword. Attribute 0: row
/// 0 (TEXCOORD0.xy, TEXCOORD8.xy), the stage data naming no second row,
/// then the next 16-bit row no entry names (TEXCOORD1, row 1), as the
/// producer packs it. Attribute 1: row 2 (TEXCOORD2.xyz, a half of
/// padding), then the row the stage data names, `named` (TEXCOORD3 at row
/// 3, or TEXCOORD4 at 4 after it). The program reads halves 4 and 7 of
/// attribute 0, half 3 of it and half 4 of attribute 1, and writes them as
/// floats.
fn packed_reader(named: u8) -> Result<HostShader, ShaderError> {
    let mut w = Vec::new();
    w.extend(interp_f16(2, attr(0, 2, false)));
    w.extend(interp_f16(3, attr(0, 3, true)));
    w.extend(interp_f16(8, attr(0, 1, true)));
    w.extend(interp_f16(9, attr(1, 2, false)));
    for (dst, src) in [(4, 2), (5, 3), (6, 8), (7, 9)] {
        w.push(vop1(V_CVT_F32_F16, dst, vgpr(src)));
    }
    w.extend(EXP_V4_V7);
    w.push(S_ENDPGM);
    let mut sig = vec![
        half_input("TEXCOORD", 0, 0, 0, 2),
        half_input("TEXCOORD", 8, 0, 2, 2),
        half_input("TEXCOORD", 1, 1, 0, 4),
        half_input("TEXCOORD", 2, 2, 0, 3),
        half_input("TEXCOORD", 3, 3, 0, 2),
    ];
    if named == 4 {
        sig.push(half_input("TEXCOORD", 4, 4, 0, 2));
    }
    translate_pixel(&w, vec![packed(0, 0, 0), packed(1, 2, named)], &sig)
}

/// The partner the stage data leaves unnamed is the row the producer packs
/// there: TEXCOORD1.x and .w (4.5, 7.5), TEXCOORD8.y (33.5), and through
/// the named partner TEXCOORD3.x (12.5).
#[test]
fn a_packed_input_finds_the_row_its_stage_data_leaves_unnamed() {
    let _serial = serial();
    let ps = packed_reader(3).unwrap();
    draw_pixel(&ps, [4.5, 7.5, 33.5, 12.5]);
}

/// Where the producer's rule does not give a second row the stage data
/// names (here TEXCOORD4, not TEXCOORD3), it is not taken for the
/// unnamed ones either.
#[test]
fn an_unnamed_partner_the_named_ones_contradict_stays_refused() {
    let _serial = serial();
    assert!(
        matches!(
            packed_reader(4),
            Err(ShaderError::Unsupported {
                what: "a packed input half no element holds",
                ..
            })
        ),
        "{:?}",
        packed_reader(4).map(|s| s.stage)
    );
}

/// Nor where a 16-bit row is left over. The producer packs rows 0 and 1
/// with rows 2 and 3; this stage data leaves out the parameter at row 0
/// (unread), so row 0 joins the free rows and the rule would pair row 1
/// with row 2.
#[test]
fn an_unnamed_partner_beside_a_row_left_over_stays_refused() {
    let _serial = serial();
    let mut w = Vec::new();
    w.extend(interp_f16(2, attr(0, 2, false)));
    w.push(vop1(V_CVT_F32_F16, 4, vgpr(2)));
    for r in 5..8 {
        w.push(vop1(V_MOV_B32, r, k(0)));
    }
    w.extend(EXP_V4_V7);
    w.push(S_ENDPGM);
    let sig: Vec<_> = (0..4).map(|r| half_input("TEXCOORD", r, r, 0, 4)).collect();
    let ps = translate_pixel(&w, vec![packed(0, 1, 1)], &sig);
    assert!(
        matches!(
            ps,
            Err(ShaderError::Unsupported {
                what: "a packed input half no element holds",
                ..
            })
        ),
        "{:?}",
        ps.map(|s| s.stage)
    );
}

/// A pixel program whose `v_fma_mixlo_f16` writes f16(1.5 × 1.5 + 0.5) =
/// 2.75 into the low half of v0, which holds the barycentric i; `read`
/// then reads v0 into v4-v7, which it writes.
fn mixed_over_barycentric(read: &[u32]) -> Result<HostShader, ShaderError> {
    let mut w = vec![vop1(V_MOV_B32, 2, LIT), 1.5f32.to_bits()];
    w.push(vop1(V_MOV_B32, 3, INLINE_HALF));
    w.extend(vop3p(V_FMA_MIXLO_F16, 0, vgpr(2), vgpr(2), vgpr(3)));
    w.extend(read);
    w.extend(EXP_V4_V7);
    w.push(S_ENDPGM);
    translate_pixel(&w, vec![], &[])
}

/// The barycentric's half stays out of the result: f16 2.75 (0x4180) read
/// as an f16, shifted to the top (16.0 as a float), masked and as SDWA's
/// low word (16768 converted).
#[test]
fn a_result_beside_a_barycentric_half_is_read_alone() {
    let _serial = serial();
    let read = [
        vop1(V_CVT_F32_F16, 4, vgpr(0)),
        vop2(V_LSHLREV_B32, 5, k(16), 0),
        vop2(V_AND_B32, 6, LIT, 0),
        0xFFFF,
        vop1(V_CVT_F32_U32, 6, vgpr(6)),
        vop1(V_MOV_B32, 7, SDWA),
        sdwa_src0(0, 4),
        vop1(V_CVT_F32_U32, 7, vgpr(7)),
    ];
    let ps = mixed_over_barycentric(&read).unwrap();
    assert_eq!(f16_bits(2.75), 0x4180);
    draw_pixel(&ps, [2.75, 16.0, 16768.0, 16768.0]);
}

/// Reading the barycentric's half stays refused: which two of the three
/// weights the console's i and j are depends on the vertex order its
/// hardware rasterised in.
#[test]
fn a_barycentric_half_read_stays_refused() {
    let _serial = serial();
    let reads: [&[u32]; 2] = [
        &[vop2(V_LSHRREV_B32, 4, k(16), 0)],
        &[vop1(V_MOV_B32, 4, vgpr(0))],
    ];
    for read in reads {
        assert!(
            matches!(
                mixed_over_barycentric(read),
                Err(ShaderError::Unsupported {
                    what: "barycentrics read directly",
                    ..
                })
            ),
            "{read:x?}"
        );
    }
}

/// SOPP `s_cbranch_vccnz` over `n` dwords.
const fn s_cbranch_vccnz(n: u32) -> u32 {
    0xBF87_0000 | n
}

const S_ANDN2_B32: u32 = 0x14;
const S_NOP: u32 = 0xBF80_0000;
const INLINE_MINUS_1: u16 = 193;

/// The console compiler's skip over a leftover: a raw buffer's descriptor
/// sits in s[12:15]; where the root constant is not 0 the code reads two
/// dwords of the buffer into s12 and s13 and sets the flag in v2, else it
/// skips both, leaving the descriptor's first two dwords there. After the
/// join, lanes with the flag clear jump past the read of s12 and s13:
/// 7 then, else s12 + s13, stored at 4 × thread. `flag` is what v2 holds
/// on the skip.
fn leftover(flag: u16) -> NativeProgram {
    let mut w = vec![s_mov_b32(1, k(0))];
    w.extend(s_load(2, 8, 0, 0));
    w.extend(s_load(2, 12, 0, 0x20));
    w.push(vop1(V_MOV_B32, 2, flag));
    w.push(s_cmp_eq_u32(2, k(0)));
    w.push(s_cbranch_scc1(4));
    // s_buffer_load_dwordx2 s[12:13], s[12:15], 0
    w.extend([0xF424_0000 | 12 << 6 | 12 >> 1, 125 << 25]);
    w.push(vop1(V_MOV_B32, 2, INLINE_MINUS_1));
    w.push(s_branch(1));
    w.push(S_NOP);
    // The join.
    w.push(vopc(V_CMP_EQ_U32, INLINE_MINUS_1, 2));
    w.push(sop2(S_ANDN2_B32, VCC_LO as u8, EXEC_LO, VCC_LO));
    w.push(vop1(V_MOV_B32, 3, k(7)));
    w.push(s_cbranch_vccnz(2));
    w.push(vop1(V_MOV_B32, 3, 12));
    w.push(vop2(V_ADD_NC_U32, 3, 13, 3));
    w.push(vop2(V_LSHLREV_B32, 5, k(2), 0));
    w.extend(buffer_store_dword(3, 5, 8, 0));
    w.push(S_ENDPGM);
    let raw = UsedResource {
        stride: 1,
        ..used(ResourceType::Srv, 8, 4, 0)
    };
    program(
        &w,
        false,
        [4, 1, 1],
        vec![table(0, 0), constant_at_s2()],
        vec![used(ResourceType::Uav, 0, 4, 0), raw],
    )
}

/// The lanes that bring the descriptor's dwords to the join never read
/// them: the program translates, and each path gives its own result.
#[test]
fn a_leftover_descriptor_no_lane_reads_joins_a_stand_in() {
    let _serial = serial();
    let shader = translate(&leftover(k(0))).unwrap();
    let data = [0x100, 0x20, 0x3, 0x4];
    let views = [
        View::RawUav(4),
        View::Buffer {
            data: &data,
            stride: super::host::Stride::Raw,
        },
    ];
    for constant in [0, 1] {
        run(&shader, &views, constant, |label, out| {
            let want = if constant == 0 { 7 } else { 0x120 };
            assert_eq!(out, [want; 4], "{label}, {constant}");
        });
    }
}

/// Where the skip sets the flag, or leaves it unknown, those lanes read
/// the leftover: refused as before.
#[test]
fn a_leftover_some_lane_may_read_stays_refused() {
    let _serial = serial();
    for flag in [INLINE_MINUS_1, vgpr(0)] {
        assert_eq!(
            refusal(&leftover(flag)),
            "a descriptor's address or format read as data",
            "{flag}"
        );
    }
}

/// A buffer descriptor only read as numbers (its third dword: the record
/// count, 0 for a null view): 4 × thread gets the count, then 5 where it
/// is 0, else 9.
fn records_only(stride: u32) -> NativeProgram {
    let mut w = vec![s_mov_b32(1, k(0))];
    w.extend(s_load(2, 8, 0, 0));
    w.extend(s_load(2, 12, 0, 0x20));
    w.push(vop1(V_MOV_B32, 3, 14));
    w.push(vop1(V_MOV_B32, 4, k(5)));
    w.push(s_cmp_eq_u32(14, k(0)));
    w.push(s_cbranch_scc1(1));
    w.push(vop1(V_MOV_B32, 4, k(9)));
    w.push(vop2(V_LSHLREV_B32, 6, k(3), 0));
    w.extend(buffer_store_dword(3, 6, 8, 0));
    w.extend(buffer_store_dword(4, 6, 8, 4));
    w.push(S_ENDPGM);
    let entry = UsedResource {
        stride,
        ..used(ResourceType::Srv, 8, 4, 0)
    };
    program(
        &w,
        false,
        [2, 1, 1],
        vec![table(0, 0)],
        vec![used(ResourceType::Uav, 0, 4, 0), entry],
    )
}

/// Its table entry says what the view is: structured, its records; raw,
/// its bytes; null, none.
#[test]
fn a_buffer_only_read_as_data_is_the_view_its_entry_names() {
    let _serial = serial();
    let data: Vec<u32> = (0..32).collect();
    let cases = [
        (16, super::host::Stride::Structured(16), Some(&data[..]), 8),
        (1, super::host::Stride::Raw, Some(&data[..]), 128),
        (16, super::host::Stride::Structured(16), None, 0),
        (1, super::host::Stride::Raw, None, 0),
    ];
    for (entry, stride, data, want) in cases {
        let shader = translate(&records_only(entry)).unwrap();
        let view = match data {
            Some(data) => View::Buffer { data, stride },
            None => View::NullBuffer(stride),
        };
        let flag = if want == 0 { 5 } else { 9 };
        run(&shader, &[View::RawUav(4), view], 0, |label, out| {
            assert_eq!(
                out,
                [want, flag, want, flag],
                "{label}, entry stride {entry}"
            );
        });
    }
}

/// A stride of 4 is also a typed buffer's, whose view a raw or structured
/// handle may not read: refused.
#[test]
fn a_buffer_only_read_as_data_with_a_typed_stride_stays_refused() {
    let _serial = serial();
    for stride in [4, 2, 0] {
        assert_eq!(
            refusal(&records_only(stride)),
            "a descriptor read only as data",
            "{stride}"
        );
    }
}

const V_BFE_U32: u32 = 0x148;
const V_ALIGNBIT_B32: u32 = 0x14E;

/// An image's descriptor (s[12:19]) used by an `image_load` of
/// dimension `dim` (1 2D, 5 2D array; layer 0) and read as numbers by
/// `read`, which leaves four results in v3-v6, stored at 16 × thread.
fn image_fields(dim: u32, read: &[u32]) -> NativeProgram {
    let mut w = vec![s_mov_b32(1, k(0))];
    w.extend(s_load(2, 8, 0, 0));
    w.extend(s_load(3, 12, 0, 0x20));
    w.push(vop1(V_MOV_B32, 2, k(0)));
    w.extend(mimg(IMAGE_LOAD, 7, 0, 12, 0, 1, dim, false));
    w.extend(read);
    w.push(vop2(V_LSHLREV_B32, 8, k(4), 0));
    for (r, at) in [(3, 0), (4, 4), (5, 8), (6, 12)] {
        w.extend(buffer_store_dword(r, 8, 8, at));
    }
    w.push(S_ENDPGM);
    program(
        &w,
        false,
        [2, 1, 1],
        vec![table(0, 0)],
        vec![
            used(ResourceType::Uav, 0, 4, 0),
            used(ResourceType::Srv, 8, 8, 0),
        ],
    )
}

/// The fields the console compiler reads sizes from, as it reads them:
/// the last level (dword 3 bits 16-19), the depth or last layer (dword 4,
/// masked by a register it moved 0xFFFF into), the width (bits 30-31 of
/// dword 1 and 0-13 of dword 2 by `v_alignbit_b32`), the height (dword 2
/// bits 14-29 by `v_bfe_u32`).
fn size_read() -> Vec<u32> {
    let mut read = vec![sop2(S_BFE_U32, 2, 15, LIT), 0x0004_0010];
    read.extend([s_mov_b32(0, LIT), 0xFFFF, sop2(S_AND_B32, 3, 16, 0)]);
    read.extend(vop3(V_ALIGNBIT_B32, 5, 14, 13, k(30)));
    read.extend([vop2(V_AND_B32, 5, LIT, 5), 0xFFFF]);
    read.extend(vop3(V_BFE_U32, 6, 14, k(14), k(16)));
    read.extend([vop1(V_MOV_B32, 3, 2), vop1(V_MOV_B32, 4, 3)]);
    read
}

#[test]
fn image_size_fields_read_as_the_compiler_reads_them() {
    let _serial = serial();
    let shader = translate(&image_fields(5, &size_read())).unwrap();
    let texels: Vec<[f32; 4]> = (0..96).map(|i| [i as f32, 0.0, 0.0, 1.0]).collect();
    let views = [View::RawUav(8), View::Layers(8, 4, 3, &texels)];
    run(&shader, &views, 0, |label, out| {
        assert_eq!(out, [0, 2, 7, 3, 0, 2, 7, 3], "{label}");
    });
    // The last level, with levels and no layers.
    let mut read = vec![sop2(S_BFE_U32, 2, 15, LIT), 0x0004_0010];
    read.extend([vop1(V_MOV_B32, 3, 2), vop1(V_MOV_B32, 4, k(0))]);
    read.extend([vop1(V_MOV_B32, 5, k(0)), vop1(V_MOV_B32, 6, k(0))]);
    let shader = translate(&image_fields(1, &read)).unwrap();
    let views = [View::RawUav(8), View::Levels(8, 4, 3, &texels[..42])];
    run(&shader, &views, 0, |label, out| {
        assert_eq!(out, [2, 0, 0, 0, 2, 0, 0, 0], "{label}");
    });
}

/// The rest of those dwords (format, swizzle, tiling, the address's high
/// bits, a 2D view's depth field) has no host value: reading it stays
/// refused.
#[test]
fn other_bits_of_image_size_dwords_stay_refused() {
    let _serial = serial();
    let reads: [(u32, &[u32]); 5] = [
        (1, &[vop1(V_MOV_B32, 3, 13)]),
        (
            1,
            &[sop2(S_AND_B32, 2, 15, LIT), 0xFFF, vop1(V_MOV_B32, 3, 2)],
        ),
        (
            5,
            &[
                s_mov_b32(0, LIT),
                0x1_FFFF,
                sop2(S_AND_B32, 2, 16, 0),
                vop1(V_MOV_B32, 3, 2),
            ],
        ),
        (1, &[sop2(S_LSHR_B32, 2, 14, k(31)), vop1(V_MOV_B32, 3, 2)]),
        (1, &size_read()),
    ];
    for (dim, read) in reads {
        assert_eq!(
            refusal(&image_fields(dim, read)),
            "a descriptor's address or format read as data",
            "{dim}, {read:x?}"
        );
    }
}

/// [`leftover`] with an image's second descriptor dword, known only in
/// its width bits, as the leftover in s13 (the data from a raw buffer).
fn leftover_image_dword(flag: u16) -> NativeProgram {
    let mut w = vec![s_mov_b32(1, k(0))];
    w.extend(s_load(2, 8, 0, 0));
    w.extend(s_load(3, 12, 0, 0x20));
    w.extend(s_load(2, 20, 0, 0x40));
    w.extend(image_2d(IMAGE_LOAD, 7, 0, 12, 0));
    w.push(vop1(V_MOV_B32, 2, flag));
    w.push(s_cmp_eq_u32(2, k(0)));
    w.push(s_cbranch_scc1(4));
    // s_buffer_load_dword s13, s[20:23], 0
    w.extend([0xF420_0000 | 13 << 6 | 20 >> 1, 125 << 25]);
    w.push(vop1(V_MOV_B32, 2, INLINE_MINUS_1));
    w.push(s_branch(1));
    w.push(S_NOP);
    w.push(vopc(V_CMP_EQ_U32, INLINE_MINUS_1, 2));
    w.push(sop2(S_ANDN2_B32, VCC_LO as u8, EXEC_LO, VCC_LO));
    w.push(vop1(V_MOV_B32, 3, k(7)));
    w.push(s_cbranch_vccnz(1));
    w.push(vop1(V_MOV_B32, 3, 13));
    w.push(vop2(V_LSHLREV_B32, 5, k(2), 0));
    w.extend(buffer_store_dword(3, 5, 8, 0));
    w.push(S_ENDPGM);
    let raw = UsedResource {
        stride: 1,
        ..used(ResourceType::Srv, 16, 4, 0)
    };
    program(
        &w,
        false,
        [2, 1, 1],
        vec![table(0, 0), constant_at_s2()],
        vec![
            used(ResourceType::Uav, 0, 4, 0),
            used(ResourceType::Srv, 8, 8, 0),
            raw,
        ],
    )
}

/// A dword known only in part is a leftover like one not known at all.
#[test]
fn a_leftover_image_dword_no_lane_reads_joins_a_stand_in() {
    let _serial = serial();
    let shader = translate(&leftover_image_dword(k(0))).unwrap();
    let texels: Vec<[f32; 4]> = vec![[0.0; 4]; 8];
    let data = [0x1234_5678, 1, 2, 3];
    let views = [
        View::RawUav(2),
        View::Texture(4, 2, &texels),
        View::Buffer {
            data: &data,
            stride: super::host::Stride::Raw,
        },
    ];
    for constant in [0, 1] {
        run(&shader, &views, constant, |label, out| {
            let want = if constant == 0 { 7 } else { 0x1234_5678 };
            assert_eq!(out, [want; 2], "{label}, {constant}");
        });
    }
    assert_eq!(
        refusal(&leftover_image_dword(INLINE_MINUS_1)),
        "a descriptor's address or format read as data"
    );
}
