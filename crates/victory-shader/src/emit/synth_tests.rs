//! Hand-assembled GFX10 compute programs through the whole translator:
//! validated by `dxil.dll` when there is one, and run on the host devices
//! (hardware and WARP) against results worked out from the instruction
//! set, for what the differential tests cannot reach without the console
//! compiler: spilled SGPRs, descriptor loads read as numbers, derivatives.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::dxbc::{Container, FourCC};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
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
    let res = resource::track(&p, &native.layout, native.stage)?;
    let io = io::raise(&p, native)?;
    let st = structure::structure(&p)?;
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
    let res = resource::track(&p, &native.layout, native.stage).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let st = structure::structure(&p).unwrap();
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

/// Four constants copied into the VGPRs may be an immediate sampler, which
/// nothing stands in for: refused, as is a program that computes with a
/// held sampler's dword.
#[test]
fn held_samplers_that_cannot_be_followed_are_refused() {
    let _serial = serial();
    let mut immediate = Held::new();
    for d in 0..4u8 {
        immediate.0.push(vop1(V_MOV_B32, 4 + d, k(u16::from(d))));
    }
    immediate.sample();
    immediate.store(0);
    assert_eq!(
        refusal(&immediate.program()),
        "a sampler held in VGPRs that may be an immediate one"
    );

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
    computed.copy(1);
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
    h.copy(1);
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
/// first active host lane's: refused.
#[test]
fn a_held_sampler_read_back_outside_a_waterfall_is_refused() {
    let _serial = serial();
    let mut h = Held::new();
    h.copy(1);
    for d in 0..4u8 {
        h.0.push(vop1(V_READFIRSTLANE_B32, 28 + d, vgpr(4 + d)));
    }
    h.0.extend(image_2d(IMAGE_SAMPLE_LZ, 8, 1, 12, 28));
    h.store(0);
    assert_eq!(
        refusal(&h.program()),
        "a sampler held in VGPRs read back outside a waterfall loop"
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
        h.copy(1);
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
        h.copy(1);
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
/// lane is every lane's only if every lane ran the copy.
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
        for d in 0..8u8 {
            h.0.push(vop1(V_READFIRSTLANE_B32, 32 + d, vgpr(10 + d)));
        }
        h.0.extend(s_load(2, 20, 4, 16));
        h.0.extend(image_2d(IMAGE_SAMPLE_LZ, 8, 1, 32, 20));
        h.store(0);
        h.program()
    };
    assert_eq!(
        refusal(&build(true)),
        "a descriptor copied into VGPRs in some lanes only"
    );
    let shader = translate(&build(false)).unwrap();
    run_held(&shader, 0, |label, out| {
        assert_eq!(out[..8], [WRAP; 8], "{label}");
    });
}
