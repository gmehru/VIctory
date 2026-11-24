//! Presented frames written to PNG files, to see what reaches the screen:
//! with `VICTORY_CAPTURE=<folder>`, every frame whose number is a power of
//! two (the frames the log reports) is saved as `frame-<n>.png`, at half
//! size, and so are the frames [`listed`] by number or by how far they come
//! after the first world frame. Only in builds with diagnostics.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// The folder frames go to, if capturing.
pub(crate) fn folder() -> Option<&'static PathBuf> {
    static FOLDER: OnceLock<Option<PathBuf>> = OnceLock::new();
    FOLDER
        .get_or_init(|| std::env::var_os("VICTORY_CAPTURE").map(PathBuf::from))
        .as_ref()
}

/// Frames listed in `VICTORY_CAPTURE_FRAMES` (comma-separated), and those
/// `VICTORY_CAPTURE_AFTER` (comma-separated offsets) puts after the first
/// world frame ([`presenting`]): saved too, and their draws traced at
/// debug level.
pub(crate) fn listed(n: u64) -> bool {
    static LIST: OnceLock<Vec<u64>> = OnceLock::new();
    LIST.get_or_init(|| numbers("VICTORY_CAPTURE_FRAMES")).contains(&n)
        || after_world(n, WORLD.load(Relaxed), offsets())
}

/// The decimal numbers in variable `name`, comma-separated; what does not
/// read as one is left out.
fn numbers(name: &str) -> Vec<u64> {
    let list = std::env::var(name).unwrap_or_default();
    list.split(',').filter_map(|f| f.trim().parse().ok()).collect()
}

/// `VICTORY_CAPTURE_AFTER`'s offsets.
fn offsets() -> &'static [u64] {
    static OFFSETS: OnceLock<Vec<u64>> = OnceLock::new();
    OFFSETS.get_or_init(|| numbers("VICTORY_CAPTURE_AFTER"))
}

/// The first world frame, once seen (0 before).
static WORLD: AtomicU64 = AtomicU64::new(0);

/// Draws recorded between two presents from which the second is a world
/// frame: menu and loading frames record a dozen or so, the world's
/// thousands (its first frames that many even while most are dropped for
/// pipelines being made).
const WORLD_DRAWS: u64 = 200;

/// Whether frame `n` is one of `offsets` after the first world frame
/// `world` (0: none yet).
fn after_world(n: u64, world: u64, offsets: &[u64]) -> bool {
    world != 0 && n >= world && offsets.contains(&(n - world))
}

/// At the present of frame `frame`, before anything of it is saved: with
/// `VICTORY_CAPTURE_AFTER`, the first frame since whose previous present at
/// least [`WORLD_DRAWS`] draws were recorded is the first world frame. Its
/// own targets were recorded before that was known, so offset 0 saves its
/// picture only.
pub(crate) fn presenting(frame: u64) {
    static LAST: AtomicU64 = AtomicU64::new(0);
    if folder().is_none() || offsets().is_empty() {
        return;
    }
    let now = crate::list::recorded_draws();
    let draws = now.saturating_sub(LAST.swap(now, Relaxed));
    // The first present has no previous one: loading's draws come before.
    if frame > 1 && draws >= WORLD_DRAWS && WORLD.compare_exchange(0, frame, Relaxed, Relaxed).is_ok() {
        tracing::info!(
            frame,
            draws,
            offsets = ?offsets(),
            "first world frame: VICTORY_CAPTURE_AFTER counts from here"
        );
    }
}

/// Whether frame `n` is saved.
pub(crate) fn wanted(n: u64) -> bool {
    folder().is_some() && (n.is_power_of_two() || listed(n))
}

/// Saves 8-bit RGBA or BGRA rows (`pitch` bytes apart) as frame `n`.
pub(crate) fn save(n: u64, width: u32, height: u32, pitch: usize, bgra: bool, rows: &[u8]) {
    save_as(&format!("frame-{n}"), width, height, pitch, bgra, rows);
}

/// Saves 8-bit RGBA or BGRA rows as `<name>.png`, at half size.
fn save_as(name: &str, width: u32, height: u32, pitch: usize, bgra: bool, rows: &[u8]) {
    let Some(folder) = folder() else { return };
    let (w, h) = ((width / 2).max(1), (height / 2).max(1));
    // Filter byte 0 per row, then RGB of every other pixel of every other row.
    let mut raw = Vec::with_capacity((w as usize * 3 + 1) * h as usize);
    for y in 0..h as usize {
        raw.push(0);
        let row = &rows[(2 * y).min(height as usize - 1) * pitch..];
        for x in 0..w as usize {
            let p = &row[(2 * x).min(width as usize - 1) * 4..][..3];
            if bgra {
                raw.extend([p[2], p[1], p[0]]);
            } else {
                raw.extend_from_slice(p);
            }
        }
    }
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = Vec::new();
    header.extend(w.to_be_bytes());
    header.extend(h.to_be_bytes());
    header.extend([8, 2, 0, 0, 0]); // 8-bit RGB
    chunk(&mut png, b"IHDR", &header);
    chunk(
        &mut png,
        b"IDAT",
        &miniz_oxide::deflate::compress_to_vec_zlib(&raw, 3),
    );
    chunk(&mut png, b"IEND", &[]);
    let path = folder.join(format!("{name}.png"));
    match std::fs::write(&path, png) {
        Ok(()) => tracing::info!(path = %path.display(), "frame saved"),
        Err(e) => tracing::warn!(error = %e, path = %path.display(), "frame not saved"),
    }
}

fn chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend((data.len() as u32).to_be_bytes());
    let start = png.len();
    png.extend(kind);
    png.extend(data);
    let crc = crc32(&png[start..]);
    png.extend(crc.to_be_bytes());
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

/// Records a copy of `texture` (in the copy-source state) into a
/// readback buffer.
pub(crate) fn read_back(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
) -> Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0;
    // SAFETY: a live texture; outputs for one subresource.
    unsafe {
        let desc = texture.GetDesc();
        device.GetCopyableFootprints(
            &desc,
            0,
            1,
            0,
            Some(&mut footprint),
            None,
            None,
            Some(&mut total),
        );
    }
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: total,
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
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; an open list; locations naming live resources.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
        let buffer = buffer?;
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(texture),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        Some((buffer, footprint))
    }
}

/// Saves a frame read back by [`read_back`] (8-bit color only).
pub(crate) fn save_read_back(
    frame: u64,
    buffer: &ID3D12Resource,
    footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) {
    let f = footprint.Footprint;
    let bgra = match f.Format {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => false,
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => true,
        other => {
            tracing::warn!(format = other.0, "frame not saved: not 8-bit color");
            return;
        }
    };
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished (every submission is waited for); the
    // mapping spans the footprint's rows.
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let len = f.RowPitch as usize * f.Height as usize;
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), len);
        save(frame, f.Width, f.Height, f.RowPitch as usize, bgra, rows);
        buffer.Unmap(0, None);
    }
}

/// With `VICTORY_DUMP_TARGETS` set: title textures the frames listed in
/// `VICTORY_CAPTURE_FRAMES` render to or write, in first-use order, saved
/// at that frame's present as `frame-<n>-<i>-<resource>-<w>x<h>-f<format>.png`
/// (what each holds at the end of the frame; formats normalized to show
/// structure: floats scaled by their largest value, depth by its range).
/// The host copies are held: the title may release a target before the
/// frame is presented.
static TARGETS: std::sync::Mutex<Vec<(usize, std::sync::Arc<crate::resource::HostTexture>)>> =
    std::sync::Mutex::new(Vec::new());

fn dump_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| folder().is_some() && std::env::var_os("VICTORY_DUMP_TARGETS").is_some())
}

/// Notes a title texture the frame being built renders to or writes.
pub(crate) fn note_target(resource: crate::object::This) {
    if resource.is_null() || !dump_on() || !listed(crate::queue::presented() + 1) {
        return;
    }
    let mut targets = TARGETS.lock().unwrap_or_else(|e| e.into_inner());
    if !targets.iter().any(|&(r, _)| r == resource as usize)
        && let Some(copy) = crate::resource::host_copy(resource)
    {
        targets.push((resource as usize, copy));
    }
}

/// How long one frame's dumps may take: they hold the title's render
/// thread in its present, and the title ends a frame that takes about 30 s.
const DUMP_BUDGET: std::time::Duration = std::time::Duration::from_secs(15);

/// Targets narrower or lower than this are not dumped (tile grids, lookup
/// tables, probes).
const DUMP_SMALLEST: u64 = 32;

/// At the present of frame `frame`: saves the noted targets, largest first
/// (the scene's before small passes'), until [`DUMP_BUDGET`] is spent; file
/// names keep the first-use order. Logs how many were saved, how long it
/// took and which one took longest, and which were left out for the budget.
pub(crate) fn dump_targets(frame: u64) {
    if !dump_on() {
        return;
    }
    let targets = std::mem::take(&mut *TARGETS.lock().unwrap_or_else(|e| e.into_inner()));
    let Some(host) = crate::host::host() else { return };
    let started = std::time::Instant::now();
    let mut order: Vec<(usize, D3D12_RESOURCE_DESC)> = targets
        .iter()
        // SAFETY: a live resource.
        .map(|(_, copy)| unsafe { copy.texture.GetDesc() })
        .enumerate()
        .collect();
    order.sort_by_key(|(i, d)| (std::cmp::Reverse(d.Width * u64::from(d.Height)), *i));
    let (mut dumped, mut not_read, mut not_2d, mut small) = (0, 0, 0, 0);
    let mut late = Vec::new();
    // The target that took longest (ms, its number in the file names).
    let mut slowest = (0, None);
    for (i, d) in order {
        if d.SampleDesc.Count > 1 || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
            not_2d += 1;
            continue;
        }
        if d.Width < DUMP_SMALLEST || u64::from(d.Height) < DUMP_SMALLEST {
            small += 1;
            continue;
        }
        let (resource, copy) = &targets[i];
        if started.elapsed() >= DUMP_BUDGET {
            late.push(format!("{i:02} {resource:#x} {}x{} f{}", d.Width, d.Height, d.Format.0));
            continue;
        }
        // Every submission has run; the dump's list moves it there and back.
        let state = copy.state.load(Relaxed);
        let resource = *resource as crate::object::This;
        let one = std::time::Instant::now();
        if dump_one(host, &copy.texture, state, frame, i, resource, &d) {
            dumped += 1;
        } else {
            not_read += 1;
        }
        let ms = one.elapsed().as_millis();
        if ms >= slowest.0 {
            slowest = (ms, Some(i));
        }
    }
    tracing::info!(
        frame,
        dumped,
        ms = started.elapsed().as_millis(),
        slowest_ms = slowest.0,
        slowest = ?slowest.1,
        not_read,
        not_2d,
        smaller_than_32 = small,
        past_the_budget = late.len(),
        "targets dumped"
    );
    if !late.is_empty() {
        tracing::warn!(
            frame,
            budget_s = DUMP_BUDGET.as_secs(),
            skipped = ?late,
            "targets not dumped: the frame's dump time ran out"
        );
    }
}

/// Saves one target; false if nothing was saved (no list, no read-back, a
/// format not handled).
fn dump_one(
    host: &crate::host::Host,
    texture: &ID3D12Resource,
    state: u32,
    frame: u64,
    i: usize,
    resource: crate::object::This,
    d: &D3D12_RESOURCE_DESC,
) -> bool {
    let Ok(list) = host.new_list() else { return false };
    let state = D3D12_RESOURCE_STATES(crate::host::legal_state(texture, state) as i32);
    let copy = D3D12_RESOURCE_STATE_COPY_SOURCE;
    // SAFETY: an open list of ours and a live texture.
    let read = unsafe {
        if state != copy {
            list.list
                .ResourceBarrier(&[crate::host::transition(texture, state, copy)]);
        }
        let read = read_back(&host.device, &list.list, texture);
        if state != copy {
            list.list
                .ResourceBarrier(&[crate::host::transition(texture, copy, state)]);
        }
        read
    };
    list.close();
    host.execute(&[&list]);
    let Some((buffer, footprint)) = read else { return false };
    let f = footprint.Footprint;
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished; the mapping spans the footprint's rows.
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return false;
        }
        let pitch = f.RowPitch as usize;
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), pitch * f.Height as usize);
        let rgba = to_rgba(f.Format.0 as u32, f.Width as usize, f.Height as usize, pitch, rows);
        if let Some(rgba) = &rgba {
            let name = format!(
                "frame-{frame}-{i:02}-{resource:?}-{}x{}-f{}",
                d.Width, d.Height, d.Format.0
            );
            save_as(&name, f.Width, f.Height, f.Width as usize * 4, false, rgba);
        } else {
            tracing::info!(format = f.Format.0, ?resource, "target not dumped: format");
        }
        buffer.Unmap(0, None);
        rgba.is_some()
    }
}

fn half(h: u16) -> f32 {
    let (sign, exp, man) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as f32);
    let v = match exp {
        0 => man * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    };
    if sign == 1 { -v } else { v }
}

/// A float packed in `bits` with a 5-bit exponent and `man` mantissa bits
/// (the 11- and 10-bit floats of R11G11B10).
fn small_float(bits: u32, man: u32) -> f32 {
    let (exp, m) = ((bits >> man) & 0x1f, (bits & ((1 << man) - 1)) as f32);
    match exp {
        0 => m / (1 << man) as f32 * 2f32.powi(-14),
        31 => f32::INFINITY,
        e => (1.0 + m / (1 << man) as f32) * 2f32.powi(e as i32 - 15),
    }
}

/// Rows of a host format as 8-bit RGBA: 8-bit color as is, wide integer
/// formats a colour per value ([`id_colour`]), everything else as channel
/// values scaled so the largest finite one is white (single channels:
/// lowest black, highest white, as grey). `None` for formats not handled.
fn to_rgba(format: u32, w: usize, h: usize, pitch: usize, rows: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(w * h * 4);
    let px = |x: usize, y: usize, n: usize| &rows[y * pitch + x * n..][..n];
    let u16_at = |b: &[u8], i: usize| u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
    let u32_at = |b: &[u8], i: usize| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap_or([0; 4]));
    // Integers, four or two channels of 32 or 16 bits (ids: a visibility
    // buffer's triangle and instance, say): a colour per value, zero black.
    // Scaled by the largest, all but the largest ids would look black.
    let integers = match format {
        3 | 4 => Some((4, 4)),
        12 | 14 => Some((4, 2)),
        17 | 18 => Some((2, 4)),
        _ => None,
    };
    if let Some((channels, bytes)) = integers {
        let (mut low, mut high, mut nonzero) = (u32::MAX, 0, 0u64);
        for y in 0..h {
            for x in 0..w {
                let p = px(x, y, channels * bytes);
                let v: [u32; 4] = std::array::from_fn(|c| match (c < channels, bytes) {
                    (false, _) => 0,
                    (true, 4) => u32_at(p, c),
                    (true, _) => u32::from(u16_at(p, c)),
                });
                for &c in &v[..channels] {
                    (low, high) = (low.min(c), high.max(c));
                }
                nonzero += u64::from(v != [0; 4]);
                out.extend(id_colour(v));
            }
        }
        tracing::info!(format, low, high, nonzero, shown = "a colour per value", "target values");
        return Some(out);
    }
    // Channel values as floats, then scaled.
    type Decode = Box<dyn Fn(&[u8]) -> [f32; 3]>;
    let (n, decode): (usize, Decode) = match format {
        27..=32 => {
            for y in 0..h {
                for x in 0..w {
                    let p = px(x, y, 4);
                    out.extend([p[0], p[1], p[2], 255]);
                }
            }
            return Some(out);
        }
        // B8G8R8A8 and B8G8R8X8 (89 is R10G10B10_XR_BIAS_A2, below).
        87 | 88 | 90..=93 => {
            for y in 0..h {
                for x in 0..w {
                    let p = px(x, y, 4);
                    out.extend([p[2], p[1], p[0], 255]);
                }
            }
            return Some(out);
        }
        23..=25 | 89 => (4, Box::new(move |b| {
            let v = u32_at(b, 0);
            [(v & 0x3ff) as f32, (v >> 10 & 0x3ff) as f32, (v >> 20 & 0x3ff) as f32]
        })),
        26 => (4, Box::new(move |b| {
            let v = u32_at(b, 0);
            [small_float(v & 0x7ff, 6), small_float(v >> 11 & 0x7ff, 6), small_float(v >> 22, 5)]
        })),
        9..=10 => (8, Box::new(move |b| [half(u16_at(b, 0)), half(u16_at(b, 1)), half(u16_at(b, 2))])),
        // R16G16B16A16_SNORM, signed (before the unsigned ones below).
        13 => (8, Box::new(move |b| [0, 1, 2].map(|i| f32::from(u16_at(b, i) as i16) / 32767.0))),
        // R16G16B16A16_UNORM (UINT and SINT are integers, above).
        11..=14 => (8, Box::new(move |b| [0, 1, 2].map(|i| f32::from(u16_at(b, i))))),
        33..=34 => (4, Box::new(move |b| [half(u16_at(b, 0)), half(u16_at(b, 1)), 0.0])),
        1..=2 => (16, Box::new(move |b| {
            [0, 1, 2].map(|i| f32::from_bits(u32_at(b, i)))
        })),
        15..=16 => (8, Box::new(move |b| [f32::from_bits(u32_at(b, 0)), f32::from_bits(u32_at(b, 1)), 0.0])),
        35..=38 => (4, Box::new(move |b| [f32::from(u16_at(b, 0)), f32::from(u16_at(b, 1)), 0.0])),
        48..=52 => (2, Box::new(move |b| [f32::from(b[0]), f32::from(b[1]), 0.0])),
        // Single channels (and depth planes): shown as grey.
        39..=43 => (4, Box::new(move |b| [f32::from_bits(u32_at(b, 0)); 3])),
        19..=22 => (8, Box::new(move |b| [f32::from_bits(u32_at(b, 0)); 3])),
        44..=47 => (4, Box::new(move |b| [(u32_at(b, 0) & 0xff_ffff) as f32; 3])),
        53..=54 => (2, Box::new(move |b| [half(u16_at(b, 0)); 3])),
        55..=59 => (2, Box::new(move |b| [f32::from(u16_at(b, 0)); 3])),
        60..=65 => (1, Box::new(move |b| [f32::from(b[0]); 3])),
        _ => return None,
    };
    let values: Vec<[f32; 3]> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| decode(px(x, y, n)))
        .collect();
    let grey = matches!(format, 19..=22 | 39..=47 | 53..=65);
    let finite = values.iter().flatten().copied().filter(|v| v.is_finite());
    let (low, high) = if grey {
        finite.fold((f32::MAX, f32::MIN), |(l, h), v| (l.min(v), h.max(v)))
    } else {
        (0.0, finite.fold(0.0f32, f32::max))
    };
    let range = (high - low).max(1e-20);
    tracing::info!(format, low, high, "target values");
    for v in values {
        let c = v.map(|c| (((c - low) / range).clamp(0.0, 1.0) * 255.0) as u8);
        out.extend([c[0], c[1], c[2], 255]);
    }
    Some(out)
}

/// The colour an integer pixel (ids) is shown in: black for zero, else a
/// bright one from a hash of the value, so neighbouring ids differ.
fn id_colour(v: [u32; 4]) -> [u8; 4] {
    if v == [0; 4] {
        return [0, 0, 0, 255];
    }
    let mut h = 0x811c_9dc5u32;
    for c in v {
        h = (h ^ c).wrapping_mul(0x0100_0193);
        h ^= h >> 15;
        h = h.wrapping_mul(0x2c1b_3c6d);
        h ^= h >> 12;
    }
    let [r, g, b, _] = h.to_le_bytes();
    [r | 0x40, g | 0x40, b | 0x40, 255]
}

/// Pixels per side of the grid [`probe`] reads.
const GRID: u32 = 4;
/// Bytes between the grid's pixels in the readback buffer (placement
/// alignment).
const SLOT: u64 = 512;

/// Records copies of a grid of pixels of `texture` (8-bit color, in the
/// copy-source state) into a readback buffer, to tell black frames.
pub(crate) fn probe(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
) -> Option<ID3D12Resource> {
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    if !matches!(d.Format.0, 27..=29 | 87 | 90 | 91) {
        return None;
    }
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: SLOT * u64::from(GRID * GRID),
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
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; an open list; locations naming live resources.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
        let buffer = buffer?;
        for i in 0..GRID * GRID {
            let (x, y) = (
                (2 * (i % GRID) + 1) * d.Width as u32 / (2 * GRID),
                (2 * (i / GRID) + 1) * d.Height / (2 * GRID),
            );
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&buffer),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: SLOT * u64::from(i),
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: d.Format,
                            Width: 1,
                            Height: 1,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let pixel = D3D12_BOX {
                left: x,
                top: y,
                front: 0,
                right: x + 1,
                bottom: y + 1,
                back: 1,
            };
            list.CopyTextureRegion(&to, 0, 0, 0, &from, Some(&pixel));
        }
        Some(buffer)
    }
}

/// Whether every pixel [`probe`] read is (near) black.
pub(crate) fn black(buffer: &ID3D12Resource) -> bool {
    let mut data = std::ptr::null_mut();
    // SAFETY: the copies finished (every submission is waited for).
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return false;
        }
        let bytes =
            std::slice::from_raw_parts(data.cast::<u8>(), (SLOT * u64::from(GRID * GRID)) as usize);
        let level = bytes
            .chunks(SLOT as usize)
            .flat_map(|pixel| pixel[..3].iter().copied())
            .max()
            .unwrap_or(0);
        buffer.Unmap(0, None);
        let mut levels = LEVELS.lock().unwrap_or_else(|e| e.into_inner());
        if levels.len() == 12 {
            levels.pop_front();
        }
        levels.push_back(level);
        level < 8
    }
}

/// The brightest probed channel of the last frames, oldest first: a fade
/// darkens over several, a glitch drops at once.
static LEVELS: std::sync::Mutex<std::collections::VecDeque<u8>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());

pub(crate) fn recent_levels() -> Vec<u8> {
    let levels = LEVELS.lock().unwrap_or_else(|e| e.into_inner());
    levels.iter().copied().collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn crc_matches_the_png_check_value() {
        assert_eq!(super::crc32(b"IEND"), 0xae42_6082);
    }

    /// R16G16B16A16_UNORM is integers, not half floats; 89 is 10:10:10:2,
    /// not B8G8R8A8.
    #[test]
    fn unorm16_and_xr_bias_targets_decode_as_their_formats() {
        // Red 0x8000 (a half-float -0), green 0x4000, blue 0.
        let unorm16 = [0x00, 0x80, 0x00, 0x40, 0, 0, 0xff, 0xff];
        assert_eq!(super::to_rgba(11, 1, 1, 8, &unorm16), Some(vec![255, 127, 0, 255]));
        // Red 0x3ff, green 0x1ff, blue 0, alpha 3.
        let xr = (0x3ffu32 | 0x1ff << 10 | 3 << 30).to_le_bytes();
        assert_eq!(super::to_rgba(89, 1, 1, 4, &xr), Some(vec![255, 127, 0, 255]));
        // B8G8R8A8 still swaps to RGBA.
        assert_eq!(super::to_rgba(87, 1, 1, 4, &[1, 2, 3, 4]), Some(vec![3, 2, 1, 255]));
    }

    /// Two-channel 32-bit integer targets (a visibility buffer's ids):
    /// cleared pixels black, each id a colour of its own.
    #[test]
    fn integer_targets_get_a_colour_per_value() {
        const R32G32_UINT: u32 = 17;
        let pixels: [[u32; 2]; 4] = [[0, 0], [5, 7], [5, 7], [5, 8]];
        let rows: Vec<u8> = pixels.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        let rgba = super::to_rgba(R32G32_UINT, 4, 1, rows.len(), &rows).unwrap();
        let p: Vec<&[u8]> = rgba.chunks(4).collect();
        assert_eq!(p[0], [0, 0, 0, 255]);
        assert_ne!(p[1], p[0]);
        assert_eq!(p[1], p[2]);
        assert_ne!(p[1], p[3]);
        // Sixteen-bit, four channels: the same.
        const R16G16B16A16_UINT: u32 = 12;
        let rows: Vec<u8> = [[0u16; 4], [1, 2, 3, 4]].iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        let rgba = super::to_rgba(R16G16B16A16_UINT, 2, 1, rows.len(), &rows).unwrap();
        assert_eq!(rgba[..4], [0, 0, 0, 255]);
        assert_ne!(rgba[4..], [0, 0, 0, 255]);
    }

    /// `VICTORY_CAPTURE_AFTER` offsets count from the first world frame,
    /// once there is one.
    #[test]
    fn frames_count_from_the_first_world_frame() {
        let offsets = [0, 3, 10];
        assert!(!super::after_world(3, 0, &offsets));
        assert!(super::after_world(100, 100, &offsets));
        assert!(super::after_world(103, 100, &offsets));
        assert!(super::after_world(110, 100, &offsets));
        assert!(!super::after_world(104, 100, &offsets));
        assert!(!super::after_world(97, 100, &offsets));
    }
}
