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
    // The folder need not exist yet: a run lost its first capture to that.
    if let Err(e) = std::fs::create_dir_all(folder) {
        tracing::warn!(error = %e, folder = %folder.display(), "capture folder not made");
    }
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
    read_back_sub(device, list, texture, 0)
}

/// As [`read_back`], for subresource `sub` (a depth-stencil texture's
/// stencil plane, say).
fn read_back_sub(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
    sub: u32,
) -> Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0;
    // SAFETY: a live texture; outputs for one subresource.
    unsafe {
        let desc = texture.GetDesc();
        device.GetCopyableFootprints(
            &desc,
            sub,
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
                SubresourceIndex: sub,
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

pub(crate) fn dump_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| folder().is_some() && std::env::var_os("VICTORY_DUMP_TARGETS").is_some())
}

/// Whether a listed frame was presented since the last call (the frame
/// statistics then print every drop reason, not the first 40).
pub(crate) fn take_listed() -> bool {
    LISTED_SEEN.swap(false, Relaxed)
}
static LISTED_SEEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

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
/// Each target not dumped for its kind or size, or not read, is logged
/// with its description; float targets under [`DUMP_SMALLEST`] get their
/// values logged ([`small_values`]); depth-stencil targets get their
/// stencil plane saved too (`…-stencil.png`).
pub(crate) fn dump_targets(frame: u64) {
    // Called for listed frames only.
    LISTED_SEEN.store(true, Relaxed);
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
        let (resource, copy) = &targets[i];
        let resource = *resource as crate::object::This;
        let skipped = |why: &str| {
            tracing::info!(
                frame,
                index = i,
                ?resource,
                why,
                dimension = d.Dimension.0,
                width = d.Width,
                height = d.Height,
                depth_or_array = d.DepthOrArraySize,
                format = d.Format.0,
                mips = d.MipLevels,
                samples = d.SampleDesc.Count,
                "listed frame: target not dumped"
            );
        };
        if d.SampleDesc.Count > 1 || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
            not_2d += 1;
            skipped("not_2d");
            continue;
        }
        // Every submission has run; the dump's list moves it there and back.
        let state = copy.state.load(Relaxed);
        if d.Width < DUMP_SMALLEST || u64::from(d.Height) < DUMP_SMALLEST {
            small += 1;
            skipped("smaller_than_32");
            if started.elapsed() < DUMP_BUDGET {
                small_values(host, &copy.texture, state, frame, i, resource, &d);
            }
            continue;
        }
        if started.elapsed() >= DUMP_BUDGET {
            late.push(format!("{i:02} {resource:?} {}x{} f{}", d.Width, d.Height, d.Format.0));
            continue;
        }
        let one = std::time::Instant::now();
        if dump_one(host, &copy.texture, state, frame, i, resource, &d) {
            dumped += 1;
        } else {
            not_read += 1;
            skipped("not_read");
        }
        // The two-plane resource formats: R32G8X24_TYPELESS,
        // D32_FLOAT_S8X24_UINT, R24G8_TYPELESS, D24_UNORM_S8_UINT.
        if matches!(d.Format.0, 19 | 20 | 44 | 45) {
            dump_stencil(host, &copy.texture, state, frame, i, resource, &d);
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
    let Some((f, rows)) = read_rows(host, texture, state, 0) else { return false };
    let pitch = f.RowPitch as usize;
    let rgba = to_rgba(f.Format.0 as u32, f.Width as usize, f.Height as usize, pitch, &rows);
    if let Some(rgba) = &rgba {
        let name = format!(
            "frame-{frame}-{i:02}-{resource:?}-{}x{}-f{}",
            d.Width, d.Height, d.Format.0
        );
        save_as(&name, f.Width, f.Height, f.Width as usize * 4, false, rgba);
    } else {
        tracing::info!(format = f.Format.0, ?resource, "target not dumped: format");
    }
    rgba.is_some()
}

/// Subresource `sub` of `texture` (in host state `state`, moved to the
/// copy source and back), read back: its footprint and rows.
fn read_rows(
    host: &crate::host::Host,
    texture: &ID3D12Resource,
    state: u32,
    sub: u32,
) -> Option<(D3D12_SUBRESOURCE_FOOTPRINT, Vec<u8>)> {
    let list = host.new_list().ok()?;
    let state = D3D12_RESOURCE_STATES(crate::host::legal_state(texture, state) as i32);
    let copy = D3D12_RESOURCE_STATE_COPY_SOURCE;
    // SAFETY: an open list of ours and a live texture.
    let read = unsafe {
        if state != copy {
            list.list
                .ResourceBarrier(&[crate::host::transition(texture, state, copy)]);
        }
        let read = read_back_sub(&host.device, &list.list, texture, sub);
        if state != copy {
            list.list
                .ResourceBarrier(&[crate::host::transition(texture, copy, state)]);
        }
        read
    };
    list.close();
    host.execute(&[&list]);
    let (buffer, footprint) = read?;
    let f = footprint.Footprint;
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished; the mapping spans the footprint's rows.
    unsafe {
        buffer.Map(0, None, Some(&mut data)).ok()?;
        let len = f.RowPitch as usize * f.Height as usize;
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), len).to_vec();
        buffer.Unmap(0, None);
        Some((f, rows))
    }
}

/// A depth-stencil target's stencil plane (plane slice 1, first mip and
/// array slice), saved beside its depth as `…-stencil.png` (grey from the
/// lowest value to the highest, as the 'target values' line gives them),
/// and its values counted on a 'stencil values' line.
fn dump_stencil(
    host: &crate::host::Host,
    texture: &ID3D12Resource,
    state: u32,
    frame: u64,
    i: usize,
    resource: crate::object::This,
    d: &D3D12_RESOURCE_DESC,
) {
    let sub = u32::from(d.MipLevels.max(1)) * u32::from(d.DepthOrArraySize.max(1));
    let Some((f, rows)) = read_rows(host, texture, state, sub) else {
        tracing::info!(frame, index = i, ?resource, "listed frame: stencil not read");
        return;
    };
    let (w, h, pitch) = (f.Width as usize, f.Height as usize, f.RowPitch as usize);
    // One byte a pixel (R8 footprints, the only stencil planes there are).
    if matches!(f.Format.0, 60..=65) {
        let counts = stencil_counts(w, h, pitch, &rows);
        tracing::info!(
            frame,
            index = i,
            ?resource,
            width = w,
            height = h,
            footprint = f.Format.0,
            counts = ?counts,
            "listed frame: stencil values (value, pixels; most first)"
        );
    }
    // A 24-bit depth plane's raw values counted too: a pass that tests
    // depth for equality (the tiled lighting against the material depth)
    // can then be compared with what the plane really holds.
    if matches!(d.Format.0, 44 | 45)
        && let Some((f0, rows0)) = read_rows(host, texture, state, 0)
        && matches!(f0.Format.0, 39..=47)
    {
        let (w0, h0, p0) = (f0.Width as usize, f0.Height as usize, f0.RowPitch as usize);
        let mut seen = std::collections::HashMap::<u32, u32>::new();
        for y in 0..h0 {
            for x in 0..w0 {
                let o = y * p0 + x * 4;
                if let Some(b) = rows0.get(o..o + 4) {
                    let v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & 0x00ff_ffff;
                    *seen.entry(v).or_default() += 1;
                }
            }
        }
        let distinct = seen.len();
        let mut counts: Vec<(u32, u32)> = seen.into_iter().collect();
        counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        counts.truncate(48);
        tracing::info!(
            frame,
            index = i,
            ?resource,
            distinct,
            counts = ?counts,
            "listed frame: 24-bit depth values (raw value, pixels; most first)"
        );
    }
    let Some(rgba) = to_rgba(f.Format.0 as u32, w, h, pitch, &rows) else {
        tracing::info!(frame, index = i, ?resource, footprint = f.Format.0, "listed frame: stencil not dumped: format");
        return;
    };
    let name = format!(
        "frame-{frame}-{i:02}-{resource:?}-{}x{}-f{}-stencil",
        d.Width, d.Height, d.Format.0
    );
    save_as(&name, f.Width, f.Height, w * 4, false, &rgba);
}

/// How many pixels of a one-byte plane hold each value, most first.
fn stencil_counts(w: usize, h: usize, pitch: usize, rows: &[u8]) -> Vec<(u8, u64)> {
    let mut counts = [0u64; 256];
    for y in 0..h {
        for &v in &rows[y * pitch..][..w] {
            counts[usize::from(v)] += 1;
        }
    }
    let mut out: Vec<(u8, u64)> = (0..=255u8)
        .zip(counts)
        .filter(|&(_, n)| n > 0)
        .collect();
    out.sort_by_key(|&(v, n)| (std::cmp::Reverse(n), v));
    out
}

/// Pixels whose values a small float target gets logged with, at most
/// (16 x 16: exposure, luminance and tile grids).
const SMALL_VALUES: usize = 256;

/// A float target under [`DUMP_SMALLEST`] (first mip and slice): its
/// values on a line, pixel by pixel in rows (at most [`SMALL_VALUES`]
/// pixels; channels as the format has them), with the counts of
/// [`float_counts`]. Other formats are left out.
fn small_values(
    host: &crate::host::Host,
    texture: &ID3D12Resource,
    state: u32,
    frame: u64,
    i: usize,
    resource: crate::object::This,
    d: &D3D12_RESOURCE_DESC,
) {
    if float_pixel(d.Format.0 as u32).is_none() {
        return;
    }
    let Some((f, rows)) = read_rows(host, texture, state, 0) else {
        tracing::info!(frame, index = i, ?resource, "listed frame: small target not read");
        return;
    };
    let format = f.Format.0 as u32;
    let Some(bytes) = float_pixel(format) else {
        tracing::info!(frame, index = i, ?resource, footprint = format, "listed frame: small target not read: format");
        return;
    };
    let (w, h, pitch) = (f.Width as usize, f.Height as usize, f.RowPitch as usize);
    let pixels: Vec<Vec<f32>> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| float_channels(format, &rows[y * pitch + x * bytes..][..bytes]))
        .collect();
    // Alpha included.
    let counts = float_counts(&pixels);
    let text = if w * h <= SMALL_VALUES {
        pixels
            .chunks(w.max(1))
            .map(|row| {
                row.iter()
                    .map(|p| {
                        let p: Vec<String> = p.iter().map(|v| format!("{v}")).collect();
                        format!("({})", p.join(" "))
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect::<Vec<_>>()
            .join(" | ")
    } else {
        format!("({} pixels: more than {SMALL_VALUES}, not listed)", w * h)
    };
    tracing::info!(
        frame,
        index = i,
        ?resource,
        width = d.Width,
        height = d.Height,
        format = d.Format.0,
        footprint = format,
        negative = counts.negative,
        nan = counts.nan,
        infinite = counts.infinite,
        nonzero = counts.nonzero,
        values = %text,
        "listed frame: small target values (rows split by |)"
    );
}

/// Bytes per pixel of a float format whose channels [`float_channels`]
/// reads (typeless families read as floats, as the dumps show them).
fn float_pixel(format: u32) -> Option<usize> {
    Some(match format {
        1..=2 => 16,
        9..=10 | 15..=16 => 8,
        26 | 33..=34 | 39..=41 => 4,
        53..=54 => 2,
        _ => return None,
    })
}

/// A pixel of a format [`float_pixel`] takes, every channel it has.
fn float_channels(format: u32, b: &[u8]) -> Vec<f32> {
    let u16_at = |i: usize| u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
    let f32_at = |i: usize| f32::from_bits(u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]));
    match format {
        1..=2 => (0..4).map(f32_at).collect(),
        9..=10 => (0..4).map(|i| half(u16_at(i))).collect(),
        15..=16 => (0..2).map(f32_at).collect(),
        26 => {
            let v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            vec![small_float(v & 0x7ff, 6), small_float(v >> 11 & 0x7ff, 6), small_float(v >> 22, 5)]
        }
        33..=34 => (0..2).map(|i| half(u16_at(i))).collect(),
        39..=41 => vec![f32_at(0)],
        53..=54 => vec![half(u16_at(0))],
        _ => Vec::new(),
    }
}

fn half(h: u16) -> f32 {
    let (sign, exp, man) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as f32);
    let v = match exp {
        0 => man * 2f32.powi(-24),
        31 if man != 0.0 => f32::NAN,
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
        31 if m != 0.0 => f32::NAN,
        31 => f32::INFINITY,
        e => (1.0 + m / (1 << man) as f32) * 2f32.powi(e as i32 - 15),
    }
}

/// Pixels of a float target with a channel below zero (-inf included), a
/// NaN, an infinity, a value other than zero (NaN included): what its
/// lowest and highest finite values do not tell ('high=0' holds for a
/// target of zeros, of negatives and of NaN alike).
#[derive(Debug, Default, PartialEq)]
struct FloatCounts {
    negative: u64,
    nan: u64,
    infinite: u64,
    nonzero: u64,
}

fn float_counts<P: AsRef<[f32]>>(pixels: &[P]) -> FloatCounts {
    let mut n = FloatCounts::default();
    for p in pixels {
        let p = p.as_ref();
        n.negative += u64::from(p.iter().any(|&c| c < 0.0));
        n.nan += u64::from(p.iter().any(|c| c.is_nan()));
        n.infinite += u64::from(p.iter().any(|c| c.is_infinite()));
        n.nonzero += u64::from(p.iter().any(|&c| c != 0.0));
    }
    n
}

/// How an integer format is shown ([`id_colour`]): channels, bytes a
/// channel, signed.
fn integer_layout(format: u32) -> Option<(usize, usize, bool)> {
    Some(match format {
        3 => (4, 4, false),
        4 => (4, 4, true),
        12 => (4, 2, false),
        14 => (4, 2, true),
        17 => (2, 4, false),
        18 => (2, 4, true),
        // R32_UINT and R32_SINT (R32_TYPELESS, D32_FLOAT and R32_FLOAT,
        // 39 to 41, are floats: depth planes come as those).
        42 => (1, 4, false),
        43 => (1, 4, true),
        _ => return None,
    })
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
    if let Some((channels, bytes, signed)) = integer_layout(format) {
        // Lowest and highest channel values (signed formats as such).
        let (mut low, mut high, mut nonzero) = (i64::MAX, i64::MIN, 0u64);
        for y in 0..h {
            for x in 0..w {
                let p = px(x, y, channels * bytes);
                let v: [u32; 4] = std::array::from_fn(|c| match (c < channels, bytes) {
                    (false, _) => 0,
                    (true, 4) => u32_at(p, c),
                    (true, _) => u32::from(u16_at(p, c)),
                });
                for &c in &v[..channels] {
                    let c = match (signed, bytes) {
                        (false, _) => i64::from(c),
                        (true, 4) => i64::from(c as i32),
                        (true, _) => i64::from(c as u16 as i16),
                    };
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
        // Single channels (and depth planes): shown as grey (42 and 43,
        // R32_UINT and R32_SINT, are integers, above).
        39..=41 => (4, Box::new(move |b| [f32::from_bits(u32_at(b, 0)); 3])),
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
    let grey = matches!(format, 19..=22 | 39..=41 | 44..=47 | 53..=65);
    let finite = values.iter().flatten().copied().filter(|v| v.is_finite());
    let (low, high) = if grey {
        finite.fold((f32::MAX, f32::MIN), |(l, h), v| (l.min(v), h.max(v)))
    } else {
        (0.0, finite.fold(0.0f32, f32::max))
    };
    let range = (high - low).max(1e-20);
    let n = float_counts(&values);
    tracing::info!(
        format,
        low,
        high,
        negative = n.negative,
        nan = n.nan,
        infinite = n.infinite,
        nonzero = n.nonzero,
        "target values"
    );
    for v in values {
        // NaN white, as +inf: seen, not taken for zero.
        let c = v.map(|c| if c.is_nan() { 255 } else { (((c - low) / range).clamp(0.0, 1.0) * 255.0) as u8 });
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

    /// R32_UINT and R32_SINT are integers (a colour per value, zero
    /// black), not floats; R32_TYPELESS, D32_FLOAT and R32_FLOAT stay
    /// floats (grey from lowest to highest).
    #[test]
    fn r32_integer_targets_decode_as_integers() {
        assert_eq!(super::integer_layout(42), Some((1, 4, false)));
        assert_eq!(super::integer_layout(43), Some((1, 4, true)));
        for float in 39..=41 {
            assert_eq!(super::integer_layout(float), None);
        }
        let rows: Vec<u8> = [0u32, 7, 7, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
        let rgba = super::to_rgba(42, 4, 1, rows.len(), &rows).unwrap();
        let p: Vec<&[u8]> = rgba.chunks(4).collect();
        assert_eq!(p[0], [0, 0, 0, 255]);
        assert_eq!(p[1], super::id_colour([7, 0, 0, 0]));
        assert_eq!(p[1], p[2]);
        assert_ne!(p[1], p[3]);
        let rows: Vec<u8> = [0i32, -1].iter().flat_map(|v| v.to_le_bytes()).collect();
        let rgba = super::to_rgba(43, 2, 1, rows.len(), &rows).unwrap();
        assert_eq!(rgba[4..], super::id_colour([u32::MAX, 0, 0, 0]));
        // R32_FLOAT: 0.0 black, 1.0 white.
        let rows: Vec<u8> = [0.0f32, 1.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let rgba = super::to_rgba(41, 2, 1, rows.len(), &rows).unwrap();
        assert_eq!(rgba, [0, 0, 0, 255, 255, 255, 255, 255]);
    }

    /// Negative, NaN and infinite values are counted by pixel; half and
    /// small floats with a mantissa at the top exponent are NaN, shown
    /// white.
    #[test]
    fn float_targets_count_negatives_nan_and_infinities() {
        let (inf, nan) = (f32::INFINITY, f32::NAN);
        let pixels = [
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [nan, 0.0, 0.0],
            [inf, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [-inf, 0.0, 0.0],
            [-0.0, 0.0, 0.0],
        ];
        let n = super::float_counts(&pixels);
        assert_eq!(
            n,
            super::FloatCounts {
                negative: 2,
                nan: 1,
                infinite: 2,
                nonzero: 5
            }
        );
        assert!(super::half(0x7e00).is_nan());
        assert!(super::half(0xfe00).is_nan());
        assert_eq!(super::half(0x7c00), inf);
        assert_eq!(super::half(0xfc00), -inf);
        assert_eq!(super::half(0xbc00), -1.0);
        assert!(super::small_float(0x7c0 | 1, 6).is_nan());
        assert_eq!(super::small_float(0x7c0, 6), inf);
        // R16G16B16A16_FLOAT: a NaN pixel white, a negative one black.
        let rows: Vec<u8> = [[0x7e00u16, 0, 0, 0], [0xbc00, 0xbc00, 0xbc00, 0], [0x3c00, 0x3c00, 0x3c00, 0]]
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let rgba = super::to_rgba(10, 3, 1, rows.len(), &rows).unwrap();
        assert_eq!(rgba[..4], [255, 0, 0, 255]);
        assert_eq!(rgba[4..8], [0, 0, 0, 255]);
        assert_eq!(rgba[8..], [255, 255, 255, 255]);
        // Small targets' values: every channel the format has.
        let pixel: Vec<u8> = [0x3c00u16, 0xbc00, 0, 0x7e00].iter().flat_map(|v| v.to_le_bytes()).collect();
        let channels = super::float_channels(10, &pixel);
        assert_eq!(channels[..3], [1.0, -1.0, 0.0]);
        assert!(channels[3].is_nan());
        assert_eq!(super::float_pixel(10), Some(8));
        assert_eq!(super::float_pixel(42), None);
    }

    /// A stencil plane's values, most pixels first.
    #[test]
    fn stencil_values_are_counted() {
        // Two rows of three, pitch 4.
        let rows = [1, 1, 0, 9, 1, 2, 2, 9];
        assert_eq!(super::stencil_counts(3, 2, 4, &rows), [(1, 3), (2, 2), (0, 1)]);
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
