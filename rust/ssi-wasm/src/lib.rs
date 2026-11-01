//! solidcore's `ssi` for the knitter (js/knit/ssi.js): two fitted surfaces in, every curve they meet in out, each
//! point with its parameters on both. One call per pair of surfaces, so the wasm boundary is crossed once a pair.
//! A plain module (no wasm-bindgen glue, no imports): the caller writes numbers into memory from `ssi_alloc`,
//! calls `ssi_intersect`, and reads the result or the error from the pointers it is given.
//!
//! A surface is a flat array of f64, its kind first (angles in degrees, lengths in model units):
//!
//! ```text
//! 0 plane     origin(3) xaxis(3) yaxis(3) u0 u1 v0 v1            origin + a xaxis + b yaxis
//! 1 cylinder  origin(3) xaxis(3) yaxis(3) radius a0 a1 h0 h1     axis xaxis x yaxis through origin
//! 2 cone      apex(3)   xaxis(3) yaxis(3) half_angle h0 h1       heights along the axis from the apex
//! 3 sphere    center(3) xaxis(3) yaxis(3) radius lon0 lon1 lat0 lat1
//! 4 torus     center(3) xaxis(3) yaxis(3) major minor maj0 maj1 min0 min1
//! ```
//!
//! The result is flat too: the curve count, then per curve `closed`, its point count and 7 numbers a point
//! (x y z ua va ub vb, the parameters on a's and b's patches), then the isolated points' count and theirs.
//! A refusal (a tangent contact `ssi` cannot prove, a stalled march) is an error message, its code first
//! (`BOOLEAN_TANGENT: ...`), so the caller can fall back for that pair.

use ssi::{intersect_bodies, shapes, CurvePoint, Options, Patch};
use std::cell::RefCell;

thread_local! {
    static OUT: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static ERR: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn at(spec: &[f64], i: usize) -> Result<f64, String> {
    spec.get(i).copied().ok_or_else(|| "BAD_SPEC: too few numbers".to_string())
}

fn v3(spec: &[f64], i: usize) -> Result<[f64; 3], String> {
    Ok([at(spec, i)?, at(spec, i + 1)?, at(spec, i + 2)?])
}

fn patch(spec: &[f64]) -> Result<Patch, String> {
    let (o, x, y) = (v3(spec, 1)?, v3(spec, 4)?, v3(spec, 7)?);
    let pair = |i: usize| -> Result<[f64; 2], String> { Ok([at(spec, i)?, at(spec, i + 1)?]) };
    Ok(match at(spec, 0)? as i32 {
        0 => shapes::plane(o, x, y, pair(10)?, pair(12)?),
        1 => shapes::cylinder(o, x, y, at(spec, 10)?, pair(11)?, pair(13)?),
        2 => shapes::cone(o, x, y, at(spec, 10)?, pair(11)?),
        3 => shapes::sphere(o, x, y, at(spec, 10)?, pair(11)?, pair(13)?),
        4 => shapes::torus(o, x, y, at(spec, 10)?, at(spec, 11)?, pair(12)?, pair(14)?),
        k => return Err(format!("BAD_SPEC: no surface kind {k}")),
    })
}

fn push(out: &mut Vec<f64>, q: &CurvePoint) {
    out.extend_from_slice(&q.p);
    out.extend_from_slice(&[q.a.u, q.a.v, q.b.u, q.b.v]);
}

fn run(a: &[f64], b: &[f64], feature: f64) -> Result<Vec<f64>, String> {
    let (pa, pb) = (patch(a)?, patch(b)?);
    let options = if feature > 0.0 { shapes::resolving(feature, &[&pa, &pb]) } else { Options::default() };
    let result = intersect_bodies(&[pa], &[pb], &options).map_err(|e| format!("{}: {e}", e.code()))?;
    let mut out = vec![result.curves.len() as f64];
    for curve in &result.curves {
        out.push(if curve.closed { 1.0 } else { 0.0 });
        out.push(curve.points.len() as f64);
        for q in &curve.points {
            push(&mut out, q);
        }
    }
    out.push(result.points.len() as f64);
    for q in &result.points {
        push(&mut out, q);
    }
    Ok(out)
}

/// Room for `n` f64 the caller fills; give it back with `ssi_free`.
#[no_mangle]
pub extern "C" fn ssi_alloc(n: usize) -> *mut f64 {
    let mut v = vec![0.0_f64; n];
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// # Safety
/// `p` and `n` must come from one `ssi_alloc` call, freed once.
#[no_mangle]
pub unsafe extern "C" fn ssi_free(p: *mut f64, n: usize) {
    drop(Vec::from_raw_parts(p, n, n));
}

/// Intersect surfaces `a` (`na` numbers) and `b`; 0 when the result is ready (`ssi_out_*`), 1 for an error
/// (`ssi_err_*`). `feature` > 0 resolves features down to that length (`shapes::resolving`).
///
/// # Safety
/// `a` and `b` must point at `na` and `nb` initialized f64.
#[no_mangle]
pub unsafe extern "C" fn ssi_intersect(a: *const f64, na: usize, b: *const f64, nb: usize, feature: f64) -> i32 {
    let (a, b) = (std::slice::from_raw_parts(a, na), std::slice::from_raw_parts(b, nb));
    match run(a, b, feature) {
        Ok(v) => {
            OUT.with(|o| *o.borrow_mut() = v);
            0
        }
        Err(msg) => {
            ERR.with(|e| *e.borrow_mut() = msg.into_bytes());
            1
        }
    }
}

/// The last result: its first number and its length.
#[no_mangle]
pub extern "C" fn ssi_out_ptr() -> *const f64 {
    OUT.with(|o| o.borrow().as_ptr())
}
#[no_mangle]
pub extern "C" fn ssi_out_len() -> usize {
    OUT.with(|o| o.borrow().len())
}

/// The last error message, UTF-8.
#[no_mangle]
pub extern "C" fn ssi_err_ptr() -> *const u8 {
    ERR.with(|e| e.borrow().as_ptr())
}
#[no_mangle]
pub extern "C" fn ssi_err_len() -> usize {
    ERR.with(|e| e.borrow().len())
}
