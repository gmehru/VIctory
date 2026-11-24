//! Leftovers no lane reads: proof that the lanes entering a block over one
//! edge never use what that edge brings to some of the block's phis, so
//! the emitter may join a stand-in there for a value it cannot express.
//!
//! The console compiler skips work on a uniform test of data and keeps,
//! across the skip, whatever the registers held (a descriptor's dwords
//! loaded for other work, say). The join after the skip takes that
//! leftover as a phi argument, and a second test, which every lane coming
//! from the skip fails, guards the code that reads the phi. The walk here
//! follows those lanes through the per-lane program of structurisation
//! (decision 0003), tracking which values may hold the leftover (poisoned)
//! and which are the same constant in all of them. It succeeds when, along
//! every path they can take while a poisoned value is still live:
//!
//! - each branch is decided without a poisoned value (by constants, or by
//!   values the leftover does not reach, with both ways then followed);
//! - each instruction with an effect, or reading other lanes (wave
//!   operations, ballots, whole-wave sections, derivatives in samples),
//!   reads no poisoned value and runs under no guard that depends on one;
//! - no back edge is taken (a loop could carry a poisoned value round).
//!
//! What a poisoned lane computes otherwise only feeds values that are dead
//! where the walk stops following it, so the stand-in is never seen.

use std::collections::{HashMap, HashSet};

use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, Value};
use crate::structure::{Atom, Bool, LaneExit, Structured};

type Why = &'static str;

/// What a value is in the lanes the walk follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Abs {
    /// This number in every one of them.
    Const(u32),
    /// Not computed from the leftover; not known.
    Unknown,
    /// May be computed from the leftover.
    Poison,
}

fn join(a: Abs, b: Abs) -> Abs {
    match (a, b) {
        (Abs::Poison, _) | (_, Abs::Poison) => Abs::Poison,
        (Abs::Const(x), Abs::Const(y)) if x == y => a,
        _ => Abs::Unknown,
    }
}

/// The per-lane program the walk runs over.
pub(super) struct Walk<'a> {
    pub p: &'a Program,
    pub st: &'a Structured,
    /// Reverse-postorder index per block (`usize::MAX`: unreachable).
    pub rpo: &'a [usize],
    /// Whether instruction `i` of block `b` reads other lanes' values (a
    /// whole-wave section, a lane mask read as a number).
    pub lanes: &'a dyn Fn(usize, usize) -> bool,
}

/// Where values are defined and read.
struct Index {
    /// Block and instruction writing each value.
    defs: HashMap<Value, (usize, usize)>,
    /// Block of each phi value.
    phis: HashMap<Value, usize>,
    /// Blocks reading each value: an instruction's block, or for a phi
    /// argument the predecessor it comes from (read on leaving it).
    uses: HashMap<Value, Vec<usize>>,
}

impl Index {
    fn new(p: &Program) -> Self {
        let mut ix = Index {
            defs: HashMap::new(),
            phis: HashMap::new(),
            uses: HashMap::new(),
        };
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                ix.phis.insert(phi.value, b);
                for (&a, &q) in phi.args.iter().zip(&block.preds) {
                    ix.uses.entry(a).or_default().push(q);
                }
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, w) in &inst.writes {
                    ix.defs.insert(w, (b, i));
                }
                for &(_, r) in &inst.reads {
                    ix.uses.entry(r).or_default().push(b);
                }
            }
        }
        ix
    }

    fn def_block(&self, v: Value) -> Option<usize> {
        self.defs
            .get(&v)
            .map(|d| d.0)
            .or(self.phis.get(&v).copied())
    }

    /// The blocks `v` is live into: some read of it is reachable from
    /// there without passing its definition.
    fn live_in(&self, p: &Program, v: Value) -> HashSet<usize> {
        let def = self.def_block(v);
        let mut live = HashSet::new();
        let mut work = Vec::new();
        for &u in self.uses.get(&v).into_iter().flatten() {
            // A read in the defining block follows the definition.
            if Some(u) != def && live.insert(u) {
                work.push(u);
            }
        }
        while let Some(z) = work.pop() {
            for &q in &p.blocks[z].preds {
                if Some(q) != def && live.insert(q) {
                    work.push(q);
                }
            }
        }
        live
    }
}

/// How a branch goes in the lanes followed: (can fall, can take, decided
/// by a poisoned value on some way there).
type Ways = (bool, bool, bool);

struct State<'w, 'a> {
    w: &'w Walk<'a>,
    ix: Index,
    abs: HashMap<Value, Abs>,
    live: HashMap<Value, HashSet<usize>>,
}

impl Walk<'_> {
    /// Ok when the lanes entering `block` over its predecessor `n` read
    /// none of `seeds` (phis of `block`) where it could be seen; the error
    /// says what may read one.
    pub fn unseen(&self, block: usize, n: usize, seeds: &HashSet<Value>) -> Result<(), Why> {
        if self.rpo[block] == usize::MAX {
            return Err("an unreachable join");
        }
        let mut s = State {
            w: self,
            ix: Index::new(self.p),
            abs: HashMap::new(),
            live: HashMap::new(),
        };
        let mut order: Vec<usize> = (0..self.p.blocks.len())
            .filter(|&b| self.rpo[b] != usize::MAX)
            .collect();
        order.sort_by_key(|&b| self.rpo[b]);
        // Per block reached: the predecessors (by index) lanes arrive from.
        let mut incoming: HashMap<usize, Vec<usize>> = HashMap::from([(block, vec![n])]);
        for &b in order.iter().skip_while(|&&b| b != block) {
            let Some(edges) = incoming.remove(&b) else {
                continue;
            };
            for phi in &self.p.blocks[b].phis {
                let a = if b == block && seeds.contains(&phi.value) {
                    Abs::Poison
                } else {
                    edges
                        .iter()
                        .map(|&k| s.val(phi.args[k]))
                        .reduce(join)
                        .unwrap_or(Abs::Unknown)
                };
                s.abs.insert(phi.value, a);
            }
            for i in 0..self.p.blocks[b].insts.len() {
                s.step(b, i)?;
            }
            for t in s.successors(b)? {
                let ks: Vec<usize> = (0..self.p.blocks[t].preds.len())
                    .filter(|&k| self.p.blocks[t].preds[k] == b)
                    .collect();
                if !s.poison_live(t, &ks) {
                    continue;
                }
                if self.rpo[t] <= self.rpo[b] {
                    return Err("a leftover live around a loop");
                }
                incoming.entry(t).or_default().extend(ks);
            }
        }
        Ok(())
    }
}

impl State<'_, '_> {
    fn val(&self, v: Value) -> Abs {
        match self.abs.get(&v) {
            Some(&a) => a,
            None => self.constant(v).map_or(Abs::Unknown, Abs::Const),
        }
    }

    /// `v` when every lane that has it got it from a constant move.
    fn constant(&self, v: Value) -> Option<u32> {
        let st = self.w.st;
        if st.masks.contains_key(&v) || st.highs.contains_key(&v) {
            return None;
        }
        let &(b, i) = self.ix.defs.get(&v)?;
        // A guarded move leaves the old contents in some lanes.
        if st.guards[b][i].is_some() {
            return None;
        }
        let inst = &self.w.p.blocks[b].insts[i];
        match (&inst.fields, inst.name()) {
            (Fields::Sop { src, .. }, "s_mov_b32") => imm(src.first()?),
            (Fields::Sop { imm: Some(k), .. }, "s_movk_i32") => Some(i32::from(*k as i16) as u32),
            (Fields::Valu(f), "v_mov_b32") if plain(f) => imm(f.src.first()?),
            _ => None,
        }
    }

    fn operand(&self, inst: &Inst, op: &Operand) -> Abs {
        let read = |r: Reg| inst.read(r).map_or(Abs::Unknown, |v| self.val(v));
        match *op {
            Operand::Scalar { code: NULL, .. } => Abs::Const(0),
            Operand::Scalar { code, .. } => read(Reg::S(code)),
            Operand::Vector { reg, .. } => read(Reg::V(reg)),
            _ => imm(op).map_or(Abs::Unknown, Abs::Const),
        }
    }

    /// Whether `inst` reads a poisoned value. EXEC read implicitly aside:
    /// whether a lane runs an instruction is its guard's business. Named
    /// as an operand (a mask operation, a copy, a ballot), it is data.
    fn reads_poison(&self, inst: &Inst) -> bool {
        let implicit = match &inst.fields {
            Fields::Sop { .. } => false,
            Fields::Valu(f) => !f.src.iter().any(
                |o| matches!(*o, Operand::Scalar { code, .. } if code == EXEC || code == EXEC + 1),
            ),
            _ => true,
        };
        inst.reads.iter().any(|&(r, v)| {
            let exec = matches!(r, Reg::S(c) if c == EXEC || c == EXEC + 1);
            !(implicit && exec) && self.val(v) == Abs::Poison
        })
    }

    /// One instruction run by the lanes followed.
    fn step(&mut self, b: usize, i: usize) -> Result<(), Why> {
        let w = self.w;
        let block = &w.p.blocks[b];
        let inst = &block.insts[i];
        // The closing branch: [`Self::successors`] decides it.
        if i + 1 == block.insts.len() && matches!(block.exit, Exit::Branch { .. }) {
            return Ok(());
        }
        let ways = match w.st.guards[b][i] {
            None => (false, true, false),
            Some(g) => self.ways(g),
        };
        let poison = self.reads_poison(inst) || ways.0 && ways.1 && ways.2;
        if poison && (!pure(inst) || (w.lanes)(b, i)) {
            return Err(match inst.fields {
                Fields::Valu(_) | Fields::Sop { .. } => "a leftover read across lanes",
                _ => "a leftover read by an instruction with an effect",
            });
        }
        let runs = ways.1;
        for &(reg, v) in &inst.writes {
            let x = if poison {
                Abs::Poison
            } else if runs && !ways.0 {
                self.fold(inst, reg)
            } else {
                // Lanes it skips keep the old contents.
                Abs::Unknown
            };
            self.abs.insert(v, x);
        }
        Ok(())
    }

    /// The value `inst` writes to `reg` in a lane that runs it, its
    /// operands not poisoned.
    fn fold(&self, inst: &Inst, reg: Reg) -> Abs {
        let st = self.w.st;
        let Some(v) = inst.write(reg) else {
            return Abs::Unknown;
        };
        // Lane masks: their lane bits are atoms ([`Self::atom`]).
        if st.masks.contains_key(&v) || st.highs.contains_key(&v) {
            return Abs::Unknown;
        }
        let name = inst.name();
        match &inst.fields {
            Fields::Sop { src, imm, .. } if reg == Reg::Scc => {
                let Some(op) = name.strip_prefix("s_cmpk_").or(name.strip_prefix("s_cmp_")) else {
                    return Abs::Unknown;
                };
                let b = match (name.starts_with("s_cmpk_"), imm) {
                    (true, Some(k)) if op.ends_with("_i32") => {
                        Abs::Const(i32::from(*k as i16) as u32)
                    }
                    (true, Some(k)) => Abs::Const(u32::from(*k)),
                    (true, None) => return Abs::Unknown,
                    (false, _) => match src.get(1) {
                        Some(s1) => self.operand(inst, s1),
                        None => return Abs::Unknown,
                    },
                };
                match (src.first().map(|s0| self.operand(inst, s0)), b) {
                    (Some(Abs::Const(x)), Abs::Const(y)) => {
                        compare(op, x, y).map_or(Abs::Unknown, |r| Abs::Const(u32::from(r)))
                    }
                    _ => Abs::Unknown,
                }
            }
            Fields::Sop { src, .. } if name == "s_mov_b32" => src
                .first()
                .map_or(Abs::Unknown, |s0| self.operand(inst, s0)),
            Fields::Sop { imm: Some(k), .. } if name == "s_movk_i32" => {
                Abs::Const(i32::from(*k as i16) as u32)
            }
            Fields::Sop { src, .. } if name == "s_cselect_b32" => {
                match inst.read(Reg::Scc).map(|c| self.val(c)) {
                    Some(Abs::Const(c)) => src
                        .get(usize::from(c == 0))
                        .map_or(Abs::Unknown, |s| self.operand(inst, s)),
                    _ => Abs::Unknown,
                }
            }
            Fields::Valu(f) if name == "v_mov_b32" && plain(f) => f
                .src
                .first()
                .map_or(Abs::Unknown, |s0| self.operand(inst, s0)),
            _ => Abs::Unknown,
        }
    }

    /// An atom of structurisation's lane conditions, in the lanes followed.
    fn atom(&self, a: Atom) -> Abs {
        match a {
            // The lane's bit of a compare: the compare of its operands.
            Atom::Lane(v) => {
                if self.val(v) == Abs::Poison {
                    return Abs::Poison;
                }
                let Some(&(b, i)) = self.ix.defs.get(&v) else {
                    return Abs::Unknown;
                };
                let inst = &self.w.p.blocks[b].insts[i];
                if self.reads_poison(inst) {
                    return Abs::Poison;
                }
                let name = inst.name();
                let (Fields::Valu(f), Some(op)) = (
                    &inst.fields,
                    name.strip_prefix("v_cmpx_").or(name.strip_prefix("v_cmp_")),
                ) else {
                    return Abs::Unknown;
                };
                if !plain(f) {
                    return Abs::Unknown;
                }
                match (
                    f.src.first().map(|s| self.operand(inst, s)),
                    f.src.get(1).map(|s| self.operand(inst, s)),
                ) {
                    (Some(Abs::Const(x)), Some(Abs::Const(y))) => {
                        compare(op, x, y).map_or(Abs::Unknown, |r| Abs::Const(u32::from(r)))
                    }
                    _ => Abs::Unknown,
                }
            }
            Atom::Uniform(v) => match self.val(v) {
                Abs::Const(c) => Abs::Const(u32::from(c != 0)),
                x => x,
            },
            Atom::Phi(v) | Atom::Quad(v) | Atom::Opaque(v) => match self.val(v) {
                Abs::Poison => Abs::Poison,
                _ => Abs::Unknown,
            },
            Atom::Constant(_) => Abs::Unknown,
        }
    }

    /// The ways lane condition `b` goes in the lanes followed.
    fn ways(&self, b: Bool) -> Ways {
        let mut memo = HashMap::new();
        self.ways_in(b, &mut memo)
    }

    fn ways_in(&self, b: Bool, memo: &mut HashMap<Bool, Ways>) -> Ways {
        if b == Bool::FALSE {
            return (true, false, false);
        }
        if b == Bool::TRUE {
            return (false, true, false);
        }
        if let Some(&w) = memo.get(&b) {
            return w;
        }
        let (atom, lo, hi) = self.w.st.bools.node(b).expect("not a constant");
        let w = match self.atom(atom) {
            Abs::Const(0) => self.ways_in(lo, memo),
            Abs::Const(_) => self.ways_in(hi, memo),
            a => {
                let (l, h) = (self.ways_in(lo, memo), self.ways_in(hi, memo));
                (l.0 || h.0, l.1 || h.1, l.2 || h.2 || a == Abs::Poison)
            }
        };
        memo.insert(b, w);
        w
    }

    /// Where the lanes followed leave block `b` for.
    fn successors(&self, b: usize) -> Result<Vec<usize>, Why> {
        match self.w.st.exits[b] {
            LaneExit::Jump(t) => Ok(vec![t]),
            LaneExit::End => Ok(vec![]),
            LaneExit::Indirect => Err("an indirect jump"),
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => match self.ways(when) {
                (true, true, true) => Err("a branch on a leftover"),
                (fall, take, _) => Ok([(take, taken), (fall, not_taken)]
                    .into_iter()
                    .filter_map(|(go, t)| go.then_some(t))
                    .collect()),
            },
        }
    }

    /// Whether lanes entering `t` over its predecessors `ks` may still read
    /// a poisoned value: a phi of `t` joins one, or one is live into `t`.
    fn poison_live(&mut self, t: usize, ks: &[usize]) -> bool {
        let p = self.w.p;
        for phi in &p.blocks[t].phis {
            if ks.iter().any(|&k| self.val(phi.args[k]) == Abs::Poison)
                && self.ix.uses.contains_key(&phi.value)
            {
                return true;
            }
        }
        let poisoned: Vec<Value> = self
            .abs
            .iter()
            .filter(|(_, a)| **a == Abs::Poison)
            .map(|(&v, _)| v)
            .collect();
        for v in poisoned {
            if !self.live.contains_key(&v) {
                let l = self.ix.live_in(p, v);
                self.live.insert(v, l);
            }
            if self.live[&v].contains(&t) {
                return true;
            }
        }
        false
    }
}

/// A VALU encoding with no modifier that changes its operands or result.
fn plain(f: &crate::gfx10::Valu) -> bool {
    f.dpp.is_none() && f.sdwa.is_none() && f.neg == 0 && f.abs == 0 && !f.clamp && f.omod == 0
}

/// The number an inline constant or literal operand holds.
fn imm(op: &Operand) -> Option<u32> {
    match *op {
        Operand::Const(c) => crate::resource::inline(c),
        Operand::Literal(x) => Some(x),
        _ => None,
    }
}

/// A 32-bit integer compare (`eq_u32`, `lt_i32`, ...) of `a` with `b`.
fn compare(op: &str, a: u32, b: u32) -> Option<bool> {
    let (cmp, ty) = op.split_once('_')?;
    let signed = match ty {
        "u32" => false,
        "i32" => true,
        _ => return None,
    };
    let (sa, sb) = (a as i32, b as i32);
    Some(match cmp {
        "f" => false,
        "t" => true,
        "eq" => a == b,
        "ne" | "lg" => a != b,
        "lt" if signed => sa < sb,
        "le" if signed => sa <= sb,
        "gt" if signed => sa > sb,
        "ge" if signed => sa >= sb,
        "lt" => a < b,
        "le" => a <= b,
        "gt" => a > b,
        "ge" => a >= b,
        _ => return None,
    })
}

/// Whether `inst`, its operands aside, only computes its results: no
/// memory, export, message or interpolation, and no other lane's value.
fn pure(inst: &Inst) -> bool {
    let name = inst.name();
    match &inst.fields {
        Fields::Sop { .. } => {
            !matches!(
                name,
                "s_setpc_b64"
                    | "s_swappc_b64"
                    | "s_getpc_b64"
                    | "s_setreg_b32"
                    | "s_setreg_imm32_b32"
            ) && !name.starts_with("s_movrel")
        }
        Fields::Valu(f) => {
            f.dpp.is_none()
                && !name.starts_with("v_readlane")
                && !name.starts_with("v_readfirstlane")
                && !name.starts_with("v_writelane")
                && !name.starts_with("v_permlane")
                && !name.starts_with("v_mbcnt")
                && !name.starts_with("v_interp")
                && !name.starts_with("v_movrel")
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;
    /// The skip: v2 gets `flag` (a VOP1 `v_mov_b32 v2` source), and where
    /// s3 is 0 the code jumps (from block 0) to the join at byte 20; else
    /// s4 gets s5 and v2 -1.
    const fn head(flag: u32) -> [u32; 5] {
        [
            0x7E04_0200 | flag, // v_mov_b32 v2, <flag>
            0xBF06_8003,        // s_cmp_eq_u32 s3, 0
            0xBF85_0002,        // s_cbranch_scc1 -> 20
            0xBE84_0305,        // s_mov_b32 s4, s5
            0x7E04_02C1,        // v_mov_b32 v2, -1
        ]
    }
    /// At the join: lanes whose v2 is not -1 jump over the next `n` dwords.
    const fn test(n: u32) -> [u32; 3] {
        [
            0x7D84_04C1,     // v_cmp_eq_u32 vcc, -1, v2
            0x8A6A_6A7E,     // s_andn2_b32 vcc_lo, exec_lo, vcc_lo
            0xBF87_0000 | n, // s_cbranch_vccnz
        ]
    }
    const ZERO: u32 = 0x80;
    const MINUS_1: u32 = 0xC1;
    const V0: u32 = 0x100;
    const READ_S4: u32 = 0x7E06_0204; // v_mov_b32 v3, s4
    /// buffer_store_dword v3, v5, s[8:11], 0 offen
    const STORE_V3: [u32; 2] = [0xE070_1000, 0x8002_0305];

    /// Whether lanes entering the join at byte 20 from block 0 never see
    /// s4's phi there.
    fn walk(words: &[u32]) -> Result<(), Why> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = ir::build(&code, false, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let order = ir::reverse_postorder(&p.blocks);
        let mut rpo = vec![usize::MAX; p.blocks.len()];
        for (n, &b) in order.iter().enumerate() {
            rpo[b] = n;
        }
        let join = p.blocks.iter().position(|b| b.offset == 20).unwrap();
        let n = p.blocks[join]
            .preds
            .iter()
            .position(|&q| p.blocks[q].offset == 0)
            .unwrap();
        let phi = p.blocks[join].phis.iter().find(|f| f.reg == Reg::S(4));
        let seeds = HashSet::from([phi.expect("s4 joined").value]);
        let lanes = |_: usize, _: usize| false;
        Walk {
            p: &p,
            st: &st,
            rpo: &rpo,
            lanes: &lanes,
        }
        .unseen(join, n, &seeds)
    }

    fn program(flag: u32, join: &[u32]) -> Vec<u32> {
        let mut w = head(flag).to_vec();
        w.extend(join);
        w.extend(STORE_V3);
        w.push(S_ENDPGM);
        w
    }

    /// The flag is 0 on the skip: those lanes jump past the read, and what
    /// they keep in v3 is not the leftover.
    #[test]
    fn lanes_the_test_turns_away_never_read_the_leftover() {
        let mut join = test(1).to_vec();
        join.push(READ_S4);
        assert_eq!(walk(&program(ZERO, &join)), Ok(()));
        // A copy no one reads is no read either.
        let mut join = vec![0x7E0C_0204]; // v_mov_b32 v6, s4
        join.extend(test(1));
        join.push(READ_S4);
        assert_eq!(walk(&program(ZERO, &join)), Ok(()));
    }

    /// The flag set, or not known, on the skip: the store may see it.
    #[test]
    fn lanes_that_may_pass_the_test_read_the_leftover() {
        let mut join = test(1).to_vec();
        join.push(READ_S4);
        for flag in [MINUS_1, V0] {
            assert_eq!(
                walk(&program(flag, &join)),
                Err("a leftover read by an instruction with an effect"),
                "{flag:#x}"
            );
        }
    }

    #[test]
    fn a_branch_on_the_leftover_is_a_read() {
        let join = [
            0xBF06_8004, // s_cmp_eq_u32 s4, 0
            0xBF85_0001, // s_cbranch_scc1 over the copy
            0x7E06_0280, // v_mov_b32 v3, 0
        ];
        assert_eq!(walk(&program(ZERO, &join)), Err("a branch on a leftover"));
    }

    /// The lanes turned away read v3 in every lane (`v_readfirstlane`):
    /// the leftover copied into v3 before the test is seen.
    #[test]
    fn a_read_across_lanes_is_a_read() {
        let mut join = vec![READ_S4, 0x7E0E_0503]; // v_readfirstlane_b32 s7, v3
        join.extend(test(0));
        assert_eq!(
            walk(&program(ZERO, &join)),
            Err("a leftover read across lanes")
        );
    }

    /// Lanes compare s4 and store where it matches: whether they store
    /// depends on the leftover.
    #[test]
    fn a_guard_on_the_leftover_is_a_read() {
        let mut join = vec![
            0x7D84_0004, // v_cmp_eq_u32 vcc, s4, v0
            0xBE8A_3C6A, // s_and_saveexec_b32 s10, vcc_lo
        ];
        join.extend(STORE_V3);
        join.push(0x887E_0A7E); // s_or_b32 exec_lo, exec_lo, s10
        join.extend(test(1));
        join.push(READ_S4);
        assert_eq!(
            walk(&program(ZERO, &join)),
            Err("a leftover read by an instruction with an effect")
        );
    }

    /// Turned away into a loop back to the join, the lanes carry the
    /// leftover round: the walk does not follow them.
    #[test]
    fn a_leftover_live_around_a_loop_stays_unproven() {
        let mut join = test(1).to_vec();
        join.extend([
            READ_S4,
            0x8106_8106, // s_add_i32 s6, s6, 1
            0xBF0A_8406, // s_cmp_lt_u32 s6, 4
            0xBF85_FFF9, // s_cbranch_scc1 -> 20
        ]);
        assert_eq!(
            walk(&program(ZERO, &join)),
            Err("a leftover live around a loop")
        );
    }

    #[test]
    fn compares_fold_as_32_bit_integers() {
        assert_eq!(compare("eq_u32", u32::MAX, 0), Some(false));
        assert_eq!(compare("lt_i32", u32::MAX, 0), Some(true));
        assert_eq!(compare("lt_u32", u32::MAX, 0), Some(false));
        assert_eq!(compare("lg_u32", 1, 1), Some(false));
        assert_eq!(compare("ge_i32", 0x8000_0000, 1), Some(false));
        assert_eq!(compare("eq_f32", 0, 0), None);
        assert_eq!(compare("eq_u64", 0, 0), None);
    }
}
