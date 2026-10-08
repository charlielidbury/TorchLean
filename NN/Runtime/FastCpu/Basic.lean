/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor.Core

/-!
# Native CPU fast paths: dispatch support

The eager runtime is generic in its scalar type, so its tensor operations run
the proof-facing specifications through boxed scalar dictionaries. For the
packed `Float` backend, `NN.Runtime.FastCpu.*` modules replace selected
specifications in compiled code (`@[csimp]`) with dispatchers that call native
kernels (`csrc/cpu/torchlean_fastcpu.c`) and otherwise run the specification.

## Trust boundary

The logical definitions are untouched: each `@[csimp]` lemma is `rfl` against
a wrapper whose body *is* the specification. The wrapper's compiled code comes
from an `unsafe` dispatcher (`@[implemented_by]`). That dispatcher is trusted,
not proved, on two points:

* **Backend identification.** The native path runs only when the storage and
  scalar dictionaries are *structurally identical at runtime* to
  `instFloatStorage` and Lean's IEEE `Float` instances (same closure code
  pointers and captured values, same constructor fields; `sameValue` in C).
  Equal runtime representations evaluate identically, so this is a sound but
  incomplete test: any other dictionary takes the specification path.
* **Kernel agreement.** Each C kernel performs the specification's IEEE
  binary64 operations in the specification's order (sequential reductions from
  `+0.0`, no fused multiply-add). Results are bit-identical for non-NaN values;
  NaN payloads are unspecified on both sides. This is validated by tests, not
  proved.

Set `TORCHLEAN_FASTCPU=0` to run every specification in Lean, and
`TORCHLEAN_FASTCPU_STATS=1` to print native/reference call counts at exit.
-/

@[expose] public section

universe u v

namespace TorchLean.FastCpu

open Spec

/-- Whether native CPU kernels are enabled (`TORCHLEAN_FASTCPU`, default on). -/
@[extern "torchlean_fastcpu_enabled"]
opaque enabled : Unit → Bool

/-- Logically the identity; counts one dispatch that ran the Lean reference. -/
@[extern "torchlean_fastcpu_note_fallback"]
def noteFallback {β : Type} (op : UInt32) (value : β) : β := value

/-- Whether a boxed scalar is the binary64 value `+0.0`. -/
@[extern "torchlean_fastcpu_is_positive_zero"]
opaque isPositiveZero {α : Type} (value : @& α) : Bool

/--
Whether two runtime values have identical representations (C `tl_same`):
equal pointers, closures with the same code and equal captures, or
constructors with equal fields. Set `TORCHLEAN_FASTCPU_DEBUG=1` to print
mismatches.
-/
@[extern "torchlean_fastcpu_same"]
opaque sameValue {α : Type u} {β : Type v} (a : @& α) (b : @& β) : Bool

/-- Kernel identifiers shared with `csrc/cpu/torchlean_fastcpu.c`. -/
def Op.matmul : UInt32 := 0
/-- See `Op.matmul`. -/
def Op.matmulBackward : UInt32 := 1
/-- See `Op.matmul`. -/
def Op.add : UInt32 := 2
/-- See `Op.matmul`. -/
def Op.sub : UInt32 := 3
/-- See `Op.matmul`. -/
def Op.mul : UInt32 := 4
/-- See `Op.matmul`. -/
def Op.scale : UInt32 := 5
/-- See `Op.matmul`. -/
def Op.exp : UInt32 := 6
/-- See `Op.matmul`. -/
def Op.abs : UInt32 := 7
/-- See `Op.matmul`. -/
def Op.inv : UInt32 := 8
/-- See `Op.matmul`. -/
def Op.sign : UInt32 := 9
/-- See `Op.matmul`. -/
def Op.relu : UInt32 := 10
/-- See `Op.matmul`. -/
def Op.reluDeriv : UInt32 := 11
/-- See `Op.matmul`. -/
def Op.broadcast : UInt32 := 12
/-- See `Op.matmul`. -/
def Op.reduceBroadcast : UInt32 := 13
/-- See `Op.matmul`. -/
def Op.swap : UInt32 := 14
/-- See `Op.matmul`. -/
def Op.softmax : UInt32 := 15
/-- See `Op.matmul`. -/
def Op.softmaxBackward : UInt32 := 16
/-- See `Op.matmul`. -/
def Op.layerNorm : UInt32 := 17
/-- See `Op.matmul`. -/
def Op.layerNormBackward : UInt32 := 18

/-- Native kernels are enabled and `storage` is the packed `Float` storage. -/
unsafe def floatStorage {α : Type u} (storage : Storage α) : Bool :=
  enabled () && sameValue storage instFloatStorage

/-- `add` is Lean's IEEE `Float` addition. -/
unsafe def floatAdd {α : Type} (add : Add α) : Bool :=
  sameValue add instAddFloat ||
    sameValue add (Context.toAdd (self := instContextFloat))

/-- `sub` is Lean's IEEE `Float` subtraction. -/
unsafe def floatSub {α : Type} (sub : Sub α) : Bool :=
  sameValue sub instSubFloat ||
    sameValue sub (Context.toSub (self := instContextFloat))

/-- `mul` is Lean's IEEE `Float` multiplication. -/
unsafe def floatMul {α : Type} (mul : Mul α) : Bool :=
  sameValue mul instMulFloat ||
    sameValue mul (Context.toMul (self := instContextFloat))

/-- `max` is Lean's `Float` maximum (`if x ≤ y then y else x`). -/
unsafe def floatMax {α : Type} (max : Max α) : Bool :=
  sameValue max instMaxFloat ||
    sameValue max (Context.toMax (self := instContextFloat))

/-- `beq` is Lean's IEEE `Float` equality. -/
unsafe def floatBEq {α : Type} (beq : BEq α) : Bool :=
  sameValue beq instBEqFloat ||
    sameValue beq (Context.toBEq (self := instContextFloat))

/--
The `DecidableRel (· > ·)` that generic code derives from a `Context`
(`Context.decidableGT`, as a partial application). Kept out of line so that
`gtOfContext Float` builds the same closure shape the eager runtime passes.
-/
@[noinline, instance_reducible] def gtOfContext (α : Type) [Context α] :
    DecidableRel ((· > ·) : α → α → Prop) :=
  inferInstance

/-- `gt` decides Lean's IEEE `Float` strict order. -/
unsafe def floatGt {α : Type} [LT α]
    (gt : DecidableRel ((· > ·) : α → α → Prop)) : Bool :=
  sameValue gt
      (inferInstance : DecidableRel ((· > ·) : Float → Float → Prop)) ||
    sameValue gt (Context.decidableGT (self := instContextFloat)) ||
    sameValue gt (gtOfContext Float)

/-- The zero is the binary64 value `+0.0`. -/
unsafe def floatZero {α : Type} (zero : Zero α) : Bool :=
  isPositiveZero (@Zero.zero α zero)

/-- The one is the binary64 value `1.0`. -/
unsafe def floatOne {α : Type} (one : One α) : Bool :=
  sameValue (@One.one α one) (1.0 : Float)

/-- The scalar context is TorchLean's `Float` context. -/
unsafe def floatContext {α : Type} (context : Context α) : Bool :=
  sameValue context instContextFloat

/-- Product of a list of dimensions. -/
def dimsProduct (dims : List Nat) : Nat :=
  dims.foldl (· * ·) 1

/-- The packed buffer of a tensor already identified as `Float`-backed. -/
@[inline] unsafe def floatBuffer {α : Type} [storage : Storage α]
    {s : Shape} (tensor : Tensor α s) : FloatArray :=
  unsafeCast tensor.buffer

/-- Wrap a native result buffer; the kernel guarantees its length. -/
@[inline] unsafe def ofFloatBuffer {α : Type} [storage : Storage α]
    {s : Shape} (buffer : FloatArray) : Tensor α s :=
  { buffer := unsafeCast buffer, size_eq := lcProof }

end TorchLean.FastCpu
