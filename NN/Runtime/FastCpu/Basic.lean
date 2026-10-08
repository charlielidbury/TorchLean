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
  scalar dictionaries are *pointer-identical* to `instFloatStorage` and Lean's
  IEEE `Float` operations, and the zero is `+0.0`. Any other dictionary,
  including an extensionally equal copy, takes the specification path.
* **Kernel agreement.** Each C kernel performs the specification's IEEE
  binary64 operations in the specification's order (sequential reductions from
  `+0.0`, no fused multiply-add). Results are bit-identical for non-NaN values;
  NaN payloads are unspecified on both sides. This is validated by tests, not
  proved.

Set `TORCHLEAN_FASTCPU=0` to run every specification in Lean, and
`TORCHLEAN_FASTCPU_STATS=1` to print native/reference call counts at exit.
-/

@[expose] public section

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

/-- Kernel identifiers shared with `csrc/cpu/torchlean_fastcpu.c`. -/
def Op.matmul : UInt32 := 0
/-- See `Op.matmul`. -/
def Op.matmulBackward : UInt32 := 1

/--
Whether the storage and scalar dictionaries are exactly the packed `Float`
backend with IEEE addition, multiplication and a `+0.0` zero.

Pointer identity is a sufficient condition only: dictionaries that are equal
but separately allocated fail the test and use the specification.
-/
unsafe def isFloatBackend {α : Type} (storage : Storage α) (add : Add α)
    (mul : Mul α) (zero : Zero α) : Bool :=
  enabled () &&
    ptrAddrUnsafe storage == ptrAddrUnsafe instFloatStorage &&
    (ptrAddrUnsafe add == ptrAddrUnsafe instAddFloat ||
      ptrAddrUnsafe add ==
        ptrAddrUnsafe (Context.toAdd (self := instContextFloat))) &&
    (ptrAddrUnsafe mul == ptrAddrUnsafe instMulFloat ||
      ptrAddrUnsafe mul ==
        ptrAddrUnsafe (Context.toMul (self := instContextFloat))) &&
    isPositiveZero (@Zero.zero α zero)

/-- The packed buffer of a tensor already identified as `Float`-backed. -/
@[inline] unsafe def floatBuffer {α : Type} [storage : Storage α]
    {s : Shape} (tensor : Tensor α s) : FloatArray :=
  unsafeCast tensor.buffer

/-- Wrap a native result buffer; the kernel guarantees its length. -/
@[inline] unsafe def ofFloatBuffer {α : Type} [storage : Storage α]
    {s : Shape} (buffer : FloatArray) : Tensor α s :=
  { buffer := unsafeCast buffer, size_eq := lcProof }

end TorchLean.FastCpu
