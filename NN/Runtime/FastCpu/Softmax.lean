/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Runtime.FastCpu.Basic
public import NN.Spec.Layers.Activation

/-!
# Native CPU last-axis softmax

Compiled-code replacements for `softmaxInnermostSpec` and its VJP. Per row the
kernel evaluates `softmaxVecSpec` exactly: the running `max` from the first
entry, `exp (x - m)`, a `+0.0`-seeded sum, then one division per entry; the
backward pass recomputes `y` and returns `y * (dy - Σ dy * y)`. See
`NN.Runtime.FastCpu.Basic` for the dispatch rule and trust boundary.
-/

@[expose] public section

namespace TorchLean.FastCpu

open Spec

/-- Row-wise softmax of a `[rows, n]` buffer (`n > 0`). -/
@[extern "torchlean_fastcpu_softmax"]
opaque softmaxNative (x : @& FloatArray) (rows n : USize) : FloatArray

/-- Row-wise softmax VJP of a `[rows, n]` buffer (`n > 0`). -/
@[extern "torchlean_fastcpu_softmax_backward"]
opaque softmaxBackwardNative (x dy : @& FloatArray) (rows n : USize) :
    FloatArray

/-- `(rows, n)` of a shape whose last axis is nonempty. -/
def lastAxis? (s : Shape) : Option (Nat × Nat) :=
  match s.getLast? with
  | some (n + 1) => some (s.size / (n + 1), n + 1)
  | _ => none

/-- Compiled implementation of `softmaxInnermostSpecFast`. -/
unsafe def softmaxInnermostSpecImpl {α : Type} [storage : Storage α]
    [context : Context α] {s : Shape} (x : Tensor α s) : Tensor α s :=
  match floatStorage storage && floatContext context, lastAxis? s with
  | true, some (rows, n) =>
      ofFloatBuffer (softmaxNative (floatBuffer x) rows.toUSize n.toUSize)
  | _, _ =>
      noteFallback Op.softmax (Activation.Internal.softmaxInnermostSpec x)

/-- `softmaxInnermostSpec`, compiled through `softmaxInnermostSpecImpl`. -/
@[implemented_by softmaxInnermostSpecImpl]
def softmaxInnermostSpecFast {α : Type} [Storage α] [Context α] {s : Shape}
    (x : Tensor α s) : Tensor α s :=
  Activation.Internal.softmaxInnermostSpec x

/-- Compiled implementation of `softmaxInnermostBackwardSpecFast`. -/
unsafe def softmaxInnermostBackwardSpecImpl {α : Type} [storage : Storage α]
    [context : Context α] {s : Shape} (x dy : Tensor α s) : Tensor α s :=
  match floatStorage storage && floatContext context, lastAxis? s with
  | true, some (rows, n) =>
      ofFloatBuffer <| softmaxBackwardNative (floatBuffer x) (floatBuffer dy)
        rows.toUSize n.toUSize
  | _, _ =>
      noteFallback Op.softmaxBackward
        (Activation.Internal.softmaxInnermostBackwardSpec x dy)

/-- `softmaxInnermostBackwardSpec`, compiled through its native dispatcher. -/
@[implemented_by softmaxInnermostBackwardSpecImpl]
def softmaxInnermostBackwardSpecFast {α : Type} [Storage α] [Context α]
    {s : Shape} (x dy : Tensor α s) : Tensor α s :=
  Activation.Internal.softmaxInnermostBackwardSpec x dy

/-- Compile `softmaxInnermostSpec` through the native dispatcher. -/
@[csimp] theorem softmaxInnermostSpec_eq_fast :
    @Activation.Internal.softmaxInnermostSpec = @softmaxInnermostSpecFast :=
  rfl
/-- Compile `softmaxInnermostBackwardSpec` through the native dispatcher. -/
@[csimp] theorem softmaxInnermostBackwardSpec_eq_fast :
    @Activation.Internal.softmaxInnermostBackwardSpec =
      @softmaxInnermostBackwardSpecFast := rfl

end TorchLean.FastCpu
