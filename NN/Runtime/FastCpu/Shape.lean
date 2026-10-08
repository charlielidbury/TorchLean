/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Runtime.FastCpu.Basic
public import NN.Spec.Core.TensorReductionShape.LinearAlgebra

/-!
# Native CPU data movement and broadcast adjoints

Compiled-code replacements for `broadcastTo` and `swapAdjacentAxes` (pure
copies) and for `reduceFromBroadcastTo` when the broadcast only prepends axes
(`target = lead ++ source`); the latter sums the outermost axis away one axis
at a time, each output from `+0.0` in increasing index order, as the
specification's `reduceOuterAxis`/`sumSpec` recursion does. See
`NN.Runtime.FastCpu.Basic` for the dispatch rule and trust boundary.
-/

@[expose] public section

universe u

namespace TorchLean.FastCpu

open Spec

/-- `broadcastTo` as a strided copy; empty when the shapes do not broadcast. -/
@[extern "torchlean_fastcpu_broadcast"]
opaque broadcastNative (x : @& FloatArray) (source target : @& List Nat) :
    FloatArray

/-- Sum away the leading axes `lead` of `lead ++ [inner elements]`. -/
@[extern "torchlean_fastcpu_reduce_leading"]
opaque reduceLeadingNative (x : @& FloatArray) (lead : @& List Nat)
    (inner : USize) : FloatArray

/-- `n` copies of a boxed `Float`. -/
@[extern "torchlean_fastcpu_fill"]
opaque fillNative {α : Type u} (n : USize) (value : @& α) : FloatArray

/-- `[outer, a, b, inner] → [outer, b, a, inner]`. -/
@[extern "torchlean_fastcpu_swap"]
opaque swapNative (x : @& FloatArray) (outer a b inner : USize) : FloatArray

/-- Compiled implementation of `constFast`. -/
unsafe def constImpl {α : Type u} [storage : Storage α] {s : Shape} (a : α) :
    TorchLean.Tensor.Internal.Rep α s :=
  if floatStorage storage then
    { buffer := unsafeCast (fillNative s.size.toUSize a), size_eq := lcProof }
  else TorchLean.Tensor.Internal.Rep.const a

/-- `Rep.const` (`Tensor.full`, `replicate`), compiled through `constImpl`. -/
@[implemented_by constImpl]
def constFast {α : Type u} [Storage α] {s : Shape} (a : α) :
    TorchLean.Tensor.Internal.Rep α s :=
  TorchLean.Tensor.Internal.Rep.const a

/-- Compiled implementation of `fullFast`. -/
unsafe def fullImpl {α : Type} [storage : Storage α] (shape : Shape)
    (value : α) : Tensor α shape :=
  if floatStorage storage then
    ofFloatBuffer (fillNative shape.size.toUSize value)
  else TorchLean.Tensor.full shape value

/-- `Tensor.full`, compiled through `fullImpl`. -/
@[implemented_by fullImpl]
def fullFast {α : Type} [Storage α] (shape : Shape) (value : α) :
    Tensor α shape :=
  TorchLean.Tensor.full shape value

/-- Compiled implementation of `replicateFast`. -/
unsafe def replicateImpl {α : Type} [storage : Storage α] {shape : Shape}
    (t : Tensor α .scalar) : Tensor α shape :=
  if floatStorage storage then
    ofFloatBuffer (fillNative shape.size.toUSize t.item)
  else Spec.replicate t

/-- `Spec.replicate`, compiled through `replicateImpl`. -/
@[implemented_by replicateImpl]
def replicateFast {α : Type} [Storage α] {shape : Shape}
    (t : Tensor α .scalar) : Tensor α shape :=
  Spec.replicate t

/-- Compiled implementation of `broadcastToFast`. -/
unsafe def broadcastToImpl {α : Type} [storage : Storage α] {s₁ s₂ : Shape}
    (h : Shape.CanBroadcastTo s₁ s₂) (t : Tensor α s₁) : Tensor α s₂ :=
  let reference := fun (_ : Unit) =>
    noteFallback Op.broadcast (TorchLean.Tensor.broadcastTo h t)
  if floatStorage storage then
    let out := broadcastNative (floatBuffer t) s₁ s₂
    if out.size == s₂.size then ofFloatBuffer out else reference ()
  else reference ()

/-- `broadcastTo`, compiled through `broadcastToImpl`. -/
@[implemented_by broadcastToImpl]
def broadcastToFast {α : Type} [Storage α] {s₁ s₂ : Shape}
    (h : Shape.CanBroadcastTo s₁ s₂) (t : Tensor α s₁) : Tensor α s₂ :=
  TorchLean.Tensor.broadcastTo h t

/-- Compiled implementation of `reduceFromBroadcastToFast`. -/
unsafe def reduceFromBroadcastToImpl {α : Type} [storage : Storage α]
    [add : Add α] [zero : Zero α] {s₁ s₂ : Shape}
    (h : Shape.CanBroadcastTo s₁ s₂) (t : Tensor α s₂) : Tensor α s₁ :=
  let lead := s₂.length - s₁.length
  if floatStorage storage && floatAdd add && floatZero zero &&
      s₁.length ≤ s₂.length && s₂.drop lead == s₁ then
    if lead == 0 then ofFloatBuffer (floatBuffer t)
    else
      ofFloatBuffer <| reduceLeadingNative (floatBuffer t) (s₂.take lead)
        s₁.size.toUSize
  else
    noteFallback Op.reduceBroadcast (TorchLean.Tensor.reduceFromBroadcastTo h t)

/-- `reduceFromBroadcastTo`, compiled through `reduceFromBroadcastToImpl`. -/
@[implemented_by reduceFromBroadcastToImpl]
def reduceFromBroadcastToFast {α : Type} [Storage α] [Add α] [Zero α]
    {s₁ s₂ : Shape} (h : Shape.CanBroadcastTo s₁ s₂) (t : Tensor α s₂) :
    Tensor α s₁ :=
  TorchLean.Tensor.reduceFromBroadcastTo h t

/-- Compiled implementation of `swapAdjacentAxesFast`. -/
unsafe def swapAdjacentAxesImpl {β : Type} [storage : Storage β]
    {shape : Shape} (t : Tensor β shape) (depth : Nat) :
    Tensor β (shape.swapAdjacentAtDepth depth) :=
  match floatStorage storage, shape.drop depth with
  | true, a :: b :: rest =>
      ofFloatBuffer <| swapNative (floatBuffer t)
        (dimsProduct (shape.take depth)).toUSize a.toUSize b.toUSize
        (dimsProduct rest).toUSize
  | _, _ => noteFallback Op.swap (TorchLean.Tensor.swapAdjacentAxes t depth)

/-- `swapAdjacentAxes`, compiled through `swapAdjacentAxesImpl`. -/
@[implemented_by swapAdjacentAxesImpl]
def swapAdjacentAxesFast {β : Type} [Storage β] {shape : Shape}
    (t : Tensor β shape) (depth : Nat) :
    Tensor β (shape.swapAdjacentAtDepth depth) :=
  TorchLean.Tensor.swapAdjacentAxes t depth

/-- Compile constant tensors through the native fill. -/
@[csimp] theorem const_eq_fast :
    @TorchLean.Tensor.Internal.Rep.const = @constFast := rfl
/-- Compile `Tensor.full` through the native fill. -/
@[csimp] theorem full_eq_fast : @TorchLean.Tensor.full = @fullFast := rfl
/-- Compile `Spec.replicate` through the native fill. -/
@[csimp] theorem replicate_eq_fast : @Spec.replicate = @replicateFast := rfl
/-- Compile `broadcastTo` through the native dispatcher. -/
@[csimp] theorem broadcastTo_eq_fast :
    @TorchLean.Tensor.broadcastTo = @broadcastToFast := rfl
/-- Compile `reduceFromBroadcastTo` through the native dispatcher. -/
@[csimp] theorem reduceFromBroadcastTo_eq_fast :
    @TorchLean.Tensor.reduceFromBroadcastTo = @reduceFromBroadcastToFast := rfl
/--
Compile `swapAdjacentAxes` through the native dispatcher. This entry is
imported after (and so supersedes) `swapAdjacentAxes_eq_swapAdjacentAxesDirect`
in modules that import `NN.Runtime.FastCpu`.
-/
@[csimp] theorem swapAdjacentAxes_eq_fast :
    @TorchLean.Tensor.swapAdjacentAxes = @swapAdjacentAxesFast := rfl

end TorchLean.FastCpu
