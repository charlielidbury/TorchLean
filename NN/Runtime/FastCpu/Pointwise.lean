/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Runtime.FastCpu.Basic
public import NN.Spec.Layers.Activation

/-!
# Native CPU pointwise operations

Compiled-code replacements for the pointwise specifications the eager tape
records: `addSpec`, `subSpec`, `mulSpec`, `scaleSpec`, `expSpec`, `absSpec`,
`invSpec`, `signSpec` and the tensor `reluSpec`/`reluDerivSpec`. Each native
loop evaluates the specification's scalar expression once per element
(`csrc/cpu/torchlean_fastcpu.c`, `tl_ew_task`). See
`NN.Runtime.FastCpu.Basic` for the dispatch rule and trust boundary.
-/

@[expose] public section

namespace TorchLean.FastCpu

open Spec

/-- `a op b` elementwise (`Op.add`, `Op.sub`, `Op.mul`). -/
@[extern "torchlean_fastcpu_zip"]
opaque zipNative (op : UInt32) (a b : @& FloatArray) : FloatArray

/-- A unary elementwise kernel (`Op.exp` … `Op.reluDeriv`). -/
@[extern "torchlean_fastcpu_map"]
opaque mapNative (op : UInt32) (a : @& FloatArray) : FloatArray

/-- `a * c` elementwise; `c` is a boxed `Float`. -/
@[extern "torchlean_fastcpu_scale"]
opaque scaleNative {α : Type} (a : @& FloatArray) (c : @& α) : FloatArray

section Binary

variable {α : Type}

/-- Compiled implementation of `addSpecFast`. -/
unsafe def addSpecImpl [storage : Storage α] [add : Add α]
    {s : Shape} (a b : Tensor α s) : Tensor α s :=
  if floatStorage storage && floatAdd add then
    ofFloatBuffer (zipNative Op.add (floatBuffer a) (floatBuffer b))
  else noteFallback Op.add (TorchLean.Tensor.addSpec a b)

/-- `addSpec`, compiled through `addSpecImpl`. -/
@[implemented_by addSpecImpl]
def addSpecFast [Storage α] [Add α]
    {s : Shape} (a b : Tensor α s) : Tensor α s :=
  TorchLean.Tensor.addSpec a b

/-- Compiled implementation of `subSpecFast`. -/
unsafe def subSpecImpl [storage : Storage α] [sub : Sub α]
    {s : Shape} (a b : Tensor α s) : Tensor α s :=
  if floatStorage storage && floatSub sub then
    ofFloatBuffer (zipNative Op.sub (floatBuffer a) (floatBuffer b))
  else noteFallback Op.sub (TorchLean.Tensor.subSpec a b)

/-- `subSpec`, compiled through `subSpecImpl`. -/
@[implemented_by subSpecImpl]
def subSpecFast [Storage α] [Sub α]
    {s : Shape} (a b : Tensor α s) : Tensor α s :=
  TorchLean.Tensor.subSpec a b

/-- Compiled implementation of `mulSpecFast`. -/
unsafe def mulSpecImpl [storage : Storage α] [mul : Mul α]
    {s : Shape} (a b : Tensor α s) : Tensor α s :=
  if floatStorage storage && floatMul mul then
    ofFloatBuffer (zipNative Op.mul (floatBuffer a) (floatBuffer b))
  else noteFallback Op.mul (TorchLean.Tensor.mulSpec a b)

/-- `mulSpec`, compiled through `mulSpecImpl`. -/
@[implemented_by mulSpecImpl]
def mulSpecFast [Storage α] [Mul α]
    {s : Shape} (a b : Tensor α s) : Tensor α s :=
  TorchLean.Tensor.mulSpec a b

/-- Compiled implementation of `scaleSpecFast`. -/
unsafe def scaleSpecImpl [storage : Storage α] [mul : Mul α]
    {s : Shape} (t : Tensor α s) (c : α) : Tensor α s :=
  if floatStorage storage && floatMul mul then
    ofFloatBuffer (scaleNative (floatBuffer t) c)
  else noteFallback Op.scale (TorchLean.Tensor.scaleSpec t c)

/-- `scaleSpec`, compiled through `scaleSpecImpl`. -/
@[implemented_by scaleSpecImpl]
def scaleSpecFast [Storage α] [Mul α] {s : Shape} (t : Tensor α s) (c : α) :
    Tensor α s :=
  TorchLean.Tensor.scaleSpec t c

end Binary

section Unary

variable {α : Type}

/-- Run a `Context`-based unary kernel, or the specification. -/
@[inline] unsafe def contextUnary [storage : Storage α] [context : Context α]
    {s : Shape} (op : UInt32) (reference : Tensor α s → Tensor α s)
    (t : Tensor α s) :
    Tensor α s :=
  if floatStorage storage && floatContext context then
    ofFloatBuffer (mapNative op (floatBuffer t))
  else noteFallback op (reference t)

/-- Compiled implementation of `expSpecFast`. -/
unsafe def expSpecImpl [Storage α] [Context α] {s : Shape} (t : Tensor α s) :
    Tensor α s :=
  contextUnary Op.exp TorchLean.Tensor.expSpec t

/-- `expSpec`, compiled through `expSpecImpl`. -/
@[implemented_by expSpecImpl]
def expSpecFast [Storage α] [Context α]
    {s : Shape} (t : Tensor α s) : Tensor α s :=
  TorchLean.Tensor.expSpec t

/-- Compiled implementation of `absSpecFast`. -/
unsafe def absSpecImpl [Storage α] [Context α] {s : Shape} (t : Tensor α s) :
    Tensor α s :=
  contextUnary Op.abs TorchLean.Tensor.absSpec t

/-- `absSpec`, compiled through `absSpecImpl`. -/
@[implemented_by absSpecImpl]
def absSpecFast [Storage α] [Context α]
    {s : Shape} (t : Tensor α s) : Tensor α s :=
  TorchLean.Tensor.absSpec t

/-- Compiled implementation of `invSpecFast`. -/
unsafe def invSpecImpl [Storage α] [Context α] {s : Shape} (t : Tensor α s) :
    Tensor α s :=
  contextUnary Op.inv TorchLean.Tensor.invSpec t

/-- `invSpec`, compiled through `invSpecImpl`. -/
@[implemented_by invSpecImpl]
def invSpecFast [Storage α] [Context α]
    {s : Shape} (t : Tensor α s) : Tensor α s :=
  TorchLean.Tensor.invSpec t

/-- Compiled implementation of `signSpecFast`. -/
unsafe def signSpecImpl [Storage α] [Context α] {s : Shape} (t : Tensor α s) :
    Tensor α s :=
  contextUnary Op.sign TorchLean.Tensor.signSpec t

/-- `signSpec`, compiled through `signSpecImpl`. -/
@[implemented_by signSpecImpl]
def signSpecFast [Storage α] [Context α]
    {s : Shape} (t : Tensor α s) : Tensor α s :=
  TorchLean.Tensor.signSpec t

/-- Compiled implementation of `reluSpecFast`. -/
unsafe def reluSpecImpl [storage : Storage α] [zero : Zero α] [max : Max α]
    [beq : BEq α] {s : Shape} (t : Tensor α s) : Tensor α s :=
  if floatStorage storage && floatZero zero && floatMax max && floatBEq beq
  then ofFloatBuffer (mapNative Op.relu (floatBuffer t))
  else noteFallback Op.relu (Activation.reluSpec t)

/-- Tensor `Activation.reluSpec`, compiled through `reluSpecImpl`. -/
@[implemented_by reluSpecImpl]
def reluSpecFast [Storage α] [Zero α] [Max α] [BEq α]
    {s : Shape} (t : Tensor α s) :
    Tensor α s :=
  Activation.reluSpec t

/-- Compiled implementation of `reluDerivSpecFast`. -/
unsafe def reluDerivSpecImpl [storage : Storage α] [zero : Zero α]
    [one : One α] [LT α] [gt : DecidableRel ((· > ·) : α → α → Prop)]
    {s : Shape} (t : Tensor α s) : Tensor α s :=
  if floatStorage storage && floatZero zero && floatOne one && floatGt gt
  then ofFloatBuffer (mapNative Op.reluDeriv (floatBuffer t))
  else noteFallback Op.reluDeriv (Activation.reluDerivSpec t)

/-- Tensor `Activation.reluDerivSpec`, compiled through `reluDerivSpecImpl`. -/
@[implemented_by reluDerivSpecImpl]
def reluDerivSpecFast [Storage α] [Zero α] [One α] [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape} (t : Tensor α s) :
    Tensor α s :=
  Activation.reluDerivSpec t

end Unary

/-- Compile `addSpec` through the native dispatcher. -/
@[csimp] theorem addSpec_eq_fast :
    @TorchLean.Tensor.addSpec = @addSpecFast := rfl
/-- Compile `subSpec` through the native dispatcher. -/
@[csimp] theorem subSpec_eq_fast :
    @TorchLean.Tensor.subSpec = @subSpecFast := rfl
/-- Compile `mulSpec` through the native dispatcher. -/
@[csimp] theorem mulSpec_eq_fast :
    @TorchLean.Tensor.mulSpec = @mulSpecFast := rfl
/-- Compile `scaleSpec` through the native dispatcher. -/
@[csimp] theorem scaleSpec_eq_fast :
    @TorchLean.Tensor.scaleSpec = @scaleSpecFast := rfl
/-- Compile `expSpec` through the native dispatcher. -/
@[csimp] theorem expSpec_eq_fast :
    @TorchLean.Tensor.expSpec = @expSpecFast := rfl
/-- Compile `absSpec` through the native dispatcher. -/
@[csimp] theorem absSpec_eq_fast :
    @TorchLean.Tensor.absSpec = @absSpecFast := rfl
/-- Compile `invSpec` through the native dispatcher. -/
@[csimp] theorem invSpec_eq_fast :
    @TorchLean.Tensor.invSpec = @invSpecFast := rfl
/-- Compile `signSpec` through the native dispatcher. -/
@[csimp] theorem signSpec_eq_fast :
    @TorchLean.Tensor.signSpec = @signSpecFast := rfl
/-- Compile tensor `reluSpec` through the native dispatcher. -/
@[csimp] theorem reluSpec_eq_fast :
    @Activation.reluSpec = @reluSpecFast := rfl
/-- Compile tensor `reluDerivSpec` through the native dispatcher. -/
@[csimp] theorem reluDerivSpec_eq_fast :
    @Activation.reluDerivSpec = @reluDerivSpecFast := rfl

end TorchLean.FastCpu
