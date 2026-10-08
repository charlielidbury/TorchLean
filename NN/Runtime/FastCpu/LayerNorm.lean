/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Runtime.FastCpu.Basic
public import NN.Spec.Layers.Normalization.Core

/-!
# Native CPU layer normalization

Compiled-code replacements for `Spec.layerNorm` and `Spec.layerNormBackward`
on `[seqLen, embedDim]`. The kernels reproduce the specifications' operation
sequence per row, including `reduceVar` re-centring its already-centred input,
the `max var 0` clamp, and row-ordered column sums for the affine gradients
(`csrc/cpu/torchlean_fastcpu.c`, `tl_ln_task`). See
`NN.Runtime.FastCpu.Basic` for the dispatch rule and trust boundary.
-/

@[expose] public section

namespace TorchLean.FastCpu

open Spec

/-- Layer norm of a `[rows, width]` buffer; `eps` is a boxed `Float`. -/
@[extern "torchlean_fastcpu_layer_norm"]
opaque layerNormNative {α : Type} (x gamma beta : @& FloatArray)
    (rows width : USize) (eps : @& α) : FloatArray

/-- `(dx, dgamma, dbeta)` of layer norm; `eps` is a boxed `Float`. -/
@[extern "torchlean_fastcpu_layer_norm_backward"]
opaque layerNormBackwardNative {α : Type} (x gamma dy : @& FloatArray)
    (rows width : USize) (eps : @& α) :
    FloatArray × FloatArray × FloatArray

/-- Compiled implementation of `layerNormFast`. -/
unsafe def layerNormImpl {α : Type} [storage : Storage α]
    [context : Context α] {seqLen embedDim : Nat}
    (x : Tensor α [seqLen, embedDim]) (gamma beta : Tensor α [embedDim])
    (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0) (epsilon : α) :
    Tensor α [seqLen, embedDim] :=
  if floatStorage storage && floatContext context then
    ofFloatBuffer <| layerNormNative (floatBuffer x) (floatBuffer gamma)
      (floatBuffer beta) seqLen.toUSize embedDim.toUSize epsilon
  else
    noteFallback Op.layerNorm
      (Spec.layerNorm x gamma beta h_seq_pos h_embed_pos epsilon)

/-- `Spec.layerNorm`, compiled through `layerNormImpl`. -/
@[implemented_by layerNormImpl]
def layerNormFast {α : Type} [Storage α] [Context α] {seqLen embedDim : Nat}
    (x : Tensor α [seqLen, embedDim]) (gamma beta : Tensor α [embedDim])
    (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0) (epsilon : α) :
    Tensor α [seqLen, embedDim] :=
  Spec.layerNorm x gamma beta h_seq_pos h_embed_pos epsilon

/-- Compiled implementation of `layerNormBackwardFast`. -/
unsafe def layerNormBackwardImpl {α : Type} [storage : Storage α]
    [context : Context α] {seqLen embedDim : Nat}
    (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
    (input : Tensor α [seqLen, embedDim]) (scale : Tensor α [embedDim])
    (outputGradient : Tensor α [seqLen, embedDim]) (epsilon : α) :
    NormalizationGradients α [seqLen, embedDim] [embedDim] :=
  if floatStorage storage && floatContext context then
    let (dx, dgamma, dbeta) := layerNormBackwardNative (floatBuffer input)
      (floatBuffer scale) (floatBuffer outputGradient) seqLen.toUSize
      embedDim.toUSize epsilon
    { inputGradient := ofFloatBuffer dx
      scaleGradient := ofFloatBuffer dgamma
      biasGradient := ofFloatBuffer dbeta }
  else
    noteFallback Op.layerNormBackward
      (Spec.layerNormBackward h_seq_pos h_embed_pos input scale
        outputGradient epsilon)

/-- `Spec.layerNormBackward`, compiled through `layerNormBackwardImpl`. -/
@[implemented_by layerNormBackwardImpl]
def layerNormBackwardFast {α : Type} [Storage α] [Context α]
    {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0)
    (h_embed_pos : embedDim > 0) (input : Tensor α [seqLen, embedDim])
    (scale : Tensor α [embedDim]) (outputGradient : Tensor α [seqLen, embedDim])
    (epsilon : α) : NormalizationGradients α [seqLen, embedDim] [embedDim] :=
  Spec.layerNormBackward h_seq_pos h_embed_pos input scale outputGradient
    epsilon

/-- Compile `Spec.layerNorm` through the native dispatcher. -/
@[csimp] theorem layerNorm_eq_fast : @Spec.layerNorm = @layerNormFast := rfl
/-- Compile `Spec.layerNormBackward` through the native dispatcher. -/
@[csimp] theorem layerNormBackward_eq_fast :
    @Spec.layerNormBackward = @layerNormBackwardFast := rfl

end TorchLean.FastCpu
