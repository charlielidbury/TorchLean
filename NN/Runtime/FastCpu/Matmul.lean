/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Runtime.FastCpu.Basic
public import NN.Spec.Core.TensorReductionShape.LinearAlgebra

/-!
# Native CPU matrix products

`TorchLean.Tensor.matmulSpec` and `matmulBackwardSpec` are what the eager tape
records for `matmul`. For the packed `Float` backend their compiled code calls
one strided, multithreaded C kernel (`torchlean_fastcpu_bmm`) that keeps the
specification's reduction order: every output is `foldl (· + ·) 0` over the
products in increasing contraction index, as in `Spec.matMulSpec`.

Covered layouts (everything else runs the specification):

* forward: each operand's batch prefix either has the size of the result batch
  (the broadcast is a no-op on the row-major data) or size one (shared by
  every batch entry);
* backward: both batch prefixes equal the result batch, so
  `reduceFromBroadcastTo` only copies.

See `NN.Runtime.FastCpu.Basic` for the trust boundary.
-/

@[expose] public section

namespace TorchLean.FastCpu

open Spec

/--
`out[b] = L[b] @ R[b]` over strided views of packed buffers (strides in
elements: batch, row, column). The result is a contiguous `[batch, m, n]`
buffer. Only reachable from `unsafe` dispatchers.
-/
@[extern "torchlean_fastcpu_bmm"]
opaque bmmNative (lhs : @& FloatArray) (lBatch lRow lCol : USize)
    (rhs : @& FloatArray) (rBatch rRow rCol : USize)
    (batch m k n : USize) : FloatArray

/-- `bmmNative`, counted as a backward product in the statistics. -/
@[extern "torchlean_fastcpu_bmm_backward"]
opaque bmmBackwardNative (lhs : @& FloatArray) (lBatch lRow lCol : USize)
    (rhs : @& FloatArray) (rBatch rRow rCol : USize)
    (batch m k n : USize) : FloatArray

/--
Batch stride of an operand whose batch prefix `source` broadcasts to
`target`: the matrix size when the broadcast leaves the row-major data
unchanged, zero when one matrix is shared, `none` otherwise.
-/
def batchStride? (source target : Shape) (matrix : Nat) : Option Nat :=
  if source.size == target.size then some matrix
  else if source.size == 1 then some 0
  else none

/-- Compiled implementation of `matmulSpecFast`. -/
unsafe def matmulSpecImpl {α : Type} [storage : Storage α]
    [add : Add α] [mul : Mul α] [zero : Zero α]
    {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (A : Tensor α (batchA.concat [m, n]))
    (B : Tensor α (batchB.concat [n, p])) :
    Tensor α (batch.concat [m, p]) :=
  let reference := fun (_ : Unit) =>
    noteFallback Op.matmul
      (TorchLean.Tensor.matmulSpec broadcastA broadcastB A B)
  if isFloatBackend storage add mul zero then
    match batchStride? batchA batch (m * n),
        batchStride? batchB batch (n * p) with
    | some strideA, some strideB =>
        ofFloatBuffer <| bmmNative
          (floatBuffer A) strideA.toUSize n.toUSize 1
          (floatBuffer B) strideB.toUSize p.toUSize 1
          batch.size.toUSize m.toUSize n.toUSize p.toUSize
    | _, _ => reference ()
  else reference ()

/--
`TorchLean.Tensor.matmulSpec`, compiled through `matmulSpecImpl`.

Logically this is the specification itself, so `matmulSpec_eq_fast` is `rfl`.
-/
@[implemented_by matmulSpecImpl]
def matmulSpecFast {α : Type} [Storage α] [Add α] [Mul α] [Zero α]
    {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (A : Tensor α (batchA.concat [m, n]))
    (B : Tensor α (batchB.concat [n, p])) :
    Tensor α (batch.concat [m, p]) :=
  TorchLean.Tensor.matmulSpec broadcastA broadcastB A B

/-- Compiled implementation of `matmulBackwardSpecFast`. -/
unsafe def matmulBackwardSpecImpl {α : Type} [storage : Storage α]
    [add : Add α] [mul : Mul α] [zero : Zero α]
    {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (A : Tensor α (batchA.concat [m, n]))
    (B : Tensor α (batchB.concat [n, p]))
    (dC : Tensor α (batch.concat [m, p])) :
    Tensor α (batchA.concat [m, n]) × Tensor α (batchB.concat [n, p]) :=
  if isFloatBackend storage add mul zero && batchA == batch &&
      batchB == batch then
    let count := batch.size.toUSize
    let (m', n', p') := (m.toUSize, n.toUSize, p.toUSize)
    -- dA[b] = dC[b] @ B[b]ᵀ: R(j, k) = B[b, k, j].
    let dA := bmmBackwardNative
      (floatBuffer dC) (m' * p') p' 1
      (floatBuffer B) (n' * p') 1 p'
      count m' p' n'
    -- dB[b] = A[b]ᵀ @ dC[b]: L(k, i) = A[b, i, k].
    let dB := bmmBackwardNative
      (floatBuffer A) (m' * n') 1 n'
      (floatBuffer dC) (m' * p') p' 1
      count n' m' p'
    (ofFloatBuffer dA, ofFloatBuffer dB)
  else
    noteFallback Op.matmulBackward
      (TorchLean.Tensor.matmulBackwardSpec broadcastA broadcastB A B dC)

/--
`TorchLean.Tensor.matmulBackwardSpec`, compiled through
`matmulBackwardSpecImpl`.
-/
@[implemented_by matmulBackwardSpecImpl]
def matmulBackwardSpecFast {α : Type} [Storage α] [Add α] [Mul α] [Zero α]
    {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (A : Tensor α (batchA.concat [m, n]))
    (B : Tensor α (batchB.concat [n, p]))
    (dC : Tensor α (batch.concat [m, p])) :
    Tensor α (batchA.concat [m, n]) × Tensor α (batchB.concat [n, p]) :=
  TorchLean.Tensor.matmulBackwardSpec broadcastA broadcastB A B dC

/-- Compile `matmulSpec` through the native dispatcher. -/
@[csimp] theorem matmulSpec_eq_fast :
    @TorchLean.Tensor.matmulSpec = @matmulSpecFast := rfl

/-- Compile `matmulBackwardSpec` through the native dispatcher. -/
@[csimp] theorem matmulBackwardSpec_eq_fast :
    @TorchLean.Tensor.matmulBackwardSpec = @matmulBackwardSpecFast := rfl

end TorchLean.FastCpu
