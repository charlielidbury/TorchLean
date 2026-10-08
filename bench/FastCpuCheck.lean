import NN.API

/-!
# End-to-end bit-exactness check for the fastcpu kernels

Evaluates small objectives that exercise every natively dispatched operation
(forward and backward) and prints, per objective, the loss bits and a hash of
every gradient bit pattern. Compare the output of

```bash
TORCHLEAN_FASTCPU=0 lake exe fastcpu_check > ref.txt
lake exe fastcpu_check > fast.txt && diff ref.txt fast.txt
```
-/

open TorchLean Spec Runtime.Autograd.Torch

namespace FastCpuCheck

/-- Pseudo-random values in `[-2, 2)`, with exact `0.0` and `-0.0` mixed in. -/
def value (seed i : Nat) : Float :=
  let x := (i * 2654435761 + seed * 40503 + 17) % 1000003
  if x % 11 == 0 then 0.0
  else if x % 13 == 0 then -0.0
  else x.toFloat / 250000.75 - 2.0

/-- A deterministic tensor with special values. -/
def tensor (s : Shape) (seed : Nat) : Tensor Float s :=
  TorchLean.Tensor.Internal.Rep.ofFlatFn fun i => value seed i.val

/-- FNV-style mix of 64-bit words. -/
def mix (h w : UInt64) : UInt64 := (h ^^^ w) * 1099511628211

/-- Hash of every bit pattern in a tensor pack. -/
def hashPack : {ss : List Shape} → TensorPack Float ss → UInt64 → UInt64
  | [], .nil, h => h
  | _ :: _, .cons t rest, h =>
      hashPack rest (t.data.foldl (fun h x => mix h x.toBits) h)

/-- Print the loss bits and gradient hash of one objective. -/
def check {ss : List Shape} (label : String)
    (definition : Module.ObjectiveDefinition Unit ss []) : IO Unit := do
  let objective ← Module.instantiate definition {} (α := Float)
  let (g, l) ← objective.grad Arguments.empty Arguments.empty (value := true)
  let h := hashPack (nn.State.Internal.toTensorPack g) 14695981039346656037
  IO.println s!"{label}: loss {l.item} bits {l.item.toBits} grad {h}"

section

variable {α : Type} [TorchLean.Storage α] [Context α]
  {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]

local notation "Ref'" s:max => Ref (m := m) (α := α) s

/-- `exp(-x²/2)·|x|/(1+|x|) + relu(x) - 0.5·x`, then a weighted sum. -/
def pointwise {s : Shape} (x w : Ref' s) : m (Ref' []) := do
  let a ← abs (m := m) (α := α) x
  let one ← const (m := m) (α := α) (Tensor.full s (1 : α))
  let t ← inv (m := m) (α := α) (← add (m := m) (α := α) a one)
  let x2 ← mul (m := m) (α := α) x x
  let e ← exp (m := m) (α := α) (← scale (m := m) (α := α) x2 (-1 / 2))
  let tail ← mul (m := m) (α := α) (← mul (m := m) (α := α) e a) t
  let r ← relu (m := m) (α := α) x
  let y ← sub (m := m) (α := α) (← add (m := m) (α := α) r tail)
    (← scale (m := m) (α := α) x (1 / 2))
  sum (m := m) (α := α) (← mul (m := m) (α := α) y w)

end

/-- Wrap a two-state loss as an objective. -/
def obj2 {s₁ s₂ : Shape} (a : Tensor Float s₁) (b : Tensor Float s₂)
    (loss : ∀ {α : Type} [TorchLean.Storage α] [Context α]
      {m : Type → Type} [Monad m] [Ops (m := m) (α := α)],
      Ref (m := m) (α := α) s₁ → Ref (m := m) (α := α) s₂ →
        m (Ref (m := m) (α := α) [])) :
    Module.ObjectiveDefinition Unit [s₁, s₂] [] :=
  { initState := .cons a (.cons b .nil)
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        CurriedRef.curry (Ref := Ref (m := m) (α := α)) (ss := [s₁, s₂])
          (β := m (Ref (m := m) (α := α) ([] : Shape))) fun args => do
            let .cons x (.cons y .nil) := args
            loss x y }

/-- Wrap a three-state loss as an objective. -/
def obj3 {s₁ s₂ s₃ : Shape} (a : Tensor Float s₁) (b : Tensor Float s₂)
    (c : Tensor Float s₃)
    (loss : ∀ {α : Type} [TorchLean.Storage α] [Context α]
      {m : Type → Type} [Monad m] [Ops (m := m) (α := α)],
      Ref (m := m) (α := α) s₁ → Ref (m := m) (α := α) s₂ →
        Ref (m := m) (α := α) s₃ → m (Ref (m := m) (α := α) [])) :
    Module.ObjectiveDefinition Unit [s₁, s₂, s₃] [] :=
  { initState := .cons a (.cons b (.cons c .nil))
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        CurriedRef.curry (Ref := Ref (m := m) (α := α)) (ss := [s₁, s₂, s₃])
          (β := m (Ref (m := m) (α := α) ([] : Shape))) fun args => do
            let .cons x (.cons y (.cons z .nil)) := args
            loss x y z }

def main : IO Unit := do
  check "matmul 2d" <| obj2 (tensor [37, 19] 1) (tensor [19, 23] 2)
    fun x w => do
      let y ← matmul (batchA := []) (batchB := []) (batch := [])
        (mDim := 37) (nDim := 19) (pDim := 23) x w
      sum (← mul y y)
  check "matmul batched 4d" <|
    obj2 (tensor [2, 3, 5, 7] 3) (tensor [2, 3, 7, 6] 4) fun x w => do
      let y ← matmul (batchA := [2, 3]) (batchB := [2, 3]) (batch := [2, 3])
        (mDim := 5) (nDim := 7) (pDim := 6) x w
      sum (← mul y y)
  check "matmul shared weight" <|
    obj2 (tensor [3, 5, 7] 5) (tensor [7, 4] 6) fun x w => do
      let y ← matmul (batchA := [3]) (batchB := []) (batch := [3])
        (mDim := 5) (nDim := 7) (pDim := 4) x w
      sum (← mul y y)
  check "pointwise" <| obj2 (tensor [9, 31] 7) (tensor [9, 31] 8)
    fun x w => pointwise x w
  check "softmax" <| obj2 (tensor [4, 6, 9] 9) (tensor [4, 6, 9] 10)
    fun x w => do
      sum (← mul (← softmaxLast (← scale x 3)) w)
  check "layerNorm" <| obj3 (tensor [13, 8] 11) (tensor [8] 12) (tensor [8] 13)
    fun x g b => do
      let y ← layerNorm (seqLen := 13) (embedDim := 8) (by decide) (by decide)
        x g b
      sum (← mul y x)
  check "broadcast + swap" <|
    obj3 (tensor [4, 3, 5] 14) (tensor [5] 15) (tensor [5, 4, 3] 16)
    fun x b w => do
      let bb ← broadcastTo (s₁ := [5]) (s₂ := [4, 3, 5])
        Shape.BroadcastTo.proof b
      let y ← add x bb
      let z ← swapAdjacentAtDepth 0 (← swapAdjacentAtDepth 1 y)
      let z : Ref (s := [5, 4, 3]) := z
      sum (← mul (← mul z z) w)

end FastCpuCheck

def main : IO Unit := FastCpuCheck.main
