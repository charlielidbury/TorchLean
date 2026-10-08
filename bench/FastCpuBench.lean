import NN.API

/-!
# Eager CPU micro-benchmark (fastcpu track)

Times TorchLean's eager CPU path through the public objective API
(`Module.instantiate`, `Objective.loss`, `Objective.grad`) on the shapes
that dominate a nanoTabPFN training step. Every tensor is trainable state,
so `grad` differentiates both operands of each op.

```bash
lake exe fastcpu_bench mm 28800 96 96
lake exe fastcpu_bench bmm 192 150 24 150
lake exe fastcpu_bench ln 28800 96
lake exe fastcpu_bench softmax 192 150 150
lake exe fastcpu_bench add 28800 96
```
-/

open TorchLean Runtime.Autograd.Torch

namespace FastCpuBench

/-- Deterministic pseudo-random values in `[-1, 1)`. -/
def pseudo (seed i : Nat) : Float :=
  let x := (i * 2654435761 + seed * 40503 + 17) % 1000003
  x.toFloat / 500001.5 - 1.0

/-- A deterministic pseudo-random `Float` tensor. -/
def randTensor (s : Shape) (seed : Nat) : Tensor Float s :=
  TorchLean.Tensor.Internal.Rep.ofFlatFn fun i => pseudo seed i.val

/-- Run `action` `reps` times; return the best and mean wall time in ms. -/
def timeIt (reps : Nat) (action : IO Unit) : IO (Float × Float) := do
  let mut best := 1e30
  let mut total := 0.0
  for _ in [0:reps] do
    let t0 ← IO.monoNanosNow
    action
    let t1 ← IO.monoNanosNow
    let ms := (t1 - t0).toFloat / 1e6
    best := min best ms
    total := total + ms
  pure (best, total / reps.toFloat)

/-- Time the forward loss and the full gradient of one objective. -/
def benchObjective {ss : List Shape} (label : String) (flops : Float)
    (reps : Nat) (definition : Module.ObjectiveDefinition Unit ss []) :
    IO Unit := do
  let objective ← Module.instantiate definition {} (α := Float)
  let lossValue ← objective.loss Arguments.empty Arguments.empty
  let (fwdBest, fwdMean) ← timeIt reps do
    let _ ← objective.loss Arguments.empty Arguments.empty
  let (gradBest, gradMean) ← timeIt reps do
    let g ← objective.grad Arguments.empty Arguments.empty
    -- Force the gradient pack so lazy work cannot escape the timer.
    let _ := g
  let gflops (ms : Float) := flops / (ms * 1e6)
  IO.println s!"{label}: loss={lossValue.item}"
  IO.println s!"  fwd      best {fwdBest} ms  mean {fwdMean} ms  \
    ({gflops fwdBest} GFLOP/s on {flops / 1e9} GFLOP)"
  IO.println s!"  fwd+bwd  best {gradBest} ms  mean {gradMean} ms  \
    ({gflops gradBest} GFLOP/s on {3 * flops / 1e9} GFLOP)"

/-- `sum(A @ W)` with `A : [M, K]`, `W : [K, N]`. -/
def matmulDef (M K N : Nat) :
    Module.ObjectiveDefinition Unit [[M, K], [K, N]] [] :=
  { initState := .cons (randTensor [M, K] 1) (.cons (randTensor [K, N] 2) .nil)
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        CurriedRef.curry (Ref := Ref (m := m) (α := α))
          (ss := [[M, K], [K, N]])
          (β := m (Ref (m := m) (α := α) ([] : Shape))) fun args => do
            let .cons a (.cons w .nil) := args
            let y ← matmul (m := m) (α := α) (batchA := []) (batchB := [])
              (batch := []) (mDim := M) (nDim := K) (pDim := N) a w
            sum (m := m) (α := α) y }

/-- `sum(A @ B)` with `A : [B, M, K]`, `B : [B, K, N]` (attention scores). -/
def bmmDef (B M K N : Nat) :
    Module.ObjectiveDefinition Unit [[B, M, K], [B, K, N]] [] :=
  { initState := .cons (randTensor [B, M, K] 1) (.cons (randTensor [B, K, N] 2) .nil)
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        CurriedRef.curry (Ref := Ref (m := m) (α := α))
          (ss := [[B, M, K], [B, K, N]])
          (β := m (Ref (m := m) (α := α) ([] : Shape))) fun args => do
            let .cons a (.cons b .nil) := args
            let y ← matmul (m := m) (α := α) (batchA := [B]) (batchB := [B])
              (batch := [B]) (mDim := M) (nDim := K) (pDim := N) a b
            sum (m := m) (α := α) y }

/-- `sum(layerNorm(x) * x)` over rows of `x : [R, E]`. -/
def layerNormDef (R E : Nat) (hR : R > 0) (hE : E > 0) :
    Module.ObjectiveDefinition Unit [[R, E], [E], [E]] [] :=
  { initState := .cons (randTensor [R, E] 1)
      (.cons (randTensor [E] 2) (.cons (randTensor [E] 3) .nil))
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        CurriedRef.curry (Ref := Ref (m := m) (α := α))
          (ss := [[R, E], [E], [E]])
          (β := m (Ref (m := m) (α := α) ([] : Shape))) fun args => do
            let .cons x (.cons g (.cons b .nil)) := args
            let y ← layerNorm (m := m) (α := α) hR hE x g b
            let z ← mul (m := m) (α := α) y x
            sum (m := m) (α := α) z }

/-- `sum(softmax(x) * x)` over the last axis of `x : [B, M, N]`. -/
def softmaxDef (B M N : Nat) :
    Module.ObjectiveDefinition Unit [[B, M, N]] [] :=
  { initState := .cons (randTensor [B, M, N] 1) .nil
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        CurriedRef.curry (Ref := Ref (m := m) (α := α))
          (ss := [[B, M, N]])
          (β := m (Ref (m := m) (α := α) ([] : Shape))) fun args => do
            let .cons x .nil := args
            let y ← softmaxLast (m := m) (α := α) x
            let z ← mul (m := m) (α := α) y x
            sum (m := m) (α := α) z }

/-- `sum(a + b)` with `a b : [R, E]`. -/
def addDef (R E : Nat) :
    Module.ObjectiveDefinition Unit [[R, E], [R, E]] [] :=
  { initState := .cons (randTensor [R, E] 1) (.cons (randTensor [R, E] 2) .nil)
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        CurriedRef.curry (Ref := Ref (m := m) (α := α))
          (ss := [[R, E], [R, E]])
          (β := m (Ref (m := m) (α := α) ([] : Shape))) fun args => do
            let .cons a (.cons b .nil) := args
            let y ← add (m := m) (α := α) a b
            sum (m := m) (α := α) y }

/-- `sum(gelu(x))` with `x : [R, E]`. -/
def geluDef (R E : Nat) :
    Module.ObjectiveDefinition Unit [[R, E]] [] :=
  { initState := .cons (randTensor [R, E] 1) .nil
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        CurriedRef.curry (Ref := Ref (m := m) (α := α))
          (ss := [[R, E]])
          (β := m (Ref (m := m) (α := α) ([] : Shape))) fun args => do
            let .cons x .nil := args
            let y ← gelu (m := m) (α := α) x
            sum (m := m) (α := α) y }

def usage : String :=
  "usage: fastcpu_bench [--reps N] (mm M K N | bmm B M K N | ln R E | \
   softmax B M N | add R E | gelu R E)"

def parseNats (args : List String) : Option (List Nat) :=
  args.mapM String.toNat?

def run (reps : Nat) : List String → IO Unit
  | "mm" :: rest => do
      let some [M, K, N] := parseNats rest | throw <| IO.userError usage
      benchObjective s!"mm [{M}x{K}]@[{K}x{N}]"
        (2 * M.toFloat * K.toFloat * N.toFloat) reps (matmulDef M K N)
  | "bmm" :: rest => do
      let some [B, M, K, N] := parseNats rest | throw <| IO.userError usage
      benchObjective s!"bmm {B}x[{M}x{K}]@[{K}x{N}]"
        (2 * B.toFloat * M.toFloat * K.toFloat * N.toFloat) reps
        (bmmDef B M K N)
  | "ln" :: rest => do
      let some [R, E] := parseNats rest | throw <| IO.userError usage
      if hR : R > 0 then
        if hE : E > 0 then
          benchObjective s!"layerNorm [{R}x{E}]" (R.toFloat * E.toFloat) reps
            (layerNormDef R E hR hE)
        else throw <| IO.userError "E must be positive"
      else throw <| IO.userError "R must be positive"
  | "softmax" :: rest => do
      let some [B, M, N] := parseNats rest | throw <| IO.userError usage
      benchObjective s!"softmax [{B}x{M}x{N}]"
        (B.toFloat * M.toFloat * N.toFloat) reps (softmaxDef B M N)
  | "add" :: rest => do
      let some [R, E] := parseNats rest | throw <| IO.userError usage
      benchObjective s!"add [{R}x{E}]" (R.toFloat * E.toFloat) reps (addDef R E)
  | "gelu" :: rest => do
      let some [R, E] := parseNats rest | throw <| IO.userError usage
      benchObjective s!"gelu [{R}x{E}]" (R.toFloat * E.toFloat) reps
        (geluDef R E)
  | _ => throw <| IO.userError usage

end FastCpuBench

def main (args : List String) : IO Unit := do
  match args with
  | "--reps" :: n :: rest =>
      FastCpuBench.run (n.toNat?.getD 3) rest
  | _ => FastCpuBench.run 3 args
