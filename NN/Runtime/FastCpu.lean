/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Runtime.FastCpu.Basic
public import NN.Runtime.FastCpu.Matmul
public import NN.Runtime.FastCpu.Pointwise
public import NN.Runtime.FastCpu.Shape
public import NN.Runtime.FastCpu.Softmax
public import NN.Runtime.FastCpu.LayerNorm

/-!
# Native CPU fast paths

Compiled-code replacements (`@[csimp]`) for the specifications the eager tape
evaluates, active in every module that imports this one. See
`NN.Runtime.FastCpu.Basic` for the dispatch rule and the trust boundary.
-/
