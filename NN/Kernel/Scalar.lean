/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Kernel.Cuda
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Value.CoreProof

/-!
# Exact scalar encodings for custom computations

Native floats use hardware arithmetic. Configured binary values use their complete interchange
words; the device selects compatible native operations or software arithmetic from the format.
Wide values are never narrowed through binary64.
The packing law concerns Lean values, not the foreign transfer or arithmetic implementation.
-/

@[expose] public section

namespace NN.Kernel

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

/-- Arithmetic representation selected by a scalar type, independently of the execution device. -/
inductive Precision where
  | native (format : Cuda.Format)
  | binary (format : FloatFormat)

/-- Complete unsigned encodings and their lossless reconstruction. -/
class Scalar (α : Type) where
  precision : Precision
  encode : α → Nat
  decode : Nat → α
  decode_encode : ∀ x, decode (encode x) = x

instance : Scalar Float32 where
  precision := .native .binary32
  encode x := x.toBits.toNat
  decode n := Float32.ofBits n.toUInt32
  decode_encode x := by
    simpa only [Nat.toUInt32, UInt32.ofNat_toNat, Cuda.Literal.ofValue, Cuda.Literal.value] using
      Cuda.Literal.value_ofValue .binary32 x

instance : Scalar Float where
  precision := .native .binary64
  encode x := x.toBits.toNat
  decode n := Float.ofBits n.toUInt64
  decode_encode x := by
    simpa only [Nat.toUInt64, UInt64.ofNat_toNat, Cuda.Literal.ofValue, Cuda.Literal.value] using
      Cuda.Literal.value_ofValue .binary64 x

instance {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code] :
    Scalar (ExecFloat (Configured.Family format code plan)) where
  precision := .binary format
  encode := ExecFloat.Binary.toNatBits
  decode := ExecFloat.Binary.ofNatBits
  decode_encode := ExecFloat.Binary.ofNatBits_toNatBits

/-- Byte width of one complete value, padded only to a whole number of bytes. -/
def Precision.bytes : Precision → Nat
  | .native .binary32 => 4
  | .native .binary64 => 8
  | .binary format => (format.bitWidth + 7) / 8

/-- Whether the custom GPU arithmetic implementation accepts this complete scalar format.
This is a type/format capability check, not a claim that a GPU or enough memory is available. -/
def Precision.supportsGpu : Precision → Bool
  | .native _ => true
  | .binary format => format.expWidth ≤ 30 && format.fracWidth ≤ 4096

end NN.Kernel
