module One : sig
  type t
  val dim : t Tensor.dim
  val size : int
end

module Two : sig
  type t
  val dim : t Tensor.dim
  val size : int
end

module Three : sig
  type t
  val dim : t Tensor.dim
  val size : int
end

module Four : sig
  type t
  val dim : t Tensor.dim
  val size : int
end

module Batch : sig
  type t
  val dim : t Tensor.dim
  val size : int
end

module Samples : sig
  type t
  val dim : t Tensor.dim
  val size : int
end

module Feature : sig
  type t
  val dim : t Tensor.dim
  val size : int
end

module Output : sig
  type t
  val dim : t Tensor.dim
  val size : int
end

module Seq : sig
  type t
  val dim : t Tensor.dim
  val size : int
end

module D_model : sig
  type t
  val dim : t Tensor.dim
  val size : int
end

module Heads : sig
  type t
  val dim : t Tensor.dim
  val size : int
end

module Head_dim : sig
  type t
  val dim : t Tensor.dim
  val size : int
end

module Vocab : sig
  type t
  val dim : t Tensor.dim
  val size : int
end
