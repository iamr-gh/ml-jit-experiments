type 'd dim

val named : name:string -> size:int -> 'd dim
val dim_name : 'd dim -> string
val dim_size : 'd dim -> int

type ('n, 'a) vec
type ('m, 'n, 'a) mat
type ('b, 'm, 'n, 'a) batch3

module Vec : sig
  val dim : ('n, 'a) vec -> 'n dim
  val shape : ('n, 'a) vec -> int list
  val length : ('n, 'a) vec -> int
  val zeros : 'n dim -> zero:'a -> ('n, 'a) vec
  val of_array : 'n dim -> 'a array -> ('n, 'a) vec
  val of_list : 'n dim -> 'a list -> ('n, 'a) vec
  val to_array : ('n, 'a) vec -> 'a array
  val get : ('n, 'a) vec -> int -> 'a
  val set : ('n, 'a) vec -> int -> 'a -> unit
  val map : f:('a -> 'b) -> ('n, 'a) vec -> ('n, 'b) vec
  val add : add:('a -> 'a -> 'a) -> ('n, 'a) vec -> ('n, 'a) vec -> ('n, 'a) vec
  val sub : sub:('a -> 'a -> 'a) -> ('n, 'a) vec -> ('n, 'a) vec -> ('n, 'a) vec
  val mul_scalar : mul:('a -> 'a -> 'a) -> ('n, 'a) vec -> 'a -> ('n, 'a) vec
  val div_scalar : div:('a -> 'a -> 'a) -> ('n, 'a) vec -> 'a -> ('n, 'a) vec
  val equal : eq:('a -> 'a -> bool) -> ('n, 'a) vec -> ('n, 'a) vec -> bool
  val pp : (Format.formatter -> 'a -> unit) -> Format.formatter -> ('n, 'a) vec -> unit
  val to_string : ('a -> string) -> ('n, 'a) vec -> string
end

module Mat : sig
  val rows : ('m, 'n, 'a) mat -> 'm dim
  val cols : ('m, 'n, 'a) mat -> 'n dim
  val shape : ('m, 'n, 'a) mat -> int list
  val length : ('m, 'n, 'a) mat -> int
  val zeros : 'm dim -> 'n dim -> zero:'a -> ('m, 'n, 'a) mat
  val of_array : 'm dim -> 'n dim -> 'a array -> ('m, 'n, 'a) mat
  val of_list : 'm dim -> 'n dim -> 'a list -> ('m, 'n, 'a) mat
  val to_array : ('m, 'n, 'a) mat -> 'a array
  val get : ('m, 'n, 'a) mat -> int -> int -> 'a
  val set : ('m, 'n, 'a) mat -> int -> int -> 'a -> unit
  val map : f:('a -> 'b) -> ('m, 'n, 'a) mat -> ('m, 'n, 'b) mat
  val add : add:('a -> 'a -> 'a) -> ('m, 'n, 'a) mat -> ('m, 'n, 'a) mat -> ('m, 'n, 'a) mat
  val sub : sub:('a -> 'a -> 'a) -> ('m, 'n, 'a) mat -> ('m, 'n, 'a) mat -> ('m, 'n, 'a) mat
  val mul_scalar : mul:('a -> 'a -> 'a) -> ('m, 'n, 'a) mat -> 'a -> ('m, 'n, 'a) mat
  val div_scalar : div:('a -> 'a -> 'a) -> ('m, 'n, 'a) mat -> 'a -> ('m, 'n, 'a) mat
  val softmax : ('m, 'n, float) mat -> ('m, 'n, float) mat
  val reshape : 'x dim -> 'y dim -> ('m, 'n, 'a) mat -> ('x, 'y, 'a) mat
  val transpose : ('m, 'n, 'a) mat -> ('n, 'm, 'a) mat
  val matmul :
    zero:'a ->
    add:('a -> 'a -> 'a) ->
    mul:('a -> 'a -> 'a) ->
    ('m, 'k, 'a) mat ->
    ('k, 'n, 'a) mat ->
    ('m, 'n, 'a) mat
  val equal : eq:('a -> 'a -> bool) -> ('m, 'n, 'a) mat -> ('m, 'n, 'a) mat -> bool
  val pp : (Format.formatter -> 'a -> unit) -> Format.formatter -> ('m, 'n, 'a) mat -> unit
  val to_string : ('a -> string) -> ('m, 'n, 'a) mat -> string
end

module Batch3 : sig
  val batch : ('b, 'm, 'n, 'a) batch3 -> 'b dim
  val rows : ('b, 'm, 'n, 'a) batch3 -> 'm dim
  val cols : ('b, 'm, 'n, 'a) batch3 -> 'n dim
  val shape : ('b, 'm, 'n, 'a) batch3 -> int list
  val length : ('b, 'm, 'n, 'a) batch3 -> int
  val zeros : 'b dim -> 'm dim -> 'n dim -> zero:'a -> ('b, 'm, 'n, 'a) batch3
  val of_array : 'b dim -> 'm dim -> 'n dim -> 'a array -> ('b, 'm, 'n, 'a) batch3
  val of_list : 'b dim -> 'm dim -> 'n dim -> 'a list -> ('b, 'm, 'n, 'a) batch3
  val to_array : ('b, 'm, 'n, 'a) batch3 -> 'a array
  val get : ('b, 'm, 'n, 'a) batch3 -> int -> int -> int -> 'a
  val set : ('b, 'm, 'n, 'a) batch3 -> int -> int -> int -> 'a -> unit
  val map : f:('a -> 'b) -> ('b, 'm, 'n, 'a) batch3 -> ('b, 'm, 'n, 'b) batch3
  val add :
    add:('a -> 'a -> 'a) ->
    ('b, 'm, 'n, 'a) batch3 ->
    ('b, 'm, 'n, 'a) batch3 ->
    ('b, 'm, 'n, 'a) batch3
  val sub :
    sub:('a -> 'a -> 'a) ->
    ('b, 'm, 'n, 'a) batch3 ->
    ('b, 'm, 'n, 'a) batch3 ->
    ('b, 'm, 'n, 'a) batch3
  val transpose : ('b, 'm, 'n, 'a) batch3 -> ('b, 'n, 'm, 'a) batch3
  val equal :
    eq:('a -> 'a -> bool) ->
    ('b, 'm, 'n, 'a) batch3 ->
    ('b, 'm, 'n, 'a) batch3 ->
    bool
  val pp :
    (Format.formatter -> 'a -> unit) ->
    Format.formatter ->
    ('b, 'm, 'n, 'a) batch3 ->
    unit
  val to_string : ('a -> string) -> ('b, 'm, 'n, 'a) batch3 -> string
end

module Int : sig
  module Vec : sig
    val add : ('n, int) vec -> ('n, int) vec -> ('n, int) vec
    val sub : ('n, int) vec -> ('n, int) vec -> ('n, int) vec
    val mul_scalar : ('n, int) vec -> int -> ('n, int) vec
    val div_scalar : ('n, int) vec -> int -> ('n, int) vec
    val equal : ('n, int) vec -> ('n, int) vec -> bool
    val pp : Format.formatter -> ('n, int) vec -> unit
  end

  module Mat : sig
    val add : ('m, 'n, int) mat -> ('m, 'n, int) mat -> ('m, 'n, int) mat
    val sub : ('m, 'n, int) mat -> ('m, 'n, int) mat -> ('m, 'n, int) mat
    val mul_scalar : ('m, 'n, int) mat -> int -> ('m, 'n, int) mat
    val div_scalar : ('m, 'n, int) mat -> int -> ('m, 'n, int) mat
    val matmul : ('m, 'k, int) mat -> ('k, 'n, int) mat -> ('m, 'n, int) mat
    val equal : ('m, 'n, int) mat -> ('m, 'n, int) mat -> bool
    val pp : Format.formatter -> ('m, 'n, int) mat -> unit
  end

  module Batch3 : sig
    val add : ('b, 'm, 'n, int) batch3 -> ('b, 'm, 'n, int) batch3 -> ('b, 'm, 'n, int) batch3
    val sub : ('b, 'm, 'n, int) batch3 -> ('b, 'm, 'n, int) batch3 -> ('b, 'm, 'n, int) batch3
    val equal : ('b, 'm, 'n, int) batch3 -> ('b, 'm, 'n, int) batch3 -> bool
    val pp : Format.formatter -> ('b, 'm, 'n, int) batch3 -> unit
  end
end

module Float : sig
  val relu : float -> float
  val sigmoid : float -> float

  module Vec : sig
    val add : ('n, float) vec -> ('n, float) vec -> ('n, float) vec
    val sub : ('n, float) vec -> ('n, float) vec -> ('n, float) vec
    val mul_scalar : ('n, float) vec -> float -> ('n, float) vec
    val div_scalar : ('n, float) vec -> float -> ('n, float) vec
    val equal : ('n, float) vec -> ('n, float) vec -> bool
    val pp : Format.formatter -> ('n, float) vec -> unit
  end

  module Mat : sig
    val add : ('m, 'n, float) mat -> ('m, 'n, float) mat -> ('m, 'n, float) mat
    val sub : ('m, 'n, float) mat -> ('m, 'n, float) mat -> ('m, 'n, float) mat
    val mul_scalar : ('m, 'n, float) mat -> float -> ('m, 'n, float) mat
    val div_scalar : ('m, 'n, float) mat -> float -> ('m, 'n, float) mat
    val matmul : ('m, 'k, float) mat -> ('k, 'n, float) mat -> ('m, 'n, float) mat
    val softmax : ('m, 'n, float) mat -> ('m, 'n, float) mat
    val equal : ('m, 'n, float) mat -> ('m, 'n, float) mat -> bool
    val pp : Format.formatter -> ('m, 'n, float) mat -> unit
  end

  module Batch3 : sig
    val add : ('b, 'm, 'n, float) batch3 -> ('b, 'm, 'n, float) batch3 -> ('b, 'm, 'n, float) batch3
    val sub : ('b, 'm, 'n, float) batch3 -> ('b, 'm, 'n, float) batch3 -> ('b, 'm, 'n, float) batch3
    val equal : ('b, 'm, 'n, float) batch3 -> ('b, 'm, 'n, float) batch3 -> bool
    val pp : Format.formatter -> ('b, 'm, 'n, float) batch3 -> unit
  end
end
