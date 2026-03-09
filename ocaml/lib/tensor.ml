type 'd dim = {
  name : string;
  size : int;
}

let named ~name ~size =
  if size < 0 then invalid_arg "dimension size must be non-negative";
  { name; size }

let dim_name dim = dim.name
let dim_size dim = dim.size

type ('n, 'a) vec = {
  vec_dim : 'n dim;
  vec_data : 'a array;
}

type ('m, 'n, 'a) mat = {
  mat_rows : 'm dim;
  mat_cols : 'n dim;
  mat_data : 'a array;
}

type ('b, 'm, 'n, 'a) batch3 = {
  batch_dim : 'b dim;
  batch_rows : 'm dim;
  batch_cols : 'n dim;
  batch_data : 'a array;
}

let check_index dim idx =
  if idx < 0 || idx >= dim then invalid_arg "tensor index out of bounds"

let check_data_length expected actual =
  if actual <> expected then invalid_arg "tensor data length does not match shape"

let map2_array_exn ~f left right =
  let left_len = Array.length left in
  let right_len = Array.length right in
  if left_len <> right_len then invalid_arg "tensor shape mismatch";
  Array.init left_len (fun i -> f left.(i) right.(i))

let pp_dims fmt dims =
  List.iteri
    (fun i dim ->
      if i > 0 then Format.pp_print_char fmt 'x';
      Format.pp_print_int fmt dim)
    dims

let pp_array pp_elem fmt data =
  Array.iteri
    (fun i value ->
      if i > 0 then Format.pp_print_string fmt ", ";
      pp_elem fmt value)
    data

module Vec = struct
  let dim t = t.vec_dim
  let shape t = [t.vec_dim.size]
  let length t = Array.length t.vec_data

  let zeros dim ~zero =
    { vec_dim = dim; vec_data = Array.make dim.size zero }

  let of_array dim data =
    check_data_length dim.size (Array.length data);
    { vec_dim = dim; vec_data = Array.copy data }

  let of_list dim values =
    of_array dim (Array.of_list values)

  let to_array t = Array.copy t.vec_data

  let get t i =
    check_index t.vec_dim.size i;
    t.vec_data.(i)

  let set t i value =
    check_index t.vec_dim.size i;
    t.vec_data.(i) <- value

  let map ~f t =
    { vec_dim = t.vec_dim; vec_data = Array.map f t.vec_data }

  let add ~add left right =
    { vec_dim = left.vec_dim; vec_data = map2_array_exn ~f:add left.vec_data right.vec_data }

  let sub ~sub left right =
    { vec_dim = left.vec_dim; vec_data = map2_array_exn ~f:sub left.vec_data right.vec_data }

  let mul_scalar ~mul t scalar =
    map ~f:(fun value -> mul value scalar) t

  let div_scalar ~div t scalar =
    map ~f:(fun value -> div value scalar) t

  let equal ~eq left right =
    left.vec_dim.size = right.vec_dim.size && Array.for_all2 eq left.vec_data right.vec_data

  let pp pp_elem fmt t =
    Format.pp_print_string fmt "Tensor(";
    pp_dims fmt (shape t);
    Format.pp_print_string fmt ")[";
    pp_array pp_elem fmt t.vec_data;
    Format.pp_print_char fmt ']'

  let to_string show_elem t =
    let pp_elem fmt value = Format.pp_print_string fmt (show_elem value) in
    Format.asprintf "%a" (pp pp_elem) t
end

module Mat = struct
  let rows t = t.mat_rows
  let cols t = t.mat_cols
  let shape t = [t.mat_rows.size; t.mat_cols.size]
  let length t = Array.length t.mat_data

  let zeros rows cols ~zero =
    { mat_rows = rows; mat_cols = cols; mat_data = Array.make (rows.size * cols.size) zero }

  let of_array rows cols data =
    check_data_length (rows.size * cols.size) (Array.length data);
    { mat_rows = rows; mat_cols = cols; mat_data = Array.copy data }

  let of_list rows cols values =
    of_array rows cols (Array.of_list values)

  let to_array t = Array.copy t.mat_data

  let get t i j =
    check_index t.mat_rows.size i;
    check_index t.mat_cols.size j;
    t.mat_data.((i * t.mat_cols.size) + j)

  let set t i j value =
    check_index t.mat_rows.size i;
    check_index t.mat_cols.size j;
    t.mat_data.((i * t.mat_cols.size) + j) <- value

  let map ~f t =
    { mat_rows = t.mat_rows; mat_cols = t.mat_cols; mat_data = Array.map f t.mat_data }

  let add ~add left right =
    if left.mat_rows.size <> right.mat_rows.size || left.mat_cols.size <> right.mat_cols.size then
      invalid_arg "tensor shape mismatch";
    {
      mat_rows = left.mat_rows;
      mat_cols = left.mat_cols;
      mat_data = map2_array_exn ~f:add left.mat_data right.mat_data;
    }

  let sub ~sub left right =
    if left.mat_rows.size <> right.mat_rows.size || left.mat_cols.size <> right.mat_cols.size then
      invalid_arg "tensor shape mismatch";
    {
      mat_rows = left.mat_rows;
      mat_cols = left.mat_cols;
      mat_data = map2_array_exn ~f:sub left.mat_data right.mat_data;
    }

  let mul_scalar ~mul t scalar =
    map ~f:(fun value -> mul value scalar) t

  let div_scalar ~div t scalar =
    map ~f:(fun value -> div value scalar) t

  let softmax t =
    if t.mat_cols.size = 0 then invalid_arg "softmax requires non-empty last dimension";
    let out = Array.make (Array.length t.mat_data) 0.0 in
    for row = 0 to t.mat_rows.size - 1 do
      let base = row * t.mat_cols.size in
      let max_value = ref t.mat_data.(base) in
      for j = 1 to t.mat_cols.size - 1 do
        let value = t.mat_data.(base + j) in
        if value > !max_value then max_value := value
      done;
      let sum = ref 0.0 in
      for j = 0 to t.mat_cols.size - 1 do
        let exp_value = exp (t.mat_data.(base + j) -. !max_value) in
        out.(base + j) <- exp_value;
        sum := !sum +. exp_value
      done;
      for j = 0 to t.mat_cols.size - 1 do
        out.(base + j) <- out.(base + j) /. !sum
      done
    done;
    { mat_rows = t.mat_rows; mat_cols = t.mat_cols; mat_data = out }

  let reshape rows cols t =
    check_data_length (rows.size * cols.size) (Array.length t.mat_data);
    { mat_rows = rows; mat_cols = cols; mat_data = Array.copy t.mat_data }

  let transpose t =
    let out_len = t.mat_rows.size * t.mat_cols.size in
    let out = if out_len = 0 then [||] else Array.make out_len t.mat_data.(0) in
    for i = 0 to t.mat_rows.size - 1 do
      for j = 0 to t.mat_cols.size - 1 do
        out.(j * t.mat_rows.size + i) <- t.mat_data.(i * t.mat_cols.size + j)
      done
    done;
    { mat_rows = t.mat_cols; mat_cols = t.mat_rows; mat_data = out }

  let matmul ~zero ~add ~mul left right =
    if left.mat_cols.size <> right.mat_rows.size then invalid_arg "matmul shape mismatch";
    let out = Array.make (left.mat_rows.size * right.mat_cols.size) zero in
    for i = 0 to left.mat_rows.size - 1 do
      for j = 0 to right.mat_cols.size - 1 do
        let sum = ref zero in
        for k = 0 to left.mat_cols.size - 1 do
          let left_value = left.mat_data.(i * left.mat_cols.size + k) in
          let right_value = right.mat_data.(k * right.mat_cols.size + j) in
          sum := add !sum (mul left_value right_value)
        done;
        out.(i * right.mat_cols.size + j) <- !sum
      done
    done;
    { mat_rows = left.mat_rows; mat_cols = right.mat_cols; mat_data = out }

  let equal ~eq left right =
    left.mat_rows.size = right.mat_rows.size &&
    left.mat_cols.size = right.mat_cols.size &&
    Array.for_all2 eq left.mat_data right.mat_data

  let pp pp_elem fmt t =
    Format.pp_print_string fmt "Tensor(";
    pp_dims fmt (shape t);
    Format.pp_print_string fmt ")[";
    pp_array pp_elem fmt t.mat_data;
    Format.pp_print_char fmt ']'

  let to_string show_elem t =
    let pp_elem fmt value = Format.pp_print_string fmt (show_elem value) in
    Format.asprintf "%a" (pp pp_elem) t
end

module Batch3 = struct
  let batch t = t.batch_dim
  let rows t = t.batch_rows
  let cols t = t.batch_cols
  let shape t = [t.batch_dim.size; t.batch_rows.size; t.batch_cols.size]
  let length t = Array.length t.batch_data

  let zeros batch rows cols ~zero =
    {
      batch_dim = batch;
      batch_rows = rows;
      batch_cols = cols;
      batch_data = Array.make (batch.size * rows.size * cols.size) zero;
    }

  let of_array batch rows cols data =
    check_data_length (batch.size * rows.size * cols.size) (Array.length data);
    { batch_dim = batch; batch_rows = rows; batch_cols = cols; batch_data = Array.copy data }

  let of_list batch rows cols values =
    of_array batch rows cols (Array.of_list values)

  let to_array t = Array.copy t.batch_data

  let get t b i j =
    check_index t.batch_dim.size b;
    check_index t.batch_rows.size i;
    check_index t.batch_cols.size j;
    t.batch_data.(((b * t.batch_rows.size) + i) * t.batch_cols.size + j)

  let set t b i j value =
    check_index t.batch_dim.size b;
    check_index t.batch_rows.size i;
    check_index t.batch_cols.size j;
    t.batch_data.(((b * t.batch_rows.size) + i) * t.batch_cols.size + j) <- value

  let map ~f t =
    {
      batch_dim = t.batch_dim;
      batch_rows = t.batch_rows;
      batch_cols = t.batch_cols;
      batch_data = Array.map f t.batch_data;
    }

  let add ~add left right =
    if left.batch_dim.size <> right.batch_dim.size ||
       left.batch_rows.size <> right.batch_rows.size ||
       left.batch_cols.size <> right.batch_cols.size
    then invalid_arg "tensor shape mismatch";
    {
      batch_dim = left.batch_dim;
      batch_rows = left.batch_rows;
      batch_cols = left.batch_cols;
      batch_data = map2_array_exn ~f:add left.batch_data right.batch_data;
    }

  let sub ~sub left right =
    if left.batch_dim.size <> right.batch_dim.size ||
       left.batch_rows.size <> right.batch_rows.size ||
       left.batch_cols.size <> right.batch_cols.size
    then invalid_arg "tensor shape mismatch";
    {
      batch_dim = left.batch_dim;
      batch_rows = left.batch_rows;
      batch_cols = left.batch_cols;
      batch_data = map2_array_exn ~f:sub left.batch_data right.batch_data;
    }

  let transpose t =
    let out_len = Array.length t.batch_data in
    let out = if out_len = 0 then [||] else Array.make out_len t.batch_data.(0) in
    for batch_idx = 0 to t.batch_dim.size - 1 do
      for i = 0 to t.batch_rows.size - 1 do
        for j = 0 to t.batch_cols.size - 1 do
          let src = (((batch_idx * t.batch_rows.size) + i) * t.batch_cols.size) + j in
          let dst = (((batch_idx * t.batch_cols.size) + j) * t.batch_rows.size) + i in
          out.(dst) <- t.batch_data.(src)
        done
      done
    done;
    { batch_dim = t.batch_dim; batch_rows = t.batch_cols; batch_cols = t.batch_rows; batch_data = out }

  let equal ~eq left right =
    left.batch_dim.size = right.batch_dim.size &&
    left.batch_rows.size = right.batch_rows.size &&
    left.batch_cols.size = right.batch_cols.size &&
    Array.for_all2 eq left.batch_data right.batch_data

  let pp pp_elem fmt t =
    Format.pp_print_string fmt "Tensor(";
    pp_dims fmt (shape t);
    Format.pp_print_string fmt ")[";
    pp_array pp_elem fmt t.batch_data;
    Format.pp_print_char fmt ']'

  let to_string show_elem t =
    let pp_elem fmt value = Format.pp_print_string fmt (show_elem value) in
    Format.asprintf "%a" (pp pp_elem) t
end

module Int = struct
  module Vec = struct
    let add left right = Vec.add ~add:( + ) left right
    let sub left right = Vec.sub ~sub:( - ) left right
    let mul_scalar t scalar = Vec.mul_scalar ~mul:( * ) t scalar
    let div_scalar t scalar = Vec.div_scalar ~div:( / ) t scalar
    let equal left right = Vec.equal ~eq:Int.equal left right
    let pp fmt t = Vec.pp Format.pp_print_int fmt t
  end

  module Mat = struct
    let add left right = Mat.add ~add:( + ) left right
    let sub left right = Mat.sub ~sub:( - ) left right
    let mul_scalar t scalar = Mat.mul_scalar ~mul:( * ) t scalar
    let div_scalar t scalar = Mat.div_scalar ~div:( / ) t scalar
    let matmul left right = Mat.matmul ~zero:0 ~add:( + ) ~mul:( * ) left right
    let equal left right = Mat.equal ~eq:Int.equal left right
    let pp fmt t = Mat.pp Format.pp_print_int fmt t
  end

  module Batch3 = struct
    let add left right = Batch3.add ~add:( + ) left right
    let sub left right = Batch3.sub ~sub:( - ) left right
    let equal left right = Batch3.equal ~eq:Int.equal left right
    let pp fmt t = Batch3.pp Format.pp_print_int fmt t
  end
end

module Float = struct
  let relu x = if x > 0.0 then x else 0.0
  let sigmoid x = 1.0 /. (1.0 +. exp (-.x))

  module Vec = struct
    let add left right = Vec.add ~add:( +. ) left right
    let sub left right = Vec.sub ~sub:( -. ) left right
    let mul_scalar t scalar = Vec.mul_scalar ~mul:( *. ) t scalar
    let div_scalar t scalar = Vec.div_scalar ~div:( /. ) t scalar
    let equal left right = Vec.equal ~eq:Float.equal left right
    let pp fmt t = Vec.pp Format.pp_print_float fmt t
  end

  module Mat = struct
    let add left right = Mat.add ~add:( +. ) left right
    let sub left right = Mat.sub ~sub:( -. ) left right
    let mul_scalar t scalar = Mat.mul_scalar ~mul:( *. ) t scalar
    let div_scalar t scalar = Mat.div_scalar ~div:( /. ) t scalar
    let matmul left right = Mat.matmul ~zero:0.0 ~add:( +. ) ~mul:( *. ) left right
    let softmax = Mat.softmax
    let equal left right = Mat.equal ~eq:Float.equal left right
    let pp fmt t = Mat.pp Format.pp_print_float fmt t
  end

  module Batch3 = struct
    let add left right = Batch3.add ~add:( +. ) left right
    let sub left right = Batch3.sub ~sub:( -. ) left right
    let equal left right = Batch3.equal ~eq:Float.equal left right
    let pp fmt t = Batch3.pp Format.pp_print_float fmt t
  end
end
