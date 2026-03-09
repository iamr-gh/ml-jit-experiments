open Ocaml

let approx_equal a b =
  Stdlib.Float.abs (a -. b) < 1e-4

let expect condition message =
  if not condition then failwith message

let () =
  let matrix_2x2 = Tensor.Mat.of_list Dims.Two.dim Dims.Two.dim [1; 2; 3; 4] in
  let matrix_2x3 = Tensor.Mat.of_list Dims.Two.dim Dims.Three.dim [1; 2; 3; 4; 5; 6] in
  let matrix_3x2 = Tensor.Mat.of_list Dims.Three.dim Dims.Two.dim [1; 2; 3; 4; 5; 6] in
  let batch_2x2x3 =
    Tensor.Batch3.of_list Dims.Batch.dim Dims.Two.dim Dims.Three.dim
      [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]
  in
  let vector_2 = Tensor.Vec.of_list Dims.Two.dim [0.0; 1.0] in

  let a = matrix_2x2 in
  let b = Tensor.Mat.of_list Dims.Two.dim Dims.Two.dim [10; 20; 30; 40] in
  let c = Tensor.Int.Mat.add a b in
  expect
    (Tensor.Int.Mat.equal c (Tensor.Mat.of_list Dims.Two.dim Dims.Two.dim [11; 22; 33; 44]))
    "tensor add";

  let reshaped = Tensor.Mat.reshape Dims.Three.dim Dims.Two.dim matrix_2x3 in
  expect
    (Tensor.Int.Mat.equal reshaped matrix_3x2)
    "tensor reshape";

  let transposed = Tensor.Mat.transpose matrix_2x3 in
  expect
    (Tensor.Int.Mat.equal transposed (Tensor.Mat.of_list Dims.Three.dim Dims.Two.dim [1; 4; 2; 5; 3; 6]))
    "tensor transpose last two dims";

  let batched_t = Tensor.Batch3.transpose batch_2x2x3 in
  expect
    (Tensor.Int.Batch3.equal batched_t
       (Tensor.Batch3.of_list Dims.Batch.dim Dims.Three.dim Dims.Two.dim
          [1; 4; 2; 5; 3; 6; 7; 10; 8; 11; 9; 12]))
    "tensor transpose batched last two dims";

  let fa = Tensor.Mat.of_list Dims.Two.dim Dims.Three.dim [1.0; 2.0; 3.0; 4.0; 5.0; 6.0] in
  let fb = Tensor.Mat.of_list Dims.Three.dim Dims.Two.dim [7.0; 8.0; 9.0; 10.0; 11.0; 12.0] in
  let fc = Tensor.Float.Mat.matmul fa fb in
  let (_ : (Dims.Two.t, Dims.Two.t, float) Tensor.mat) = fc in
  expect
    (Tensor.Float.Mat.equal fc (Tensor.Mat.of_list Dims.Two.dim Dims.Two.dim [58.0; 64.0; 139.0; 154.0]))
    "tensor matmul";

  let helper_a = Tensor.Mat.of_list Dims.Two.dim Dims.Two.dim [1.0; 2.0; 3.0; 4.0] in
  let helper_b = Tensor.Mat.of_list Dims.Two.dim Dims.Two.dim [10.0; 20.0; 30.0; 40.0] in
  expect
    (Tensor.Float.Mat.equal
       (Tensor.Float.Mat.add helper_a helper_b)
       (Tensor.Mat.of_list Dims.Two.dim Dims.Two.dim [11.0; 22.0; 33.0; 44.0]))
    "tensor helper constructor";

  let relu_actual =
    Tensor.Mat.map ~f:Tensor.Float.relu
      (Tensor.Mat.of_list Dims.Two.dim Dims.Two.dim [-1.0; 2.0; -3.0; 4.0])
  in
  expect
    (Tensor.Float.Mat.equal relu_actual (Tensor.Mat.of_list Dims.Two.dim Dims.Two.dim [0.0; 2.0; 0.0; 4.0]))
    "tensor elementwise relu";

  let sigmoid_actual = Tensor.Vec.map ~f:Tensor.Float.sigmoid vector_2 in
  expect (approx_equal (Tensor.Vec.get sigmoid_actual 0) 0.5) "tensor elementwise sigmoid first";
  expect (approx_equal (Tensor.Vec.get sigmoid_actual 1) 0.7310586) "tensor elementwise sigmoid second";

  let divided =
    Tensor.Float.Mat.div_scalar
      (Tensor.Mat.of_list Dims.Two.dim Dims.Two.dim [2.0; 4.0; 6.0; 8.0])
      2.0
  in
  expect
    (Tensor.Float.Mat.equal divided (Tensor.Mat.of_list Dims.Two.dim Dims.Two.dim [1.0; 2.0; 3.0; 4.0]))
    "tensor divide by scalar";

  let softmax_actual =
    Tensor.Float.Mat.softmax
      (Tensor.Mat.of_list Dims.Two.dim Dims.Three.dim [1.0; 2.0; 3.0; 1.0; 1.0; 1.0])
  in
  expect (approx_equal (Tensor.Mat.get softmax_actual 0 0) 0.09003057) "tensor softmax 0";
  expect (approx_equal (Tensor.Mat.get softmax_actual 0 1) 0.24472848) "tensor softmax 1";
  expect (approx_equal (Tensor.Mat.get softmax_actual 0 2) 0.66524094) "tensor softmax 2";
  expect (approx_equal (Tensor.Mat.get softmax_actual 1 0) 0.33333334) "tensor softmax 3";
  expect (approx_equal (Tensor.Mat.get softmax_actual 1 1) 0.33333334) "tensor softmax 4";
  expect (approx_equal (Tensor.Mat.get softmax_actual 1 2) 0.33333334) "tensor softmax 5";

  expect
    (Tensor.Mat.to_string string_of_int (Tensor.Mat.of_list Dims.Two.dim Dims.Two.dim [1; 2; 3; 4])
     = "Tensor(2x2)[1, 2, 3, 4]")
    "tensor format";

  let named = Tensor.Mat.of_list Dims.Two.dim Dims.One.dim [5; 6] in
  expect (Tensor.Mat.to_string string_of_int named = "Tensor(2x1)[5, 6]") "named shape format";

  let named_float = Tensor.Mat.of_list Dims.Two.dim Dims.Two.dim [1.0; 2.0; 3.0; 4.0] in
  let (_ : (Dims.Two.t, Dims.Two.t, float) Tensor.mat) = named_float in
  expect (Tensor.Mat.shape named_float = [2; 2]) "named shape rank";
  expect (Tensor.Mat.length named_float = 4) "named shape length";

  let mutable_vec = Tensor.Vec.of_list Dims.Two.dim [1; 2] in
  Tensor.Vec.set mutable_vec 1 9;
  expect (Tensor.Vec.get mutable_vec 1 = 9) "set/get vec";

  let mutable_mat = Tensor.Mat.of_list Dims.Two.dim Dims.Two.dim [1; 2; 3; 4] in
  Tensor.Mat.set mutable_mat 1 0 8;
  expect (Tensor.Mat.get mutable_mat 1 0 = 8) "set/get mat";

  let mutable_batch = Tensor.Batch3.of_list Dims.Batch.dim Dims.One.dim Dims.Two.dim [1; 2; 3; 4] in
  Tensor.Batch3.set mutable_batch 1 0 1 7;
  expect (Tensor.Batch3.get mutable_batch 1 0 1 = 7) "set/get batch3";

  expect (Tensor.dim_size Dims.Vocab.dim = 100000) "large generated dimension"
