open Ocaml

let mean_squared_error predicted target =
  let diff = Tensor.Float.Mat.sub predicted target in
  let squared = Tensor.Mat.map ~f:(fun value -> value *. value) diff in
  let total =
    Tensor.Mat.to_array squared |> Array.fold_left ( +. ) 0.0
  in
  total /. float_of_int (Tensor.Mat.length squared)

let () =
  let x : (Dims.Samples.t, Dims.Feature.t, float) Tensor.mat =
    Tensor.Mat.of_list Dims.Samples.dim Dims.Feature.dim [1.0; 2.0; 3.0; 4.0]
  in
  let y : (Dims.Samples.t, Dims.Output.t, float) Tensor.mat =
    Tensor.Mat.of_list Dims.Samples.dim Dims.Output.dim [2.0; 4.0; 6.0; 8.0]
  in
  let w = ref (Tensor.Mat.of_list Dims.Feature.dim Dims.Output.dim [0.0]) in
  let learning_rate = 0.05 in
  let sample_scale = 2.0 /. float_of_int Dims.Samples.size in
  let initial_predictions = Tensor.Float.Mat.matmul x !w in
  let initial_loss = mean_squared_error initial_predictions y in
  for _ = 1 to 200 do
    let predictions = Tensor.Float.Mat.matmul x !w in
    let errors = Tensor.Float.Mat.sub predictions y in
    let gradient =
      Tensor.Float.Mat.matmul (Tensor.Mat.transpose x) errors
      |> fun value -> Tensor.Float.Mat.mul_scalar value sample_scale
    in
    w := Tensor.Float.Mat.sub !w (Tensor.Float.Mat.mul_scalar gradient learning_rate)
  done;
  let predictions : (Dims.Samples.t, Dims.Output.t, float) Tensor.mat =
    Tensor.Float.Mat.matmul x !w
  in
  let learned_weight = Tensor.Mat.get !w 0 0 in
  let final_loss = mean_squared_error predictions y in
  Printf.printf "initial loss: %.6f\nfinal loss: %.6f\nlearned weight: %.6f\npredictions: %s\n"
    initial_loss
    final_loss
    learned_weight
    (Tensor.Mat.to_string string_of_float predictions)
