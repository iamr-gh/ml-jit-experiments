let trim = String.trim

let split_fields line =
  line
  |> String.split_on_char ' '
  |> List.filter (fun field -> field <> "")

let read_entries path =
  let lines = In_channel.with_open_bin path In_channel.input_lines in
  let parse_line line =
    match split_fields (trim line) with
    | [] -> None
    | [module_name; size; label] -> Some (module_name, int_of_string size, label)
    | _ -> invalid_arg "each dims manifest line must be: ModuleName size label"
  in
  List.filter_map parse_line lines

let write_file path contents =
  Out_channel.with_open_bin path (fun channel -> output_string channel contents)

let render_mli entries =
  let render_entry (module_name, _, _) =
    Printf.sprintf
      "module %s : sig\n  type t\n  val dim : t Tensor.dim\n  val size : int\nend\n"
      module_name
  in
  String.concat "\n" (List.map render_entry entries)

let render_ml entries =
  let render_entry (module_name, size, label) =
    Printf.sprintf
      "module %s = struct\n  type t = unit\n  let dim : t Tensor.dim = Tensor.named ~name:%S ~size:%d\n  let size = %d\nend\n"
      module_name label size size
  in
  String.concat "\n" (List.map render_entry entries)

let () =
  match Array.to_list Sys.argv with
  | [_; manifest_path; mli_path; ml_path] ->
      let entries = read_entries manifest_path in
      write_file mli_path (render_mli entries);
      write_file ml_path (render_ml entries)
  | _ ->
      prerr_endline "usage: ocaml tools/gen_dims.ml <manifest> <dims.mli> <dims.ml>";
      exit 1
