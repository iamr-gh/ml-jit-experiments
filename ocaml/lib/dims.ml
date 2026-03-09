module One = struct
  type t = unit
  let dim : t Tensor.dim = Tensor.named ~name:"one" ~size:1
  let size = 1
end

module Two = struct
  type t = unit
  let dim : t Tensor.dim = Tensor.named ~name:"two" ~size:2
  let size = 2
end

module Three = struct
  type t = unit
  let dim : t Tensor.dim = Tensor.named ~name:"three" ~size:3
  let size = 3
end

module Four = struct
  type t = unit
  let dim : t Tensor.dim = Tensor.named ~name:"four" ~size:4
  let size = 4
end

module Batch = struct
  type t = unit
  let dim : t Tensor.dim = Tensor.named ~name:"batch" ~size:2
  let size = 2
end

module Samples = struct
  type t = unit
  let dim : t Tensor.dim = Tensor.named ~name:"samples" ~size:4
  let size = 4
end

module Feature = struct
  type t = unit
  let dim : t Tensor.dim = Tensor.named ~name:"feature" ~size:1
  let size = 1
end

module Output = struct
  type t = unit
  let dim : t Tensor.dim = Tensor.named ~name:"output" ~size:1
  let size = 1
end

module Seq = struct
  type t = unit
  let dim : t Tensor.dim = Tensor.named ~name:"seq" ~size:128
  let size = 128
end

module D_model = struct
  type t = unit
  let dim : t Tensor.dim = Tensor.named ~name:"d_model" ~size:768
  let size = 768
end

module Heads = struct
  type t = unit
  let dim : t Tensor.dim = Tensor.named ~name:"heads" ~size:12
  let size = 12
end

module Head_dim = struct
  type t = unit
  let dim : t Tensor.dim = Tensor.named ~name:"head_dim" ~size:64
  let size = 64
end

module Vocab = struct
  type t = unit
  let dim : t Tensor.dim = Tensor.named ~name:"vocab" ~size:100000
  let size = 100000
end
