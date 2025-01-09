# 🧮 Algebraic Positional Encodings

This repository implements the methods and experiments described [here](https://arxiv.org/abs/2312.16045) [spotlight @ NeurIPS 2024].

Long story short, we substitute the dot-product attention with a position-dependent bilinear scalar function.

We obtain such functions by a homomorphic interpretation of the IO data types/structures onto subgroups of the orthogonal group.
We examine and provide implementations for the following cases:
* **Sequences**. 
We have $`α(q, k) = qW^dk`$ where $`W`$ is a parameterized orthogonal matrix and $`d`$ is the relative distance
between the query and the key.
* **Grids**. 
We have $`a(q, k) = q (W_1^{d_1} \oplus W_2^{d_2} \oplus \dots) k`$ where $`W_i`$ is an orthogonal matrix,
$`d_i`$ the distance between query and key on axis $`i`$ and $`\oplus`$ the matrix direct sum.  
* **Trees**. 
We have $`α(q, k) = q(W_{|p[0]|}^{\mathrm{sgn}(p[0])}W_{|p[1]|}^{\mathrm{sgn}(p[1])}...W_{|p[t]|}^{\mathrm{sgn}(p[t])})k`$ 
where $`W`$ is a 4-dimensional tensor containing an orthogonal matrix for each tree branch, 
and $`p`$ is a vector denoting the minimal *path* of signed steps from a query node to a key node.

Parallelism is maintained by decomposing the bilinear function into two linear functions applied independently (batched)
on the queries/keys. 

Composites of existing cases can be obtained by taking the direct sum of the appropriate primitives (DIY).

See the paper for more details.

## Implementation
If you want to use this with your own work, you will need to make a few simple changes to your transformer's codebase.
The current implementation allows for generic Transformer layers by having them accept the attention function they are
to use as an extra argument in their forward pass. The pipeline is as follows:
1. obtain *absolute* positional encoding matrices through some algebraic encoder (see `ape.nn.positions.algebraic`)
2. ask the positional encoder for an attention function given the absolute positional encodings of the queries/keys
   (see `ape.nn.positions.schemes` if writing your own)
3. pass the attention function on the Transformer encoder, where you can propagate it across layers or apply it once
   (see `ape.nn.encoder` for instance)

Concrete end-to-end examples in `eval.models` -- navigate to the modality of interest.

Alternatively, you may want to consider tying each Transformer layer to its own positional encoder / attention function.
It still makes sense to precompute positional encodings externally, so you can parallelize their computation. 

If you're trying to pull something off, and it's not working, or if you need clarifications with anything, feel free 
to get in touch/open an issue.

## Experiments
The scripts under `scripts/` should allow you to replicate any of the experiments detailed in the paper.
For instance
```bash
#!/bin/bash
python scripts/image.py --model Algebraic --dataset cifar10 --data_dir $DATA_DIR --store_path $STORE_DIR --seed 1312 
```
Will do an image classification run on cifar10 using default parameters (make sure to substitute `$DATA_DIR` and
`$STORE_DIR`).

More experiments are likely tbd.


## License
The software is published under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

```
You are free to:
* Share — copy and redistribute the material in any medium or format for any purpose, even commercially.
* Adapt — remix, transform, and build upon the material for any purpose, even commercially.
The licensor cannot revoke these freedoms as long as you follow the license terms.

Under the following terms:
* Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
* ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
* No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
```

## Citing
Cite this arxiv entry if you utilize this work in a scholarly context.
```
@misc{kogkalidis2023algebraic,
      title={Algebraic Positional Encodings}, 
      author={Konstantinos Kogkalidis and Jean-Philippe Bernardy and Vikas Garg},
      year={2023},
      eprint={2312.16045},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
