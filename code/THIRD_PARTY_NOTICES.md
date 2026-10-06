# Third-party notices

## OpenEvolve

`src/dispatchevolve/optimizer/genetic/` contains code adapted from
[OpenEvolve](https://github.com/codelion/openevolve), licensed under Apache-2.0.
The bundled [license](src/dispatchevolve/optimizer/genetic/LICENSE_OPenevolve)
was verified against upstream revision
`4f4b0c4f40906f434d64fe5089aff24e927f7e24`.
This is the license verification revision; the original import revision was not
recorded. Local changes include repository-based candidate evolution, evaluator
integration, durable iteration state and replay-budget controls.

## ShinkaEvolve

The embedding and novelty utilities identify
[ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) as their source in their
module headers. Its [Apache-2.0 license](licenses/ShinkaEvolve-Apache-2.0.txt)
was verified against upstream revision
`8adc053a2ce4511ad2ac310e004c530a73fb974a`.
These utilities have been adapted to the local optimizer interfaces.

## Project code

DispatchEvolve is licensed under [Apache-2.0](LICENSE). The third-party
components above retain their upstream attribution and license notices.
