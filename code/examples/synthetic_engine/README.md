# Synthetic example

The example uses generated request-resource pairs and an engine with one ranking
policy. Scripted model responses replace the initial travel-cost rule with an
inverse-cost rule. Use `--live` to generate decisions with a model endpoint.

Run commands from the project directory containing `pyproject.toml`:

```bash
uv run python -m dispatchevolve.synthetic_demo
```

Search uses `synthetic.csv`; the final evaluation uses the separate `test.csv`.
`workspaces/synthetic_demo/summary.json` contains baseline and selected-engine
metrics, improvements and replay cost. Some objectives remain unchanged in this
example, so its FAI is zero.

Use `--output-dir workspaces/another_demo` for another run. Each run requires an
unused output directory.
