# Jetson AI Lab devpi (PyPI indexes)

Root URL: [https://pypi.jetson-ai-lab.io/](https://pypi.jetson-ai-lab.io/)

The site is a **devpi** mirror with **per–JetPack / CUDA** subtrees (for example `jp6/cu126`, `jp6/cu128`). Paths evolve with JetPack lines — **open the root in a browser** and pick the tree that matches:

- Your **L4T / JetPack** line (`cat /etc/nv_tegra_release`)
- The **CUDA userland** on the device (see JetPack release notes)

## pip usage pattern

After you identify the correct index URL from the devpi tree:

```bash
python -m pip install --dry-run --report /tmp/jetson-package-plan.json --extra-index-url '<paste-index-url-here>' '<package>'
```

pip merges candidates across indexes; `--extra-index-url` does not prioritize Jetson builds. Inspect the resolver report, then pin a verified wheel URL/hash or use an appropriately controlled index. Match Python ABI, CUDA/driver requirements, JetPack release and GPU code support before installation. See [pip package selection](https://pip.pypa.io/en/stable/cli/pip_install/#finding-packages).

## Why not only PyPI?

Public `manylinux2014_aarch64` wheel tags do not establish SM support. Check native code for Orin SM 8.7 / Thor SM 11.0 or a compatible PTX JIT path. A Jetson-oriented index is a useful discovery source, not proof that every artifact matches the installed release.
