# Core Concepts

The package is easier to use once you separate the major abstractions:

- **Spaces** define the structure, dtype, bounds, and backend placement of data.
- **Environments** define runtime interaction through `reset`, `step`, and `render`.
- **Worlds and nodes** let you compose an environment out of reusable subsystems.
- **Wrappers and transformations** adapt an environment or data stream without rewriting the source implementation.
- **Batches and replay buffers** bring the same typed-space mindset to stored data.

## Read These First

<div class="grid cards" markdown>

-   **Environments**

    Stateful `Env`, functional `FuncEnv`, and the adapter between them.

    [Open the page](environments.md)

-   **Spaces and Backends**

    The contract that keeps data portable across simulators, wrappers, and storages.

    [Open the page](spaces-and-backends.md)

-   **World Composition**

    `World`, `WorldNode`, and `WorldEnv` for building richer environments from parts.

    [Open the page](world-composition.md)

</div>

## Design Direction

UniEnv keeps two ideas front and center:

- **Structure should stay explicit.** Observations, actions, contexts, and stored transitions should remain described by spaces instead of becoming anonymous tensors too early.
- **Backend choice should stay late.** The same logic should be able to target NumPy, PyTorch, JAX, and storage-layer backends with minimal translation code.
