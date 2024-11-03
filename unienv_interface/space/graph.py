"""Implementation of a space that represents graph information where nodes and edges can be represented with euclidean space."""
from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, Tuple, Type, Literal, NamedTuple
import numpy as np
from .space import Space
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
import array_api_compat
import gymnasium as gym
import dataclasses
from .discrete import Discrete
from .multi_discrete import MultiDiscrete
from .box import Box

@dataclasses.dataclass
class GraphInstance(Generic[BArrayType]):
    """A Graph space instance.

    * nodes (np.ndarray): an (n x ...) sized array representing the features for n nodes, (...) must adhere to the shape of the node space.
    * edges (Optional[np.ndarray]): an (m x ...) sized array representing the features for m edges, (...) must adhere to the shape of the edge space.
    * edge_links (Optional[np.ndarray]): an (m x 2) sized array of ints representing the indices of the two nodes that each edge connects.
    """

    nodes: BArrayType
    edges: Optional[BArrayType] = None
    edge_links: Optional[BArrayType] = None

class Graph(Space[GraphInstance[BArrayType], gym.spaces.GraphInstance, BDeviceType, BDtypeType, BRNGType]):
    def __init__(
        self,
        backend : Type[ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]],
        node_space: Box | Discrete,
        edge_space: None | Box | Discrete,
        device : Optional[BDeviceType] = None,
    ):
        device = device if device is not None else node_space.device
        assert isinstance(
            node_space, (Box, Discrete)
        ), f"Values of the node_space should be instances of Box or Discrete, got {type(node_space)}"
        assert backend == node_space.backend, f"Expects the backend of the node space to be the same as the Graph backend, actual backend: {node_space.backend}, expected backend: {backend}"
        if edge_space is not None:
            assert isinstance(
                edge_space, (Box, Discrete)
            ), f"Values of the edge_space should be instances of None Box or Discrete, got {type(edge_space)}"
            assert backend == edge_space.backend, f"Expects the backend of the edge space to be the same as the Graph backend, actual backend: {edge_space.backend}, expected backend: {backend}"

        self.node_space = node_space if device is None else node_space.to_device(device)
        self.edge_space = edge_space if edge_space is None else edge_space.to_device(device)

        super().__init__(
            backend=backend,
            shape=None,
            device=device,
            dtype=None,
        )

    @property
    def is_flattenable(self):
        return False
    
    @property
    def flat_dim(self) -> None:
        """Return the shape of the space as an immutable property."""
        return None
    
    def flatten(self, data : GraphInstance[BArrayType]) -> BArrayType:
        raise NotImplementedError("Graph space is not flattenable.")
    
    def unflatten(self, data : BArrayType) -> GraphInstance[BArrayType]:
        raise NotImplementedError("Graph space is not flattenable.")

    def to_device(self, device : BDeviceType) -> "Graph[BDeviceType, BDtypeType, BRNGType]":
        return Graph(
            backend=self.backend,
            node_space=self.node_space.to_device(device),
            edge_space=self.edge_space.to_device(device) if self.edge_space is not None else None,
            device=device
        )

    def to_backend(self, backend : Type[ComputeBackend], device : Optional[Any]) -> "Graph":
        return Graph(
            backend=backend,
            node_space=self.node_space.to_backend(backend, device),
            edge_space=self.edge_space.to_backend(backend, device) if self.edge_space is not None else None,
            device=device
        )

    def _generate_sample_space(
        self, base_space: None | Box | Discrete, num: int
    ) -> Box | MultiDiscrete | None:
        if num == 0 or base_space is None:
            return None

        if isinstance(base_space, Box):
            return Box(
                backend=self.backend,
                low=self.backend.array_api_namespace.stack(num * [base_space.low]),
                high=self.backend.array_api_namespace.stack(num * [base_space.high]),
                dtype=base_space.dtype,
                shape=(num,) + base_space.shape,
                device=base_space.device
            )
        elif isinstance(base_space, Discrete):
            return MultiDiscrete(
                backend=self.backend,
                nvec=self.backend.array_api_namespace.ones(num, dtype=base_space.dtype) * base_space.n,
                start=None if base_space.start==0 else self.backend.array_api_namespace.ones(num, dtype=base_space.dtype) * base_space.start,
                dtype=base_space.dtype,
                device=base_space.device,
            )
        elif base_space is None:
            return None
        else:
            raise TypeError(
                f"Expects base space to be Box and Discrete, actual space: {type(base_space)}."
            )

    def sample(
        self,
        rng: BRNGType,
        num_nodes: int = 10,
        num_edges: int | None = None,
    ) -> Tuple[BRNGType, GraphInstance]:
        # we only have edges when we have at least 2 nodes
        if num_edges is None:
            if num_nodes > 1:
                # maximal number of edges is `n*(n-1)` allowing self connections and two-way is allowed
                rng, num_edges = self.backend.random_discrete_uniform(
                    rng,
                    shape=(1,),
                    from_num=0,
                    to_num=num_nodes * (num_nodes - 1),
                    device=None
                )[0]
            else:
                num_edges = 0
        else:
            assert (
                num_edges >= 0
            ), f"Expects the number of edges to be greater than 0, actual value: {num_edges}"

        sampled_node_space = self._generate_sample_space(self.node_space, num_nodes)
        sampled_edge_space = self._generate_sample_space(self.edge_space, num_edges)

        assert sampled_node_space is not None
        sampled_nodes = sampled_node_space.sample()
        sampled_edges = (
            sampled_edge_space.sample()
            if sampled_edge_space is not None
            else None
        )

        sampled_edge_links = None
        if sampled_edges is not None and num_edges > 0:
            rng, sampled_edge_links = self.backend.random_discrete_uniform(
                rng, shape=(num_edges, 2), from_num=0, to_num=num_nodes, device=self.node_space.device
            )

        return rng, GraphInstance(sampled_nodes, sampled_edges, sampled_edge_links)

    def contains(self, x: GraphInstance) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, GraphInstance):
            # Checks the nodes
            if self.backend.is_backendarray(x.nodes):
                if all(x.nodes[i] in self.node_space for i in x.nodes.shape[0]):
                    # Check the edges and edge links which are optional
                    if self.backend.is_backendarray(x.edges) and self.backend.is_backendarray(x.edge_links):
                        assert x.edges is not None
                        assert x.edge_links is not None
                        if self.edge_space is not None:
                            return (
                                all(x.edges[i] in self.edge_space for i in x.edges.shape[0]) and
                                self.backend.dtype_is_real_integer(x.edge_links.dtype) and
                                x.edge_links.shape == (len(x.edges), 2) and
                                self.backend.array_api_namespace.all(
                                    self.backend.array_api_namespace.logical_and(
                                        x.edge_links >= 0,
                                        x.edge_links < len(x.nodes),
                                    )
                                )
                            ) 
                    else:
                        return x.edges is None and x.edge_links is None
        return False

    def __repr__(self) -> str:
        return f"Graph({self.node_space}, {self.edge_space})"

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance."""
        return (
            isinstance(other, Graph)
            and (self.node_space == other.node_space)
            and (self.edge_space == other.edge_space)
        )

    def to_jsonable(
        self, sample_n: Sequence[GraphInstance]
    ) -> list[dict[str, list[int | float]]]:
        """Convert a batch of samples from this space to a JSONable data type."""
        ret_n = []
        for sample in sample_n:
            ret = {"nodes": self.node_space.to_jsonable([sample.nodes[i] for i in sample.nodes.shape[0]])}
            if sample.edges is not None and sample.edge_links is not None:
                ret["edges"] = self.edge_space.to_jsonable([sample.edges[i] for i in sample.edges.shape[0]])
                ret["edge_links"] = self.backend.to_numpy(sample.edge_links).tolist()
            ret_n.append(ret)
        return ret_n

    def from_jsonable(
        self, sample_n: Sequence[dict[str, list[list[int] | list[float]]]]
    ) -> list[GraphInstance]:
        """Convert a JSONable data type to a batch of samples from this space."""
        ret: list[GraphInstance] = []
        for sample in sample_n:
            if "edges" in sample:
                assert self.edge_space is not None
                ret_n = GraphInstance(
                    self.node_space.from_jsonable(sample["nodes"]),
                    self.edge_space.from_jsonable(sample["edges"]),
                    self.backend.from_numpy(np.asarray(sample["edge_links"], dtype=np.int32)),
                )
            else:
                ret_n = GraphInstance(
                    self.node_space.from_jsonable(sample["nodes"]),
                    None,
                    None,
                )
            ret.append(ret_n)
        return ret
    
    def from_gym_data(self, gym_data : gym.spaces.GraphInstance) -> GraphInstance[BArrayType]:
        return GraphInstance(
            self.backend.from_numpy(gym_data.nodes, dtype=self.node_space.dtype, device=self.node_space.device),
            self.backend.from_numpy(gym_data.edges, dtype=self.edge_space.dtype, device=self.edge_space.device) if gym_data.edges is not None else None,
            self.backend.from_numpy(gym_data.edge_links, device=self.device) if gym_data.edge_links is not None else None,
        )
    
    def to_gym_data(self, data : GraphInstance[BArrayType]) -> gym.spaces.GraphInstance:
        return gym.spaces.GraphInstance(
            nodes=self.backend.to_numpy(data.nodes),
            edges=self.backend.to_numpy(data.edges) if data.edges is not None else None,
            edge_links=self.backend.to_numpy(data.edge_links) if data.edge_links is not None else None,
        )
    
    def from_other_backend(self, other_data : GraphInstance[Any]) -> GraphInstance[BArrayType]:
        new_node = self.backend.from_dlpack(other_data.nodes)
        new_edge = self.backend.from_dlpack(other_data.edges) if other_data.edges is not None else None
        new_edge_links = self.backend.from_dlpack(other_data.edge_links) if other_data.edge_links is not None else None
        if self.node_space.device is not None:
            new_node = array_api_compat.to_device(new_node, self.node_space.device)
        if new_edge is not None and self.edge_space is not None and self.edge_space.device is not None:
            new_edge = array_api_compat.to_device(new_edge, self.edge_space.device)
        if new_edge_links is not None and self.device is not None:
            new_edge_links = array_api_compat.to_device(new_edge_links, self.device)

        return GraphInstance(
            nodes=new_node,
            edges=new_edge,
            edge_links=new_edge_links,
        )
    
    def from_same_backend(self, other_data : GraphInstance[BArrayType]) -> GraphInstance[BArrayType]:
        new_node = other_data.nodes
        new_edge = other_data.edges
        new_edge_links = other_data.edge_links
        if self.node_space.device is not None:
            new_node = array_api_compat.to_device(new_node, self.node_space.device)
        if new_edge is not None and self.edge_space is not None and self.edge_space.device is not None:
            new_edge = array_api_compat.to_device(new_edge, self.edge_space.device)
        if new_edge_links is not None and self.device is not None:
            new_edge_links = array_api_compat.to_device(new_edge_links, self.device)
        return GraphInstance(
            nodes=new_node,
            edges=new_edge,
            edge_links=new_edge_links,
        )

    def to_gym_space(self) -> gym.Space:
        """Convert this space to a gym space."""
        return gym.spaces.Graph(
            node_space=self.node_space.to_gym_space(),
            edge_space=self.edge_space.to_gym_space() if self.edge_space is not None else None
        )
