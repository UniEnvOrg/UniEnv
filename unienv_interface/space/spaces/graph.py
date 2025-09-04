from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, Tuple, Union
from typing_extensions import TypedDict # for Python < 3.11, otherwise we can use typing.TypedDict
import numpy as np
from ..space import Space
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType, ArrayAPIArray
from .box import BoxSpace
import dataclasses

@dataclasses.dataclass
class GraphInstance(Generic[BArrayType]):
    
    n_nodes: BArrayType
    """Number of nodes in the graph, shape (*batch_shape)"""

    n_edges: Optional[BArrayType] = None
    """Number of edges in the graph, shape (*batch_shape) or None if no edges are present."""

    nodes_features: Optional[BArrayType] = None
    """Node features, shape (*batch_shape, max(n_nodes), *node_feature_space.shape) if node_feature_space is not None, otherwise None."""

    edges_features: Optional[BArrayType] = None
    """Edge features, shape (*batch_shape, max(n_edges), *edge_feature_space.shape) if edge_feature_space is not None, otherwise None."""

    edges: Optional[BArrayType] = None
    """Edges in the graph, shape (*batch_shape, max(n_edges), 2) where each edge is represented by a pair of node indices, or None if no edges are present."""

class GraphSpace(Space[GraphInstance[BArrayType], BDeviceType, BDtypeType, BRNGType], Generic[BArrayType, BDeviceType, BDtypeType, BRNGType]):
    def __init__(
        self,
        backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
        node_feature_space: Optional[BoxSpace[BArrayType, BDeviceType, BDtypeType, BRNGType]],
        edge_feature_space: Optional[BoxSpace[BArrayType, BDeviceType, BDtypeType, BRNGType]] = None,
        is_edge : bool = False,
        min_nodes : int = 1,
        max_nodes : Optional[int] = None,
        min_edges : int = 1,
        max_edges : Optional[int] = None,
        batch_shape : Sequence[int] = (),
        device : Optional[BDeviceType] = None,
    ):
        device = device or node_feature_space.device

        assert all(
            np.issubdtype(type(dim), np.integer) for dim in batch_shape
        ), f"Expect all batch_shape elements to be an integer, actual type: {tuple(type(dim) for dim in batch_shape)}"
        self.batch_shape = tuple(int(dim) for dim in batch_shape)  # This changes any np types to int

        assert isinstance(
            node_feature_space, BoxSpace
        ), f"Values of the node_space should be instances of BoxSpace, got {type(node_feature_space)}"
        assert backend == node_feature_space.backend, f"Backend mismatch for node feature space: {backend} != {node_feature_space.backend}"
        if edge_feature_space is not None:
            assert is_edge, "Expects edge_feature_space to be provided only when is_edge is True"
            assert isinstance(
                edge_feature_space, BoxSpace
            ), f"Values of the edge_space should be instances of BoxSpace, got {type(edge_space)}"
            assert backend == edge_feature_space.backend, f"Backend mismatch for edge feature space: {backend} != {edge_feature_space.backend}"

        self.is_edge = is_edge
        self.node_feature_space = node_feature_space if (device is None or node_feature_space is None) else node_feature_space.to(device=device)
        self.edge_feature_space = edge_feature_space if (device is None or edge_feature_space is None) else edge_feature_space.to(device=device)
        
        assert min_nodes >= 1, f"Expects the minimum number of nodes to be at least 1, actual value: {min_nodes}"
        assert max_nodes >= min_nodes or max_nodes is None, f"Expects the maximum number of nodes to be at least the minimum number of nodes, actual values: {max_nodes} < {min_nodes}"
        assert min_edges >= 1, f"Expects the minimum number of edges to be at least 1, actual value: {min_edges}"
        assert min_edges <= max_edges or max_edges is None, f"Expects the minimum number of edges to be at most the maximum number of edges, actual values: {min_edges} > {max_edges}"
        assert max_edges is None or max_edges >= min_edges, f"Expects the maximum number of edges to be at least the minimum number of edges, actual values: {max_edges} < {min_edges}"
        max_possible_edges = max_nodes * (max_nodes - 1) if max_nodes is not None else None
        if max_possible_edges is not None:
            assert min_edges <= max_possible_edges, f"Expects the minimum number of edges to be at most {max_possible_edges}, actual value: {min_edges}"
            assert max_edges is None or max_edges <= max_possible_edges, f"Expects the maximum number of edges to be at most {max_possible_edges}, actual value: {max_edges}"
        
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_edges = min_edges
        self.max_edges = max_edges

        super().__init__(
            backend=backend,
            shape=None,
            device=device,
            dtype=None,
        )

    def to(
        self,
        backend: Optional[ComputeBackend] = None,
        device: Optional[Union[BDeviceType, Any]] = None,
    ) -> Union["GraphSpace[BArrayType, BDeviceType, BDtypeType, BRNGType]", "GraphSpace"]:
        if (backend is None or backend==self.backend) and (device is None or device==self.device):
            return self
    
        new_backend = backend or self.backend
        new_device = device if backend is not None else (device or self.device)
        return GraphSpace(
            new_backend,
            self.max_nodes,
            node_feature_space=self.node_feature_space.to(backend=new_backend, device=new_device) if self.node_feature_space is not None else None,
            edge_feature_space=self.edge_feature_space.to(backend=new_backend, device=new_device) if self.edge_feature_space is not None else None,
            min_nodes=self.min_nodes,
            max_nodes=self.max_nodes,
            min_edges=self.min_edges,
            max_edges=self.max_edges,
            is_edge=self.is_edge,
            batch_shape=self.batch_shape,
            device=new_device,
        )
    
    def _make_batched_node_feature_space(
        self,
        num_nodes: int,
    ) -> Optional[BoxSpace[BArrayType, BDeviceType, BDtypeType, BRNGType]]:
        """Create a batched node feature space based on the number of nodes."""
        if self.node_feature_space is None or num_nodes < 0:
            return None
        return BoxSpace(
            self.backend,
            low=self.backend.reshape(
                self.node_feature_space._low, tuple([1] * (len(self.batch_shape) + 1)) + tuple(self.node_feature_space._low.shape)
            ),
            high=self.backend.reshape(
                self.node_feature_space._high, tuple([1] * (len(self.batch_shape) + 1)) + tuple(self.node_feature_space._high.shape)
            ),
            dtype=self.node_feature_space.dtype,
            device=self.node_feature_space.device,
            shape=self.batch_shape + (num_nodes,) + self.node_feature_space.shape,
        )
    
    def _make_batched_edge_feature_space(
        self,
        num_edges: int,
    ) -> Optional[BoxSpace[BArrayType, BDeviceType, BDtypeType, BRNGType]]:
        if self.edge_feature_space is None or num_edges < 0:
            return None

        return BoxSpace(
            self.backend,
            low=self.backend.reshape(
                self.edge_feature_space._low, tuple([1] * (len(self.batch_shape) + 1)) + tuple(self.edge_feature_space._low.shape)
            ),
            high=self.backend.reshape(
                self.edge_feature_space._high, tuple([1] * (len(self.batch_shape) + 1)) + tuple(self.edge_feature_space._high.shape)
            ),
            dtype=self.edge_feature_space.dtype,
            device=self.edge_feature_space.device,
            shape=self.batch_shape + (num_edges,) + self.edge_feature_space.shape,
        )

    def sample(
        self,
        rng: BRNGType,
    ) -> Tuple[BRNGType, GraphInstance]:
        # we only have edges when we have at least 2 nodes
        if self.max_nodes is not None:
            rng, num_nodes = self.backend.random.random_discrete_uniform(
                self.batch_shape,
                self.min_nodes,
                self.max_nodes + 1,
                rng=rng,
                dtype=self.backend.default_integer_dtype,
                device=self.device,
            )
        else:
            rng, num_nodes = self.backend.random.random_exponential(
                self.batch_shape,
                rng=rng,
                device=self.device,
            )
            num_nodes = self.backend.astype(self.backend.floor(num_nodes), self.backend.default_integer_dtype) + self.min_nodes

        if self.is_edge:
            max_edge = self.max_edges or (num_nodes ** 2) # Assume we allow node to have edges to itself
            rng, num_edges = self.backend.random.random_uniform(
                self.batch_shape,
                rng=rng,
                dtype=self.backend.default_floating_dtype,
                device=self.device,
            )
            num_edges = self.backend.astype(self.backend.floor(num_edges * (max_edge - self.min_edges + 1)), self.backend.default_integer_dtype) + self.min_edges
        else:
            num_edges = None

        if self.node_feature_space is None:
            batched_node_space = self._make_batched_node_feature_space(int(self.backend.max(num_nodes)))
            rng, node_features = batched_node_space.sample(rng=rng)
        else:
            node_features = None
        
        num_edges_batch_max = int(self.backend.max(num_edges)) if num_edges is not None else 0
        if num_edges is not None and num_edges_batch_max > 0:
            rng, edges = self.backend.random.random_uniform(
                self.batch_shape + (num_edges_batch_max, 2),
                rng=rng,
                device=self.device,
            )
            edges = self.backend.astype(self.backend.floor(edges * num_nodes), self.backend.default_integer_dtype)
        else:
            edges = None

        if num_edges_batch_max > 0 and self.edge_feature_space is not None:
            batched_edge_space = self._make_batched_edge_feature_space(num_edges_batch_max)
            rng, edge_features = batched_edge_space.sample(rng=rng)
        else:
            edge_features = None

        return rng, GraphInstance(
            n_nodes=num_nodes,
            n_edges=num_edges if num_edges > 0 else None,
            nodes_features=node_features,
            edges_features=edge_features,
            edges=edges,
        )

    def create_empty(self) -> GraphInstance[BArrayType]:
        raise NotImplementedError(
            "GraphSpace does not support create_empty method. Use GraphInstance with appropriate shapes instead."
        )

    def is_bounded(self, manner = "both"):
        return (
            (self.node_feature_space is None or self.node_feature_space.is_bounded(manner)) 
            and (self.edge_feature_space is None or self.edge_feature_space.is_bounded(manner))
        )

    def contains(self, x: GraphInstance[BArrayType]) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if not isinstance(x, GraphInstance):
            return False
        
        if not self.backend.is_backendarray(x.n_nodes) or not self.backend.dtype_is_real_integer(x.n_nodes.dtype) or x.n_nodes.shape != self.batch_shape:
            return False
        
        if bool(self.backend.any(x.n_nodes < self.min_nodes)) or (self.max_nodes is not None and bool(self.backend.any(x.n_nodes > self.max_nodes))):
            return False

        if self.node_feature_space is not None:
            if x.nodes_features is None:
                return False
            batched_node_space = self._make_batched_node_feature_space(int(self.backend.max(x.n_nodes)))
            if not batched_node_space.contains(x.nodes_features):
                return False
        
        if self.is_edge and (x.n_edges is not None):
            if not self.backend.is_backendarray(x.n_edges) or not self.backend.dtype_is_real_integer(x.n_edges.dtype) or x.n_edges.shape != self.batch_shape:
                return False
            
            if bool(self.backend.any(x.n_edges < self.min_edges)) or (self.max_edges is not None and bool(self.backend.any(x.n_edges > self.max_edges))):
                return False
            
            if x.edges is None or not self.backend.is_backendarray(x.edges) or x.edges.shape != self.batch_shape + (int(self.backend.max(x.n_edges)), 2) or not self.backend.dtype_is_real_integer(x.edges.dtype):
                return False
            
            if x.edges_features is not None:
                batched_edge_space = self._make_batched_edge_feature_space(int(self.backend.max(x.n_edges)))
                if not batched_edge_space.contains(x.edges_features):
                    return False
        elif self.is_edge and self.min_edges > 0 and (x.n_edges is None  or x.edges is None):
            return False
        elif not self.is_edge and (x.n_edges is not None or x.edges is not None or x.edges_features is not None):
            return False

        return True

    def get_repr(
        self, 
        abbreviate = False,
        include_backend = True, 
        include_device = True, 
        include_dtype = True
    ):
        next_include_device = self.device is None and include_device
        if abbreviate:
            ret = f"G("
        else:
            ret = f"GraphSpace("
        ret += f"V={{{self.min_nodes}, {self.max_nodes}, {self.node_feature_space.get_repr(abbreviate, False, next_include_device, include_dtype)}}})"
        if self.is_edge:
            ret += f"E={{{self.min_edges}, {self.max_edges}"
            if self.edge_feature_space is not None:
                ret += self.edge_feature_space.get_repr(abbreviate, False, next_include_device, include_dtype)
            ret += "}"
        
        if include_backend:
            ret += f", backend={self.backend}"
        if include_device and self.device is not None:
            ret += f", device={self.device}"
        ret += ")"
        return ret

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance."""
        return (
            isinstance(other, GraphSpace)
            and (self.backend == other.backend)
            and (self.is_edge == other.is_edge)
            and (self.min_nodes == other.min_nodes)
            and (self.max_nodes == other.max_nodes)
            and (self.min_edges == other.min_edges)
            and (self.max_edges == other.max_edges)
            and (self.node_feature_space == other.node_feature_space)
            and (self.edge_feature_space == other.edge_feature_space)
        )
    
    def data_to(self, data, backend = None, device = None):
        if backend is not None and backend != self.backend:
            data = backend.from_other_backend(self.backend, data)
        if device is not None:
            data = (backend or self.backend).to_device(data,device)
        return data
