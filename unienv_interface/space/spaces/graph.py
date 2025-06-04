"""Implementation of a space that represents graph information where nodes and edges can be represented with euclidean space."""
from typing import Any, Generic, Iterable, SupportsFloat, Mapping, Sequence, TypeVar, Optional, Tuple, Union
from typing_extensions import TypedDict # for Python < 3.11, otherwise we can use typing.TypedDict
import numpy as np
from ..space import Space
from xarray import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType, ArrayAPIArray
from .box import BoxSpace

class GraphInstance(Generic[BArrayType], TypedDict):
    """A Graph space instance.

    * nodes : an (*B, n, ...) sized array representing the features for n nodes, (...) must adhere to the shape of the node space.
    * edges : an optional (*B, m, ...) sized array representing the features for m edges, (...) must adhere to the shape of the edge space.
    * edges : an optional (m x 2) sized array of ints representing the indices of the two nodes that each edge connects.
    """

    n_nodes: int
    n_edges: Optional[int] = None
    nodes_features: Optional[BArrayType] = None
    edges_features: Optional[BArrayType] = None
    edges: Optional[BArrayType] = None

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
                1,
                self.min_nodes,
                self.max_nodes + 1,
                rng=rng,
                dtype=self.backend.default_integer_dtype
            )
            num_nodes = int(num_nodes[0])
        else:
            rng, num_nodes = self.backend.random.random_exponential(
                1,
                rng=rng,
                dtype=self.backend.default_floating_dtype,
            )
            num_nodes = int(num_nodes[0]) + self.min_nodes

        if self.is_edge and num_nodes >= 2:
            max_edge = self.max_edges or (num_nodes * (num_nodes - 1))
            rng, num_edges = self.backend.random.random_discrete_uniform(
                1,
                self.min_edges,
                max_edge + 1,
                rng=rng,
                dtype=self.backend.default_integer_dtype
            )
            num_edges = int(num_edges[0])
        else:
            num_edges = 0

        if self.node_feature_space is None:
            batched_node_space = self._make_batched_node_feature_space(num_nodes)
            rng, node_features = batched_node_space.sample(rng=rng)
        else:
            node_features = None
        
        if num_edges > 0:
            rng, edges = self.backend.random.random_discrete_uniform(
                self.batch_shape + (num_edges, 2),
                from_num=0,
                to_num=num_nodes,
                rng=rng,
                dtype=self.backend.default_integer_dtype,
                device=self.device
            )
            edges = self.backend.at(edges)[edges[..., 1] == edges[..., 0], 1].add(1)
            edges = self.backend.at(edges)[edges[..., 1] >= num_nodes, 1].set(0)
        else:
            edges = None

        if num_edges > 0 and self.edge_feature_space is not None:
            batched_edge_space = self._make_batched_edge_feature_space(num_edges)
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

    def contains(self, x: GraphInstance) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if not isinstance(x, Mapping):
            return False
        
        if any(k not in x for k in ["n_nodes", "n_edges", "nodes_features", "edges_features", "edges"]):
            return False
        
        if not isinstance(x['n_nodes'], int):
            return False

        if x['n_nodes'] < self.min_nodes or (self.max_nodes is not None and x['n_nodes'] > self.max_nodes):
            return False

        if self.node_feature_space is not None:
            if x['nodes_features'] is None:
                return False
            batched_node_space = self._make_batched_node_feature_space(x['n_nodes'])
            if not batched_node_space.contains(x['nodes_features']):
                return False
        
        if self.is_edge and x["n_edges"] is not None:
            if not isinstance(x['n_edges'], int):
                return False
            if x['n_edges'] < self.min_edges or (self.max_edges is not None and x['n_edges'] > self.max_edges):
                return False
            
            if self.edge_feature_space is not None:
                if x['edges_features'] is None:
                    return False
                batched_edge_space = self._make_batched_edge_feature_space(x['n_edges'])
                if not batched_edge_space.contains(x['edges_features']):
                    return False
            
            if x['edges'] is None or not self.backend.is_backendarray(x['edges']) or x['edges'].shape != self.batch_shape + (x['n_edges'], 2) or not self.backend.dtype_is_real_integer(x['edges'].dtype):
                return False
        elif self.is_edge and self.min_edges > 0:
            return False
        elif (not self.is_edge) or self.min_edges >= 0:
            if x['n_edges'] is not None:
                return False
            if x['edges'] is not None:
                return False
            if x['edges_features'] is not None:
                return False
        else:
            # self.is_edge and self.min_edges >= 0
            pass

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
        if backend is not None:
            data = backend.from_other_backend(self.backend, data)
        if device is not None:
            data = (backend or self.backend).to_device(data,device)
        return data
