from typing import Protocol, TypeVar, Optional, Any, Tuple, Union, Type, TypedDict, List, Literal, Sequence
from abc import abstractmethod
from dataclasses import dataclass
from ._array_typing import *
from . import _api_constant as const
from ._api_return_typing import *
from ._api_linalg_typing import ArrayAPILinalgNamespace
from ._api_fft_typing import ArrayAPIFFTNamespace

_NAMESPACE_C = TypeVar("_NAMESPACE_C", bound="ArrayAPINamespace")
_NAMESPACE_ARRAY = TypeVar("_NAMESPACE_ARRAY", bound=Array)
_NAMESPACE_DEVICE = TypeVar("_NAMESPACE_DEVICE", bound=Device)
_NAMESPACE_DTYPE = TypeVar("_NAMESPACE_DTYPE", bound=DType)
class ArrayAPINamespace(Protocol[_NAMESPACE_ARRAY, _NAMESPACE_DEVICE, _NAMESPACE_DTYPE]):
    """
    linalg namespace
    """
    linalg: ArrayAPILinalgNamespace[_NAMESPACE_ARRAY, _NAMESPACE_DEVICE, _NAMESPACE_DTYPE]

    """
    fft namespace
    """
    fft: ArrayAPIFFTNamespace[_NAMESPACE_ARRAY, _NAMESPACE_DEVICE, _NAMESPACE_DTYPE]

    """
    Constants
    https://github.com/data-apis/array-api/blob/main/src/array_api_stubs/_2024_12/constants.py
    """
    @property
    def e(self : _NAMESPACE_C) -> float:
        return const.e
    
    @property
    def inf(self : _NAMESPACE_C) -> float:
        return const.inf
    
    @property
    def nan(self : _NAMESPACE_C) -> float:
        return const.nan
    
    @property
    def newaxis(self : _NAMESPACE_C) -> None:
        return const.newaxis
    
    @property
    def pi(self : _NAMESPACE_C) -> float:
        return const.pi

    """
    __array_namespace_info__ typings
    https://github.com/data-apis/array-api/blob/main/src/array_api_stubs/_2024_12/info.py
    """
    @abstractmethod
    def __array_namespace_info__(
        self : _NAMESPACE_C
    ) -> Info:
        """
        Returns a namespace with Array API namespace inspection utilities.

        See :ref:`inspection` for a list of inspection APIs.

        Returns
        -------
        out: Info
            An object containing Array API namespace inspection utilities.

        Notes
        -----

        The returned object may be either a namespace or a class, so long as an Array API user can access inspection utilities as follows:

        ::

        info = xp.__array_namespace_info__()
        info.capabilities()
        info.devices()
        info.dtypes()
        info.default_dtypes()
        # ...

        .. versionadded: 2023.12
        """

    """
    Creation Functions
    https://github.com/data-apis/array-api/blob/main/src/array_api_stubs/_2024_12/creation_functions.py
    """

    @abstractmethod
    def arange(
        self : _NAMESPACE_C,
        start: Union[int, float],
        /,
        stop: Optional[Union[int, float]] = None,
        step: Union[int, float] = 1,
        *,
        dtype: Optional[_NAMESPACE_DTYPE] = None,
        device: Optional[_NAMESPACE_DEVICE] = None,
    ) -> _NAMESPACE_ARRAY:
        """
        Returns evenly spaced values within the half-open interval ``[start, stop)`` as a one-dimensional array.

        Parameters
        ----------
        start: Union[int, float]
            if ``stop`` is specified, the start of interval (inclusive); otherwise, the end of the interval (exclusive). If ``stop`` is not specified, the default starting value is ``0``.
        stop: Optional[Union[int, float]]
            the end of the interval. Default: ``None``.
        step: Union[int, float]
            the distance between two adjacent elements (``out[i+1] - out[i]``). Must not be ``0``; may be negative, this results in an empty array if ``stop >= start``. Default: ``1``.
        dtype: Optional[dtype]
            output array data type. If ``dtype`` is ``None``, the output array data type must be inferred from ``start``, ``stop`` and ``step``. If those are all integers, the output array dtype must be the default integer dtype; if one or more have type ``float``, then the output array dtype must be the default real-valued floating-point data type. Default: ``None``.
        device: Optional[device]
            device on which to place the created array. Default: ``None``.


        .. note::
        This function cannot guarantee that the interval does not include the ``stop`` value in those cases where ``step`` is not an integer and floating-point rounding errors affect the length of the output array.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            a one-dimensional array containing evenly spaced values. The length of the output array must be ``ceil((stop-start)/step)`` if ``stop - start`` and ``step`` have the same sign, and length ``0`` otherwise.
        """
        

    @abstractmethod
    def asarray(
        self : _NAMESPACE_C,
        obj: Union[
            _NAMESPACE_ARRAY, bool, int, float, complex, NestedSequence, SupportsBufferProtocol
        ],
        /,
        *,
        dtype: Optional[_NAMESPACE_DTYPE] = None,
        device: Optional[_NAMESPACE_DEVICE] = None,
        copy: Optional[bool] = None,
    ) -> _NAMESPACE_C:
        r"""
        Convert the input to an array.

        Parameters
        ----------
        obj: Union[array, bool, int, float, complex, NestedSequence[bool | int | float | complex], SupportsBufferProtocol]
            object to be converted to an array. May be a Python scalar, a (possibly nested) sequence of Python scalars, or an object supporting the Python buffer protocol.

            .. admonition:: Tip
            :class: important

            An object supporting the buffer protocol can be turned into a memoryview through ``memoryview(obj)``.

        dtype: Optional[dtype]
            output array data type. If ``dtype`` is ``None``, the output array data type must be inferred from the data type(s) in ``obj``. If all input values are Python scalars, then, in order of precedence,

            -   if all values are of type ``bool``, the output data type must be ``bool``.
            -   if all values are of type ``int`` or are a mixture of ``bool`` and ``int``, the output data type must be the default integer data type.
            -   if one or more values are ``complex`` numbers, the output data type must be the default complex floating-point data type.
            -   if one or more values are ``float``\s, the output data type must be the default real-valued floating-point data type.

            Default: ``None``.

            .. admonition:: Note
            :class: note

            If ``dtype`` is not ``None``, then array conversions should obey :ref:`type-promotion` rules. Conversions not specified according to :ref:`type-promotion` rules may or may not be permitted by a conforming array library. To perform an explicit cast, use :func:`array_api.astype`.

            .. note::
            If an input value exceeds the precision of the resolved output array data type, behavior is left unspecified and, thus, implementation-defined.

        device: Optional[device]
            device on which to place the created array. If ``device`` is ``None`` and ``obj`` is an array, the output array device must be inferred from ``obj``. Default: ``None``.
        copy: Optional[bool]
            boolean indicating whether or not to copy the input. If ``True``, the function must always copy (see :ref:`copy-keyword-argument`). If ``False``, the function must never copy for input which supports the buffer protocol and must raise a ``ValueError`` in case a copy would be necessary. If ``None``, the function must reuse existing memory buffer if possible and copy otherwise. Default: ``None``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the data from ``obj``.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.
        """
        

    @abstractmethod
    def empty(
        self : _NAMESPACE_C,
        shape: Union[int, Tuple[int, ...]],
        *,
        dtype: Optional[_NAMESPACE_DTYPE] = None,
        device: Optional[_NAMESPACE_DEVICE] = None,
    ) -> _NAMESPACE_ARRAY:
        """
        Returns an uninitialized array having a specified `shape`.

        Parameters
        ----------
        shape: Union[int, Tuple[int, ...]]
            output array shape.
        dtype: Optional[dtype]
            output array data type. If ``dtype`` is ``None``, the output array data type must be the default real-valued floating-point data type. Default: ``None``.
        device: Optional[device]
            device on which to place the created array. Default: ``None``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing uninitialized data.
        """
        
    @abstractmethod
    def empty_like(
        self : _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /, *, dtype: Optional[_NAMESPACE_DTYPE] = None, device: Optional[_NAMESPACE_DEVICE] = None
    ) -> _NAMESPACE_DEVICE:
        """
        Returns an uninitialized array with the same ``shape`` as an input array ``x``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array from which to derive the output array shape.
        dtype: Optional[dtype]
            output array data type. If ``dtype`` is ``None``, the output array data type must be inferred from ``x``. Default: ``None``.
        device: Optional[device]
            device on which to place the created array. If ``device`` is ``None``, the output array device must be inferred from ``x``. Default: ``None``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array having the same shape as ``x`` and containing uninitialized data.
        """
        
    @abstractmethod
    def eye(
        self : _NAMESPACE_C,
        n_rows: int,
        n_cols: Optional[int] = None,
        /,
        *,
        k: int = 0,
        dtype: Optional[_NAMESPACE_DTYPE] = None,
        device: Optional[_NAMESPACE_DEVICE] = None,
    ) -> _NAMESPACE_C:
        r"""
        Returns a two-dimensional array with ones on the ``k``\th diagonal and zeros elsewhere.

        .. note::
        An output array having a complex floating-point data type must have the value ``1 + 0j`` along the ``k``\th diagonal and ``0 + 0j`` elsewhere.

        Parameters
        ----------
        n_rows: int
            number of rows in the output array.
        n_cols: Optional[int]
            number of columns in the output array. If ``None``, the default number of columns in the output array is equal to ``n_rows``. Default: ``None``.
        k: int
            index of the diagonal. A positive value refers to an upper diagonal, a negative value to a lower diagonal, and ``0`` to the main diagonal. Default: ``0``.
        dtype: Optional[dtype]
            output array data type. If ``dtype`` is ``None``, the output array data type must be the default real-valued floating-point data type. Default: ``None``.
        device: Optional[device]
            device on which to place the created array. Default: ``None``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array where all elements are equal to zero, except for the ``k``\th diagonal, whose values are equal to one.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.
        """
        
    @abstractmethod
    def from_dlpack(
        self : _NAMESPACE_C,
        x: object,
        /,
        *,
        device: Optional[_NAMESPACE_DEVICE] = None,
        copy: Optional[bool] = None,
    ) -> _NAMESPACE_ARRAY:
        """
        Returns a new array containing the data from another (array) object with a ``__dlpack__`` method.

        Parameters
        ----------
        x: object
            input (array) object.
        device: Optional[device]
            device on which to place the created array. If ``device`` is ``None`` and ``x`` supports DLPack, the output array must be on the same device as ``x``. Default: ``None``.

            The v2023.12 standard only mandates that a compliant library should offer a way for ``from_dlpack`` to return an array
            whose underlying memory is accessible to the Python interpreter, when the corresponding ``device`` is provided. If the
            array library does not support such cases at all, the function must raise ``BufferError``. If a copy must be made to
            enable this support but ``copy`` is set to ``False``, the function must raise ``ValueError``.

            Other device kinds will be considered for standardization in a future version of this API standard.
        copy: Optional[bool]
            boolean indicating whether or not to copy the input. If ``True``, the function must always copy. If ``False``, the function must never copy, and raise ``BufferError`` in case a copy is deemed necessary (e.g.  if a cross-device data movement is requested, and it is not possible without a copy). If ``None``, the function must reuse the existing memory buffer if possible and copy otherwise. Default: ``None``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the data in ``x``.

            .. admonition:: Note
            :class: note

            The returned array may be either a copy or a view. See :ref:`data-interchange` for details.

        Raises
        ------
        BufferError
            The ``__dlpack__`` and ``__dlpack_device__`` methods on the input array
            may raise ``BufferError`` when the data cannot be exported as DLPack
            (e.g., incompatible dtype, strides, or device). It may also raise other errors
            when export fails for other reasons (e.g., not enough memory available
            to materialize the data). ``from_dlpack`` must propagate such
            exceptions.
        AttributeError
            If the ``__dlpack__`` and ``__dlpack_device__`` methods are not present
            on the input array. This may happen for libraries that are never able
            to export their data with DLPack.
        ValueError
            If data exchange is possible via an explicit copy but ``copy`` is set to ``False``.

        Notes
        -----
        See :meth:`array.__dlpack__` for implementation suggestions for `from_dlpack` in
        order to handle DLPack versioning correctly.

        A way to move data from two array libraries to the same device (assumed supported by both libraries) in
        a library-agnostic fashion is illustrated below:

        .. code:: python

            def func(x, y):
                xp_x = x.__array_namespace__()
                xp_y = y.__array_namespace__()

                # Other functions than `from_dlpack` only work if both arrays are from the same library. So if
                # `y` is from a different one than `x`, let's convert `y` into an array of the same type as `x`:
                if not xp_x == xp_y:
                    y = xp_x.from_dlpack(y, copy=True, device=x.device)

                # From now on use `xp_x.xxxxx` functions, as both arrays are from the library `xp_x`
                ...


        .. versionchanged:: 2023.12
        Required exceptions to address unsupported use cases.

        .. versionchanged:: 2023.12
        Added device and copy support.
        """
        
    @abstractmethod
    def full(
        self : _NAMESPACE_C,
        shape: Union[int, Tuple[int, ...]],
        fill_value: Union[bool, int, float, complex],
        *,
        dtype: Optional[_NAMESPACE_DTYPE] = None,
        device: Optional[_NAMESPACE_DEVICE] = None,
    ) -> _NAMESPACE_ARRAY:
        """
        Returns a new array having a specified ``shape`` and filled with ``fill_value``.

        Parameters
        ----------
        shape: Union[int, Tuple[int, ...]]
            output array shape.
        fill_value: Union[bool, int, float, complex]
            fill value.
        dtype: Optional[dtype]
            output array data type. If ``dtype`` is ``None``, the output array data type must be inferred from ``fill_value`` according to the following rules:

            - If the fill value is an ``int``, the output array data type must be the default integer data type.
            - If the fill value is a ``float``, the output array data type must be the default real-valued floating-point data type.
            - If the fill value is a ``complex`` number, the output array data type must be the default complex floating-point data type.
            - If the fill value is a ``bool``, the output array must have a boolean data type. Default: ``None``.

            .. note::
            If the ``fill_value`` exceeds the precision of the resolved default output array data type, behavior is left unspecified and, thus, implementation-defined.

        device: Optional[device]
            device on which to place the created array. Default: ``None``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array where every element is equal to ``fill_value``.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.
        """
        
    @abstractmethod
    def full_like(
        self : _NAMESPACE_C,
        x: _NAMESPACE_ARRAY,
        /,
        fill_value: Union[bool, int, float, complex],
        *,
        dtype: Optional[_NAMESPACE_DTYPE] = None,
        device: Optional[_NAMESPACE_DEVICE] = None,
    ) -> _NAMESPACE_ARRAY:
        """
        Returns a new array filled with ``fill_value`` and having the same ``shape`` as an input array ``x``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array from which to derive the output array shape.
        fill_value: Union[bool, int, float, complex]
            fill value.
        dtype: Optional[dtype]
            output array data type. If ``dtype`` is ``None``, the output array data type must be inferred from ``x``. Default: ``None``.

            .. note::
            If the ``fill_value`` exceeds the precision of the resolved output array data type, behavior is unspecified and, thus, implementation-defined.

            .. note::
            If the ``fill_value`` has a data type which is not of the same data type kind (boolean, integer, or floating-point) as the resolved output array data type (see :ref:`type-promotion`), behavior is unspecified and, thus, implementation-defined.

        device: Optional[device]
            device on which to place the created array. If ``device`` is ``None``, the output array device must be inferred from ``x``. Default: ``None``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array having the same shape as ``x`` and where every element is equal to ``fill_value``.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.
        """
        
    @abstractmethod
    def linspace(
        self : _NAMESPACE_C,
        start: Union[int, float, complex],
        stop: Union[int, float, complex],
        /,
        num: int,
        *,
        dtype: Optional[_NAMESPACE_DTYPE] = None,
        device: Optional[_NAMESPACE_DEVICE] = None,
        endpoint: bool = True,
    ) -> _NAMESPACE_ARRAY:
        r"""
        Returns evenly spaced numbers over a specified interval.

        Let :math:`N` be the number of generated values (which is either ``num`` or ``num+1`` depending on whether ``endpoint`` is ``True`` or ``False``, respectively). For real-valued output arrays, the spacing between values is given by

        .. math::
        \Delta_{\textrm{real}} = \frac{\textrm{stop} - \textrm{start}}{N - 1}

        For complex output arrays, let ``a = real(start)``, ``b = imag(start)``, ``c = real(stop)``, and ``d = imag(stop)``. The spacing between complex values is given by

        .. math::
        \Delta_{\textrm{complex}} = \frac{c-a}{N-1} + \frac{d-b}{N-1} j

        Parameters
        ----------
        start: Union[int, float, complex]
            the start of the interval.
        stop: Union[int, float, complex]
            the end of the interval. If ``endpoint`` is ``False``, the function must generate a sequence of ``num+1`` evenly spaced numbers starting with ``start`` and ending with ``stop`` and exclude the ``stop`` from the returned array such that the returned array consists of evenly spaced numbers over the half-open interval ``[start, stop)``. If ``endpoint`` is ``True``, the output array must consist of evenly spaced numbers over the closed interval ``[start, stop]``. Default: ``True``.

            .. note::
            The step size changes when `endpoint` is `False`.

        num: int
            number of samples. Must be a nonnegative integer value.
        dtype: Optional[dtype]
            output array data type. Should be a floating-point data type. If ``dtype`` is ``None``,

            -   if either ``start`` or ``stop`` is a ``complex`` number, the output data type must be the default complex floating-point data type.
            -   if both ``start`` and ``stop`` are real-valued, the output data type must be the default real-valued floating-point data type.

            Default: ``None``.

            .. admonition:: Note
            :class: note

            If ``dtype`` is not ``None``, conversion of ``start`` and ``stop`` should obey :ref:`type-promotion` rules. Conversions not specified according to :ref:`type-promotion` rules may or may not be permitted by a conforming array library.

        device: Optional[device]
            device on which to place the created array. Default: ``None``.
        endpoint: bool
            boolean indicating whether to include ``stop`` in the interval. Default: ``True``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            a one-dimensional array containing evenly spaced values.

        Notes
        -----

        .. note::
        While this specification recommends that this function only return arrays having a floating-point data type, specification-compliant array libraries may choose to support output arrays having an integer data type (e.g., due to backward compatibility concerns). However, function behavior when generating integer output arrays is unspecified and, thus, is implementation-defined. Accordingly, using this function to generate integer output arrays is not portable.

        .. note::
        As mixed data type promotion is implementation-defined, behavior when ``start`` or ``stop`` exceeds the maximum safe integer of an output floating-point data type is implementation-defined. An implementation may choose to overflow or raise an exception.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """
        
    @abstractmethod
    def meshgrid(
        self: _NAMESPACE_C,
        *arrays: _NAMESPACE_ARRAY, indexing: Literal["xy", "ij"] = "xy"
    ) -> List[_NAMESPACE_ARRAY]:
        """
        Returns coordinate matrices from coordinate vectors.

        Parameters
        ----------
        arrays: _NAMESPACE_ARRAY
            an arbitrary number of one-dimensional arrays representing grid coordinates. Each array should have the same numeric data type.
        indexing: Literal["xy", "ij"]
            Cartesian ``'xy'`` or matrix ``'ij'`` indexing of output. If provided zero or one one-dimensional vector(s) (i.e., the zero- and one-dimensional cases, respectively), the ``indexing`` keyword has no effect and should be ignored. Default: ``'xy'``.

        Returns
        -------
        out: List[array]
            list of N arrays, where ``N`` is the number of provided one-dimensional input arrays. Each returned array must have rank ``N``. For ``N`` one-dimensional arrays having lengths ``Ni = len(xi)``,

            - if matrix indexing ``ij``, then each returned array must have the shape ``(N1, N2, N3, ..., Nn)``.
            - if Cartesian indexing ``xy``, then each returned array must have shape ``(N2, N1, N3, ..., Nn)``.

            Accordingly, for the two-dimensional case with input one-dimensional arrays of length ``M`` and ``N``, if matrix indexing ``ij``, then each returned array must have shape ``(M, N)``, and, if Cartesian indexing ``xy``, then each returned array must have shape ``(N, M)``.

            Similarly, for the three-dimensional case with input one-dimensional arrays of length ``M``, ``N``, and ``P``, if matrix indexing ``ij``, then each returned array must have shape ``(M, N, P)``, and, if Cartesian indexing ``xy``, then each returned array must have shape ``(N, M, P)``.

            Each returned array should have the same data type as the input arrays.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.
        """
        
    @abstractmethod
    def ones(
        self: _NAMESPACE_C,
        shape: Union[int, Tuple[int, ...]],
        *,
        dtype: Optional[_NAMESPACE_DTYPE] = None,
        device: Optional[_NAMESPACE_DEVICE] = None,
    ) -> _NAMESPACE_ARRAY:
        """
        Returns a new array having a specified ``shape`` and filled with ones.

        .. note::
        An output array having a complex floating-point data type must contain complex numbers having a real component equal to one and an imaginary component equal to zero (i.e., ``1 + 0j``).

        Parameters
        ----------
        shape: Union[int, Tuple[int, ...]]
            output array shape.
        dtype: Optional[dtype]
            output array data type. If ``dtype`` is ``None``, the output array data type must be the default real-valued floating-point data type. Default: ``None``.
        device: Optional[device]
            device on which to place the created array. Default: ``None``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing ones.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.
        """
        
    @abstractmethod
    def ones_like(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /, *, dtype: Optional[_NAMESPACE_DTYPE] = None, device: Optional[_NAMESPACE_DEVICE] = None
    ) -> _NAMESPACE_ARRAY:
        """
        Returns a new array filled with ones and having the same ``shape`` as an input array ``x``.

        .. note::
        An output array having a complex floating-point data type must contain complex numbers having a real component equal to one and an imaginary component equal to zero (i.e., ``1 + 0j``).

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array from which to derive the output array shape.
        dtype: Optional[dtype]
            output array data type. If ``dtype`` is ``None``, the output array data type must be inferred from ``x``. Default: ``None``.
        device: Optional[device]
            device on which to place the created array. If ``device`` is ``None``, the output array device must be inferred from ``x``. Default: ``None``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array having the same shape as ``x`` and filled with ones.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.
        """
        
    @abstractmethod
    def tril(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /, *, k: int = 0
    ) -> _NAMESPACE_ARRAY:
        """
        Returns the lower triangular part of a matrix (or a stack of matrices) ``x``.

        .. note::
        The lower triangular part of the matrix is defined as the elements on and below the specified diagonal ``k``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array having shape ``(..., M, N)`` and whose innermost two dimensions form ``MxN`` matrices.
        k: int
            diagonal above which to zero elements. If ``k = 0``, the diagonal is the main diagonal. If ``k < 0``, the diagonal is below the main diagonal. If ``k > 0``, the diagonal is above the main diagonal. Default: ``0``.

            .. note::
            The main diagonal is defined as the set of indices ``{(i, i)}`` for ``i`` on the interval ``[0, min(M, N) - 1]``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the lower triangular part(s). The returned array must have the same shape and data type as ``x``. All elements above the specified diagonal ``k`` must be zeroed. The returned array should be allocated on the same device as ``x``.
        """
        
    @abstractmethod
    def triu(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /, *, k: int = 0
    ) -> _NAMESPACE_ARRAY:
        """
        Returns the upper triangular part of a matrix (or a stack of matrices) ``x``.

        .. note::
        The upper triangular part of the matrix is defined as the elements on and above the specified diagonal ``k``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array having shape ``(..., M, N)`` and whose innermost two dimensions form ``MxN`` matrices.
        k: int
            diagonal below which to zero elements. If ``k = 0``, the diagonal is the main diagonal. If ``k < 0``, the diagonal is below the main diagonal. If ``k > 0``, the diagonal is above the main diagonal. Default: ``0``.

            .. note::
            The main diagonal is defined as the set of indices ``{(i, i)}`` for ``i`` on the interval ``[0, min(M, N) - 1]``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the upper triangular part(s). The returned array must have the same shape and data type as ``x``. All elements below the specified diagonal ``k`` must be zeroed. The returned array should be allocated on the same device as ``x``.
        """
        
    @abstractmethod
    def zeros(
        self: _NAMESPACE_C,
        shape: Union[int, Tuple[int, ...]],
        *,
        dtype: Optional[_NAMESPACE_DTYPE] = None,
        device: Optional[_NAMESPACE_DEVICE] = None,
    ) -> _NAMESPACE_ARRAY:
        """
        Returns a new array having a specified ``shape`` and filled with zeros.

        Parameters
        ----------
        shape: Union[int, Tuple[int, ...]]
            output array shape.
        dtype: Optional[dtype]
            output array data type. If ``dtype`` is ``None``, the output array data type must be the default real-valued floating-point data type. Default: ``None``.
        device: Optional[device]
            device on which to place the created array. Default: ``None``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing zeros.
        """
        
    @abstractmethod
    def zeros_like(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /, *, dtype: Optional[_NAMESPACE_DTYPE] = None, device: Optional[_NAMESPACE_DEVICE] = None
    ) -> _NAMESPACE_ARRAY:
        """
        Returns a new array filled with zeros and having the same ``shape`` as an input array ``x``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array from which to derive the output array shape.
        dtype: Optional[dtype]
            output array data type. If ``dtype`` is ``None``, the output array data type must be inferred from ``x``. Default: ``None``.
        device: Optional[device]
            device on which to place the created array. If ``device`` is ``None``, the output array device must be inferred from ``x``. Default: ``None``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array having the same shape as ``x`` and filled with zeros.
        """
        
    """
    Data Type Functions
    https://github.com/data-apis/array-api/blob/main/src/array_api_stubs/_2024_12/data_type_functions.py
    """
    @abstractmethod
    def astype(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, dtype: _NAMESPACE_DTYPE, /, *, copy: bool = True, device: Optional[_NAMESPACE_DEVICE] = None
    ) -> _NAMESPACE_ARRAY:
        """
        Copies an array to a specified data type irrespective of :ref:`type-promotion` rules.

        .. note::
        Casting floating-point ``NaN`` and ``infinity`` values to integral data types is not specified and is implementation-dependent.

        .. note::
        Casting a complex floating-point array to a real-valued data type should not be permitted.

        Historically, when casting a complex floating-point array to a real-valued data type, libraries such as NumPy have discarded imaginary components such that, for a complex floating-point array ``x``, ``astype(x)`` equals ``astype(real(x))``). This behavior is considered problematic as the choice to discard the imaginary component is arbitrary and introduces more than one way to achieve the same outcome (i.e., for a complex floating-point array ``x``, ``astype(x)`` and ``astype(real(x))`` versus only ``astype(imag(x))``). Instead, in order to avoid ambiguity and to promote clarity, this specification requires that array API consumers explicitly express which component should be cast to a specified real-valued data type.

        .. note::
        When casting a boolean input array to a real-valued data type, a value of ``True`` must cast to a real-valued number equal to ``1``, and a value of ``False`` must cast to a real-valued number equal to ``0``.

        When casting a boolean input array to a complex floating-point data type, a value of ``True`` must cast to a complex number equal to ``1 + 0j``, and a value of ``False`` must cast to a complex number equal to ``0 + 0j``.

        .. note::
        When casting a real-valued input array to ``bool``, a value of ``0`` must cast to ``False``, and a non-zero value must cast to ``True``.

        When casting a complex floating-point array to ``bool``, a value of ``0 + 0j`` must cast to ``False``, and all other values must cast to ``True``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            array to cast.
        dtype: dtype
            desired data type.
        copy: bool
            specifies whether to copy an array when the specified ``dtype`` matches the data type of the input array ``x``. If ``True``, a newly allocated array must always be returned (see :ref:`copy-keyword-argument`). If ``False`` and the specified ``dtype`` matches the data type of the input array, the input array must be returned; otherwise, a newly allocated array must be returned. Default: ``True``.
        device: Optional[device]
            device on which to place the returned array. If ``device`` is ``None``, the output array device must be inferred from ``x``. Default: ``None``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array having the specified data type. The returned array must have the same shape as ``x``.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.

        .. versionchanged:: 2023.12
        Added device keyword argument support.
        """
        
    @abstractmethod
    def can_cast(
        self: _NAMESPACE_C,
        from_: Union[_NAMESPACE_DTYPE, _NAMESPACE_ARRAY], to: _NAMESPACE_DTYPE, /
    ) -> bool:
        """
        Determines if one data type can be cast to another data type according to type promotion rules (see :ref:`type-promotion`).

        Parameters
        ----------
        from_: Union[dtype, array]
            input data type or array from which to cast.
        to: dtype
            desired data type.

        Returns
        -------
        out: bool
            ``True`` if the cast can occur according to type promotion rules (see :ref:`type-promotion`); otherwise, ``False``.

        Notes
        -----

        -   When ``from_`` is a data type, the function must determine whether the data type can be cast to another data type according to the complete type promotion rules (see :ref:`type-promotion`) described in this specification, irrespective of whether a conforming array library supports devices which do not have full data type support.
        -   When ``from_`` is an array, the function must determine whether the data type of the array can be cast to the desired data type according to the type promotion graph of the array device. As not all devices can support all data types, full support for type promotion rules (see :ref:`type-promotion`) may not be possible. Accordingly, the output of ``can_cast(array, dtype)`` may differ from ``can_cast(array.dtype, dtype)``.

        .. versionchanged:: 2024.12
        Required that the application of type promotion rules must account for device context.
        """
        
    @abstractmethod
    def finfo(
        self: _NAMESPACE_C,
        type: Union[_NAMESPACE_DTYPE, _NAMESPACE_ARRAY], /
    ) -> finfo_object:
        """
        Machine limits for floating-point data types.

        Parameters
        ----------
        type: Union[dtype, array]
            the kind of floating-point data-type about which to get information. If complex, the information is about its component data type.

            .. note::
            Complex floating-point data types are specified to always use the same precision for both its real and imaginary components, so the information should be true for either component.

        Returns
        -------
        out: finfo object
            an object having the following attributes:

            - **bits**: *int*

            number of bits occupied by the real-valued floating-point data type.

            - **eps**: *float*

            difference between 1.0 and the next smallest representable real-valued floating-point number larger than 1.0 according to the IEEE-754 standard.

            - **max**: *float*

            largest representable real-valued number.

            - **min**: *float*

            smallest representable real-valued number.

            - **smallest_normal**: *float*

            smallest positive real-valued floating-point number with full precision.

            - **dtype**: dtype

            real-valued floating-point data type.

            .. versionadded:: 2022.12

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.
        """
        
    @abstractmethod
    def iinfo(
        self: _NAMESPACE_C,
        type: Union[_NAMESPACE_DTYPE, _NAMESPACE_ARRAY], /
    ) -> iinfo_object:
        """
        Machine limits for integer data types.

        Parameters
        ----------
        type: Union[dtype, array]
            the kind of integer data-type about which to get information.

        Returns
        -------
        out: iinfo object
            an object having the following attributes:

            - **bits**: *int*

            number of bits occupied by the type.

            - **max**: *int*

            largest representable number.

            - **min**: *int*

            smallest representable number.

            - **dtype**: dtype

            integer data type.

            .. versionadded:: 2022.12
        """
        
    @abstractmethod
    def isdtype(
        self: _NAMESPACE_C,
        dtype: _NAMESPACE_DTYPE, kind: Union[_NAMESPACE_DTYPE, str, Tuple[Union[_NAMESPACE_DTYPE, str], ...]]
    ) -> bool:
        """
        Returns a boolean indicating whether a provided dtype is of a specified data type "kind".

        Parameters
        ----------
        dtype: dtype
            the input dtype.
        kind: Union[str, dtype, Tuple[Union[str, dtype], ...]]
            data type kind.

            -   If ``kind`` is a dtype, the function must return a boolean indicating whether the input ``dtype`` is equal to the dtype specified by ``kind``.
            -   If ``kind`` is a string, the function must return a boolean indicating whether the input ``dtype`` is of a specified data type kind. The following dtype kinds must be supported:

                -   ``'bool'``: boolean data types (e.g., ``bool``).
                -   ``'signed integer'``: signed integer data types (e.g., ``int8``, ``int16``, ``int32``, ``int64``).
                -   ``'unsigned integer'``: unsigned integer data types (e.g., ``uint8``, ``uint16``, ``uint32``, ``uint64``).
                -   ``'integral'``: integer data types. Shorthand for ``('signed integer', 'unsigned integer')``.
                -   ``'real floating'``: real-valued floating-point data types (e.g., ``float32``, ``float64``).
                -   ``'complex floating'``: complex floating-point data types (e.g., ``complex64``, ``complex128``).
                -   ``'numeric'``: numeric data types. Shorthand for ``('integral', 'real floating', 'complex floating')``.

            -   If ``kind`` is a tuple, the tuple specifies a union of dtypes and/or kinds, and the function must return a boolean indicating whether the input ``dtype`` is either equal to a specified dtype or belongs to at least one specified data type kind.

            .. note::
            A conforming implementation of the array API standard is **not** limited to only including the dtypes described in this specification in the required data type kinds. For example, implementations supporting ``float16`` and ``bfloat16`` can include ``float16`` and ``bfloat16`` in the ``real floating`` data type kind. Similarly, implementations supporting ``int128`` can include ``int128`` in the ``signed integer`` data type kind.

            In short, conforming implementations may extend data type kinds; however, data type kinds must remain consistent (e.g., only integer dtypes may belong to integer data type kinds and only floating-point dtypes may belong to floating-point data type kinds), and extensions must be clearly documented as such in library documentation.

        Returns
        -------
        out: bool
            boolean indicating whether a provided dtype is of a specified data type kind.

        Notes
        -----

        .. versionadded:: 2022.12
        """
        
    @abstractmethod
    def result_type(
        self: _NAMESPACE_C,
        *arrays_and_dtypes: Union[_NAMESPACE_ARRAY, int, float, complex, bool, _NAMESPACE_DTYPE]
    ) -> _NAMESPACE_DTYPE:
        """
        Returns the dtype that results from applying type promotion rules (see :ref:`type-promotion`) to the arguments.

        Parameters
        ----------
        arrays_and_dtypes: Union[array, int, float, complex, bool, dtype]
            an arbitrary number of input arrays, scalars, and/or dtypes.

        Returns
        -------
        out: dtype
            the dtype resulting from an operation involving the input arrays, scalars, and/or dtypes.

        Notes
        -----

        -   At least one argument must be an array or a dtype.
        -   If provided array and/or dtype arguments having mixed data type kinds (e.g., integer and floating-point), the returned dtype is unspecified and thus implementation-dependent.
        -   If at least one argument is an array, the function must determine the resulting dtype according to the type promotion graph of the array device which is shared among all array arguments. As not all devices can support all data types, full support for type promotion rules (see :ref:`type-promotion`) may not be possible. Accordingly, the returned dtype may differ from that determined from the complete type promotion graph defined in this specification (see :ref:`type-promotion`).
        -   If two or more arguments are arrays belonging to different devices, behavior is unspecified and thus implementation-dependent. Conforming implementations may choose to ignore device attributes, raise an exception, or some other behavior.

        .. versionchanged:: 2024.12
        Added scalar argument support.

        .. versionchanged:: 2024.12
        Required that the application of type promotion rules must account for device context.
        """
        
    """
    Elementwise Functions
    https://github.com/data-apis/array-api/blob/main/src/array_api_stubs/_2024_12/elementwise_functions.py
    """
    @abstractmethod
    def abs(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Calculates the absolute value for each element ``x_i`` of the input array ``x``.

        For real-valued input arrays, the element-wise result has the same magnitude as the respective element in ``x`` but has positive sign.

        .. note::
        For signed integer data types, the absolute value of the minimum representable integer is implementation-dependent.

        .. note::
        For complex floating-point operands, the complex absolute value is known as the norm, modulus, or magnitude and, for a complex number :math:`z = a + bj` is computed as

        .. math::
            \operatorname{abs}(z) = \sqrt{a^2 + b^2}

        .. note::
        For complex floating-point operands, conforming implementations should take care to avoid undue overflow or underflow during intermediate stages of computation.

        ..
        TODO: once ``hypot`` is added to the specification, remove the special cases for complex floating-point operands and the note concerning guarding against undue overflow/underflow, and state that special cases must be handled as if implemented as ``hypot(real(x), imag(x))``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a numeric data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the absolute value of each element in ``x``. If ``x`` has a real-valued data type, the returned array must have the same data type as ``x``. If ``x`` has a complex floating-point data type, the returned array must have a real-valued floating-point data type whose precision matches the precision of ``x`` (e.g., if ``x`` is ``complex128``, then the returned array must have a ``float64`` data type).

        Notes
        -----

        **Special Cases**

        For real-valued floating-point operands,

        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If ``x_i`` is ``-0``, the result is ``+0``.
        - If ``x_i`` is ``-infinity``, the result is ``+infinity``.

        For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

        - If ``a`` is either ``+infinity`` or ``-infinity`` and ``b`` is any value (including ``NaN``), the result is ``+infinity``.
        - If ``a`` is any value (including ``NaN``) and ``b`` is either ``+infinity`` or ``-infinity``, the result is ``+infinity``.
        - If ``a`` is either ``+0`` or ``-0``, the result is equal to ``abs(b)``.
        - If ``b`` is either ``+0`` or ``-0``, the result is equal to ``abs(a)``.
        - If ``a`` is ``NaN`` and ``b`` is a finite number, the result is ``NaN``.
        - If ``a`` is a finite number and ``b`` is ``NaN``, the result is ``NaN``.
        - If ``a`` is ``NaN`` and ``b`` is ``NaN``, the result is ``NaN``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def acos(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Calculates an implementation-dependent approximation of the principal value of the inverse cosine for each element ``x_i`` of the input array ``x``.

        Each element-wise result is expressed in radians.

        .. note::
        The principal value of the arc cosine of a complex number :math:`z` is

        .. math::
            \operatorname{acos}(z) = \frac{1}{2}\pi + j\ \ln(zj + \sqrt{1-z^2})

        For any :math:`z`,

        .. math::
            \operatorname{acos}(z) = \pi - \operatorname{acos}(-z)

        .. note::
        For complex floating-point operands, ``acos(conj(x))`` must equal ``conj(acos(x))``.

        .. note::
        The inverse cosine (or arc cosine) is a multi-valued function and requires a branch cut on the complex plane. By convention, a branch cut is placed at the line segments :math:`(-\infty, -1)` and :math:`(1, \infty)` of the real axis.

        Accordingly, for complex arguments, the function returns the inverse cosine in the range of a strip unbounded along the imaginary axis and in the interval :math:`[0, \pi]` along the real axis.

        *Note: branch cuts follow C99 and have provisional status* (see :ref:`branch-cuts`).

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the inverse cosine of each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Special cases**

        For real-valued floating-point operands,

        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If ``x_i`` is greater than ``1``, the result is ``NaN``.
        - If ``x_i`` is less than ``-1``, the result is ``NaN``.
        - If ``x_i`` is ``1``, the result is ``+0``.

        For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

        - If ``a`` is either ``+0`` or ``-0`` and ``b`` is ``+0``, the result is ``π/2 - 0j``.
        - If ``a`` is either ``+0`` or ``-0`` and ``b`` is ``NaN``, the result is ``π/2 + NaN j``.
        - If ``a`` is a finite number and ``b`` is ``+infinity``, the result is ``π/2 - infinity j``.
        - If ``a`` is a nonzero finite number and ``b`` is ``NaN``, the result is ``NaN + NaN j``.
        - If ``a`` is ``-infinity`` and ``b`` is a positive (i.e., greater than ``0``) finite number, the result is ``π - infinity j``.
        - If ``a`` is ``+infinity`` and ``b`` is a positive (i.e., greater than ``0``) finite number, the result is ``+0 - infinity j``.
        - If ``a`` is ``-infinity`` and ``b`` is ``+infinity``, the result is ``3π/4 - infinity j``.
        - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``, the result is ``π/4 - infinity j``.
        - If ``a`` is either ``+infinity`` or ``-infinity`` and ``b`` is ``NaN``, the result is ``NaN ± infinity j`` (sign of the imaginary component is unspecified).
        - If ``a`` is ``NaN`` and ``b`` is a finite number, the result is ``NaN + NaN j``.
        - If ``a`` is ``NaN`` and ``b`` is ``+infinity``, the result is ``NaN - infinity j``.
        - If ``a`` is ``NaN`` and ``b`` is ``NaN``, the result is ``NaN + NaN j``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def acosh(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Calculates an implementation-dependent approximation to the inverse hyperbolic cosine for each element ``x_i`` of the input array ``x``.

        .. note::
        The principal value of the inverse hyperbolic cosine of a complex number :math:`z` is

        .. math::
            \operatorname{acosh}(z) = \ln(z + \sqrt{z+1}\sqrt{z-1})

        For any :math:`z`,

        .. math::
            \operatorname{acosh}(z) = \frac{\sqrt{z-1}}{\sqrt{1-z}}\operatorname{acos}(z)

        or simply

        .. math::
            \operatorname{acosh}(z) = j\ \operatorname{acos}(z)

        in the upper half of the complex plane.

        .. note::
        For complex floating-point operands, ``acosh(conj(x))`` must equal ``conj(acosh(x))``.

        .. note::
        The inverse hyperbolic cosine is a multi-valued function and requires a branch cut on the complex plane. By convention, a branch cut is placed at the line segment :math:`(-\infty, 1)` of the real axis.

        Accordingly, for complex arguments, the function returns the inverse hyperbolic cosine in the interval :math:`[0, \infty)` along the real axis and in the interval :math:`[-\pi j, +\pi j]` along the imaginary axis.

        *Note: branch cuts follow C99 and have provisional status* (see :ref:`branch-cuts`).

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array whose elements each represent the area of a hyperbolic sector. Should have a floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the inverse hyperbolic cosine of each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Special cases**

        For real-valued floating-point operands,

        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If ``x_i`` is less than ``1``, the result is ``NaN``.
        - If ``x_i`` is ``1``, the result is ``+0``.
        - If ``x_i`` is ``+infinity``, the result is ``+infinity``.

        For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

        - If ``a`` is either ``+0`` or ``-0`` and ``b`` is ``+0``, the result is ``+0 + πj/2``.
        - If ``a`` is a finite number and ``b`` is ``+infinity``, the result is ``+infinity + πj/2``.
        - If ``a`` is a nonzero finite number and ``b`` is ``NaN``, the result is ``NaN + NaN j``.
        - If ``a`` is ``+0`` and ``b`` is ``NaN``, the result is ``NaN ± πj/2`` (sign of imaginary component is unspecified).
        - If ``a`` is ``-infinity`` and ``b`` is a positive (i.e., greater than ``0``) finite number, the result is ``+infinity + πj``.
        - If ``a`` is ``+infinity`` and ``b`` is a positive (i.e., greater than ``0``) finite number, the result is ``+infinity + 0j``.
        - If ``a`` is ``-infinity`` and ``b`` is ``+infinity``, the result is ``+infinity + 3πj/4``.
        - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``, the result is ``+infinity + πj/4``.
        - If ``a`` is either ``+infinity`` or ``-infinity`` and ``b`` is ``NaN``, the result is ``+infinity + NaN j``.
        - If ``a`` is ``NaN`` and ``b`` is a finite number, the result is ``NaN + NaN j``.
        - If ``a`` is ``NaN`` and ``b`` is ``+infinity``, the result is ``+infinity + NaN j``.
        - If ``a`` is ``NaN`` and ``b`` is ``NaN``, the result is ``NaN + NaN j``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def add(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, float, complex], x2: Union[_NAMESPACE_ARRAY, int, float, complex], /
    ) -> _NAMESPACE_ARRAY:
        """
        Calculates the sum for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

        Parameters
        ----------
        x1: Union[array, int, float, complex]
            first input array. Should have a numeric data type.
        x2: Union[array, int, float, complex]
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise sums. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.

        **Special cases**

        For real-valued floating-point operands,

        - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
        - If ``x1_i`` is ``+infinity`` and ``x2_i`` is ``-infinity``, the result is ``NaN``.
        - If ``x1_i`` is ``-infinity`` and ``x2_i`` is ``+infinity``, the result is ``NaN``.
        - If ``x1_i`` is ``+infinity`` and ``x2_i`` is ``+infinity``, the result is ``+infinity``.
        - If ``x1_i`` is ``-infinity`` and ``x2_i`` is ``-infinity``, the result is ``-infinity``.
        - If ``x1_i`` is ``+infinity`` and ``x2_i`` is a finite number, the result is ``+infinity``.
        - If ``x1_i`` is ``-infinity`` and ``x2_i`` is a finite number, the result is ``-infinity``.
        - If ``x1_i`` is a finite number and ``x2_i`` is ``+infinity``, the result is ``+infinity``.
        - If ``x1_i`` is a finite number and ``x2_i`` is ``-infinity``, the result is ``-infinity``.
        - If ``x1_i`` is ``-0`` and ``x2_i`` is ``-0``, the result is ``-0``.
        - If ``x1_i`` is ``-0`` and ``x2_i`` is ``+0``, the result is ``+0``.
        - If ``x1_i`` is ``+0`` and ``x2_i`` is ``-0``, the result is ``+0``.
        - If ``x1_i`` is ``+0`` and ``x2_i`` is ``+0``, the result is ``+0``.
        - If ``x1_i`` is either ``+0`` or ``-0`` and ``x2_i`` is a nonzero finite number, the result is ``x2_i``.
        - If ``x1_i`` is a nonzero finite number and ``x2_i`` is either ``+0`` or ``-0``, the result is ``x1_i``.
        - If ``x1_i`` is a nonzero finite number and ``x2_i`` is ``-x1_i``, the result is ``+0``.
        - In the remaining cases, when neither ``infinity``, ``+0``, ``-0``, nor a ``NaN`` is involved, and the operands have the same mathematical sign or have different magnitudes, the sum must be computed and rounded to the nearest representable value according to IEEE 754-2019 and a supported round mode. If the magnitude is too large to represent, the operation overflows and the result is an `infinity` of appropriate mathematical sign.

        .. note::
        Floating-point addition is a commutative operation, but not always associative.

        For complex floating-point operands, addition is defined according to the following table. For real components ``a`` and ``c`` and imaginary components ``b`` and ``d``,

        +------------+------------+------------+----------------+
        |            | c          | dj         | c + dj         |
        +============+============+============+================+
        | **a**      | a + c      | a + dj     | (a+c) + dj     |
        +------------+------------+------------+----------------+
        | **bj**     | c + bj     | (b+d)j     | c + (b+d)j     |
        +------------+------------+------------+----------------+
        | **a + bj** | (a+c) + bj | a + (b+d)j | (a+c) + (b+d)j |
        +------------+------------+------------+----------------+

        For complex floating-point operands, real-valued floating-point special cases must independently apply to the real and imaginary component operations involving real numbers as described in the above table. For example, let ``a = real(x1_i)``, ``b = imag(x1_i)``, ``c = real(x2_i)``, ``d = imag(x2_i)``, and

        - If ``a`` is ``-0`` and ``c`` is ``-0``, the real component of the result is ``-0``.
        - Similarly, if ``b`` is ``+0`` and ``d`` is ``-0``, the imaginary component of the result is ``+0``.

        Hence, if ``z1 = a + bj = -0 + 0j`` and ``z2 = c + dj = -0 - 0j``, the result of ``z1 + z2`` is ``-0 + 0j``.

        .. versionchanged:: 2022.12
        Added complex data type support.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def asin(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Calculates an implementation-dependent approximation of the principal value of the inverse sine for each element ``x_i`` of the input array ``x``.

        Each element-wise result is expressed in radians.

        .. note::
        The principal value of the arc sine of a complex number :math:`z` is

        .. math::
            \operatorname{asin}(z) = -j\ \ln(zj + \sqrt{1-z^2})

        For any :math:`z`,

        .. math::
            \operatorname{asin}(z) = \operatorname{acos}(-z) - \frac{\pi}{2}

        .. note::
        For complex floating-point operands, ``asin(conj(x))`` must equal ``conj(asin(x))``.

        .. note::
        The inverse sine (or arc sine) is a multi-valued function and requires a branch cut on the complex plane. By convention, a branch cut is placed at the line segments :math:`(-\infty, -1)` and :math:`(1, \infty)` of the real axis.

        Accordingly, for complex arguments, the function returns the inverse sine in the range of a strip unbounded along the imaginary axis and in the interval :math:`[-\pi/2, +\pi/2]` along the real axis.

        *Note: branch cuts follow C99 and have provisional status* (see :ref:`branch-cuts`).

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the inverse sine of each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Special cases**

        For real-valued floating-point operands,

        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If ``x_i`` is greater than ``1``, the result is ``NaN``.
        - If ``x_i`` is less than ``-1``, the result is ``NaN``.
        - If ``x_i`` is ``+0``, the result is ``+0``.
        - If ``x_i`` is ``-0``, the result is ``-0``.

        For complex floating-point operands, special cases must be handled as if the operation is implemented as ``-1j * asinh(x*1j)``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def asinh(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Calculates an implementation-dependent approximation to the inverse hyperbolic sine for each element ``x_i`` in the input array ``x``.

        .. note::
        The principal value of the inverse hyperbolic sine of a complex number :math:`z` is

        .. math::
            \operatorname{asinh}(z) = \ln(z + \sqrt{1+z^2})

        For any :math:`z`,

        .. math::
            \operatorname{asinh}(z) = \frac{\operatorname{asin}(zj)}{j}

        .. note::
        For complex floating-point operands, ``asinh(conj(x))`` must equal ``conj(asinh(x))`` and ``asinh(-z)`` must equal ``-asinh(z)``.

        .. note::
        The inverse hyperbolic sine is a multi-valued function and requires a branch cut on the complex plane. By convention, a branch cut is placed at the line segments :math:`(-\infty j, -j)` and :math:`(j, \infty j)` of the imaginary axis.

        Accordingly, for complex arguments, the function returns the inverse hyperbolic sine in the range of a strip unbounded along the real axis and in the interval :math:`[-\pi j/2, +\pi j/2]` along the imaginary axis.

        *Note: branch cuts follow C99 and have provisional status* (see :ref:`branch-cuts`).

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array whose elements each represent the area of a hyperbolic sector. Should have a floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the inverse hyperbolic sine of each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Special cases**

        For real-valued floating-point operands,

        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If ``x_i`` is ``+0``, the result is ``+0``.
        - If ``x_i`` is ``-0``, the result is ``-0``.
        - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
        - If ``x_i`` is ``-infinity``, the result is ``-infinity``.

        For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

        - If ``a`` is ``+0`` and ``b`` is ``+0``, the result is ``+0 + 0j``.
        - If ``a`` is a positive (i.e., greater than ``0``) finite number and ``b`` is ``+infinity``, the result is ``+infinity + πj/2``.
        - If ``a`` is a finite number and ``b`` is ``NaN``, the result is ``NaN + NaN j``.
        - If ``a`` is ``+infinity`` and ``b`` is a positive (i.e., greater than ``0``) finite number, the result is ``+infinity + 0j``.
        - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``, the result is ``+infinity + πj/4``.
        - If ``a`` is ``NaN`` and ``b`` is ``+0``, the result is ``NaN + 0j``.
        - If ``a`` is ``NaN`` and ``b`` is a nonzero finite number, the result is ``NaN + NaN j``.
        - If ``a`` is ``NaN`` and ``b`` is ``+infinity``, the result is ``±infinity + NaN j`` (sign of the real component is unspecified).
        - If ``a`` is ``NaN`` and ``b`` is ``NaN``, the result is ``NaN + NaN j``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def atan(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Calculates an implementation-dependent approximation of the principal value of the inverse tangent for each element ``x_i`` of the input array ``x``.

        Each element-wise result is expressed in radians.

        .. note::
        The principal value of the inverse tangent of a complex number :math:`z` is

        .. math::
            \operatorname{atan}(z) = -\frac{\ln(1 - zj) - \ln(1 + zj)}{2}j

        .. note::
        For complex floating-point operands, ``atan(conj(x))`` must equal ``conj(atan(x))``.

        .. note::
        The inverse tangent (or arc tangent) is a multi-valued function and requires a branch on the complex plane. By convention, a branch cut is placed at the line segments :math:`(-\infty j, -j)` and :math:`(+j, \infty j)` of the imaginary axis.

        Accordingly, for complex arguments, the function returns the inverse tangent in the range of a strip unbounded along the imaginary axis and in the interval :math:`[-\pi/2, +\pi/2]` along the real axis.

        *Note: branch cuts follow C99 and have provisional status* (see :ref:`branch-cuts`).

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the inverse tangent of each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Special cases**

        For real-valued floating-point operands,

        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If ``x_i`` is ``+0``, the result is ``+0``.
        - If ``x_i`` is ``-0``, the result is ``-0``.
        - If ``x_i`` is ``+infinity``, the result is an implementation-dependent approximation to ``+π/2``.
        - If ``x_i`` is ``-infinity``, the result is an implementation-dependent approximation to ``-π/2``.

        For complex floating-point operands, special cases must be handled as if the operation is implemented as ``-1j * atanh(x*1j)``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def atan2(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, float], x2: Union[_NAMESPACE_ARRAY, int, float], /
    ) -> _NAMESPACE_ARRAY:
        """
        Calculates an implementation-dependent approximation of the inverse tangent of the quotient ``x1/x2``, having domain ``[-infinity, +infinity] x [-infinity, +infinity]`` (where the ``x`` notation denotes the set of ordered pairs of elements ``(x1_i, x2_i)``) and codomain ``[-π, +π]``, for each pair of elements ``(x1_i, x2_i)`` of the input arrays ``x1`` and ``x2``, respectively. Each element-wise result is expressed in radians.

        The mathematical signs of ``x1_i`` and ``x2_i`` determine the quadrant of each element-wise result. The quadrant (i.e., branch) is chosen such that each element-wise result is the signed angle in radians between the ray ending at the origin and passing through the point ``(1,0)`` and the ray ending at the origin and passing through the point ``(x2_i, x1_i)``.

        .. note::
        Note the role reversal: the "y-coordinate" is the first function parameter; the "x-coordinate" is the second function parameter. The parameter order is intentional and traditional for the two-argument inverse tangent function where the y-coordinate argument is first and the x-coordinate argument is second.

        By IEEE 754 convention, the inverse tangent of the quotient ``x1/x2`` is defined for ``x2_i`` equal to positive or negative zero and for either or both of ``x1_i`` and ``x2_i`` equal to positive or negative ``infinity``.

        Parameters
        ----------
        x1: Union[array, int, float]
            input array corresponding to the y-coordinates. Should have a real-valued floating-point data type.
        x2: Union[array, int, float]
            input array corresponding to the x-coordinates. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a real-valued floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the inverse tangent of the quotient ``x1/x2``. The returned array must have a real-valued floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.

        **Special cases**

        For floating-point operands,

        - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
        - If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``+0``, the result is an implementation-dependent approximation to ``+π/2``.
        - If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``-0``, the result is an implementation-dependent approximation to ``+π/2``.
        - If ``x1_i`` is ``+0`` and ``x2_i`` is greater than ``0``, the result is ``+0``.
        - If ``x1_i`` is ``+0`` and ``x2_i`` is ``+0``, the result is ``+0``.
        - If ``x1_i`` is ``+0`` and ``x2_i`` is ``-0``, the result is an implementation-dependent approximation to ``+π``.
        - If ``x1_i`` is ``+0`` and ``x2_i`` is less than ``0``, the result is an implementation-dependent approximation to ``+π``.
        - If ``x1_i`` is ``-0`` and ``x2_i`` is greater than ``0``, the result is ``-0``.
        - If ``x1_i`` is ``-0`` and ``x2_i`` is ``+0``, the result is ``-0``.
        - If ``x1_i`` is ``-0`` and ``x2_i`` is ``-0``, the result is an implementation-dependent approximation to ``-π``.
        - If ``x1_i`` is ``-0`` and ``x2_i`` is less than ``0``, the result is an implementation-dependent approximation to ``-π``.
        - If ``x1_i`` is less than ``0`` and ``x2_i`` is ``+0``, the result is an implementation-dependent approximation to ``-π/2``.
        - If ``x1_i`` is less than ``0`` and ``x2_i`` is ``-0``, the result is an implementation-dependent approximation to ``-π/2``.
        - If ``x1_i`` is greater than ``0``, ``x1_i`` is a finite number, and ``x2_i`` is ``+infinity``, the result is ``+0``.
        - If ``x1_i`` is greater than ``0``, ``x1_i`` is a finite number, and ``x2_i`` is ``-infinity``, the result is an implementation-dependent approximation to ``+π``.
        - If ``x1_i`` is less than ``0``, ``x1_i`` is a finite number, and ``x2_i`` is ``+infinity``, the result is ``-0``.
        - If ``x1_i`` is less than ``0``, ``x1_i`` is a finite number, and ``x2_i`` is ``-infinity``, the result is an implementation-dependent approximation to ``-π``.
        - If ``x1_i`` is ``+infinity`` and ``x2_i`` is a finite number, the result is an implementation-dependent approximation to ``+π/2``.
        - If ``x1_i`` is ``-infinity`` and ``x2_i`` is a finite number, the result is an implementation-dependent approximation to ``-π/2``.
        - If ``x1_i`` is ``+infinity`` and ``x2_i`` is ``+infinity``, the result is an implementation-dependent approximation to ``+π/4``.
        - If ``x1_i`` is ``+infinity`` and ``x2_i`` is ``-infinity``, the result is an implementation-dependent approximation to ``+3π/4``.
        - If ``x1_i`` is ``-infinity`` and ``x2_i`` is ``+infinity``, the result is an implementation-dependent approximation to ``-π/4``.
        - If ``x1_i`` is ``-infinity`` and ``x2_i`` is ``-infinity``, the result is an implementation-dependent approximation to ``-3π/4``.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def atanh(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Calculates an implementation-dependent approximation to the inverse hyperbolic tangent for each element ``x_i`` of the input array ``x``.

        .. note::
        The principal value of the inverse hyperbolic tangent of a complex number :math:`z` is

        .. math::
            \operatorname{atanh}(z) = \frac{\ln(1+z)-\ln(z-1)}{2}

        For any :math:`z`,

        .. math::
            \operatorname{atanh}(z) = \frac{\operatorname{atan}(zj)}{j}

        .. note::
        For complex floating-point operands, ``atanh(conj(x))`` must equal ``conj(atanh(x))`` and ``atanh(-x)`` must equal ``-atanh(x)``.

        .. note::
        The inverse hyperbolic tangent is a multi-valued function and requires a branch cut on the complex plane. By convention, a branch cut is placed at the line segments :math:`(-\infty, 1]` and :math:`[1, \infty)` of the real axis.

        Accordingly, for complex arguments, the function returns the inverse hyperbolic tangent in the range of a half-strip unbounded along the real axis and in the interval :math:`[-\pi j/2, +\pi j/2]` along the imaginary axis.

        *Note: branch cuts follow C99 and have provisional status* (see :ref:`branch-cuts`).

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array whose elements each represent the area of a hyperbolic sector. Should have a floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the inverse hyperbolic tangent of each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Special cases**

        For real-valued floating-point operands,

        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If ``x_i`` is less than ``-1``, the result is ``NaN``.
        - If ``x_i`` is greater than ``1``, the result is ``NaN``.
        - If ``x_i`` is ``-1``, the result is ``-infinity``.
        - If ``x_i`` is ``+1``, the result is ``+infinity``.
        - If ``x_i`` is ``+0``, the result is ``+0``.
        - If ``x_i`` is ``-0``, the result is ``-0``.

        For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

        - If ``a`` is ``+0`` and ``b`` is ``+0``, the result is ``+0 + 0j``.
        - If ``a`` is ``+0`` and ``b`` is ``NaN``, the result is ``+0 + NaN j``.
        - If ``a`` is ``1`` and ``b`` is ``+0``, the result is ``+infinity + 0j``.
        - If ``a`` is a positive (i.e., greater than ``0``) finite number and ``b`` is ``+infinity``, the result is ``+0 + πj/2``.
        - If ``a`` is a nonzero finite number and ``b`` is ``NaN``, the result is ``NaN + NaN j``.
        - If ``a`` is ``+infinity`` and ``b`` is a positive (i.e., greater than ``0``) finite number, the result is ``+0 + πj/2``.
        - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``, the result is ``+0 + πj/2``.
        - If ``a`` is ``+infinity`` and ``b`` is ``NaN``, the result is ``+0 + NaN j``.
        - If ``a`` is ``NaN`` and ``b`` is a finite number, the result is ``NaN + NaN j``.
        - If ``a`` is ``NaN`` and ``b`` is ``+infinity``, the result is ``±0 + πj/2`` (sign of the real component is unspecified).
        - If ``a`` is ``NaN`` and ``b`` is ``NaN``, the result is ``NaN + NaN j``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def bitwise_and(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, bool], x2: Union[_NAMESPACE_ARRAY, int, bool], /
    ) -> _NAMESPACE_ARRAY:
        """
        Computes the bitwise AND of the underlying binary representation of each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

        Parameters
        ----------
        x1: Union[array, int, bool]
            first input array. Should have an integer or boolean data type.
        x2: Union[array, int, bool]
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have an integer or boolean data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def bitwise_left_shift(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int], x2: Union[_NAMESPACE_ARRAY, int], /
    ) -> _NAMESPACE_ARRAY:
        """
        Shifts the bits of each element ``x1_i`` of the input array ``x1`` to the left by appending ``x2_i`` (i.e., the respective element in the input array ``x2``) zeros to the right of ``x1_i``.

        Parameters
        ----------
        x1: Union[array, int]
            first input array. Should have an integer data type.
        x2: Union[array, int]
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have an integer data type. Each element must be greater than or equal to ``0``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def bitwise_invert(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        """
        Inverts (flips) each bit for each element ``x_i`` of the input array ``x``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have an integer or boolean data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have the same data type as ``x``.
        """

    @abstractmethod
    def bitwise_or(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, bool], x2: Union[_NAMESPACE_ARRAY, int, bool], /
    ) -> _NAMESPACE_ARRAY:
        """
        Computes the bitwise OR of the underlying binary representation of each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

        Parameters
        ----------
        x1: Union[array, int, bool]
            first input array. Should have an integer or boolean data type.
        x2: Union[array, int, bool]
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have an integer or boolean data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def bitwise_right_shift(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int], x2: Union[_NAMESPACE_ARRAY, int], /
    ) -> _NAMESPACE_ARRAY:
        """
        Shifts the bits of each element ``x1_i`` of the input array ``x1`` to the right according to the respective element ``x2_i`` of the input array ``x2``.

        .. note::
        This operation must be an arithmetic shift (i.e., sign-propagating) and thus equivalent to floor division by a power of two.

        Parameters
        ----------
        x1: Union[array, int]
            first input array. Should have an integer data type.
        x2: Union[array, int]
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have an integer data type. Each element must be greater than or equal to ``0``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def bitwise_xor(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, bool], x2: Union[_NAMESPACE_ARRAY, int, bool], /
    ) -> _NAMESPACE_ARRAY:
        """
        Computes the bitwise XOR of the underlying binary representation of each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

        Parameters
        ----------
        x1: Union[array, int, bool]
            first input array. Should have an integer or boolean data type.
        x2: Union[array, int, bool]
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have an integer or boolean data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def ceil(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        """
        Rounds each element ``x_i`` of the input array ``x`` to the smallest (i.e., closest to ``-infinity``) integer-valued number that is not less than ``x_i``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a real-valued data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the rounded result for each element in ``x``. The returned array must have the same data type as ``x``.

        Notes
        -----

        **Special cases**

        - If ``x_i`` is already integer-valued, the result is ``x_i``.

        For floating-point operands,

        - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
        - If ``x_i`` is ``-infinity``, the result is ``-infinity``.
        - If ``x_i`` is ``+0``, the result is ``+0``.
        - If ``x_i`` is ``-0``, the result is ``-0``.
        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        """

    @abstractmethod
    def clip(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY,
        /,
        min: Optional[Union[int, float, _NAMESPACE_ARRAY]] = None,
        max: Optional[Union[int, float, _NAMESPACE_ARRAY]] = None,
    ) -> _NAMESPACE_ARRAY:
        r"""
        Clamps each element ``x_i`` of the input array ``x`` to the range ``[min, max]``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
        input array. Should have a real-valued data type.
        min: Optional[Union[int, float, array]]
        lower-bound of the range to which to clamp. If ``None``, no lower bound must be applied. Must be compatible with ``x`` and ``max`` (see :ref:`broadcasting`). Should have the same data type as ``x``. Default: ``None``.
        max: Optional[Union[int, float, array]]
        upper-bound of the range to which to clamp. If ``None``, no upper bound must be applied. Must be compatible with ``x`` and ``min`` (see :ref:`broadcasting`). Should have the same data type as ``x``. Default: ``None``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
        an array containing element-wise results. The returned array should have the same data type as ``x``.

        Notes
        -----

        - This function is conceptually equivalent to ``maximum(minimum(x, max), min)`` when ``x``, ``min``, and ``max`` have the same data type.
        - If both ``min`` and ``max`` are ``None``, the elements of the returned array must equal the respective elements in ``x``.
        - If a broadcasted element in ``min`` is greater than a corresponding broadcasted element in ``max``, behavior is unspecified and thus implementation-dependent.
        - For scalar ``min`` and/or ``max``, the scalar values should follow type promotion rules for operations involving arrays and scalar operands (see :ref:`type-promotion`). Hence, if ``x`` and either ``min`` or ``max`` have different data type kinds (e.g., integer versus floating-point), behavior is unspecified and thus implementation-dependent.
        - If ``x`` has an integral data type and a broadcasted element in ``min`` or ``max`` is outside the bounds of the data type of ``x``, behavior is unspecified and thus implementation-dependent.
        - If either ``min`` or ``max`` is an array having a different data type than ``x``, behavior is unspecified and thus implementation-dependent.

        **Special cases**

        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If ``min_i`` is ``NaN``, the result is ``NaN``.
        - If ``max_i`` is ``NaN``, the result is ``NaN``.

        .. versionadded:: 2023.12

        .. versionchanged:: 2024.12
        Added special case behavior when one of the operands is ``NaN``.

        .. versionchanged:: 2024.12
        Clarified that behavior is only defined when ``x``, ``min``, and ``max`` resolve to arrays having the same data type.

        .. versionchanged:: 2024.12
        Clarified that behavior is only defined when elements of ``min`` and ``max`` are inside the bounds of the input array data type.
        """

    @abstractmethod
    def conj(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        """
        Returns the complex conjugate for each element ``x_i`` of the input array ``x``.

        For complex numbers of the form

        .. math::
        a + bj

        the complex conjugate is defined as

        .. math::
        a - bj

        Hence, the returned complex conjugates must be computed by negating the imaginary component of each element ``x_i``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Must have a numeric data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have the same data type as ``x``.

        Notes
        -----

        -   Whether the returned array and the input array share the same underlying memory is unspecified and thus implementation-defined.

        .. versionadded:: 2022.12

        .. versionchanged:: 2024.12
        Added support for real-valued arrays.
        """

    @abstractmethod
    def copysign(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, float], x2: Union[_NAMESPACE_ARRAY, int, float], /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Composes a floating-point value with the magnitude of ``x1_i`` and the sign of ``x2_i`` for each element of the input array ``x1``.

        Parameters
        ----------
        x1: Union[array, int, float]
        input array containing magnitudes. Should have a real-valued floating-point data type.
        x2: Union[array, int, float]
        input array whose sign bits are applied to the magnitudes of ``x1``. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a real-valued floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
        an array containing the element-wise results. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.

        **Special cases**

        For real-valued floating-point operands, let ``|x|`` be the absolute value, and if ``x1_i`` is not ``NaN``,

        - If ``x2_i`` is less than ``0``, the result is ``-|x1_i|``.
        - If ``x2_i`` is ``-0``, the result is ``-|x1_i|``.
        - If ``x2_i`` is ``+0``, the result is ``|x1_i|``.
        - If ``x2_i`` is greater than ``0``, the result is ``|x1_i|``.
        - If ``x2_i`` is ``NaN`` and the sign bit of ``x2_i`` is ``1``, the result is ``-|x1_i|``.
        - If ``x2_i`` is ``NaN`` and the sign bit of ``x2_i`` is ``0``, the result is ``|x1_i|``.

        - If ``x1_i`` is ``NaN`` and ``x2_i`` is less than ``0``, the result is ``NaN`` with a sign bit of ``1``.
        - If ``x1_i`` is ``NaN`` and ``x2_i`` is ``-0``, the result is ``NaN`` with a sign bit of ``1``.
        - If ``x1_i`` is ``NaN`` and ``x2_i`` is ``+0``, the result is ``NaN`` with a sign bit of ``0``.
        - If ``x1_i`` is ``NaN`` and ``x2_i`` is greater than ``0``, the result is ``NaN`` with a sign bit of ``0``.
        - If ``x1_i`` is ``NaN`` and ``x2_i`` is ``NaN`` and the sign bit of ``x2_i`` is ``1``, the result is ``NaN`` with a sign bit of ``1``.
        - If ``x1_i`` is ``NaN`` and ``x2_i`` is ``NaN`` and the sign bit of ``x2_i`` is ``0``, the result is ``NaN`` with a sign bit of ``0``.

        .. versionadded:: 2023.12

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def cos(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Calculates an implementation-dependent approximation to the cosine for each element ``x_i`` of the input array ``x``.

        Each element ``x_i`` is assumed to be expressed in radians.

        .. note::
        The cosine is an entire function on the complex plane and has no branch cuts.

        .. note::
        For complex arguments, the mathematical definition of cosine is

        .. math::
            \begin{align} \operatorname{cos}(x) &= \sum_{n=0}^\infty \frac{(-1)^n}{(2n)!} x^{2n} \\ &= \frac{e^{jx} + e^{-jx}}{2} \\ &= \operatorname{cosh}(jx) \end{align}

        where :math:`\operatorname{cosh}` is the hyperbolic cosine.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array whose elements are each expressed in radians. Should have a floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the cosine of each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Special cases**

        For real-valued floating-point operands,

        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If ``x_i`` is ``+0``, the result is ``1``.
        - If ``x_i`` is ``-0``, the result is ``1``.
        - If ``x_i`` is ``+infinity``, the result is ``NaN``.
        - If ``x_i`` is ``-infinity``, the result is ``NaN``.

        For complex floating-point operands, special cases must be handled as if the operation is implemented as ``cosh(x*1j)``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def cosh(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Calculates an implementation-dependent approximation to the hyperbolic cosine for each element ``x_i`` in the input array ``x``.

        The mathematical definition of the hyperbolic cosine is

        .. math::
        \operatorname{cosh}(x) = \frac{e^x + e^{-x}}{2}

        .. note::
        The hyperbolic cosine is an entire function in the complex plane and has no branch cuts. The function is periodic, with period :math:`2\pi j`, with respect to the imaginary component.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array whose elements each represent a hyperbolic angle. Should have a floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the hyperbolic cosine of each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Special cases**

        .. note::
        For all operands, ``cosh(x)`` must equal ``cosh(-x)``.

        For real-valued floating-point operands,

        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If ``x_i`` is ``+0``, the result is ``1``.
        - If ``x_i`` is ``-0``, the result is ``1``.
        - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
        - If ``x_i`` is ``-infinity``, the result is ``+infinity``.

        For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

        .. note::
        For complex floating-point operands, ``cosh(conj(x))`` must equal ``conj(cosh(x))``.

        - If ``a`` is ``+0`` and ``b`` is ``+0``, the result is ``1 + 0j``.
        - If ``a`` is ``+0`` and ``b`` is ``+infinity``, the result is ``NaN + 0j`` (sign of the imaginary component is unspecified).
        - If ``a`` is ``+0`` and ``b`` is ``NaN``, the result is ``NaN + 0j`` (sign of the imaginary component is unspecified).
        - If ``a`` is a nonzero finite number and ``b`` is ``+infinity``, the result is ``NaN + NaN j``.
        - If ``a`` is a nonzero finite number and ``b`` is ``NaN``, the result is ``NaN + NaN j``.
        - If ``a`` is ``+infinity`` and ``b`` is ``+0``, the result is ``+infinity + 0j``.
        - If ``a`` is ``+infinity`` and ``b`` is a nonzero finite number, the result is ``+infinity * cis(b)``.
        - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``, the result is ``+infinity + NaN j`` (sign of the real component is unspecified).
        - If ``a`` is ``+infinity`` and ``b`` is ``NaN``, the result is ``+infinity + NaN j``.
        - If ``a`` is ``NaN`` and ``b`` is either ``+0`` or ``-0``, the result is ``NaN + 0j`` (sign of the imaginary component is unspecified).
        - If ``a`` is ``NaN`` and ``b`` is a nonzero finite number, the result is ``NaN + NaN j``.
        - If ``a`` is ``NaN`` and ``b`` is ``NaN``, the result is ``NaN + NaN j``.

        where ``cis(v)`` is ``cos(v) + sin(v)*1j``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def divide(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, float, complex], x2: Union[_NAMESPACE_ARRAY, int, float, complex], /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Calculates the division of each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

        Parameters
        ----------
        x1: Union[array, int, float, complex]
            dividend input array. Should have a numeric data type.
        x2: Union[array, int, float, complex]
            divisor input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.

        -   If one or both of the input arrays have integer data types, the result is implementation-dependent, as type promotion between data type "kinds" (e.g., integer versus floating-point) is unspecified.

            Specification-compliant libraries may choose to raise an error or return an array containing the element-wise results. If an array is returned, the array must have a real-valued floating-point data type.

        **Special cases**

        For real-valued floating-point operands,

        - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
        - If ``x1_i`` is either ``+infinity`` or ``-infinity`` and ``x2_i`` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.
        - If ``x1_i`` is either ``+0`` or ``-0`` and ``x2_i`` is either ``+0`` or ``-0``, the result is ``NaN``.
        - If ``x1_i`` is ``+0`` and ``x2_i`` is greater than ``0``, the result is ``+0``.
        - If ``x1_i`` is ``-0`` and ``x2_i`` is greater than ``0``, the result is ``-0``.
        - If ``x1_i`` is ``+0`` and ``x2_i`` is less than ``0``, the result is ``-0``.
        - If ``x1_i`` is ``-0`` and ``x2_i`` is less than ``0``, the result is ``+0``.
        - If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``+0``, the result is ``+infinity``.
        - If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``-0``, the result is ``-infinity``.
        - If ``x1_i`` is less than ``0`` and ``x2_i`` is ``+0``, the result is ``-infinity``.
        - If ``x1_i`` is less than ``0`` and ``x2_i`` is ``-0``, the result is ``+infinity``.
        - If ``x1_i`` is ``+infinity`` and ``x2_i`` is a positive (i.e., greater than ``0``) finite number, the result is ``+infinity``.
        - If ``x1_i`` is ``+infinity`` and ``x2_i`` is a negative (i.e., less than ``0``) finite number, the result is ``-infinity``.
        - If ``x1_i`` is ``-infinity`` and ``x2_i`` is a positive (i.e., greater than ``0``) finite number, the result is ``-infinity``.
        - If ``x1_i`` is ``-infinity`` and ``x2_i`` is a negative (i.e., less than ``0``) finite number, the result is ``+infinity``.
        - If ``x1_i`` is a positive (i.e., greater than ``0``) finite number and ``x2_i`` is ``+infinity``, the result is ``+0``.
        - If ``x1_i`` is a positive (i.e., greater than ``0``) finite number and ``x2_i`` is ``-infinity``, the result is ``-0``.
        - If ``x1_i`` is a negative (i.e., less than ``0``) finite number and ``x2_i`` is ``+infinity``, the result is ``-0``.
        - If ``x1_i`` is a negative (i.e., less than ``0``) finite number and ``x2_i`` is ``-infinity``, the result is ``+0``.
        - If ``x1_i`` and ``x2_i`` have the same mathematical sign and are both nonzero finite numbers, the result has a positive mathematical sign.
        - If ``x1_i`` and ``x2_i`` have different mathematical signs and are both nonzero finite numbers, the result has a negative mathematical sign.
        - In the remaining cases, where neither ``-infinity``, ``+0``, ``-0``, nor ``NaN`` is involved, the quotient must be computed and rounded to the nearest representable value according to IEEE 754-2019 and a supported rounding mode. If the magnitude is too large to represent, the operation overflows and the result is an ``infinity`` of appropriate mathematical sign. If the magnitude is too small to represent, the operation underflows and the result is a zero of appropriate mathematical sign.

        For complex floating-point operands, division is defined according to the following table. For real components ``a`` and ``c`` and imaginary components ``b`` and ``d``,

        +------------+----------------+-----------------+--------------------------+
        |            | c              | dj              | c + dj                   |
        +============+================+=================+==========================+
        | **a**      | a / c          | -(a/d)j         | special rules            |
        +------------+----------------+-----------------+--------------------------+
        | **bj**     | (b/c)j         | b/d             | special rules            |
        +------------+----------------+-----------------+--------------------------+
        | **a + bj** | (a/c) + (b/c)j | b/d - (a/d)j    | special rules            |
        +------------+----------------+-----------------+--------------------------+

        In general, for complex floating-point operands, real-valued floating-point special cases must independently apply to the real and imaginary component operations involving real numbers as described in the above table.

        When ``a``, ``b``, ``c``, or ``d`` are all finite numbers (i.e., a value other than ``NaN``, ``+infinity``, or ``-infinity``), division of complex floating-point operands should be computed as if calculated according to the textbook formula for complex number division

        .. math::
        \frac{a + bj}{c + dj} = \frac{(ac + bd) + (bc - ad)j}{c^2 + d^2}

        When at least one of ``a``, ``b``, ``c``, or ``d`` is ``NaN``, ``+infinity``, or ``-infinity``,

        - If ``a``, ``b``, ``c``, and ``d`` are all ``NaN``, the result is ``NaN + NaN j``.
        - In the remaining cases, the result is implementation dependent.

        .. note::
        For complex floating-point operands, the results of special cases may be implementation dependent depending on how an implementation chooses to model complex numbers and complex infinity (e.g., complex plane versus Riemann sphere). For those implementations following C99 and its one-infinity model, when at least one component is infinite, even if the other component is ``NaN``, the complex value is infinite, and the usual arithmetic rules do not apply to complex-complex division. In the interest of performance, other implementations may want to avoid the complex branching logic necessary to implement the one-infinity model and choose to implement all complex-complex division according to the textbook formula. Accordingly, special case behavior is unlikely to be consistent across implementations.

        .. versionchanged:: 2022.12
        Added complex data type support.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def equal(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, float, complex, bool],
        x2: Union[_NAMESPACE_ARRAY, int, float, complex, bool],
        /,
    ) -> _NAMESPACE_ARRAY:
        r"""
        Computes the truth value of ``x1_i == x2_i`` for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

        Parameters
        ----------
        x1: Union[array, int, float, complex, bool]
            first input array. May have any data type.
        x2: Union[array, int, float, complex, bool]
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). May have any data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a data type of ``bool``.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.

        **Special Cases**

        For real-valued floating-point operands,

        - If ``x1_i`` is ``NaN`` or ``x2_i`` is ``NaN``, the result is ``False``.
        - If ``x1_i`` is ``+infinity`` and ``x2_i`` is ``+infinity``, the result is ``True``.
        - If ``x1_i`` is ``-infinity`` and ``x2_i`` is ``-infinity``, the result is ``True``.
        - If ``x1_i`` is ``-0`` and ``x2_i`` is either ``+0`` or ``-0``, the result is ``True``.
        - If ``x1_i`` is ``+0`` and ``x2_i`` is either ``+0`` or ``-0``, the result is ``True``.
        - If ``x1_i`` is a finite number, ``x2_i`` is a finite number, and ``x1_i`` equals ``x2_i``, the result is ``True``.
        - In the remaining cases, the result is ``False``.

        For complex floating-point operands, let ``a = real(x1_i)``, ``b = imag(x1_i)``, ``c = real(x2_i)``, ``d = imag(x2_i)``, and

        - If ``a``, ``b``, ``c``, or ``d`` is ``NaN``, the result is ``False``.
        - In the remaining cases, the result is the logical AND of the equality comparison between the real values ``a`` and ``c`` (real components) and between the real values ``b`` and ``d`` (imaginary components), as described above for real-valued floating-point operands (i.e., ``a == c AND b == d``).

        .. note::
        For discussion of complex number equality, see :ref:`complex-numbers`.

        .. note::
        Comparison of arrays without a corresponding promotable data type (see :ref:`type-promotion`) is undefined and thus implementation-dependent.

        .. versionchanged:: 2022.12
        Added complex data type support.

        .. versionchanged:: 2024.12
        Cross-kind comparisons are explicitly left unspecified.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def exp(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        """
        Calculates an implementation-dependent approximation to the exponential function for each element ``x_i`` of the input array ``x`` (``e`` raised to the power of ``x_i``, where ``e`` is the base of the natural logarithm).

        .. note::
        For complex floating-point operands, ``exp(conj(x))`` must equal ``conj(exp(x))``.

        .. note::
        The exponential function is an entire function in the complex plane and has no branch cuts.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the evaluated exponential function result for each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Special cases**

        For real-valued floating-point operands,

        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If ``x_i`` is ``+0``, the result is ``1``.
        - If ``x_i`` is ``-0``, the result is ``1``.
        - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
        - If ``x_i`` is ``-infinity``, the result is ``+0``.

        For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

        - If ``a`` is either ``+0`` or ``-0`` and ``b`` is ``+0``, the result is ``1 + 0j``.
        - If ``a`` is a finite number and ``b`` is ``+infinity``, the result is ``NaN + NaN j``.
        - If ``a`` is a finite number and ``b`` is ``NaN``, the result is ``NaN + NaN j``.
        - If ``a`` is ``+infinity`` and ``b`` is ``+0``, the result is ``infinity + 0j``.
        - If ``a`` is ``-infinity`` and ``b`` is a finite number, the result is ``+0 * cis(b)``.
        - If ``a`` is ``+infinity`` and ``b`` is a nonzero finite number, the result is ``+infinity * cis(b)``.
        - If ``a`` is ``-infinity`` and ``b`` is ``+infinity``, the result is ``0 + 0j`` (signs of real and imaginary components are unspecified).
        - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``, the result is ``infinity + NaN j`` (sign of real component is unspecified).
        - If ``a`` is ``-infinity`` and ``b`` is ``NaN``, the result is ``0 + 0j`` (signs of real and imaginary components are unspecified).
        - If ``a`` is ``+infinity`` and ``b`` is ``NaN``, the result is ``infinity + NaN j`` (sign of real component is unspecified).
        - If ``a`` is ``NaN`` and ``b`` is ``+0``, the result is ``NaN + 0j``.
        - If ``a`` is ``NaN`` and ``b`` is not equal to ``0``, the result is ``NaN + NaN j``.
        - If ``a`` is ``NaN`` and ``b`` is ``NaN``, the result is ``NaN + NaN j``.

        where ``cis(v)`` is ``cos(v) + sin(v)*1j``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def expm1(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        """
        Calculates an implementation-dependent approximation to ``exp(x)-1`` for each element ``x_i`` of the input array ``x``.

        .. note::
        The purpose of this function is to calculate ``exp(x)-1.0`` more accurately when `x` is close to zero. Accordingly, conforming implementations should avoid implementing this function as simply ``exp(x)-1.0``. See FDLIBM, or some other IEEE 754-2019 compliant mathematical library, for a potential reference implementation.

        .. note::
        For complex floating-point operands, ``expm1(conj(x))`` must equal ``conj(expm1(x))``.

        .. note::
        The exponential function is an entire function in the complex plane and has no branch cuts.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the evaluated result for each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Special cases**

        For real-valued floating-point operands,

        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If ``x_i`` is ``+0``, the result is ``+0``.
        - If ``x_i`` is ``-0``, the result is ``-0``.
        - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
        - If ``x_i`` is ``-infinity``, the result is ``-1``.

        For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

        - If ``a`` is either ``+0`` or ``-0`` and ``b`` is ``+0``, the result is ``0 + 0j``.
        - If ``a`` is a finite number and ``b`` is ``+infinity``, the result is ``NaN + NaN j``.
        - If ``a`` is a finite number and ``b`` is ``NaN``, the result is ``NaN + NaN j``.
        - If ``a`` is ``+infinity`` and ``b`` is ``+0``, the result is ``+infinity + 0j``.
        - If ``a`` is ``-infinity`` and ``b`` is a finite number, the result is ``+0 * cis(b) - 1.0``.
        - If ``a`` is ``+infinity`` and ``b`` is a nonzero finite number, the result is ``+infinity * cis(b) - 1.0``.
        - If ``a`` is ``-infinity`` and ``b`` is ``+infinity``, the result is ``-1 + 0j`` (sign of imaginary component is unspecified).
        - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``, the result is ``infinity + NaN j`` (sign of real component is unspecified).
        - If ``a`` is ``-infinity`` and ``b`` is ``NaN``, the result is ``-1 + 0j`` (sign of imaginary component is unspecified).
        - If ``a`` is ``+infinity`` and ``b`` is ``NaN``, the result is ``infinity + NaN j`` (sign of real component is unspecified).
        - If ``a`` is ``NaN`` and ``b`` is ``+0``, the result is ``NaN + 0j``.
        - If ``a`` is ``NaN`` and ``b`` is not equal to ``0``, the result is ``NaN + NaN j``.
        - If ``a`` is ``NaN`` and ``b`` is ``NaN``, the result is ``NaN + NaN j``.

        where ``cis(v)`` is ``cos(v) + sin(v)*1j``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def floor(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        """
        Rounds each element ``x_i`` of the input array ``x`` to the greatest (i.e., closest to ``+infinity``) integer-valued number that is not greater than ``x_i``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a real-valued data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the rounded result for each element in ``x``. The returned array must have the same data type as ``x``.

        Notes
        -----

        **Special cases**

        - If ``x_i`` is already integer-valued, the result is ``x_i``.

        For floating-point operands,

        - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
        - If ``x_i`` is ``-infinity``, the result is ``-infinity``.
        - If ``x_i`` is ``+0``, the result is ``+0``.
        - If ``x_i`` is ``-0``, the result is ``-0``.
        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        """

    @abstractmethod
    def floor_divide(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, float], x2: Union[_NAMESPACE_ARRAY, int, float], /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Rounds the result of dividing each element ``x1_i`` of the input array ``x1`` by the respective element ``x2_i`` of the input array ``x2`` to the greatest (i.e., closest to `+infinity`) integer-value number that is not greater than the division result.

        Parameters
        ----------
        x1: Union[array, int, float]
            dividend input array. Should have a real-valued data type.
        x2: Union[array, int, float]
            divisor input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a real-valued data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.
        -   For input arrays which promote to an integer data type, the result of division by zero is unspecified and thus implementation-defined.

        **Special cases**

        .. note::
        Floor division was introduced in Python via `PEP 238 <https://www.python.org/dev/peps/pep-0238/>`_ with the goal to disambiguate "true division" (i.e., computing an approximation to the mathematical operation of division) from "floor division" (i.e., rounding the result of division toward negative infinity). The former was computed when one of the operands was a ``float``, while the latter was computed when both operands were ``int``\s. Overloading the ``/`` operator to support both behaviors led to subtle numerical bugs when integers are possible, but not expected.

        To resolve this ambiguity, ``/`` was designated for true division, and ``//`` was designated for floor division. Semantically, floor division was `defined <https://www.python.org/dev/peps/pep-0238/#semantics-of-floor-division>`_ as equivalent to ``a // b == floor(a/b)``; however, special floating-point cases were left ill-defined.

        Accordingly, floor division is not implemented consistently across array libraries for some of the special cases documented below. Namely, when one of the operands is ``infinity``, libraries may diverge with some choosing to strictly follow ``floor(a/b)`` and others choosing to pair ``//`` with ``%`` according to the relation ``b = a % b + b * (a // b)``. The special cases leading to divergent behavior are documented below.

        This specification prefers floor division to match ``floor(divide(x1, x2))`` in order to avoid surprising and unexpected results; however, array libraries may choose to more strictly follow Python behavior.

        For floating-point operands,

        - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
        - If ``x1_i`` is either ``+infinity`` or ``-infinity`` and ``x2_i`` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.
        - If ``x1_i`` is either ``+0`` or ``-0`` and ``x2_i`` is either ``+0`` or ``-0``, the result is ``NaN``.
        - If ``x1_i`` is ``+0`` and ``x2_i`` is greater than ``0``, the result is ``+0``.
        - If ``x1_i`` is ``-0`` and ``x2_i`` is greater than ``0``, the result is ``-0``.
        - If ``x1_i`` is ``+0`` and ``x2_i`` is less than ``0``, the result is ``-0``.
        - If ``x1_i`` is ``-0`` and ``x2_i`` is less than ``0``, the result is ``+0``.
        - If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``+0``, the result is ``+infinity``.
        - If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``-0``, the result is ``-infinity``.
        - If ``x1_i`` is less than ``0`` and ``x2_i`` is ``+0``, the result is ``-infinity``.
        - If ``x1_i`` is less than ``0`` and ``x2_i`` is ``-0``, the result is ``+infinity``.
        - If ``x1_i`` is ``+infinity`` and ``x2_i`` is a positive (i.e., greater than ``0``) finite number, the result is ``+infinity``. (**note**: libraries may return ``NaN`` to match Python behavior.)
        - If ``x1_i`` is ``+infinity`` and ``x2_i`` is a negative (i.e., less than ``0``) finite number, the result is ``-infinity``. (**note**: libraries may return ``NaN`` to match Python behavior.)
        - If ``x1_i`` is ``-infinity`` and ``x2_i`` is a positive (i.e., greater than ``0``) finite number, the result is ``-infinity``. (**note**: libraries may return ``NaN`` to match Python behavior.)
        - If ``x1_i`` is ``-infinity`` and ``x2_i`` is a negative (i.e., less than ``0``) finite number, the result is ``+infinity``. (**note**: libraries may return ``NaN`` to match Python behavior.)
        - If ``x1_i`` is a positive (i.e., greater than ``0``) finite number and ``x2_i`` is ``+infinity``, the result is ``+0``.
        - If ``x1_i`` is a positive (i.e., greater than ``0``) finite number and ``x2_i`` is ``-infinity``, the result is ``-0``. (**note**: libraries may return ``-1.0`` to match Python behavior.)
        - If ``x1_i`` is a negative (i.e., less than ``0``) finite number and ``x2_i`` is ``+infinity``, the result is ``-0``. (**note**: libraries may return ``-1.0`` to match Python behavior.)
        - If ``x1_i`` is a negative (i.e., less than ``0``) finite number and ``x2_i`` is ``-infinity``, the result is ``+0``.
        - If ``x1_i`` and ``x2_i`` have the same mathematical sign and are both nonzero finite numbers, the result has a positive mathematical sign.
        - If ``x1_i`` and ``x2_i`` have different mathematical signs and are both nonzero finite numbers, the result has a negative mathematical sign.
        - In the remaining cases, where neither ``-infinity``, ``+0``, ``-0``, nor ``NaN`` is involved, the quotient must be computed and rounded to the greatest (i.e., closest to `+infinity`) representable integer-value number that is not greater than the division result. If the magnitude is too large to represent, the operation overflows and the result is an ``infinity`` of appropriate mathematical sign. If the magnitude is too small to represent, the operation underflows and the result is a zero of appropriate mathematical sign.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def greater(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, float], x2: Union[_NAMESPACE_ARRAY, int, float], /
    ) -> _NAMESPACE_ARRAY:
        """
        Computes the truth value of ``x1_i > x2_i`` for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

        Parameters
        ----------
        x1: Union[array, int, float]
            first input array. Should have a real-valued data type.
        x2: Union[array, int, float]
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a real-valued data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a data type of ``bool``.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.
        -   Comparison of arrays without a corresponding promotable data type (see :ref:`type-promotion`) is undefined and thus implementation-dependent.
        -   For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-dependent (see :ref:`complex-number-ordering`).

        .. versionchanged:: 2024.12
        Cross-kind comparisons are explicitly left unspecified.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def greater_equal(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, float], x2: Union[_NAMESPACE_ARRAY, int, float], /
    ) -> _NAMESPACE_ARRAY:
        """
        Computes the truth value of ``x1_i >= x2_i`` for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

        Parameters
        ----------
        x1: Union[array, int, float]
            first input array. Should have a real-valued data type.
        x2: Union[array, int, float]
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a real-valued data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a data type of ``bool``.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.
        -   Comparison of arrays without a corresponding promotable data type (see :ref:`type-promotion`) is undefined and thus implementation-dependent.
        -   For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-dependent (see :ref:`complex-number-ordering`).

        .. versionchanged:: 2024.12
        Cross-kind comparisons are explicitly left unspecified.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def hypot(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, float], x2: Union[_NAMESPACE_ARRAY, int, float], /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Computes the square root of the sum of squares for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

        .. note::
        The value computed by this function may be interpreted as the length of the hypotenuse of a right-angled triangle with sides of length ``x1_i`` and ``x2_i``, the distance of a point ``(x1_i, x2_i)`` from the origin ``(0, 0)``, or the magnitude of a complex number ``x1_i + x2_i * 1j``.

        Parameters
        ----------
        x1: Union[array, int, float]
        first input array. Should have a real-valued floating-point data type.
        x2: Union[array, int, float]
        second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a real-valued floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
        an array containing the element-wise results. The returned array must have a real-valued floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.
        -   The purpose of this function is to avoid underflow and overflow during intermediate stages of computation. Accordingly, conforming implementations should not use naive implementations.

        **Special Cases**

        For real-valued floating-point operands,

        - If ``x1_i`` is ``+infinity`` or ``-infinity`` and ``x2_i`` is any value, including ``NaN``, the result is ``+infinity``.
        - If ``x2_i`` is ``+infinity`` or ``-infinity`` and ``x1_i`` is any value, including ``NaN``, the result is ``+infinity``.
        - If ``x1_i`` is either ``+0`` or ``-0``, the result is equivalent to ``abs(x2_i)``.
        - If ``x2_i`` is either ``+0`` or ``-0``, the result is equivalent to ``abs(x1_i)``.
        - If ``x1_i`` is a finite number or ``NaN`` and ``x2_i`` is ``NaN``, the result is ``NaN``.
        - If ``x2_i`` is a finite number or ``NaN`` and ``x1_i`` is ``NaN``, the result is ``NaN``.
        - Underflow may only occur when both arguments are subnormal and the correct result is also subnormal.

        For real-valued floating-point operands, ``hypot(x1, x2)`` must equal ``hypot(x2, x1)``, ``hypot(x1, -x2)``, ``hypot(-x1, x2)``, and ``hypot(-x1, -x2)``.

        .. note::
        IEEE 754-2019 requires support for subnormal (a.k.a., denormal) numbers, which are useful for supporting gradual underflow. However, hardware support for subnormal numbers is not universal, and many platforms (e.g., accelerators) and compilers support toggling denormals-are-zero (DAZ) and/or flush-to-zero (FTZ) behavior to increase performance and to guard against timing attacks.

        Accordingly, conforming implementations may vary in their support for subnormal numbers.

        .. versionadded:: 2023.12

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def imag(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        """
        Returns the imaginary component of a complex number for each element ``x_i`` of the input array ``x``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a complex floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a floating-point data type with the same floating-point precision as ``x`` (e.g., if ``x`` is ``complex64``, the returned array must have the floating-point data type ``float32``).

        Notes
        -----

        .. versionadded:: 2022.12
        """

    @abstractmethod
    def isfinite(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        """
        Tests each element ``x_i`` of the input array ``x`` to determine if finite.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a numeric data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing test results. The returned array must have a data type of ``bool``.

        Notes
        -----

        **Special Cases**

        For real-valued floating-point operands,

        - If ``x_i`` is either ``+infinity`` or ``-infinity``, the result is ``False``.
        - If ``x_i`` is ``NaN``, the result is ``False``.
        - If ``x_i`` is a finite number, the result is ``True``.

        For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

        - If ``a`` is ``NaN`` or ``b`` is ``NaN``, the result is ``False``.
        - If ``a`` is either ``+infinity`` or ``-infinity`` and ``b`` is any value, the result is ``False``.
        - If ``a`` is any value and ``b`` is either ``+infinity`` or ``-infinity``, the result is ``False``.
        - If ``a`` is a finite number and ``b`` is a finite number, the result is ``True``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def isinf(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        """
        Tests each element ``x_i`` of the input array ``x`` to determine if equal to positive or negative infinity.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a numeric data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing test results. The returned array must have a data type of ``bool``.

        Notes
        -----

        **Special Cases**

        For real-valued floating-point operands,

        - If ``x_i`` is either ``+infinity`` or ``-infinity``, the result is ``True``.
        - In the remaining cases, the result is ``False``.

        For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

        - If ``a`` is either ``+infinity`` or ``-infinity`` and ``b`` is any value (including ``NaN``), the result is ``True``.
        - If ``a`` is either a finite number or ``NaN`` and ``b`` is either ``+infinity`` or ``-infinity``, the result is ``True``.
        - In the remaining cases, the result is ``False``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def isnan(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        """
        Tests each element ``x_i`` of the input array ``x`` to determine whether the element is ``NaN``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a numeric data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing test results. The returned array should have a data type of ``bool``.

        Notes
        -----

        **Special Cases**

        For real-valued floating-point operands,

        - If ``x_i`` is ``NaN``, the result is ``True``.
        - In the remaining cases, the result is ``False``.

        For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

        - If ``a`` or ``b`` is ``NaN``, the result is ``True``.
        - In the remaining cases, the result is ``False``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def less(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, float], x2: Union[_NAMESPACE_ARRAY, int, float], /
    ) -> _NAMESPACE_ARRAY:
        """
        Computes the truth value of ``x1_i < x2_i`` for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

        Parameters
        ----------
        x1: Union[array, int, float]
            first input array. Should have a real-valued data type.
        x2: Union[array, int, float]
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a real-valued data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a data type of ``bool``.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.
        -   Comparison of arrays without a corresponding promotable data type (see :ref:`type-promotion`) is undefined and thus implementation-dependent.
        -   For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-dependent (see :ref:`complex-number-ordering`).

        .. versionchanged:: 2024.12
        Cross-kind comparisons are explicitly left unspecified.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def less_equal(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, float], x2: Union[_NAMESPACE_ARRAY, int, float], /
    ) -> _NAMESPACE_ARRAY:
        """
        Computes the truth value of ``x1_i <= x2_i`` for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

        Parameters
        ----------
        x1: Union[array, int, float]
            first input array. Should have a real-valued data type.
        x2: Union[array, int, float]
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a real-valued data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a data type of ``bool``.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.
        -   Comparison of arrays without a corresponding promotable data type (see :ref:`type-promotion`) is undefined and thus implementation-dependent.
        -   For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-dependent (see :ref:`complex-number-ordering`).

        .. versionchanged:: 2024.12
        Cross-kind comparisons are explicitly left unspecified.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def log(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Calculates an implementation-dependent approximation to the natural (base ``e``) logarithm for each element ``x_i`` of the input array ``x``.

        .. note::
        The natural logarithm of a complex number :math:`z` with polar coordinates :math:`(r,\theta)` equals :math:`\ln r + (\theta + 2n\pi)j` with principal value :math:`\ln r + \theta j`.

        .. note::
        For complex floating-point operands, ``log(conj(x))`` must equal ``conj(log(x))``.

        .. note::
        By convention, the branch cut of the natural logarithm is the negative real axis :math:`(-\infty, 0)`.

        The natural logarithm is a continuous function from above the branch cut, taking into account the sign of the imaginary component.

        Accordingly, for complex arguments, the function returns the natural logarithm in the range of a strip in the interval :math:`[-\pi j, +\pi j]` along the imaginary axis and mathematically unbounded along the real axis.

        *Note: branch cuts follow C99 and have provisional status* (see :ref:`branch-cuts`).

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the evaluated natural logarithm for each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Special cases**

        For real-valued floating-point operands,

        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If ``x_i`` is less than ``0``, the result is ``NaN``.
        - If ``x_i`` is either ``+0`` or ``-0``, the result is ``-infinity``.
        - If ``x_i`` is ``1``, the result is ``+0``.
        - If ``x_i`` is ``+infinity``, the result is ``+infinity``.

        For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

        - If ``a`` is ``-0`` and ``b`` is ``+0``, the result is ``-infinity + πj``.
        - If ``a`` is ``+0`` and ``b`` is ``+0``, the result is ``-infinity + 0j``.
        - If ``a`` is a finite number and ``b`` is ``+infinity``, the result is ``+infinity + πj/2``.
        - If ``a`` is a finite number and ``b`` is ``NaN``, the result is ``NaN + NaN j``.
        - If ``a`` is ``-infinity`` and ``b`` is a positive (i.e., greater than ``0``) finite number, the result is ``+infinity + πj``.
        - If ``a`` is ``+infinity`` and ``b`` is a positive (i.e., greater than ``0``) finite number, the result is ``+infinity + 0j``.
        - If ``a`` is ``-infinity`` and ``b`` is ``+infinity``, the result is ``+infinity + 3πj/4``.
        - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``, the result is ``+infinity + πj/4``.
        - If ``a`` is either ``+infinity`` or ``-infinity`` and ``b`` is ``NaN``, the result is ``+infinity + NaN j``.
        - If ``a`` is ``NaN`` and ``b`` is a finite number, the result is ``NaN + NaN j``.
        - If ``a`` is ``NaN`` and ``b`` is ``+infinity``, the result is ``+infinity + NaN j``.
        - If ``a`` is ``NaN`` and ``b`` is ``NaN``, the result is ``NaN + NaN j``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def log1p(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Calculates an implementation-dependent approximation to ``log(1+x)``, where ``log`` refers to the natural (base ``e``) logarithm, for each element ``x_i`` of the input array ``x``.

        .. note::
        The purpose of this function is to calculate ``log(1+x)`` more accurately when `x` is close to zero. Accordingly, conforming implementations should avoid implementing this function as simply ``log(1+x)``. See FDLIBM, or some other IEEE 754-2019 compliant mathematical library, for a potential reference implementation.

        .. note::
        For complex floating-point operands, ``log1p(conj(x))`` must equal ``conj(log1p(x))``.

        .. note::
        By convention, the branch cut of the natural logarithm is the negative real axis :math:`(-\infty, 0)`.

        The natural logarithm is a continuous function from above the branch cut, taking into account the sign of the imaginary component.

        Accordingly, for complex arguments, the function returns the natural logarithm in the range of a strip in the interval :math:`[-\pi j, +\pi j]` along the imaginary axis and mathematically unbounded along the real axis.

        *Note: branch cuts follow C99 and have provisional status* (see :ref:`branch-cuts`).

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the evaluated result for each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Special cases**

        For real-valued floating-point operands,

        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If ``x_i`` is less than ``-1``, the result is ``NaN``.
        - If ``x_i`` is ``-1``, the result is ``-infinity``.
        - If ``x_i`` is ``-0``, the result is ``-0``.
        - If ``x_i`` is ``+0``, the result is ``+0``.
        - If ``x_i`` is ``+infinity``, the result is ``+infinity``.

        For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

        - If ``a`` is ``-1`` and ``b`` is ``+0``, the result is ``-infinity + 0j``.
        - If ``a`` is a finite number and ``b`` is ``+infinity``, the result is ``+infinity + πj/2``.
        - If ``a`` is a finite number and ``b`` is ``NaN``, the result is ``NaN + NaN j``.
        - If ``a`` is ``-infinity`` and ``b`` is a positive (i.e., greater than ``0``) finite number, the result is ``+infinity + πj``.
        - If ``a`` is ``+infinity`` and ``b`` is a positive (i.e., greater than ``0``) finite number, the result is ``+infinity + 0j``.
        - If ``a`` is ``-infinity`` and ``b`` is ``+infinity``, the result is ``+infinity + 3πj/4``.
        - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``, the result is ``+infinity + πj/4``.
        - If ``a`` is either ``+infinity`` or ``-infinity`` and ``b`` is ``NaN``, the result is ``+infinity + NaN j``.
        - If ``a`` is ``NaN`` and ``b`` is a finite number, the result is ``NaN + NaN j``.
        - If ``a`` is ``NaN`` and ``b`` is ``+infinity``, the result is ``+infinity + NaN j``.
        - If ``a`` is ``NaN`` and ``b`` is ``NaN``, the result is ``NaN + NaN j``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def log2(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Calculates an implementation-dependent approximation to the base ``2`` logarithm for each element ``x_i`` of the input array ``x``.

        .. note::
        For complex floating-point operands, ``log2(conj(x))`` must equal ``conj(log2(x))``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the evaluated base ``2`` logarithm for each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Special cases**

        For real-valued floating-point operands,

        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If ``x_i`` is less than ``0``, the result is ``NaN``.
        - If ``x_i`` is either ``+0`` or ``-0``, the result is ``-infinity``.
        - If ``x_i`` is ``1``, the result is ``+0``.
        - If ``x_i`` is ``+infinity``, the result is ``+infinity``.

        For complex floating-point operands, special cases must be handled as if the operation is implemented using the standard change of base formula

        .. math::
        \log_{2} x = \frac{\log_{e} x}{\log_{e} 2}

        where :math:`\log_{e}` is the natural logarithm, as implemented by :func:`~array_api.log`.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def log10(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Calculates an implementation-dependent approximation to the base ``10`` logarithm for each element ``x_i`` of the input array ``x``.

        .. note::
        For complex floating-point operands, ``log10(conj(x))`` must equal ``conj(log10(x))``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the evaluated base ``10`` logarithm for each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Special cases**

        For real-valued floating-point operands,

        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If ``x_i`` is less than ``0``, the result is ``NaN``.
        - If ``x_i`` is either ``+0`` or ``-0``, the result is ``-infinity``.
        - If ``x_i`` is ``1``, the result is ``+0``.
        - If ``x_i`` is ``+infinity``, the result is ``+infinity``.

        For complex floating-point operands, special cases must be handled as if the operation is implemented using the standard change of base formula

        .. math::
        \log_{10} x = \frac{\log_{e} x}{\log_{e} 10}

        where :math:`\log_{e}` is the natural logarithm, as implemented by :func:`~array_api.log`.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def logaddexp(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, float], x2: Union[_NAMESPACE_ARRAY, int, float], /
    ) -> _NAMESPACE_ARRAY:
        """
        Calculates the logarithm of the sum of exponentiations ``log(exp(x1) + exp(x2))`` for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

        Parameters
        ----------
        x1: Union[array, int, float]
            first input array. Should have a real-valued floating-point data type.
        x2: Union[array, int, float]
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a real-valued floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a real-valued floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.

        **Special cases**

        For floating-point operands,

        - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
        - If ``x1_i`` is ``+infinity`` and ``x2_i`` is not ``NaN``, the result is ``+infinity``.
        - If ``x1_i`` is not ``NaN`` and ``x2_i`` is ``+infinity``, the result is ``+infinity``.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def logical_and(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, bool], x2: Union[_NAMESPACE_ARRAY, bool], /
    ) -> _NAMESPACE_ARRAY:
        """
        Computes the logical AND for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

        .. note::
        While this specification recommends that this function only accept input arrays having a boolean data type, specification-compliant array libraries may choose to accept input arrays having real-valued data types. If non-boolean data types are supported, zeros must be considered the equivalent of ``False``, while non-zeros must be considered the equivalent of ``True``.

        Parameters
        ----------
        x1: Union[array, bool]
            first input array. Should have a boolean data type.
        x2: Union[array, bool]
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a boolean data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a data type of `bool`.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def logical_not(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        """
        Computes the logical NOT for each element ``x_i`` of the input array ``x``.

        .. note::
        While this specification recommends that this function only accept input arrays having a boolean data type, specification-compliant array libraries may choose to accept input arrays having real-valued data types. If non-boolean data types are supported, zeros must be considered the equivalent of ``False``, while non-zeros must be considered the equivalent of ``True``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a boolean data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a data type of ``bool``.
        """

    @abstractmethod
    def logical_or(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, bool], x2: Union[_NAMESPACE_ARRAY, bool], /
    ) -> _NAMESPACE_ARRAY:
        """
        Computes the logical OR for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

        .. note::
        While this specification recommends that this function only accept input arrays having a boolean data type, specification-compliant array libraries may choose to accept input arrays having real-valued data types. If non-boolean data types are supported, zeros must be considered the equivalent of ``False``, while non-zeros must be considered the equivalent of ``True``.

        Parameters
        ----------
        x1: Union[array, bool]
            first input array. Should have a boolean data type.
        x2: Union[array, bool]
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a boolean data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a data type of ``bool``.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def logical_xor(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, bool], x2: Union[_NAMESPACE_ARRAY, bool], /
    ) -> _NAMESPACE_ARRAY:
        """
        Computes the logical XOR for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

        .. note::
        While this specification recommends that this function only accept input arrays having a boolean data type, specification-compliant array libraries may choose to accept input arrays having real-valued data types. If non-boolean data types are supported, zeros must be considered the equivalent of ``False``, while non-zeros must be considered the equivalent of ``True``.

        Parameters
        ----------
        x1: Union[array, bool]
            first input array. Should have a boolean data type.
        x2: Union[array, bool]
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a boolean data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a data type of ``bool``.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def maximum(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, float], x2: Union[_NAMESPACE_ARRAY, int, float], /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Computes the maximum value for each element ``x1_i`` of the input array ``x1`` relative to the respective element ``x2_i`` of the input array ``x2``.

        Parameters
        ----------
        x1: Union[array, int, float]
        first input array. Should have a real-valued data type.
        x2: Union[array, int, float]
        second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a real-valued data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
        an array containing the element-wise maximum values. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.
        -   The order of signed zeros is unspecified and thus implementation-defined. When choosing between ``-0`` or ``+0`` as a maximum value, specification-compliant libraries may choose to return either value.
        -   For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-defined (see :ref:`complex-number-ordering`).

        **Special Cases**

        For floating-point operands,

        -   If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.

        .. versionadded:: 2023.12

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def minimum(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, float], x2: Union[_NAMESPACE_ARRAY, int, float], /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Computes the minimum value for each element ``x1_i`` of the input array ``x1`` relative to the respective element ``x2_i`` of the input array ``x2``.

        Parameters
        ----------
        x1: Union[array, int, float]
        first input array. Should have a real-valued data type.
        x2: Union[array, int, float]
        second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a real-valued data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
        an array containing the element-wise minimum values. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.
        -   The order of signed zeros is unspecified and thus implementation-defined. When choosing between ``-0`` or ``+0`` as a minimum value, specification-compliant libraries may choose to return either value.
        -   For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-defined (see :ref:`complex-number-ordering`).

        **Special Cases**

        For floating-point operands,

        -   If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.

        .. versionadded:: 2023.12

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def multiply(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, float, complex], x2: Union[_NAMESPACE_ARRAY, int, float, complex], /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Calculates the product for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

        .. note::
        Floating-point multiplication is not always associative due to finite precision.

        Parameters
        ----------
        x1: Union[array, int, float, complex]
            first input array. Should have a numeric data type.
        x2: Union[array, int, float, complex]
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise products. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.

        **Special cases**

        For real-valued floating-point operands,

        - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
        - If ``x1_i`` is either ``+infinity`` or ``-infinity`` and ``x2_i`` is either ``+0`` or ``-0``, the result is ``NaN``.
        - If ``x1_i`` is either ``+0`` or ``-0`` and ``x2_i`` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.
        - If ``x1_i`` and ``x2_i`` have the same mathematical sign, the result has a positive mathematical sign, unless the result is ``NaN``. If the result is ``NaN``, the "sign" of ``NaN`` is implementation-defined.
        - If ``x1_i`` and ``x2_i`` have different mathematical signs, the result has a negative mathematical sign, unless the result is ``NaN``. If the result is ``NaN``, the "sign" of ``NaN`` is implementation-defined.
        - If ``x1_i`` is either ``+infinity`` or ``-infinity`` and ``x2_i`` is either ``+infinity`` or ``-infinity``, the result is a signed infinity with the mathematical sign determined by the rule already stated above.
        - If ``x1_i`` is either ``+infinity`` or ``-infinity`` and ``x2_i`` is a nonzero finite number, the result is a signed infinity with the mathematical sign determined by the rule already stated above.
        - If ``x1_i`` is a nonzero finite number and ``x2_i`` is either ``+infinity`` or ``-infinity``, the result is a signed infinity with the mathematical sign determined by the rule already stated above.
        - In the remaining cases, where neither ``infinity`` nor ``NaN`` is involved, the product must be computed and rounded to the nearest representable value according to IEEE 754-2019 and a supported rounding mode. If the magnitude is too large to represent, the result is an `infinity` of appropriate mathematical sign. If the magnitude is too small to represent, the result is a zero of appropriate mathematical sign.

        For complex floating-point operands, multiplication is defined according to the following table. For real components ``a`` and ``c`` and imaginary components ``b`` and ``d``,

        +------------+----------------+-----------------+--------------------------+
        |            | c              | dj              | c + dj                   |
        +============+================+=================+==========================+
        | **a**      | a * c          | (a*d)j          | (a*c) + (a*d)j           |
        +------------+----------------+-----------------+--------------------------+
        | **bj**     | (b*c)j         | -(b*d)          | -(b*d) + (b*c)j          |
        +------------+----------------+-----------------+--------------------------+
        | **a + bj** | (a*c) + (b*c)j | -(b*d) + (a*d)j | special rules            |
        +------------+----------------+-----------------+--------------------------+

        In general, for complex floating-point operands, real-valued floating-point special cases must independently apply to the real and imaginary component operations involving real numbers as described in the above table.

        When ``a``, ``b``, ``c``, or ``d`` are all finite numbers (i.e., a value other than ``NaN``, ``+infinity``, or ``-infinity``), multiplication of complex floating-point operands should be computed as if calculated according to the textbook formula for complex number multiplication

        .. math::
        (a + bj) \cdot (c + dj) = (ac - bd) + (bc + ad)j

        When at least one of ``a``, ``b``, ``c``, or ``d`` is ``NaN``, ``+infinity``, or ``-infinity``,

        - If ``a``, ``b``, ``c``, and ``d`` are all ``NaN``, the result is ``NaN + NaN j``.
        - In the remaining cases, the result is implementation dependent.

        .. note::
        For complex floating-point operands, the results of special cases may be implementation dependent depending on how an implementation chooses to model complex numbers and complex infinity (e.g., complex plane versus Riemann sphere). For those implementations following C99 and its one-infinity model, when at least one component is infinite, even if the other component is ``NaN``, the complex value is infinite, and the usual arithmetic rules do not apply to complex-complex multiplication. In the interest of performance, other implementations may want to avoid the complex branching logic necessary to implement the one-infinity model and choose to implement all complex-complex multiplication according to the textbook formula. Accordingly, special case behavior is unlikely to be consistent across implementations.

        .. versionchanged:: 2022.12
        Added complex data type support.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def negative(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        """
        Computes the numerical negative of each element ``x_i`` (i.e., ``y_i = -x_i``) of the input array ``x``.

        .. note::
        For signed integer data types, the numerical negative of the minimum representable integer is implementation-dependent.

        .. note::
        If ``x`` has a complex floating-point data type, both the real and imaginary components for each ``x_i`` must be negated (a result which follows from the rules of complex number multiplication).

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a numeric data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the evaluated result for each element in ``x``. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def nextafter(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, float], x2: Union[_NAMESPACE_ARRAY, int, float], /
    ) -> _NAMESPACE_ARRAY:
        """
        Returns the next representable floating-point value for each element ``x1_i`` of the input array ``x1`` in the direction of the respective element ``x2_i`` of the input array ``x2``.

        Parameters
        ----------
        x1: Union[array, int, float]
            first input array. Should have a real-valued floating-point data type.
        x2: Union[array, int, float]
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have the same data type as ``x1``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have the same data type as ``x1``.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.

        **Special cases**

        For real-valued floating-point operands,

        - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
        - If ``x1_i`` is ``-0`` and ``x2_i`` is ``+0``, the result is ``+0``.
        - If ``x1_i`` is ``+0`` and ``x2_i`` is ``-0``, the result is ``-0``.

        .. versionadded:: 2024.12
        """

    @abstractmethod
    def not_equal(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, float, complex, bool],
        x2: Union[_NAMESPACE_ARRAY, int, float, complex, bool],
        /,
    ) -> _NAMESPACE_ARRAY:
        """
        Computes the truth value of ``x1_i != x2_i`` for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

        Parameters
        ----------
        x1: Union[array, int, float, complex, bool]
            first input array. May have any data type.
        x2: Union[array, int, float, complex, bool]
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`).

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a data type of ``bool``.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.

        **Special Cases**

        For real-valued floating-point operands,

        - If ``x1_i`` is ``NaN`` or ``x2_i`` is ``NaN``, the result is ``True``.
        - If ``x1_i`` is ``+infinity`` and ``x2_i`` is ``-infinity``, the result is ``True``.
        - If ``x1_i`` is ``-infinity`` and ``x2_i`` is ``+infinity``, the result is ``True``.
        - If ``x1_i`` is a finite number, ``x2_i`` is a finite number, and ``x1_i`` does not equal ``x2_i``, the result is ``True``.
        - In the remaining cases, the result is ``False``.

        For complex floating-point operands, let ``a = real(x1_i)``, ``b = imag(x1_i)``, ``c = real(x2_i)``, ``d = imag(x2_i)``, and

        - If ``a``, ``b``, ``c``, or ``d`` is ``NaN``, the result is ``True``.
        - In the remaining cases, the result is the logical OR of the equality comparison between the real values ``a`` and ``c`` (real components) and between the real values ``b`` and ``d`` (imaginary components), as described above for real-valued floating-point operands (i.e., ``a != c OR b != d``).

        .. note::
        For discussion of complex number equality, see :ref:`complex-numbers`.

        .. note::
        Comparison of arrays without a corresponding promotable data type (see :ref:`type-promotion`) is undefined and thus implementation-dependent.

        .. versionchanged:: 2022.12
        Added complex data type support.

        .. versionchanged:: 2024.12
        Cross-kind comparisons are explicitly left unspecified.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def positive(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        """
        Computes the numerical positive of each element ``x_i`` (i.e., ``y_i = +x_i``) of the input array ``x``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a numeric data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the evaluated result for each element in ``x``. The returned array must have the same data type as ``x``.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def pow(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, float, complex], x2: Union[_NAMESPACE_ARRAY, int, float, complex], /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Calculates an implementation-dependent approximation of exponentiation by raising each element ``x1_i`` (the base) of the input array ``x1`` to the power of ``x2_i`` (the exponent), where ``x2_i`` is the corresponding element of the input array ``x2``.

        Parameters
        ----------
        x1: Union[array, int, float, complex]
            first input array whose elements correspond to the exponentiation base. Should have a numeric data type.
        x2: Union[array, int, float, complex]
            second input array whose elements correspond to the exponentiation exponent. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.

        -   If both ``x1`` and ``x2`` have integer data types, the result of ``pow`` when ``x2_i`` is negative (i.e., less than zero) is unspecified and thus implementation-dependent.

        -   If ``x1`` has an integer data type and ``x2`` has a floating-point data type, behavior is implementation-dependent (type promotion between data type "kinds" (integer versus floating-point) is unspecified).

        -   By convention, the branch cut of the natural logarithm is the negative real axis :math:`(-\infty, 0)`.

            The natural logarithm is a continuous function from above the branch cut, taking into account the sign of the imaginary component. As special cases involving complex floating-point operands should be handled according to ``exp(x2*log(x1))``, exponentiation has the same branch cut for ``x1`` as the natural logarithm (see :func:`~array_api.log`).

            *Note: branch cuts follow C99 and have provisional status* (see :ref:`branch-cuts`).

        **Special cases**

        For real-valued floating-point operands,

        - If ``x1_i`` is not equal to ``1`` and ``x2_i`` is ``NaN``, the result is ``NaN``.
        - If ``x2_i`` is ``+0``, the result is ``1``, even if ``x1_i`` is ``NaN``.
        - If ``x2_i`` is ``-0``, the result is ``1``, even if ``x1_i`` is ``NaN``.
        - If ``x1_i`` is ``NaN`` and ``x2_i`` is not equal to ``0``, the result is ``NaN``.
        - If ``abs(x1_i)`` is greater than ``1`` and ``x2_i`` is ``+infinity``, the result is ``+infinity``.
        - If ``abs(x1_i)`` is greater than ``1`` and ``x2_i`` is ``-infinity``, the result is ``+0``.
        - If ``abs(x1_i)`` is ``1`` and ``x2_i`` is ``+infinity``, the result is ``1``.
        - If ``abs(x1_i)`` is ``1`` and ``x2_i`` is ``-infinity``, the result is ``1``.
        - If ``x1_i`` is ``1`` and ``x2_i`` is not ``NaN``, the result is ``1``.
        - If ``abs(x1_i)`` is less than ``1`` and ``x2_i`` is ``+infinity``, the result is ``+0``.
        - If ``abs(x1_i)`` is less than ``1`` and ``x2_i`` is ``-infinity``, the result is ``+infinity``.
        - If ``x1_i`` is ``+infinity`` and ``x2_i`` is greater than ``0``, the result is ``+infinity``.
        - If ``x1_i`` is ``+infinity`` and ``x2_i`` is less than ``0``, the result is ``+0``.
        - If ``x1_i`` is ``-infinity``, ``x2_i`` is greater than ``0``, and ``x2_i`` is an odd integer value, the result is ``-infinity``.
        - If ``x1_i`` is ``-infinity``, ``x2_i`` is greater than ``0``, and ``x2_i`` is not an odd integer value, the result is ``+infinity``.
        - If ``x1_i`` is ``-infinity``, ``x2_i`` is less than ``0``, and ``x2_i`` is an odd integer value, the result is ``-0``.
        - If ``x1_i`` is ``-infinity``, ``x2_i`` is less than ``0``, and ``x2_i`` is not an odd integer value, the result is ``+0``.
        - If ``x1_i`` is ``+0`` and ``x2_i`` is greater than ``0``, the result is ``+0``.
        - If ``x1_i`` is ``+0`` and ``x2_i`` is less than ``0``, the result is ``+infinity``.
        - If ``x1_i`` is ``-0``, ``x2_i`` is greater than ``0``, and ``x2_i`` is an odd integer value, the result is ``-0``.
        - If ``x1_i`` is ``-0``, ``x2_i`` is greater than ``0``, and ``x2_i`` is not an odd integer value, the result is ``+0``.
        - If ``x1_i`` is ``-0``, ``x2_i`` is less than ``0``, and ``x2_i`` is an odd integer value, the result is ``-infinity``.
        - If ``x1_i`` is ``-0``, ``x2_i`` is less than ``0``, and ``x2_i`` is not an odd integer value, the result is ``+infinity``.
        - If ``x1_i`` is less than ``0``, ``x1_i`` is a finite number, ``x2_i`` is a finite number, and ``x2_i`` is not an integer value, the result is ``NaN``.

        For complex floating-point operands, special cases should be handled as if the operation is implemented as ``exp(x2*log(x1))``.

        .. note::
        Conforming implementations are allowed to treat special cases involving complex floating-point operands more carefully than as described in this specification.

        .. versionchanged:: 2022.12
        Added complex data type support.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def real(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        """
        Returns the real component of a complex number for each element ``x_i`` of the input array ``x``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Must have a numeric data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a floating-point data type with the same floating-point precision as ``x`` (e.g., if ``x`` is ``complex64``, the returned array must have the floating-point data type ``float32``).

        Notes
        -----

        -   Whether the returned array and the input array share the same underlying memory is unspecified and thus implementation-defined.

        .. versionadded:: 2022.12

        .. versionchanged:: 2024.12
        Added support for real-valued arrays.
        """

    @abstractmethod
    def reciprocal(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        """
        Returns the reciprocal for each element ``x_i`` of the input array ``x``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Special cases**

        For floating-point operands, special cases must be handled as if the operation is implemented as ``1.0 / x`` (see :func:`~array_api.divide`).

        .. versionadded:: 2024.12
        """

    @abstractmethod
    def remainder(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, float], x2: Union[_NAMESPACE_ARRAY, int, float], /
    ) -> _NAMESPACE_ARRAY:
        """
        Returns the remainder of division for each element ``x1_i`` of the input array ``x1`` and the respective element ``x2_i`` of the input array ``x2``.

        .. note::
        This function is equivalent to the Python modulus operator ``x1_i % x2_i``.

        Parameters
        ----------
        x1: Union[array, int, float]
            dividend input array. Should have a real-valued data type.
        x2: Union[array, int, float]
            divisor input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a real-valued data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise results. Each element-wise result must have the same sign as the respective element ``x2_i``. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.
        -   For input arrays which promote to an integer data type, the result of division by zero is unspecified and thus implementation-defined.

        **Special cases**

        .. note::
        In general, similar to Python's ``%`` operator, this function is **not** recommended for floating-point operands as semantics do not follow IEEE 754. That this function is specified to accept floating-point operands is primarily for reasons of backward compatibility.

        For floating-point operands,

        - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
        - If ``x1_i`` is either ``+infinity`` or ``-infinity`` and ``x2_i`` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.
        - If ``x1_i`` is either ``+0`` or ``-0`` and ``x2_i`` is either ``+0`` or ``-0``, the result is ``NaN``.
        - If ``x1_i`` is ``+0`` and ``x2_i`` is greater than ``0``, the result is ``+0``.
        - If ``x1_i`` is ``-0`` and ``x2_i`` is greater than ``0``, the result is ``+0``.
        - If ``x1_i`` is ``+0`` and ``x2_i`` is less than ``0``, the result is ``-0``.
        - If ``x1_i`` is ``-0`` and ``x2_i`` is less than ``0``, the result is ``-0``.
        - If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``+0``, the result is ``NaN``.
        - If ``x1_i`` is greater than ``0`` and ``x2_i`` is ``-0``, the result is ``NaN``.
        - If ``x1_i`` is less than ``0`` and ``x2_i`` is ``+0``, the result is ``NaN``.
        - If ``x1_i`` is less than ``0`` and ``x2_i`` is ``-0``, the result is ``NaN``.
        - If ``x1_i`` is ``+infinity`` and ``x2_i`` is a positive (i.e., greater than ``0``) finite number, the result is ``NaN``.
        - If ``x1_i`` is ``+infinity`` and ``x2_i`` is a negative (i.e., less than ``0``) finite number, the result is ``NaN``.
        - If ``x1_i`` is ``-infinity`` and ``x2_i`` is a positive (i.e., greater than ``0``) finite number, the result is ``NaN``.
        - If ``x1_i`` is ``-infinity`` and ``x2_i`` is a negative (i.e., less than ``0``) finite number, the result is ``NaN``.
        - If ``x1_i`` is a positive (i.e., greater than ``0``) finite number and ``x2_i`` is ``+infinity``, the result is ``x1_i``. (**note**: this result matches Python behavior.)
        - If ``x1_i`` is a positive (i.e., greater than ``0``) finite number and ``x2_i`` is ``-infinity``, the result is ``x2_i``. (**note**: this result matches Python behavior.)
        - If ``x1_i`` is a negative (i.e., less than ``0``) finite number and ``x2_i`` is ``+infinity``, the result is ``x2_i``. (**note**: this results matches Python behavior.)
        - If ``x1_i`` is a negative (i.e., less than ``0``) finite number and ``x2_i`` is ``-infinity``, the result is ``x1_i``. (**note**: this result matches Python behavior.)
        - In the remaining cases, the result must match that of the Python ``%`` operator.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def round(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        """
        Rounds each element ``x_i`` of the input array ``x`` to the nearest integer-valued number.

        .. note::
        For complex floating-point operands, real and imaginary components must be independently rounded to the nearest integer-valued number.

        Rounded real and imaginary components must be equal to their equivalent rounded real-valued floating-point counterparts (i.e., for complex-valued ``x``, ``real(round(x))`` must equal ``round(real(x)))`` and ``imag(round(x))`` must equal ``round(imag(x))``).

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a numeric data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the rounded result for each element in ``x``. The returned array must have the same data type as ``x``.

        Notes
        -----

        **Special cases**

        .. note::
        For complex floating-point operands, the following special cases apply to real and imaginary components independently (e.g., if ``real(x_i)`` is ``NaN``, the rounded real component is ``NaN``).

        - If ``x_i`` is already integer-valued, the result is ``x_i``.

        For floating-point operands,

        - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
        - If ``x_i`` is ``-infinity``, the result is ``-infinity``.
        - If ``x_i`` is ``+0``, the result is ``+0``.
        - If ``x_i`` is ``-0``, the result is ``-0``.
        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If two integers are equally close to ``x_i``, the result is the even integer closest to ``x_i``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def sign(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Returns an indication of the sign of a number for each element ``x_i`` of the input array ``x``.

        The sign function (also known as the **signum function**) of a number :math:`x_i` is defined as

        .. math::
        \operatorname{sign}(x_i) = \begin{cases}
        0 & \textrm{if } x_i = 0 \\
        \frac{x_i}{|x_i|} & \textrm{otherwise}
        \end{cases}

        where :math:`|x_i|` is the absolute value of :math:`x_i`.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a numeric data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the evaluated result for each element in ``x``. The returned array must have the same data type as ``x``.

        Notes
        -----

        **Special cases**

        For real-valued operands,

        - If ``x_i`` is less than ``0``, the result is ``-1``.
        - If ``x_i`` is either ``-0`` or ``+0``, the result is ``0``.
        - If ``x_i`` is greater than ``0``, the result is ``+1``.
        - If ``x_i`` is ``NaN``, the result is ``NaN``.

        For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

        - If ``a`` is either ``-0`` or ``+0`` and ``b`` is either ``-0`` or ``+0``, the result is ``0 + 0j``.
        - If ``a`` is ``NaN`` or ``b`` is ``NaN``, the result is ``NaN + NaN j``.
        - In the remaining cases, special cases must be handled according to the rules of complex number division (see :func:`~array_api.divide`).

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def signbit(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Determines whether the sign bit is set for each element ``x_i`` of the input array ``x``.

        The sign bit of a real-valued floating-point number ``x_i`` is set whenever ``x_i`` is either ``-0``, less than zero, or a signed ``NaN`` (i.e., a ``NaN`` value whose sign bit is ``1``).

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a real-valued floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the evaluated result for each element in ``x``. The returned array must have a data type of ``bool``.

        Notes
        -----

        **Special cases**

        For real-valued floating-point operands,

        - If ``x_i`` is ``+0``, the result is ``False``.
        - If ``x_i`` is ``-0``, the result is ``True``.
        - If ``x_i`` is ``+infinity``, the result is ``False``.
        - If ``x_i`` is ``-infinity``, the result is ``True``.
        - If ``x_i`` is a positive (i.e., greater than ``0``) finite number, the result is ``False``.
        - If ``x_i`` is a negative (i.e., less than ``0``) finite number, the result is ``True``.
        - If ``x_i`` is ``NaN`` and the sign bit of ``x_i`` is ``0``, the result is ``False``.
        - If ``x_i`` is ``NaN`` and the sign bit of ``x_i`` is ``1``, the result is ``True``.

        .. versionadded:: 2023.12
        """

    @abstractmethod
    def sin(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Calculates an implementation-dependent approximation to the sine for each element ``x_i`` of the input array ``x``.

        Each element ``x_i`` is assumed to be expressed in radians.

        .. note::
        The sine is an entire function on the complex plane and has no branch cuts.

        .. note::
        For complex arguments, the mathematical definition of sine is

        .. math::
            \begin{align} \operatorname{sin}(x) &= \frac{e^{jx} - e^{-jx}}{2j} \\ &= \frac{\operatorname{sinh}(jx)}{j} \\ &= \frac{\operatorname{sinh}(jx)}{j} \cdot \frac{j}{j} \\ &= -j \cdot \operatorname{sinh}(jx) \end{align}

        where :math:`\operatorname{sinh}` is the hyperbolic sine.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array whose elements are each expressed in radians. Should have a floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the sine of each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Special cases**

        For real-valued floating-point operands,

        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If ``x_i`` is ``+0``, the result is ``+0``.
        - If ``x_i`` is ``-0``, the result is ``-0``.
        - If ``x_i`` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.

        For complex floating-point operands, special cases must be handled as if the operation is implemented as ``-1j * sinh(x*1j)``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def sinh(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Calculates an implementation-dependent approximation to the hyperbolic sine for each element ``x_i`` of the input array ``x``.

        The mathematical definition of the hyperbolic sine is

        .. math::
        \operatorname{sinh}(x) = \frac{e^x - e^{-x}}{2}

        .. note::
        The hyperbolic sine is an entire function in the complex plane and has no branch cuts. The function is periodic, with period :math:`2\pi j`, with respect to the imaginary component.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array whose elements each represent a hyperbolic angle. Should have a floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the hyperbolic sine of each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Special cases**

        .. note::
        For all operands, ``sinh(x)`` must equal ``-sinh(-x)``.

        For real-valued floating-point operands,

        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If ``x_i`` is ``+0``, the result is ``+0``.
        - If ``x_i`` is ``-0``, the result is ``-0``.
        - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
        - If ``x_i`` is ``-infinity``, the result is ``-infinity``.

        For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

        .. note::
        For complex floating-point operands, ``sinh(conj(x))`` must equal ``conj(sinh(x))``.

        - If ``a`` is ``+0`` and ``b`` is ``+0``, the result is ``+0 + 0j``.
        - If ``a`` is ``+0`` and ``b`` is ``+infinity``, the result is ``0 + NaN j`` (sign of the real component is unspecified).
        - If ``a`` is ``+0`` and ``b`` is ``NaN``, the result is ``0 + NaN j`` (sign of the real component is unspecified).
        - If ``a`` is a positive (i.e., greater than ``0``) finite number and ``b`` is ``+infinity``, the result is ``NaN + NaN j``.
        - If ``a`` is a positive (i.e., greater than ``0``) finite number and ``b`` is ``NaN``, the result is ``NaN + NaN j``.
        - If ``a`` is ``+infinity`` and ``b`` is ``+0``, the result is ``+infinity + 0j``.
        - If ``a`` is ``+infinity`` and ``b`` is a positive finite number, the result is ``+infinity * cis(b)``.
        - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``, the result is ``infinity + NaN j`` (sign of the real component is unspecified).
        - If ``a`` is ``+infinity`` and ``b`` is ``NaN``, the result is ``infinity + NaN j`` (sign of the real component is unspecified).
        - If ``a`` is ``NaN`` and ``b`` is ``+0``, the result is ``NaN + 0j``.
        - If ``a`` is ``NaN`` and ``b`` is a nonzero finite number, the result is ``NaN + NaN j``.
        - If ``a`` is ``NaN`` and ``b`` is ``NaN``, the result is ``NaN + NaN j``.

        where ``cis(v)`` is ``cos(v) + sin(v)*1j``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def square(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Squares each element ``x_i`` of the input array ``x``.

        The square of a number ``x_i`` is defined as

        .. math::
        x_i^2 = x_i \cdot x_i

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a numeric data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the evaluated result for each element in ``x``. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Special cases**

        For floating-point operands, special cases must be handled as if the operation is implemented as ``x * x`` (see :func:`~array_api.multiply`).

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def sqrt(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Calculates the principal square root for each element ``x_i`` of the input array ``x``.

        .. note::
        After rounding, each result must be indistinguishable from the infinitely precise result (as required by IEEE 754).

        .. note::
        For complex floating-point operands, ``sqrt(conj(x))`` must equal ``conj(sqrt(x))``.

        .. note::
        By convention, the branch cut of the square root is the negative real axis :math:`(-\infty, 0)`.

        The square root is a continuous function from above the branch cut, taking into account the sign of the imaginary component.

        Accordingly, for complex arguments, the function returns the square root in the range of the right half-plane, including the imaginary axis (i.e., the plane defined by :math:`[0, +\infty)` along the real axis and :math:`(-\infty, +\infty)` along the imaginary axis).

        *Note: branch cuts follow C99 and have provisional status* (see :ref:`branch-cuts`).

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the square root of each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Special cases**

        For real-valued floating-point operands,

        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If ``x_i`` is less than ``0``, the result is ``NaN``.
        - If ``x_i`` is ``+0``, the result is ``+0``.
        - If ``x_i`` is ``-0``, the result is ``-0``.
        - If ``x_i`` is ``+infinity``, the result is ``+infinity``.

        For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

        - If ``a`` is either ``+0`` or ``-0`` and ``b`` is ``+0``, the result is ``+0 + 0j``.
        - If ``a`` is any value (including ``NaN``) and ``b`` is ``+infinity``, the result is ``+infinity + infinity j``.
        - If ``a`` is a finite number and ``b`` is ``NaN``, the result is ``NaN + NaN j``.
        - If ``a`` ``-infinity`` and ``b`` is a positive (i.e., greater than ``0``) finite number, the result is ``NaN + NaN j``.
        - If ``a`` is ``+infinity`` and ``b`` is a positive (i.e., greater than ``0``) finite number, the result is ``+0 + infinity j``.
        - If ``a`` is ``-infinity`` and ``b`` is ``NaN``, the result is ``NaN + infinity j`` (sign of the imaginary component is unspecified).
        - If ``a`` is ``+infinity`` and ``b`` is ``NaN``, the result is ``+infinity + NaN j``.
        - If ``a`` is ``NaN`` and ``b`` is any value, the result is ``NaN + NaN j``.
        - If ``a`` is ``NaN`` and ``b`` is ``NaN``, the result is ``NaN + NaN j``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def subtract(
        self: _NAMESPACE_C,
        x1: Union[_NAMESPACE_ARRAY, int, float, complex], x2: Union[_NAMESPACE_ARRAY, int, float, complex], /
    ) -> _NAMESPACE_ARRAY:
        """
        Calculates the difference for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

        Parameters
        ----------
        x1: Union[array, int, float, complex]
            first input array. Should have a numeric data type.
        x2: Union[array, int, float, complex]
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the element-wise differences. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        -   At least one of ``x1`` or ``x2`` must be an array.
        -   The result of ``x1_i - x2_i`` must be the same as ``x1_i + (-x2_i)`` and must be governed by the same floating-point rules as addition (see :meth:`add`).

        .. versionchanged:: 2022.12
        Added complex data type support.

        .. versionchanged:: 2024.12
        Added scalar argument support.
        """

    @abstractmethod
    def tan(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Calculates an implementation-dependent approximation to the tangent for each element ``x_i`` of the input array ``x``.

        Each element ``x_i`` is assumed to be expressed in radians.

        .. note::
        Tangent is an analytical function on the complex plane and has no branch cuts. The function is periodic, with period :math:`\pi j`, with respect to the real component and has first order poles along the real line at coordinates :math:`(\pi (\frac{1}{2} + n), 0)`. However, IEEE 754 binary floating-point representation cannot represent the value :math:`\pi / 2` exactly, and, thus, no argument value is possible for which a pole error occurs.

        .. note::
        For complex arguments, the mathematical definition of tangent is

        .. math::
            \begin{align} \operatorname{tan}(x) &= \frac{j(e^{-jx} - e^{jx})}{e^{-jx} + e^{jx}} \\ &= (-1) \frac{j(e^{jx} - e^{-jx})}{e^{jx} + e^{-jx}} \\ &= -j \cdot \operatorname{tanh}(jx) \end{align}

        where :math:`\operatorname{tanh}` is the hyperbolic tangent.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array whose elements are expressed in radians. Should have a floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the tangent of each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Special cases**

        For real-valued floating-point operands,

        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If ``x_i`` is ``+0``, the result is ``+0``.
        - If ``x_i`` is ``-0``, the result is ``-0``.
        - If ``x_i`` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.

        For complex floating-point operands, special cases must be handled as if the operation is implemented as ``-1j * tanh(x*1j)``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def tanh(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        r"""
        Calculates an implementation-dependent approximation to the hyperbolic tangent for each element ``x_i`` of the input array ``x``.

        The mathematical definition of the hyperbolic tangent is

        .. math::
        \begin{align} \operatorname{tanh}(x) &= \frac{\operatorname{sinh}(x)}{\operatorname{cosh}(x)} \\ &= \frac{e^x - e^{-x}}{e^x + e^{-x}} \end{align}

        where :math:`\operatorname{sinh}(x)` is the hyperbolic sine and :math:`\operatorname{cosh}(x)` is the hyperbolic cosine.

        .. note::
        The hyperbolic tangent is an analytical function on the complex plane and has no branch cuts. The function is periodic, with period :math:`\pi j`, with respect to the imaginary component and has first order poles along the imaginary line at coordinates :math:`(0, \pi (\frac{1}{2} + n))`. However, IEEE 754 binary floating-point representation cannot represent :math:`\pi / 2` exactly, and, thus, no argument value is possible such that a pole error occurs.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array whose elements each represent a hyperbolic angle. Should have a floating-point data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the hyperbolic tangent of each element in ``x``. The returned array must have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Special cases**

        .. note::
        For all operands, ``tanh(-x)`` must equal ``-tanh(x)``.

        For real-valued floating-point operands,

        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If ``x_i`` is ``+0``, the result is ``+0``.
        - If ``x_i`` is ``-0``, the result is ``-0``.
        - If ``x_i`` is ``+infinity``, the result is ``+1``.
        - If ``x_i`` is ``-infinity``, the result is ``-1``.

        For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

        .. note::
        For complex floating-point operands, ``tanh(conj(x))`` must equal ``conj(tanh(x))``.

        - If ``a`` is ``+0`` and ``b`` is ``+0``, the result is ``+0 + 0j``.
        - If ``a`` is a nonzero finite number and ``b`` is ``+infinity``, the result is ``NaN + NaN j``.
        - If ``a`` is ``+0`` and ``b`` is ``+infinity``, the result is ``+0 + NaN j``.
        - If ``a`` is a nonzero finite number and ``b`` is ``NaN``, the result is ``NaN + NaN j``.
        - If ``a`` is ``+0`` and ``b`` is ``NaN``, the result is ``+0 + NaN j``.
        - If ``a`` is ``+infinity`` and ``b`` is a positive (i.e., greater than ``0``) finite number, the result is ``1 + 0j``.
        - If ``a`` is ``+infinity`` and ``b`` is ``+infinity``, the result is ``1 + 0j`` (sign of the imaginary component is unspecified).
        - If ``a`` is ``+infinity`` and ``b`` is ``NaN``, the result is ``1 + 0j`` (sign of the imaginary component is unspecified).
        - If ``a`` is ``NaN`` and ``b`` is ``+0``, the result is ``NaN + 0j``.
        - If ``a`` is ``NaN`` and ``b`` is a nonzero number, the result is ``NaN + NaN j``.
        - If ``a`` is ``NaN`` and ``b`` is ``NaN``, the result is ``NaN + NaN j``.

        .. warning::
        For historical reasons stemming from the C standard, array libraries may not return the expected result when ``a`` is ``+0`` and ``b`` is either ``+infinity`` or ``NaN``. The result should be ``+0 + NaN j`` in both cases; however, for libraries compiled against older C versions, the result may be ``NaN + NaN j``.

        Array libraries are not required to patch these older C versions, and, thus, users are advised that results may vary across array library implementations for these special cases.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def trunc(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        """
        Rounds each element ``x_i`` of the input array ``x`` to the nearest integer-valued number that is closer to zero than ``x_i``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a real-valued data type.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the rounded result for each element in ``x``. The returned array must have the same data type as ``x``.

        Notes
        -----

        **Special cases**

        - If ``x_i`` is already integer-valued, the result is ``x_i``.

        For floating-point operands,

        - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
        - If ``x_i`` is ``-infinity``, the result is ``-infinity``.
        - If ``x_i`` is ``+0``, the result is ``+0``.
        - If ``x_i`` is ``-0``, the result is ``-0``.
        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        """

    """
    Indexing Functions
    https://github.com/data-apis/array-api/blob/main/src/array_api_stubs/_2024_12/indexing_functions.py
    """
    @abstractmethod
    def take(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, indices: _NAMESPACE_ARRAY, /, *, axis: Optional[int] = None) -> _NAMESPACE_ARRAY:
        """
        Returns elements of an array along an axis.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have one or more dimensions (axes).
        indices: _NAMESPACE_ARRAY
            array indices. The array must be one-dimensional and have an integer data type. If an index is negative, the function must determine the element to select along a specified axis (dimension) by counting from the last element (where ``-1`` refers to the last element).
        axis: Optional[int]
            axis over which to select values. If ``axis`` is negative, the function must determine the axis along which to select values by counting from the last dimension (where ``-1`` refers to the last dimension).

            If ``x`` is a one-dimensional array, providing an ``axis`` is optional; however, if ``x`` has more than one dimension, providing an ``axis`` is required.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array having the same data type as ``x``. The output array must have the same rank (i.e., number of dimensions) as ``x`` and must have the same shape as ``x``, except for the axis specified by ``axis`` whose size must equal the number of elements in ``indices``.

        Notes
        -----

        -   Conceptually, ``take(x, indices, axis=3)`` is equivalent to ``x[:,:,:,indices,...]``; however, explicit indexing via arrays of indices is not currently supported in this specification due to concerns regarding ``__setitem__`` and array mutation semantics.
        -   This specification does not require bounds checking. The behavior for out-of-bounds indices is left unspecified.
        -   When ``x`` is a zero-dimensional array, behavior is unspecified and thus implementation-defined.

        .. versionadded:: 2022.12

        .. versionchanged:: 2023.12
        Out-of-bounds behavior is explicitly left unspecified.

        .. versionchanged:: 2024.12
        Behavior when provided a zero-dimensional input array is explicitly left unspecified.

        .. versionchanged:: 2024.12
        Clarified support for negative indices.
        """

    @abstractmethod
    def take_along_axis(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, indices: _NAMESPACE_ARRAY, /, *, axis: int = -1
    ) -> _NAMESPACE_ARRAY:
        """
        Returns elements from an array at the one-dimensional indices specified by ``indices`` along a provided ``axis``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Must be compatible with ``indices``, except for the axis (dimension) specified by ``axis`` (see :ref:`broadcasting`).
        indices: _NAMESPACE_ARRAY
            array indices. Must have the same rank (i.e., number of dimensions) as ``x``. If an index is negative, the function must determine the element to select along a specified axis (dimension) by counting from the last element (where ``-1`` refers to the last element).
        axis: int
            axis along which to select values. If ``axis`` is negative, the function must determine the axis along which to select values by counting from the last dimension (where ``-1`` refers to the last dimension). Default: ``-1``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array having the same data type as ``x``. Must have the same rank (i.e., number of dimensions) as ``x`` and must have a shape determined according to :ref:`broadcasting`, except for the axis (dimension) specified by ``axis`` whose size must equal the size of the corresponding axis (dimension) in ``indices``.

        Notes
        -----

        -   This specification does not require bounds checking. The behavior for out-of-bounds indices is left unspecified.

        .. versionadded:: 2024.12
        """
    
    """
    Linear Algebra Functions
    https://github.com/data-apis/array-api/blob/main/src/array_api_stubs/_2024_12/linear_algebra_functions.py
    """
    @abstractmethod
    def matmul(
        self: _NAMESPACE_C,
        x1: _NAMESPACE_ARRAY, x2: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        """
        Computes the matrix product.

        .. note::
        The ``matmul`` function must implement the same semantics as the built-in ``@`` operator (see `PEP 465 <https://www.python.org/dev/peps/pep-0465>`_).

        Parameters
        ----------
        x1: _NAMESPACE_ARRAY
            first input array. Should have a numeric data type. Must have at least one dimension. If ``x1`` is one-dimensional having shape ``(M,)`` and ``x2`` has more than one dimension, ``x1`` must be promoted to a two-dimensional array by prepending ``1`` to its dimensions (i.e., must have shape ``(1, M)``). After matrix multiplication, the prepended dimensions in the returned array must be removed. If ``x1`` has more than one dimension (including after vector-to-matrix promotion), ``shape(x1)[:-2]`` must be compatible with ``shape(x2)[:-2]`` (after vector-to-matrix promotion) (see :ref:`broadcasting`). If ``x1`` has shape ``(..., M, K)``, the innermost two dimensions form matrices on which to perform matrix multiplication.
        x2: _NAMESPACE_ARRAY
            second input array. Should have a numeric data type. Must have at least one dimension. If ``x2`` is one-dimensional having shape ``(N,)`` and ``x1`` has more than one dimension, ``x2`` must be promoted to a two-dimensional array by appending ``1`` to its dimensions (i.e., must have shape ``(N, 1)``). After matrix multiplication, the appended dimensions in the returned array must be removed. If ``x2`` has more than one dimension (including after vector-to-matrix promotion), ``shape(x2)[:-2]`` must be compatible with ``shape(x1)[:-2]`` (after vector-to-matrix promotion) (see :ref:`broadcasting`). If ``x2`` has shape ``(..., K, N)``, the innermost two dimensions form matrices on which to perform matrix multiplication.


        .. note::
        If either ``x1`` or ``x2`` has a complex floating-point data type, neither argument must be complex-conjugated or transposed. If conjugation and/or transposition is desired, these operations should be explicitly performed prior to computing the matrix product.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            -   if both ``x1`` and ``x2`` are one-dimensional arrays having shape ``(N,)``, a zero-dimensional array containing the inner product as its only element.
            -   if ``x1`` is a two-dimensional array having shape ``(M, K)`` and ``x2`` is a two-dimensional array having shape ``(K, N)``, a two-dimensional array containing the `conventional matrix product <https://en.wikipedia.org/wiki/Matrix_multiplication>`_ and having shape ``(M, N)``.
            -   if ``x1`` is a one-dimensional array having shape ``(K,)`` and ``x2`` is an array having shape ``(..., K, N)``, an array having shape ``(..., N)`` (i.e., prepended dimensions during vector-to-matrix promotion must be removed) and containing the `conventional matrix product <https://en.wikipedia.org/wiki/Matrix_multiplication>`_.
            -   if ``x1`` is an array having shape ``(..., M, K)`` and ``x2`` is a one-dimensional array having shape ``(K,)``, an array having shape ``(..., M)`` (i.e., appended dimensions during vector-to-matrix promotion must be removed) and containing the `conventional matrix product <https://en.wikipedia.org/wiki/Matrix_multiplication>`_.
            -   if ``x1`` is a two-dimensional array having shape ``(M, K)`` and ``x2`` is an array having shape ``(..., K, N)``, an array having shape ``(..., M, N)`` and containing the `conventional matrix product <https://en.wikipedia.org/wiki/Matrix_multiplication>`_ for each stacked matrix.
            -   if ``x1`` is an array having shape ``(..., M, K)`` and ``x2`` is a two-dimensional array having shape ``(K, N)``, an array having shape ``(..., M, N)`` and containing the `conventional matrix product <https://en.wikipedia.org/wiki/Matrix_multiplication>`_ for each stacked matrix.
            -   if either ``x1`` or ``x2`` has more than two dimensions, an array having a shape determined by :ref:`broadcasting` ``shape(x1)[:-2]`` against ``shape(x2)[:-2]`` and containing the `conventional matrix product <https://en.wikipedia.org/wiki/Matrix_multiplication>`_ for each stacked matrix.

            The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.

        **Raises**

        -   if either ``x1`` or ``x2`` is a zero-dimensional array.
        -   if ``x1`` is a one-dimensional array having shape ``(K,)``, ``x2`` is a one-dimensional array having shape ``(L,)``, and ``K != L``.
        -   if ``x1`` is a one-dimensional array having shape ``(K,)``, ``x2`` is an array having shape ``(..., L, N)``, and ``K != L``.
        -   if ``x1`` is an array having shape ``(..., M, K)``, ``x2`` is a one-dimensional array having shape ``(L,)``, and ``K != L``.
        -   if ``x1`` is an array having shape ``(..., M, K)``, ``x2`` is an array having shape ``(..., L, N)``, and ``K != L``.

        """

    @abstractmethod
    def matrix_transpose(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        """
        Transposes a matrix (or a stack of matrices) ``x``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array having shape ``(..., M, N)`` and whose innermost two dimensions form ``MxN`` matrices.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the transpose for each matrix and having shape ``(..., N, M)``. The returned array must have the same data type as ``x``.
        """

    @abstractmethod
    def tensordot(
        self: _NAMESPACE_C,
        x1: _NAMESPACE_ARRAY,
        x2: _NAMESPACE_ARRAY,
        /,
        *,
        axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2,
    ) -> _NAMESPACE_ARRAY:
        """
        Returns a tensor contraction of ``x1`` and ``x2`` over specific axes.

        .. note::
        The ``tensordot`` function corresponds to the generalized matrix product.

        Parameters
        ----------
        x1: _NAMESPACE_ARRAY
            first input array. Should have a numeric data type.
        x2: _NAMESPACE_ARRAY
            second input array. Should have a numeric data type. Corresponding contracted axes of ``x1`` and ``x2`` must be equal.

            .. note::
            Contracted axes (dimensions) must not be broadcasted.

        axes: Union[int, Tuple[Sequence[int], Sequence[int]]]
            number of axes (dimensions) to contract or explicit sequences of axis (dimension) indices for ``x1`` and ``x2``, respectively.

            If ``axes`` is an ``int`` equal to ``N``, then contraction must be performed over the last ``N`` axes of ``x1`` and the first ``N`` axes of ``x2`` in order. The size of each corresponding axis (dimension) must match. Must be nonnegative.

            -   If ``N`` equals ``0``, the result is the tensor (outer) product.
            -   If ``N`` equals ``1``, the result is the tensor dot product.
            -   If ``N`` equals ``2``, the result is the tensor double contraction (default).

            If ``axes`` is a tuple of two sequences ``(x1_axes, x2_axes)``, the first sequence must apply to ``x1`` and the second sequence to ``x2``. Both sequences must have the same length. Each axis (dimension) ``x1_axes[i]`` for ``x1`` must have the same size as the respective axis (dimension) ``x2_axes[i]`` for ``x2``. Each index referred to in a sequence must be unique. If ``x1`` has rank (i.e, number of dimensions) ``N``, a valid ``x1`` axis must reside on the half-open interval ``[-N, N)``. If ``x2`` has rank ``M``, a valid ``x2`` axis must reside on the half-open interval ``[-M, M)``.


        .. note::
        If either ``x1`` or ``x2`` has a complex floating-point data type, neither argument must be complex-conjugated or transposed. If conjugation and/or transposition is desired, these operations should be explicitly performed prior to computing the generalized matrix product.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the tensor contraction whose shape consists of the non-contracted axes (dimensions) of the first array ``x1``, followed by the non-contracted axes (dimensions) of the second array ``x2``. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.

        .. versionchanged:: 2023.12
        Allow negative axes.
        """

    @abstractmethod
    def vecdot(
        self: _NAMESPACE_C,
        x1: _NAMESPACE_ARRAY, x2: _NAMESPACE_ARRAY, /, *, axis: int = -1
    ) -> _NAMESPACE_ARRAY:
        r"""
        Computes the (vector) dot product of two arrays.

        Let :math:`\mathbf{a}` be a vector in ``x1`` and :math:`\mathbf{b}` be a corresponding vector in ``x2``. The dot product is defined as

        .. math::
        \mathbf{a} \cdot \mathbf{b} = \sum_{i=0}^{n-1} \overline{a_i}b_i

        over the dimension specified by ``axis`` and where :math:`n` is the dimension size and :math:`\overline{a_i}` denotes the complex conjugate if :math:`a_i` is complex and the identity if :math:`a_i` is real-valued.

        Parameters
        ----------
        x1: _NAMESPACE_ARRAY
            first input array. Should have a floating-point data type.
        x2: _NAMESPACE_ARRAY
            second input array. Must be compatible with ``x1`` for all non-contracted axes (see :ref:`broadcasting`). The size of the axis over which to compute the dot product must be the same size as the respective axis in ``x1``. Should have a floating-point data type.

            .. note::
            The contracted axis (dimension) must not be broadcasted.

        axis: int
            the axis (dimension) of ``x1`` and ``x2`` containing the vectors for which to compute the dot product. Should be an integer on the interval ``[-N, -1]``, where ``N`` is ``min(x1.ndim, x2.ndim)``. The function must determine the axis along which to compute the dot product by counting backward from the last dimension (where ``-1`` refers to the last dimension). By default, the function must compute the dot product over the last axis. Default: ``-1``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            if ``x1`` and ``x2`` are both one-dimensional arrays, a zero-dimensional containing the dot product; otherwise, a non-zero-dimensional array containing the dot products and having rank ``N-1``, where ``N`` is the rank (number of dimensions) of the shape determined according to :ref:`broadcasting` along the non-contracted axes. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        **Raises**

        -   if the size of the axis over which to compute the dot product is not the same (before broadcasting) for both ``x1`` and ``x2``.

        .. versionchanged:: 2022.12
        Added complex data type support.

        .. versionchanged:: 2023.12
        Restricted ``axis`` to only negative integers.
        """
    
    """
    Manipulation Functions
    https://github.com/data-apis/array-api/blob/main/src/array_api_stubs/_2024_12/manipulation_functions.py
    """
    @abstractmethod
    def broadcast_arrays(
        self: _NAMESPACE_C,
        *arrays: _NAMESPACE_ARRAY
    ) -> List[_NAMESPACE_ARRAY]:
        """
        Broadcasts one or more arrays against one another.

        Parameters
        ----------
        arrays: _NAMESPACE_ARRAY
            an arbitrary number of to-be broadcasted arrays.

        Returns
        -------
        out: List[array]
            a list of broadcasted arrays. Each array must have the same shape. Each array must have the same dtype as its corresponding input array.
        """

    @abstractmethod
    def broadcast_to(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /, shape: Tuple[int, ...]
    ) -> _NAMESPACE_ARRAY:
        """
        Broadcasts an array to a specified shape.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            array to broadcast. Must be capable of being broadcast to the specified ``shape`` (see :ref:`broadcasting`). If the array is incompatible with the specified shape, the function must raise an exception.
        shape: Tuple[int, ...]
            array shape.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array having the specified shape. Must have the same data type as ``x``.

        .. versionchanged:: 2024.12
        Clarified broadcast behavior.
        """

    @abstractmethod
    def concat(
        self: _NAMESPACE_C,
        arrays: Union[Tuple[_NAMESPACE_ARRAY, ...], List[_NAMESPACE_ARRAY]], /, *, axis: Optional[int] = 0
    ) -> _NAMESPACE_ARRAY:
        """
        Joins a sequence of arrays along an existing axis.

        Parameters
        ----------
        arrays: Union[Tuple[array, ...], List[array]]
            input arrays to join. The arrays must have the same shape, except in the dimension specified by ``axis``.
        axis: Optional[int]
            axis along which the arrays will be joined. If ``axis`` is ``None``, arrays must be flattened before concatenation. If ``axis`` is negative, the function must determine the axis along which to join by counting from the last dimension. Default: ``0``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an output array containing the concatenated values. If the input arrays have different data types, normal :ref:`type-promotion` must apply. If the input arrays have the same data type, the output array must have the same data type as the input arrays.

            .. note::
            This specification leaves type promotion between data type families (i.e., ``intxx`` and ``floatxx``) unspecified.
        """

    @abstractmethod
    def expand_dims(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /, *, axis: int = 0
    ) -> _NAMESPACE_ARRAY:
        """
        Expands the shape of an array by inserting a new axis (dimension) of size one at the position specified by ``axis``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array.
        axis: int
            axis position (zero-based). If ``x`` has rank (i.e, number of dimensions) ``N``, a valid ``axis`` must reside on the closed-interval ``[-N-1, N]``. If provided a negative ``axis``, the axis position at which to insert a singleton dimension must be computed as ``N + axis + 1``. Hence, if provided ``-1``, the resolved axis position must be ``N`` (i.e., a singleton dimension must be appended to the input array ``x``). If provided ``-N-1``, the resolved axis position must be ``0`` (i.e., a singleton dimension must be prepended to the input array ``x``).

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an expanded output array having the same data type as ``x``.

        Raises
        ------
        IndexError
            If provided an invalid ``axis`` position, an ``IndexError`` should be raised.
        """

    @abstractmethod
    def flip(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> _NAMESPACE_ARRAY:
        """
        Reverses the order of elements in an array along the given axis. The shape of the array must be preserved.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array.
        axis: Optional[Union[int, Tuple[int, ...]]]
            axis (or axes) along which to flip. If ``axis`` is ``None``, the function must flip all input array axes. If ``axis`` is negative, the function must count from the last dimension. If provided more than one axis, the function must flip only the specified axes. Default: ``None``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an output array having the same data type and shape as ``x`` and whose elements, relative to ``x``, are reordered.
        """

    @abstractmethod
    def moveaxis(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY,
        source: Union[int, Tuple[int, ...]],
        destination: Union[int, Tuple[int, ...]],
        /,
    ) -> _NAMESPACE_ARRAY:
        """
        Moves array axes (dimensions) to new positions, while leaving other axes in their original positions.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array.
        source: Union[int, Tuple[int, ...]]
            Axes to move. Provided axes must be unique. If ``x`` has rank (i.e, number of dimensions) ``N``, a valid axis must reside on the half-open interval ``[-N, N)``.
        destination: Union[int, Tuple[int, ...]]
            indices defining the desired positions for each respective ``source`` axis index. Provided indices must be unique. If ``x`` has rank (i.e, number of dimensions) ``N``, a valid axis must reside on the half-open interval ``[-N, N)``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing reordered axes. The returned array must have the same data type as ``x``.

        Notes
        -----

        .. versionadded:: 2023.12
        """

    @abstractmethod
    def permute_dims(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /, axes: Tuple[int, ...]
    ) -> _NAMESPACE_ARRAY:
        """
        Permutes the axes (dimensions) of an array ``x``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array.
        axes: Tuple[int, ...]
            tuple containing a permutation of ``(0, 1, ..., N-1)`` where ``N`` is the number of axes (dimensions) of ``x``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the axes permutation. The returned array must have the same data type as ``x``.
        """

    @abstractmethod
    def repeat(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY,
        repeats: Union[int, _NAMESPACE_ARRAY],
        /,
        *,
        axis: Optional[int] = None,
    ) -> _NAMESPACE_ARRAY:
        """
        Repeats each element of an array a specified number of times on a per-element basis.

        .. admonition:: Data-dependent output shape
            :class: important

            When ``repeats`` is an array, the shape of the output array for this function depends on the data values in the ``repeats`` array; hence, array libraries which build computation graphs (e.g., JAX, Dask, etc.) may find this function difficult to implement without knowing the values in ``repeats``. Accordingly, such libraries may choose to omit support for ``repeats`` arrays; however, conforming implementations must support providing a literal ``int``. See :ref:`data-dependent-output-shapes` section for more details.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array containing elements to repeat.
        repeats: Union[int, array]
            the number of repetitions for each element.

            If ``axis`` is ``None``, let ``N = prod(x.shape)`` and

            -   if ``repeats`` is an array, ``repeats`` must be broadcast compatible with the shape ``(N,)`` (i.e., be a one-dimensional array having shape ``(1,)`` or ``(N,)``).
            -   if ``repeats`` is an integer, ``repeats`` must be broadcasted to the shape `(N,)`.

            If ``axis`` is not ``None``, let ``M = x.shape[axis]`` and

            -   if ``repeats`` is an array, ``repeats`` must be broadcast compatible with the shape ``(M,)`` (i.e., be a one-dimensional array having shape ``(1,)`` or ``(M,)``).
            -   if ``repeats`` is an integer, ``repeats`` must be broadcasted to the shape ``(M,)``.

            If ``repeats`` is an array, the array must have an integer data type.

            .. note::
            For specification-conforming array libraries supporting hardware acceleration, providing an array for ``repeats`` may cause device synchronization due to an unknown output shape. For those array libraries where synchronization concerns are applicable, conforming array libraries are advised to include a warning in their documentation regarding potential performance degradation when ``repeats`` is an array.

        axis: Optional[int]
            the axis (dimension) along which to repeat elements. If ``axis`` is `None`, the function must flatten the input array ``x`` and then repeat elements of the flattened input array and return the result as a one-dimensional output array. A flattened input array must be flattened in row-major, C-style order. Default: ``None``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an output array containing repeated elements. The returned array must have the same data type as ``x``. If ``axis`` is ``None``, the returned array must be a one-dimensional array; otherwise, the returned array must have the same shape as ``x``, except for the axis (dimension) along which elements were repeated.

        Notes
        -----

        .. versionadded:: 2023.12
        """

    @abstractmethod
    def reshape(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /, shape: Tuple[int, ...], *, copy: Optional[bool] = None
    ) -> _NAMESPACE_ARRAY:
        """
        Reshapes an array without changing its data.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array to reshape.
        shape: Tuple[int, ...]
            a new shape compatible with the original shape. One shape dimension is allowed to be ``-1``. When a shape dimension is ``-1``, the corresponding output array shape dimension must be inferred from the length of the array and the remaining dimensions.
        copy: Optional[bool]
            whether or not to copy the input array. If ``True``, the function must always copy (see :ref:`copy-keyword-argument`). If ``False``, the function must never copy. If ``None``, the function must avoid copying, if possible, and may copy otherwise. Default: ``None``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an output array having the same data type and elements as ``x``.

        Raises
        ------
        ValueError
            If ``copy=False`` and a copy would be necessary, a ``ValueError``
            should be raised.
        """

    @abstractmethod
    def roll(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY,
        /,
        shift: Union[int, Tuple[int, ...]],
        *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> _NAMESPACE_ARRAY:
        """
        Rolls array elements along a specified axis. Array elements that roll beyond the last position are re-introduced at the first position. Array elements that roll beyond the first position are re-introduced at the last position.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array.
        shift: Union[int, Tuple[int, ...]]
            number of places by which the elements are shifted. If ``shift`` is a tuple, then ``axis`` must be a tuple of the same size, and each of the given axes must be shifted by the corresponding element in ``shift``. If ``shift`` is an ``int`` and ``axis`` a tuple, then the same ``shift`` must be used for all specified axes. If a shift is positive, then array elements must be shifted positively (toward larger indices) along the dimension of ``axis``. If a shift is negative, then array elements must be shifted negatively (toward smaller indices) along the dimension of ``axis``.
        axis: Optional[Union[int, Tuple[int, ...]]]
            axis (or axes) along which elements to shift. If ``axis`` is ``None``, the array must be flattened, shifted, and then restored to its original shape. Default: ``None``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an output array having the same data type as ``x`` and whose elements, relative to ``x``, are shifted.
        """

    @abstractmethod
    def squeeze(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /, axis: Union[int, Tuple[int, ...]]
    ) -> _NAMESPACE_ARRAY:
        """
        Removes singleton dimensions (axes) from ``x``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array.
        axis: Union[int, Tuple[int, ...]]
            axis (or axes) to squeeze.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an output array having the same data type and elements as ``x``.

        Raises
        ------
        ValueError
            If a specified axis has a size greater than one (i.e., it is not a
            singleton dimension), a ``ValueError`` should be raised.
        """

    @abstractmethod
    def stack(
        self: _NAMESPACE_C,
        arrays: Union[Tuple[_NAMESPACE_ARRAY, ...], List[_NAMESPACE_ARRAY]], /, *, axis: int = 0
    ) -> _NAMESPACE_ARRAY:
        """
        Joins a sequence of arrays along a new axis.

        Parameters
        ----------
        arrays: Union[Tuple[array, ...], List[array]]
            input arrays to join. Each array must have the same shape.
        axis: int
            axis along which the arrays will be joined. Providing an ``axis`` specifies the index of the new axis in the dimensions of the result. For example, if ``axis`` is ``0``, the new axis will be the first dimension and the output array will have shape ``(N, A, B, C)``; if ``axis`` is ``1``, the new axis will be the second dimension and the output array will have shape ``(A, N, B, C)``; and, if ``axis`` is ``-1``, the new axis will be the last dimension and the output array will have shape ``(A, B, C, N)``. A valid ``axis`` must be on the interval ``[-N, N)``, where ``N`` is the rank (number of dimensions) of ``x``. If provided an ``axis`` outside of the required interval, the function must raise an exception. Default: ``0``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an output array having rank ``N+1``, where ``N`` is the rank (number of dimensions) of ``x``. If the input arrays have different data types, normal :ref:`type-promotion` must apply. If the input arrays have the same data type, the output array must have the same data type as the input arrays.

            .. note::
            This specification leaves type promotion between data type families (i.e., ``intxx`` and ``floatxx``) unspecified.
        """

    @abstractmethod
    def tile(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, repetitions: Tuple[int, ...], /
    ) -> _NAMESPACE_ARRAY:
        """
        Constructs an array by tiling an input array.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array.
        repetitions: Tuple[int, ...]
            number of repetitions along each axis (dimension).

            Let ``N = len(x.shape)`` and ``M = len(repetitions)``.

            If ``N > M``, the function must prepend ones until all axes (dimensions) are specified (e.g., if ``x`` has shape ``(8,6,4,2)`` and ``repetitions`` is the tuple ``(3,3)``, then ``repetitions`` must be treated as ``(1,1,3,3)``).

            If ``N < M``, the function must prepend singleton axes (dimensions) to ``x`` until ``x`` has as many axes (dimensions) as ``repetitions`` specifies (e.g., if ``x`` has shape ``(4,2)`` and ``repetitions`` is the tuple ``(3,3,3,3)``, then ``x`` must be treated as if it has shape ``(1,1,4,2)``).

        Returns
        -------
        out: _NAMESPACE_ARRAY
            a tiled output array. The returned array must have the same data type as ``x`` and must have a rank (i.e., number of dimensions) equal to ``max(N, M)``. If ``S`` is the shape of the tiled array after prepending singleton dimensions (if necessary) and ``r`` is the tuple of repetitions after prepending ones (if necessary), then the number of elements along each axis (dimension) must satisfy ``S[i]*r[i]``, where ``i`` refers to the ``i`` th axis (dimension).

        Notes
        -----

        .. versionadded:: 2023.12
        """

    @abstractmethod
    def unstack(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /, *, axis: int = 0
    ) -> Tuple[_NAMESPACE_ARRAY, ...]:
        """
        Splits an array into a sequence of arrays along the given axis.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array.
        axis: int
            axis along which the array will be split. A valid ``axis`` must be on the interval ``[-N, N)``, where ``N`` is the rank (number of dimensions) of ``x``. If provided an ``axis`` outside of the required interval, the function must raise an exception. Default: ``0``.

        Returns
        -------
        out: Tuple[array, ...]
            tuple of slices along the given dimension. All the arrays have the same shape.

        Notes
        -----

        .. versionadded:: 2023.12
        """

    """
    Searching Functions
    https://github.com/data-apis/array-api/blob/main/src/array_api_stubs/_2024_12/searching_functions.py
    """
    def argmax(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /, *, axis: Optional[int] = None, keepdims: bool = False
    ) -> _NAMESPACE_ARRAY:
        """
        Returns the indices of the maximum values along a specified axis.

        When the maximum value occurs multiple times, only the indices corresponding to the first occurrence are returned.

        .. note::
        For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-dependent (see :ref:`complex-number-ordering`).

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a real-valued data type.
        axis: Optional[int]
            axis along which to search. If ``None``, the function must return the index of the maximum value of the flattened array. Default: ``None``.
        keepdims: bool
            if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            if ``axis`` is ``None``, a zero-dimensional array containing the index of the first occurrence of the maximum value; otherwise, a non-zero-dimensional array containing the indices of the maximum values. The returned array must have be the default array index data type.
        """

    @abstractmethod
    def argmin(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /, *, axis: Optional[int] = None, keepdims: bool = False
    ) -> _NAMESPACE_ARRAY:
        """
        Returns the indices of the minimum values along a specified axis.

        When the minimum value occurs multiple times, only the indices corresponding to the first occurrence are returned.

        .. note::
        For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-dependent (see :ref:`complex-number-ordering`).

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a real-valued data type.
        axis: Optional[int]
            axis along which to search. If ``None``, the function must return the index of the minimum value of the flattened array. Default: ``None``.
        keepdims: bool
            if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            if ``axis`` is ``None``, a zero-dimensional array containing the index of the first occurrence of the minimum value; otherwise, a non-zero-dimensional array containing the indices of the minimum values. The returned array must have the default array index data type.
        """

    @abstractmethod
    def count_nonzero(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY,
        /,
        *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> _NAMESPACE_ARRAY:
        """
        Counts the number of array elements which are non-zero.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array.
        axis: Optional[Union[int, Tuple[int, ...]]]
            axis or axes along which to count non-zero values. By default, the number of non-zero values must be computed over the entire array. If a tuple of integers, the number of non-zero values must be computed over multiple axes. Default: ``None``.
        keepdims: bool
            if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            if the number of non-zeros values was computed over the entire array, a zero-dimensional array containing the total number of non-zero values; otherwise, a non-zero-dimensional array containing the counts along the specified axes. The returned array must have the default array index data type.

        Notes
        -----

        -   If ``x`` has a complex floating-point data type, non-zero elements are those elements having at least one component (real or imaginary) which is non-zero.
        -   If ``x`` has a boolean data type, non-zero elements are those elements which are equal to ``True``.

        .. versionadded:: 2024.12
        """

    @abstractmethod
    def nonzero(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> Tuple[_NAMESPACE_ARRAY, ...]:
        """
        Returns the indices of the array elements which are non-zero.

        .. admonition:: Data-dependent output shape
        :class: admonition important

        The shape of the output array for this function depends on the data values in the input array; hence, array libraries which build computation graphs (e.g., JAX, Dask, etc.) may find this function difficult to implement without knowing array values. Accordingly, such libraries may choose to omit this function. See :ref:`data-dependent-output-shapes` section for more details.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Must have a positive rank. If ``x`` is zero-dimensional, the function must raise an exception.

        Returns
        -------
        out: Tuple[array, ...]
            a tuple of ``k`` arrays, one for each dimension of ``x`` and each of size ``n`` (where ``n`` is the total number of non-zero elements), containing the indices of the non-zero elements in that dimension. The indices must be returned in row-major, C-style order. The returned array must have the default array index data type.

        Notes
        -----

        -   If ``x`` has a complex floating-point data type, non-zero elements are those elements having at least one component (real or imaginary) which is non-zero.
        -   If ``x`` has a boolean data type, non-zero elements are those elements which are equal to ``True``.

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def searchsorted(
        self: _NAMESPACE_C,
        x1: _NAMESPACE_ARRAY,
        x2: _NAMESPACE_ARRAY,
        /,
        *,
        side: Literal["left", "right"] = "left",
        sorter: Optional[_NAMESPACE_ARRAY] = None,
    ) -> _NAMESPACE_ARRAY:
        """
        Finds the indices into ``x1`` such that, if the corresponding elements in ``x2`` were inserted before the indices, the order of ``x1``, when sorted in ascending order, would be preserved.

        Parameters
        ----------
        x1: _NAMESPACE_ARRAY
            input array. Must be a one-dimensional array. Should have a real-valued data type. If ``sorter`` is ``None``, must be sorted in ascending order; otherwise, ``sorter`` must be an array of indices that sort ``x1`` in ascending order.
        x2: _NAMESPACE_ARRAY
            array containing search values. Should have a real-valued data type.
        side: Literal['left', 'right']
            argument controlling which index is returned if a value lands exactly on an edge.

            Let ``v`` be an element of ``x2`` given by ``v = x2[j]``, where ``j`` refers to a valid index (see :ref:`indexing`).

            - If ``v`` is less than all elements in ``x1``, then ``out[j]`` must be ``0``.
            - If ``v`` is greater than all elements in ``x1``, then ``out[j]`` must be ``M``, where ``M`` is the number of elements in ``x1``.
            - Otherwise, each returned index ``i = out[j]`` must satisfy an index condition:

            - If ``side == 'left'``, then ``x1[i-1] < v <= x1[i]``.
            - If ``side == 'right'``, then ``x1[i-1] <= v < x1[i]``.

            Default: ``'left'``.
        sorter: Optional[array]
            array of indices that sort ``x1`` in ascending order. The array must have the same shape as ``x1`` and have an integer data type. Default: ``None``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array of indices with the same shape as ``x2``. The returned array must have the default array index data type.

        Notes
        -----

        For real-valued floating-point arrays, the sort order of NaNs and signed zeros is unspecified and thus implementation-dependent. Accordingly, when a real-valued floating-point array contains NaNs and signed zeros, what constitutes ascending order may vary among specification-conforming array libraries.

        While behavior for arrays containing NaNs and signed zeros is implementation-dependent, specification-conforming libraries should, however, ensure consistency with ``sort`` and ``argsort`` (i.e., if a value in ``x2`` is inserted into ``x1`` according to the corresponding index in the output array and ``sort`` is invoked on the resultant array, the sorted result should be an array in the same order).

        .. versionadded:: 2023.12

        .. versionchanged:: 2024.12
        Fixed incorrect boundary conditions.
        """

    @abstractmethod
    def where(
        self: _NAMESPACE_C,
        condition: _NAMESPACE_ARRAY,
        x1: Union[_NAMESPACE_ARRAY, int, float, complex, bool],
        x2: Union[_NAMESPACE_ARRAY, int, float, complex, bool],
        /,
    ) -> _NAMESPACE_ARRAY:
        """
        Returns elements chosen from ``x1`` or ``x2`` depending on ``condition``.

        Parameters
        ----------
        condition: _NAMESPACE_ARRAY
            when ``True``, yield ``x1_i``; otherwise, yield ``x2_i``. Should have a boolean data type. Must be compatible with ``x1`` and ``x2`` (see :ref:`broadcasting`).
        x1: Union[array, int, float, complex, bool]
            first input array. Must be compatible with ``condition`` and ``x2`` (see :ref:`broadcasting`).
        x2: Union[array, int, float, complex, bool]
            second input array. Must be compatible with ``condition`` and ``x1`` (see :ref:`broadcasting`).

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array with elements from ``x1`` where ``condition`` is ``True``, and elements from ``x2`` elsewhere. The returned array must have a data type determined by :ref:`type-promotion` rules with the arrays ``x1`` and ``x2``.

        Notes
        -----

        -   At least one of  ``x1`` and ``x2`` must be an array.
        -   If either ``x1`` or ``x2`` is a scalar value, the returned array must have a data type determined according to :ref:`mixing-scalars-and-arrays`.

        .. versionchanged:: 2024.12
        Added scalar argument support.

        .. versionchanged:: 2024.12
        Clarified that the ``condition`` argument should have a boolean data type.
        """
    
    """
    Set Functions
    https://github.com/data-apis/array-api/blob/main/src/array_api_stubs/_2024_12/set_functions.py
    """
    @abstractmethod
    def unique_all(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> Tuple[_NAMESPACE_ARRAY, _NAMESPACE_ARRAY, _NAMESPACE_ARRAY, _NAMESPACE_ARRAY]:
        """
        Returns the unique elements of an input array ``x``, the first occurring indices for each unique element in ``x``, the indices from the set of unique elements that reconstruct ``x``, and the corresponding counts for each unique element in ``x``.

        .. admonition:: Data-dependent output shape
            :class: important

            The shapes of two of the output arrays for this function depend on the data values in the input array; hence, array libraries which build computation graphs (e.g., JAX, Dask, etc.) may find this function difficult to implement without knowing array values. Accordingly, such libraries may choose to omit this function. See :ref:`data-dependent-output-shapes` section for more details.

        .. note::
        Uniqueness should be determined based on value equality (see :func:`~array_api.equal`). For input arrays having floating-point data types, value-based equality implies the following behavior.

        -   As ``nan`` values compare as ``False``, ``nan`` values should be considered distinct.
        -   As complex floating-point values having at least one ``nan`` component compare as ``False``, complex floating-point values having ``nan`` components should be considered distinct.
        -   As ``-0`` and ``+0`` compare as ``True``, signed zeros should not be considered distinct, and the corresponding unique element will be implementation-dependent (e.g., an implementation could choose to return ``-0`` if ``-0`` occurs before ``+0``).

        As signed zeros are not distinct, using ``inverse_indices`` to reconstruct the input array is not guaranteed to return an array having the exact same values.

        Each ``nan`` value and each complex floating-point value having a ``nan`` component should have a count of one, while the counts for signed zeros should be aggregated as a single count.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. If ``x`` has more than one dimension, the function must flatten ``x`` and return the unique elements of the flattened array.

        Returns
        -------
        out: Tuple[array, array, array, array]
            a namedtuple ``(values, indices, inverse_indices, counts)`` whose

            - first element must have the field name ``values`` and must be a one-dimensional array containing the unique elements of ``x``. The array must have the same data type as ``x``.
            - second element must have the field name ``indices`` and must be an array containing the indices (first occurrences) of a flattened ``x`` that result in ``values``. The array must have the same shape as ``values`` and must have the default array index data type.
            - third element must have the field name ``inverse_indices`` and must be an array containing the indices of ``values`` that reconstruct ``x``. The array must have the same shape as ``x`` and must have the default array index data type.
            - fourth element must have the field name ``counts`` and must be an array containing the number of times each unique element occurs in ``x``. The order of the returned counts must match the order of ``values``, such that a specific element in ``counts`` corresponds to the respective unique element in ``values``. The returned array must have same shape as ``values`` and must have the default array index data type.

            .. note::
            The order of unique elements is not specified and may vary between implementations.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.

        .. versionchanged:: 2023.12
        Clarified flattening behavior and required the order of ``counts`` match the order of ``values``.
        """

    @abstractmethod
    def unique_counts(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> Tuple[_NAMESPACE_ARRAY, _NAMESPACE_ARRAY]:
        """
        Returns the unique elements of an input array ``x`` and the corresponding counts for each unique element in ``x``.

        .. admonition:: Data-dependent output shape
            :class: important

            The shapes of two of the output arrays for this function depend on the data values in the input array; hence, array libraries which build computation graphs (e.g., JAX, Dask, etc.) may find this function difficult to implement without knowing array values. Accordingly, such libraries may choose to omit this function. See :ref:`data-dependent-output-shapes` section for more details.

        .. note::
        Uniqueness should be determined based on value equality (see :func:`~array_api.equal`). For input arrays having floating-point data types, value-based equality implies the following behavior.

        -   As ``nan`` values compare as ``False``, ``nan`` values should be considered distinct.
        -   As complex floating-point values having at least one ``nan`` component compare as ``False``, complex floating-point values having ``nan`` components should be considered distinct.
        -   As ``-0`` and ``+0`` compare as ``True``, signed zeros should not be considered distinct, and the corresponding unique element will be implementation-dependent (e.g., an implementation could choose to return ``-0`` if ``-0`` occurs before ``+0``).

        Each ``nan`` value and each complex floating-point value having a ``nan`` component should have a count of one, while the counts for signed zeros should be aggregated as a single count.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. If ``x`` has more than one dimension, the function must flatten ``x`` and return the unique elements of the flattened array.

        Returns
        -------
        out: Tuple[array, array]
            a namedtuple `(values, counts)` whose

            -   first element must have the field name ``values`` and must be a one-dimensional array containing the unique elements of ``x``. The array must have the same data type as ``x``.
            -   second element must have the field name `counts` and must be an array containing the number of times each unique element occurs in ``x``. The order of the returned counts must match the order of ``values``, such that a specific element in ``counts`` corresponds to the respective unique element in ``values``. The returned array must have same shape as ``values`` and must have the default array index data type.

            .. note::
            The order of unique elements is not specified and may vary between implementations.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.

        .. versionchanged:: 2023.12
        Clarified flattening behavior and required the order of ``counts`` match the order of ``values``.
        """

    @abstractmethod
    def unique_inverse(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> Tuple[_NAMESPACE_ARRAY, _NAMESPACE_ARRAY]:
        """
        Returns the unique elements of an input array ``x`` and the indices from the set of unique elements that reconstruct ``x``.

        .. admonition:: Data-dependent output shape
            :class: important

            The shapes of two of the output arrays for this function depend on the data values in the input array; hence, array libraries which build computation graphs (e.g., JAX, Dask, etc.) may find this function difficult to implement without knowing array values. Accordingly, such libraries may choose to omit this function. See :ref:`data-dependent-output-shapes` section for more details.

        .. note::
        Uniqueness should be determined based on value equality (see :func:`~array_api.equal`). For input arrays having floating-point data types, value-based equality implies the following behavior.

        -   As ``nan`` values compare as ``False``, ``nan`` values should be considered distinct.
        -   As complex floating-point values having at least one ``nan`` component compare as ``False``, complex floating-point values having ``nan`` components should be considered distinct.
        -   As ``-0`` and ``+0`` compare as ``True``, signed zeros should not be considered distinct, and the corresponding unique element will be implementation-dependent (e.g., an implementation could choose to return ``-0`` if ``-0`` occurs before ``+0``).

        As signed zeros are not distinct, using ``inverse_indices`` to reconstruct the input array is not guaranteed to return an array having the exact same values.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. If ``x`` has more than one dimension, the function must flatten ``x`` and return the unique elements of the flattened array.

        Returns
        -------
        out: Tuple[array, array]
            a namedtuple ``(values, inverse_indices)`` whose

            -   first element must have the field name ``values`` and must be a one-dimensional array containing the unique elements of ``x``. The array must have the same data type as ``x``.
            -   second element must have the field name ``inverse_indices`` and must be an array containing the indices of ``values`` that reconstruct ``x``. The array must have the same shape as ``x`` and have the default array index data type.

            .. note::
            The order of unique elements is not specified and may vary between implementations.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.

        .. versionchanged:: 2023.12
        Clarified flattening behavior.
        """

    @abstractmethod
    def unique_values(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /
    ) -> _NAMESPACE_ARRAY:
        """
        Returns the unique elements of an input array ``x``.

        .. admonition:: Data-dependent output shape
            :class: important

            The shapes of two of the output arrays for this function depend on the data values in the input array; hence, array libraries which build computation graphs (e.g., JAX, Dask, etc.) may find this function difficult to implement without knowing array values. Accordingly, such libraries may choose to omit this function. See :ref:`data-dependent-output-shapes` section for more details.

        .. note::
        Uniqueness should be determined based on value equality (see :func:`~array_api.equal`). For input arrays having floating-point data types, value-based equality implies the following behavior.

        -   As ``nan`` values compare as ``False``, ``nan`` values should be considered distinct.
        -   As complex floating-point values having at least one ``nan`` component compare as ``False``, complex floating-point values having ``nan`` components should be considered distinct.
        -   As ``-0`` and ``+0`` compare as ``True``, signed zeros should not be considered distinct, and the corresponding unique element will be implementation-dependent (e.g., an implementation could choose to return ``-0`` if ``-0`` occurs before ``+0``).

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. If ``x`` has more than one dimension, the function must flatten ``x`` and return the unique elements of the flattened array.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            a one-dimensional array containing the set of unique elements in ``x``. The returned array must have the same data type as ``x``.

            .. note::
            The order of unique elements is not specified and may vary between implementations.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.

        .. versionchanged:: 2023.12
        Required that the output array must be one-dimensional.
        """

    """
    Sorting Functions
    https://github.com/data-apis/array-api/blob/main/src/array_api_stubs/_2024_12/sorting_functions.py
    """
    @abstractmethod
    def argsort(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /, *, axis: int = -1, descending: bool = False, stable: bool = True
    ) -> _NAMESPACE_ARRAY:
        """
        Returns the indices that sort an array ``x`` along a specified axis.

        .. note::
        For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-dependent (see :ref:`complex-number-ordering`).

        Parameters
        ----------
        x : _NAMESPACE_ARRAY
            input array. Should have a real-valued data type.
        axis: int
            axis along which to sort. If set to ``-1``, the function must sort along the last axis. Default: ``-1``.
        descending: bool
            sort order. If ``True``, the returned indices sort ``x`` in descending order (by value). If ``False``, the returned indices sort ``x`` in ascending order (by value). Default: ``False``.
        stable: bool
            sort stability. If ``True``, the returned indices must maintain the relative order of ``x`` values which compare as equal. If ``False``, the returned indices may or may not maintain the relative order of ``x`` values which compare as equal (i.e., the relative order of ``x`` values which compare as equal is implementation-dependent). Default: ``True``.

        Returns
        -------
        out : _NAMESPACE_ARRAY
            an array of indices. The returned array must have the same shape as ``x``. The returned array must have the default array index data type.
        """

    @abstractmethod
    def sort(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /, *, axis: int = -1, descending: bool = False, stable: bool = True
    ) -> _NAMESPACE_ARRAY:
        """
        Returns a sorted copy of an input array ``x``.

        .. note::
        For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-dependent (see :ref:`complex-number-ordering`).

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a real-valued data type.
        axis: int
            axis along which to sort. If set to ``-1``, the function must sort along the last axis. Default: ``-1``.
        descending: bool
            sort order. If ``True``, the array must be sorted in descending order (by value). If ``False``, the array must be sorted in ascending order (by value). Default: ``False``.
        stable: bool
            sort stability. If ``True``, the returned array must maintain the relative order of ``x`` values which compare as equal. If ``False``, the returned array may or may not maintain the relative order of ``x`` values which compare as equal (i.e., the relative order of ``x`` values which compare as equal is implementation-dependent). Default: ``True``.

        Returns
        -------
        out : _NAMESPACE_ARRAY
            a sorted array. The returned array must have the same data type and shape as ``x``.
        """

    """
    Statistical Functions
    https://github.com/data-apis/array-api/blob/main/src/array_api_stubs/_2024_12/statistical_functions.py
    """
    @abstractmethod
    def cumulative_prod(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY,
        /,
        *,
        axis: Optional[int] = None,
        dtype: Optional[_NAMESPACE_DTYPE] = None,
        include_initial: bool = False,
    ) -> _NAMESPACE_ARRAY:
        """
        Calculates the cumulative product of elements in the input array ``x``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have one or more dimensions (axes). Should have a numeric data type.
        axis: Optional[int]
            axis along which a cumulative product must be computed. If ``axis`` is negative, the function must determine the axis along which to compute a cumulative product by counting from the last dimension.

            If ``x`` is a one-dimensional array, providing an ``axis`` is optional; however, if ``x`` has more than one dimension, providing an ``axis`` is required.

        dtype: Optional[dtype]
            data type of the returned array. If ``None``, the returned array must have the same data type as ``x``, unless ``x`` has an integer data type supporting a smaller range of values than the default integer data type (e.g., ``x`` has an ``int16`` or ``uint32`` data type and the default integer data type is ``int64``). In those latter cases:

            -   if ``x`` has a signed integer data type (e.g., ``int16``), the returned array must have the default integer data type.
            -   if ``x`` has an unsigned integer data type (e.g., ``uint16``), the returned array must have an unsigned integer data type having the same number of bits as the default integer data type (e.g., if the default integer data type is ``int32``, the returned array must have a ``uint32`` data type).

            If the data type (either specified or resolved) differs from the data type of ``x``, the input array should be cast to the specified data type before computing the product (rationale: the ``dtype`` keyword argument is intended to help prevent overflows). Default: ``None``.

        include_initial: bool
            boolean indicating whether to include the initial value as the first value in the output. By convention, the initial value must be the multiplicative identity (i.e., one). Default: ``False``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the cumulative products. The returned array must have a data type as described by the ``dtype`` parameter above.

            Let ``N`` be the size of the axis along which to compute the cumulative product. The returned array must have a shape determined according to the following rules:

            -   if ``include_initial`` is ``True``, the returned array must have the same shape as ``x``, except the size of the axis along which to compute the cumulative product must be ``N+1``.
            -   if ``include_initial`` is ``False``, the returned array must have the same shape as ``x``.

        Notes
        -----

        -   When ``x`` is a zero-dimensional array, behavior is unspecified and thus implementation-defined.

        **Special Cases**

        For both real-valued and complex floating-point operands, special cases must be handled as if the operation is implemented by successive application of :func:`~array_api.multiply`.

        .. versionadded:: 2024.12
        """

    @abstractmethod
    def cumulative_sum(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY,
        /,
        *,
        axis: Optional[int] = None,
        dtype: Optional[_NAMESPACE_DTYPE] = None,
        include_initial: bool = False,
    ) -> _NAMESPACE_ARRAY:
        """
        Calculates the cumulative sum of elements in the input array ``x``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have one or more dimensions (axes). Should have a numeric data type.
        axis: Optional[int]
            axis along which a cumulative sum must be computed. If ``axis`` is negative, the function must determine the axis along which to compute a cumulative sum by counting from the last dimension.

            If ``x`` is a one-dimensional array, providing an ``axis`` is optional; however, if ``x`` has more than one dimension, providing an ``axis`` is required.

        dtype: Optional[dtype]
            data type of the returned array. If ``None``, the returned array must have the same data type as ``x``, unless ``x`` has an integer data type supporting a smaller range of values than the default integer data type (e.g., ``x`` has an ``int16`` or ``uint32`` data type and the default integer data type is ``int64``). In those latter cases:

            -   if ``x`` has a signed integer data type (e.g., ``int16``), the returned array must have the default integer data type.
            -   if ``x`` has an unsigned integer data type (e.g., ``uint16``), the returned array must have an unsigned integer data type having the same number of bits as the default integer data type (e.g., if the default integer data type is ``int32``, the returned array must have a ``uint32`` data type).

            If the data type (either specified or resolved) differs from the data type of ``x``, the input array should be cast to the specified data type before computing the sum (rationale: the ``dtype`` keyword argument is intended to help prevent overflows). Default: ``None``.

        include_initial: bool
            boolean indicating whether to include the initial value as the first value in the output. By convention, the initial value must be the additive identity (i.e., zero). Default: ``False``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the cumulative sums. The returned array must have a data type as described by the ``dtype`` parameter above.

            Let ``N`` be the size of the axis along which to compute the cumulative sum. The returned array must have a shape determined according to the following rules:

            -   if ``include_initial`` is ``True``, the returned array must have the same shape as ``x``, except the size of the axis along which to compute the cumulative sum must be ``N+1``.
            -   if ``include_initial`` is ``False``, the returned array must have the same shape as ``x``.

        Notes
        -----

        -   When ``x`` is a zero-dimensional array, behavior is unspecified and thus implementation-defined.

        **Special Cases**

        For both real-valued and complex floating-point operands, special cases must be handled as if the operation is implemented by successive application of :func:`~array_api.add`.

        .. versionadded:: 2023.12

        .. versionchanged:: 2024.12
        Behavior when providing a zero-dimensional array is explicitly left unspecified.
        """

    @abstractmethod
    def max(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY,
        /,
        *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> _NAMESPACE_ARRAY:
        """
        Calculates the maximum value of the input array ``x``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a real-valued data type.
        axis: Optional[Union[int, Tuple[int, ...]]]
            axis or axes along which maximum values must be computed. By default, the maximum value must be computed over the entire array. If a tuple of integers, maximum values must be computed over multiple axes. Default: ``None``.
        keepdims: bool
            if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            if the maximum value was computed over the entire array, a zero-dimensional array containing the maximum value; otherwise, a non-zero-dimensional array containing the maximum values. The returned array must have the same data type as ``x``.

        Notes
        -----

        When the number of elements over which to compute the maximum value is zero, the maximum value is implementation-defined. Specification-compliant libraries may choose to raise an error, return a sentinel value (e.g., if ``x`` is a floating-point input array, return ``NaN``), or return the minimum possible value for the input array ``x`` data type (e.g., if ``x`` is a floating-point array, return ``-infinity``).

        The order of signed zeros is unspecified and thus implementation-defined. When choosing between ``-0`` or ``+0`` as a maximum value, specification-compliant libraries may choose to return either value.

        For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-defined (see :ref:`complex-number-ordering`).

        **Special Cases**

        For floating-point operands,

        -   If ``x_i`` is ``NaN``, the maximum value is ``NaN`` (i.e., ``NaN`` values propagate).

        .. versionchanged:: 2023.12
        Clarified that the order of signed zeros is implementation-defined.
        """

    @abstractmethod
    def mean(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY,
        /,
        *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> _NAMESPACE_ARRAY:
        """
        Calculates the arithmetic mean of the input array ``x``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a floating-point data type.
        axis: Optional[Union[int, Tuple[int, ...]]]
            axis or axes along which arithmetic means must be computed. By default, the mean must be computed over the entire array. If a tuple of integers, arithmetic means must be computed over multiple axes. Default: ``None``.
        keepdims: bool
            if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            if the arithmetic mean was computed over the entire array, a zero-dimensional array containing the arithmetic mean; otherwise, a non-zero-dimensional array containing the arithmetic means. The returned array must have the same data type as ``x``.

            .. note::
            While this specification recommends that this function only accept input arrays having a floating-point data type, specification-compliant array libraries may choose to accept input arrays having an integer data type. While mixed data type promotion is implementation-defined, if the input array ``x`` has an integer data type, the returned array must have the default real-valued floating-point data type.

        Notes
        -----

        **Special Cases**

        Let ``N`` equal the number of elements over which to compute the arithmetic mean. For real-valued operands,

        -   If ``N`` is ``0``, the arithmetic mean is ``NaN``.
        -   If ``x_i`` is ``NaN``, the arithmetic mean is ``NaN`` (i.e., ``NaN`` values propagate).

        For complex floating-point operands, real-valued floating-point special cases should independently apply to the real and imaginary component operations involving real numbers. For example, let ``a = real(x_i)`` and ``b = imag(x_i)``, and

        -   If ``N`` is ``0``, the arithmetic mean is ``NaN + NaN j``.
        -   If ``a`` is ``NaN``, the real component of the result is ``NaN``.
        -   Similarly, if ``b`` is ``NaN``, the imaginary component of the result is ``NaN``.

        .. note::
        Array libraries, such as NumPy, PyTorch, and JAX, currently deviate from this specification in their handling of components which are ``NaN`` when computing the arithmetic mean. In general, consumers of array libraries implementing this specification should use :func:`~array_api.isnan` to test whether the result of computing the arithmetic mean over an array have a complex floating-point data type is ``NaN``, rather than relying on ``NaN`` propagation of individual components.

        .. versionchanged:: 2024.12
        Added complex data type support.
        """

    @abstractmethod
    def min(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY,
        /,
        *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> _NAMESPACE_ARRAY:
        """
        Calculates the minimum value of the input array ``x``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a real-valued data type.
        axis: Optional[Union[int, Tuple[int, ...]]]
            axis or axes along which minimum values must be computed. By default, the minimum value must be computed over the entire array. If a tuple of integers, minimum values must be computed over multiple axes. Default: ``None``.
        keepdims: bool
            if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            if the minimum value was computed over the entire array, a zero-dimensional array containing the minimum value; otherwise, a non-zero-dimensional array containing the minimum values. The returned array must have the same data type as ``x``.

        Notes
        -----

        When the number of elements over which to compute the minimum value is zero, the minimum value is implementation-defined. Specification-compliant libraries may choose to raise an error, return a sentinel value (e.g., if ``x`` is a floating-point input array, return ``NaN``), or return the maximum possible value for the input array ``x`` data type (e.g., if ``x`` is a floating-point array, return ``+infinity``).

        The order of signed zeros is unspecified and thus implementation-defined. When choosing between ``-0`` or ``+0`` as a minimum value, specification-compliant libraries may choose to return either value.

        For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-defined (see :ref:`complex-number-ordering`).

        **Special Cases**

        For floating-point operands,

        -   If ``x_i`` is ``NaN``, the minimum value is ``NaN`` (i.e., ``NaN`` values propagate).

        .. versionchanged:: 2023.12
        Clarified that the order of signed zeros is implementation-defined.
        """

    @abstractmethod
    def prod(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY,
        /,
        *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        dtype: Optional[_NAMESPACE_DTYPE] = None,
        keepdims: bool = False,
    ) -> _NAMESPACE_ARRAY:
        """
        Calculates the product of input array ``x`` elements.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a numeric data type.
        axis: Optional[Union[int, Tuple[int, ...]]]
            axis or axes along which products must be computed. By default, the product must be computed over the entire array. If a tuple of integers, products must be computed over multiple axes. Default: ``None``.

        dtype: Optional[dtype]
            data type of the returned array. If ``None``, the returned array must have the same data type as ``x``, unless ``x`` has an integer data type supporting a smaller range of values than the default integer data type (e.g., ``x`` has an ``int16`` or ``uint32`` data type and the default integer data type is ``int64``). In those latter cases:

            -   if ``x`` has a signed integer data type (e.g., ``int16``), the returned array must have the default integer data type.
            -   if ``x`` has an unsigned integer data type (e.g., ``uint16``), the returned array must have an unsigned integer data type having the same number of bits as the default integer data type (e.g., if the default integer data type is ``int32``, the returned array must have a ``uint32`` data type).

            If the data type (either specified or resolved) differs from the data type of ``x``, the input array should be cast to the specified data type before computing the sum (rationale: the ``dtype`` keyword argument is intended to help prevent overflows). Default: ``None``.

        keepdims: bool
            if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            if the product was computed over the entire array, a zero-dimensional array containing the product; otherwise, a non-zero-dimensional array containing the products. The returned array must have a data type as described by the ``dtype`` parameter above.

        Notes
        -----

        **Special Cases**

        Let ``N`` equal the number of elements over which to compute the product.

        -   If ``N`` is ``0``, the product is `1` (i.e., the empty product).

        For both real-valued and complex floating-point operands, special cases must be handled as if the operation is implemented by successive application of :func:`~array_api.multiply`.

        .. versionchanged:: 2022.12
        Added complex data type support.

        .. versionchanged:: 2023.12
        Required the function to return a floating-point array having the same data type as the input array when provided a floating-point array.
        """

    @abstractmethod
    def std(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY,
        /,
        *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False,
    ) -> _NAMESPACE_ARRAY:
        """
        Calculates the standard deviation of the input array ``x``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a real-valued floating-point data type.
        axis: Optional[Union[int, Tuple[int, ...]]]
            axis or axes along which standard deviations must be computed. By default, the standard deviation must be computed over the entire array. If a tuple of integers, standard deviations must be computed over multiple axes. Default: ``None``.
        correction: Union[int, float]
            degrees of freedom adjustment. Setting this parameter to a value other than ``0`` has the effect of adjusting the divisor during the calculation of the standard deviation according to ``N-c`` where ``N`` corresponds to the total number of elements over which the standard deviation is computed and ``c`` corresponds to the provided degrees of freedom adjustment. When computing the standard deviation of a population, setting this parameter to ``0`` is the standard choice (i.e., the provided array contains data constituting an entire population). When computing the corrected sample standard deviation, setting this parameter to ``1`` is the standard choice (i.e., the provided array contains data sampled from a larger population; this is commonly referred to as Bessel's correction). Default: ``0``.
        keepdims: bool
            if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            if the standard deviation was computed over the entire array, a zero-dimensional array containing the standard deviation; otherwise, a non-zero-dimensional array containing the standard deviations. The returned array must have the same data type as ``x``.

            .. note::
            While this specification recommends that this function only accept input arrays having a real-valued floating-point data type, specification-compliant array libraries may choose to accept input arrays having an integer data type. While mixed data type promotion is implementation-defined, if the input array ``x`` has an integer data type, the returned array must have the default real-valued floating-point data type.

        Notes
        -----

        **Special Cases**

        Let ``N`` equal the number of elements over which to compute the standard deviation.

        -   If ``N - correction`` is less than or equal to ``0``, the standard deviation is ``NaN``.
        -   If ``x_i`` is ``NaN``, the standard deviation is ``NaN`` (i.e., ``NaN`` values propagate).
        """

    @abstractmethod
    def sum(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY,
        /,
        *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        dtype: Optional[_NAMESPACE_DTYPE] = None,
        keepdims: bool = False,
    ) -> _NAMESPACE_ARRAY:
        """
        Calculates the sum of the input array ``x``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a numeric data type.
        axis: Optional[Union[int, Tuple[int, ...]]]
            axis or axes along which sums must be computed. By default, the sum must be computed over the entire array. If a tuple of integers, sums must be computed over multiple axes. Default: ``None``.

        dtype: Optional[dtype]
            data type of the returned array. If ``None``, the returned array must have the same data type as ``x``, unless ``x`` has an integer data type supporting a smaller range of values than the default integer data type (e.g., ``x`` has an ``int16`` or ``uint32`` data type and the default integer data type is ``int64``). In those latter cases:

            -   if ``x`` has a signed integer data type (e.g., ``int16``), the returned array must have the default integer data type.
            -   if ``x`` has an unsigned integer data type (e.g., ``uint16``), the returned array must have an unsigned integer data type having the same number of bits as the default integer data type (e.g., if the default integer data type is ``int32``, the returned array must have a ``uint32`` data type).

            If the data type (either specified or resolved) differs from the data type of ``x``, the input array should be cast to the specified data type before computing the sum (rationale: the ``dtype`` keyword argument is intended to help prevent overflows). Default: ``None``.

        keepdims: bool
            if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            if the sum was computed over the entire array, a zero-dimensional array containing the sum; otherwise, an array containing the sums. The returned array must have a data type as described by the ``dtype`` parameter above.

        Notes
        -----

        **Special Cases**

        Let ``N`` equal the number of elements over which to compute the sum.

        -   If ``N`` is ``0``, the sum is ``0`` (i.e., the empty sum).

        For both real-valued and complex floating-point operands, special cases must be handled as if the operation is implemented by successive application of :func:`~array_api.add`.

        .. versionchanged:: 2022.12
        Added complex data type support.

        .. versionchanged:: 2023.12
        Required the function to return a floating-point array having the same data type as the input array when provided a floating-point array.
        """

    @abstractmethod
    def var(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY,
        /,
        *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False,
    ) -> _NAMESPACE_ARRAY:
        """
        Calculates the variance of the input array ``x``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a real-valued floating-point data type.
        axis: Optional[Union[int, Tuple[int, ...]]]
            axis or axes along which variances must be computed. By default, the variance must be computed over the entire array. If a tuple of integers, variances must be computed over multiple axes. Default: ``None``.
        correction: Union[int, float]
            degrees of freedom adjustment. Setting this parameter to a value other than ``0`` has the effect of adjusting the divisor during the calculation of the variance according to ``N-c`` where ``N`` corresponds to the total number of elements over which the variance is computed and ``c`` corresponds to the provided degrees of freedom adjustment. When computing the variance of a population, setting this parameter to ``0`` is the standard choice (i.e., the provided array contains data constituting an entire population). When computing the unbiased sample variance, setting this parameter to ``1`` is the standard choice (i.e., the provided array contains data sampled from a larger population; this is commonly referred to as Bessel's correction). Default: ``0``.
        keepdims: bool
            if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            if the variance was computed over the entire array, a zero-dimensional array containing the variance; otherwise, a non-zero-dimensional array containing the variances. The returned array must have the same data type as ``x``.


        .. note::
        While this specification recommends that this function only accept input arrays having a real-valued floating-point data type, specification-compliant array libraries may choose to accept input arrays having an integer data type. While mixed data type promotion is implementation-defined, if the input array ``x`` has an integer data type, the returned array must have the default real-valued floating-point data type.

        Notes
        -----

        **Special Cases**

        Let ``N`` equal the number of elements over which to compute the variance.

        -   If ``N - correction`` is less than or equal to ``0``, the variance is ``NaN``.
        -   If ``x_i`` is ``NaN``, the variance is ``NaN`` (i.e., ``NaN`` values propagate).
        """
    
    """
    Utility Functions
    https://github.com/data-apis/array-api/blob/main/src/array_api_stubs/_2024_12/utility_functions.py
    """
    @abstractmethod
    def all(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY,
        /,
        *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> _NAMESPACE_ARRAY:
        """
        Tests whether all input array elements evaluate to ``True`` along a specified axis.

        .. note::
        Positive infinity, negative infinity, and NaN must evaluate to ``True``.

        .. note::
        If ``x`` has a complex floating-point data type, elements having a non-zero component (real or imaginary) must evaluate to ``True``.

        .. note::
        If ``x`` is an empty array or the size of the axis (dimension) along which to evaluate elements is zero, the test result must be ``True``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array.
        axis: Optional[Union[int, Tuple[int, ...]]]
            axis or axes along which to perform a logical AND reduction. By default, a logical AND reduction must be performed over the entire array. If a tuple of integers, logical AND reductions must be performed over multiple axes. A valid ``axis`` must be an integer on the interval ``[-N, N)``, where ``N`` is the rank (number of dimensions) of ``x``. If an ``axis`` is specified as a negative integer, the function must determine the axis along which to perform a reduction by counting backward from the last dimension (where ``-1`` refers to the last dimension). If provided an invalid ``axis``, the function must raise an exception. Default: ``None``.
        keepdims: bool
            If ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            if a logical AND reduction was performed over the entire array, the returned array must be a zero-dimensional array containing the test result; otherwise, the returned array must be a non-zero-dimensional array containing the test results. The returned array must have a data type of ``bool``.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def any(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY,
        /,
        *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> _NAMESPACE_ARRAY:
        """
        Tests whether any input array element evaluates to ``True`` along a specified axis.

        .. note::
        Positive infinity, negative infinity, and NaN must evaluate to ``True``.

        .. note::
        If ``x`` has a complex floating-point data type, elements having a non-zero component (real or imaginary) must evaluate to ``True``.

        .. note::
        If ``x`` is an empty array or the size of the axis (dimension) along which to evaluate elements is zero, the test result must be ``False``.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array.
        axis: Optional[Union[int, Tuple[int, ...]]]
            axis or axes along which to perform a logical OR reduction. By default, a logical OR reduction must be performed over the entire array. If a tuple of integers, logical OR reductions must be performed over multiple axes. A valid ``axis`` must be an integer on the interval ``[-N, N)``, where ``N`` is the rank (number of dimensions) of ``x``. If an ``axis`` is specified as a negative integer, the function must determine the axis along which to perform a reduction by counting backward from the last dimension (where ``-1`` refers to the last dimension). If provided an invalid ``axis``, the function must raise an exception. Default: ``None``.
        keepdims: bool
            If ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result. Default: ``False``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            if a logical OR reduction was performed over the entire array, the returned array must be a zero-dimensional array containing the test result; otherwise, the returned array must be a non-zero-dimensional array containing the test results. The returned array must have a data type of ``bool``.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.
        """

    @abstractmethod
    def diff(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY,
        /,
        *,
        axis: int = -1,
        n: int = 1,
        prepend: Optional[_NAMESPACE_ARRAY] = None,
        append: Optional[_NAMESPACE_ARRAY] = None,
    ) -> _NAMESPACE_ARRAY:
        """
        Calculates the n-th discrete forward difference along a specified axis.

        Parameters
        ----------
        x: _NAMESPACE_ARRAY
            input array. Should have a numeric data type.
        axis: int
            axis along which to compute differences. A valid ``axis`` must be an integer on the interval ``[-N, N)``, where ``N`` is the rank (number of dimensions) of ``x``. If an ``axis`` is specified as a negative integer, the function must determine the axis along which to compute differences by counting backward from the last dimension (where ``-1`` refers to the last dimension). If provided an invalid ``axis``, the function must raise an exception. Default: ``-1``.
        n: int
            number of times to recursively compute differences. Default: ``1``.
        prepend: Optional[array]
            values to prepend to a specified axis prior to computing differences. Must have the same shape as ``x``, except for the axis specified by ``axis`` which may have any size. Should have the same data type as ``x``. Default: ``None``.
        append: Optional[array]
            values to append to a specified axis prior to computing differences. Must have the same shape as ``x``, except for the axis specified by ``axis`` which may have any size. Should have the same data type as ``x``. Default: ``None``.

        Returns
        -------
        out: _NAMESPACE_ARRAY
            an array containing the n-th differences. Should have the same data type as ``x``. Must have the same shape as ``x``, except for the axis specified by ``axis`` which must have a size determined as follows:

            -   Let ``M`` be the number of elements along an axis specified by ``axis``.
            -   Let ``N1`` be the number of prepended values along an axis specified by ``axis``.
            -   Let ``N2`` be the number of appended values along an axis specified by ``axis``.
            -   The final size of the axis specified by ``axis`` must be ``M + N1 + N2 - n``.

        Notes
        -----

        -   The first-order differences are given by ``out[i] = x[i+1] - x[i]`` along a specified axis. Higher-order differences must be calculated recursively (e.g., by calling ``diff(out, axis=axis, n=n-1)``).
        -   If a conforming implementation chooses to support ``prepend`` and ``append`` arrays which have a different data type than ``x``, behavior is unspecified and thus implementation-defined. Implementations may choose to type promote (:ref:`type-promotion`), cast ``prepend`` and/or ``append`` to the same data type as ``x``, or raise an exception.

        .. versionadded:: 2024.12
        """