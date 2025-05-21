from typing import Protocol, TypeVar, Optional, Any, Tuple, Union, Type, TypedDict, List, Literal
from abc import abstractmethod
from dataclasses import dataclass
from ._array_typing import *
from _api_return_typing import *

_NAMESPACE_C = TypeVar("_NAMESPACE_C", bound="ArrayAPINamespace")
_NAMESPACE_ARRAY = TypeVar("_NAMESPACE_ARRAY", bound=Array)
_NAMESPACE_DEVICE = TypeVar("_NAMESPACE_DEVICE", bound=Device)
_NAMESPACE_DTYPE = TypeVar("_NAMESPACE_DTYPE", bound=DType)
class ArrayAPINamespace(Protocol[_NAMESPACE_ARRAY, _NAMESPACE_DEVICE, _NAMESPACE_DTYPE]):
    """
    Constants
    https://github.com/data-apis/array-api/blob/main/src/array_api_stubs/_2024_12/constants.py
    """
    @property
    def e(self : _NAMESPACE_C) -> float:
        return 2.718281828459045

    @property
    def inf(self : _NAMESPACE_C) -> float:
        return float("inf")
    
    @property
    def nan(self : _NAMESPACE_C) -> float:
        return float("nan")
    
    @property
    def newaxis(self : _NAMESPACE_C) -> None:
        return None
    
    @property
    def pi(self : _NAMESPACE_C) -> float:
        return 3.141592653589793
    
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
        out: array
            a one-dimensional array containing evenly spaced values. The length of the output array must be ``ceil((stop-start)/step)`` if ``stop - start`` and ``step`` have the same sign, and length ``0`` otherwise.
        """
        raise NotImplementedError


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
        out: array
            an array containing the data from ``obj``.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.
        """
        raise NotImplementedError


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
        out: array
            an array containing uninitialized data.
        """
        raise NotImplementedError

    @abstractmethod
    def empty_like(
        self : _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /, *, dtype: Optional[_NAMESPACE_DTYPE] = None, device: Optional[_NAMESPACE_DEVICE] = None
    ) -> _NAMESPACE_DEVICE:
        """
        Returns an uninitialized array with the same ``shape`` as an input array ``x``.

        Parameters
        ----------
        x: array
            input array from which to derive the output array shape.
        dtype: Optional[dtype]
            output array data type. If ``dtype`` is ``None``, the output array data type must be inferred from ``x``. Default: ``None``.
        device: Optional[device]
            device on which to place the created array. If ``device`` is ``None``, the output array device must be inferred from ``x``. Default: ``None``.

        Returns
        -------
        out: array
            an array having the same shape as ``x`` and containing uninitialized data.
        """
        raise NotImplementedError

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
        out: array
            an array where all elements are equal to zero, except for the ``k``\th diagonal, whose values are equal to one.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.
        """
        raise NotImplementedError

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
        out: array
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
        raise NotImplementedError

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
        out: array
            an array where every element is equal to ``fill_value``.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.
        """
        raise NotImplementedError

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
        x: array
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
        out: array
            an array having the same shape as ``x`` and where every element is equal to ``fill_value``.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.
        """
        raise NotImplementedError

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
        out: array
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
        raise NotImplementedError

    @abstractmethod
    def meshgrid(
        self: _NAMESPACE_C,
        *arrays: _NAMESPACE_ARRAY, indexing: Literal["xy", "ij"] = "xy"
    ) -> List[_NAMESPACE_ARRAY]:
        """
        Returns coordinate matrices from coordinate vectors.

        Parameters
        ----------
        arrays: array
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
        raise NotImplementedError

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
        out: array
            an array containing ones.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.
        """
        raise NotImplementedError

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
        x: array
            input array from which to derive the output array shape.
        dtype: Optional[dtype]
            output array data type. If ``dtype`` is ``None``, the output array data type must be inferred from ``x``. Default: ``None``.
        device: Optional[device]
            device on which to place the created array. If ``device`` is ``None``, the output array device must be inferred from ``x``. Default: ``None``.

        Returns
        -------
        out: array
            an array having the same shape as ``x`` and filled with ones.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.
        """
        raise NotImplementedError

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
        x: array
            input array having shape ``(..., M, N)`` and whose innermost two dimensions form ``MxN`` matrices.
        k: int
            diagonal above which to zero elements. If ``k = 0``, the diagonal is the main diagonal. If ``k < 0``, the diagonal is below the main diagonal. If ``k > 0``, the diagonal is above the main diagonal. Default: ``0``.

            .. note::
            The main diagonal is defined as the set of indices ``{(i, i)}`` for ``i`` on the interval ``[0, min(M, N) - 1]``.

        Returns
        -------
        out: array
            an array containing the lower triangular part(s). The returned array must have the same shape and data type as ``x``. All elements above the specified diagonal ``k`` must be zeroed. The returned array should be allocated on the same device as ``x``.
        """
        raise NotImplementedError

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
        x: array
            input array having shape ``(..., M, N)`` and whose innermost two dimensions form ``MxN`` matrices.
        k: int
            diagonal below which to zero elements. If ``k = 0``, the diagonal is the main diagonal. If ``k < 0``, the diagonal is below the main diagonal. If ``k > 0``, the diagonal is above the main diagonal. Default: ``0``.

            .. note::
            The main diagonal is defined as the set of indices ``{(i, i)}`` for ``i`` on the interval ``[0, min(M, N) - 1]``.

        Returns
        -------
        out: array
            an array containing the upper triangular part(s). The returned array must have the same shape and data type as ``x``. All elements below the specified diagonal ``k`` must be zeroed. The returned array should be allocated on the same device as ``x``.
        """
        raise NotImplementedError

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
        out: array
            an array containing zeros.
        """
        raise NotImplementedError

    @abstractmethod
    def zeros_like(
        self: _NAMESPACE_C,
        x: _NAMESPACE_ARRAY, /, *, dtype: Optional[_NAMESPACE_DTYPE] = None, device: Optional[_NAMESPACE_DEVICE] = None
    ) -> _NAMESPACE_ARRAY:
        """
        Returns a new array filled with zeros and having the same ``shape`` as an input array ``x``.

        Parameters
        ----------
        x: array
            input array from which to derive the output array shape.
        dtype: Optional[dtype]
            output array data type. If ``dtype`` is ``None``, the output array data type must be inferred from ``x``. Default: ``None``.
        device: Optional[device]
            device on which to place the created array. If ``device`` is ``None``, the output array device must be inferred from ``x``. Default: ``None``.

        Returns
        -------
        out: array
            an array having the same shape as ``x`` and filled with zeros.
        """
        raise NotImplementedError

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
        x: array
            array to cast.
        dtype: dtype
            desired data type.
        copy: bool
            specifies whether to copy an array when the specified ``dtype`` matches the data type of the input array ``x``. If ``True``, a newly allocated array must always be returned (see :ref:`copy-keyword-argument`). If ``False`` and the specified ``dtype`` matches the data type of the input array, the input array must be returned; otherwise, a newly allocated array must be returned. Default: ``True``.
        device: Optional[device]
            device on which to place the returned array. If ``device`` is ``None``, the output array device must be inferred from ``x``. Default: ``None``.

        Returns
        -------
        out: array
            an array having the specified data type. The returned array must have the same shape as ``x``.

        Notes
        -----

        .. versionchanged:: 2022.12
        Added complex data type support.

        .. versionchanged:: 2023.12
        Added device keyword argument support.
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError