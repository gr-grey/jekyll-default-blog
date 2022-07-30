---
layout: "post"
title: "My Python Cheat Sheet"
date: 2022-07-29 19:00:00
categories: coding tools
---

Python is a wonderful language with powerful libraries such as numpy, pandas and matplotlib. Meanwhile, the vast number of functions and features might be hard to keep track of. 
Here's a cheat sheet covering basic python, numpy and pandas functions.

I wrote it more as a review and quick searching page for myself, so some descriptions might be vague if you are not familiar with python.
When in doubt, look to the official documentations of python and libraries for a full description!

Documentations:

- Python: [https://docs.python.org/3/](https://docs.python.org/3/)

- Numpy: [https://numpy.org/doc/stable/](https://numpy.org/doc/stable/)

- Pandas: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)

### Python built-in functions and callables 

- **`len(obj)`** return the number of items
- **`type(obj)`** return the type of an object
- **`dir(obj)`** return list of name in the current scope
- **`sum(iterable, /, start=0)`** return summation of iterables
- **`max/min(iterable, *[, key, default])`** return maximum/minimum number of iterables
- **`pow(base, exp[, mod])`** return base to the power exp, mod makes a simpler way to say pow(base, exp) % mod
- **`range(stop)` or `range(start, stop[, step])`** range is an immutable sequence type
- **`sorted(iterable, /, *, key=None, reverse=False)`** returns sorted iterables according to the key (function)
- **`//`** floor div; **`%`** mod; **`**`** exp; **`abs(num)`** absolute value

### Containers

- **`list`** \[1, 2, 3, 'a'\]

    Indexed by integers,items are stored as stack (first in, last out)

    Operations: `s[i:j:k]`; `s +t` concatenation; `s * num` adding s to itself num times; `s.index(x, i, j)` index of x first occurrence of x in s in range i to j; `x in s` check if x is in s;  `s.count(x)`; `del s[i:j]` same as `s[i:j] = []`; `s.clear()`; `s.copy()` shallow copy; `s.extend()`; `s.append()`; `s.insert(index, x)`, `s.pop(index)`, `s.remove(value)`; `s.reverse()`
filter

- **`tuple`** (1, 2, 3, 'a')

    Similar to list, but objects are not mutable
- **`set`** {1, 2, 3, 'a'}

    Objects in a set are unordered and unique, used for quick search. Set is implemented as a hash table, which supports lookup/insert/delete in O(1) time complexity.
- **`dictionary`**  { 'key1' : 1, 'key2': 2, 'key3': 3  }

    In older version of Python, dictionary is unordered, in python versions after 3.6, dictionary remembers the order of the key insertion
    Objects are key-value pairs, the keys are unique, immutable and can have different types. The values are mutable and can have different types.

    Operations:`d[key]`; `key in d` check if key exists;`d.get(key[, default=None])` returns default if key doesn't exist; `d.keys(), d.values(), d.items()` return iterables of keys, values and (key, val) paris; `iter(d)` ierate over keys; `list(d)` return list of all keys;  `del d[key]`; `d.pop(key[, default])`; `d.clear()`; `d.copy()` shallow copy; `d1 | d2` return merged dict; 
    
Strings, lists, and tuples are **sequence types** that can use the `+`, `*`, `+=`, and `*=` operators.
#### Strings

- **`count(substring[, start[, end]])`** return counts of substring within start and end range
- **`find(sub[, start[, end]])`** return the lowest index of the substring within start and end range, return -1 if sub not found, `rfind` return the highest index.
- **`join(iterable)`** return str concatenated with strings in iterable
- **`strip([chars])`** when omitting argument, remove the white space (including \n) at the beginning and end of a string, when specifying chars, remove all combination of the chars, `lstrip()` and `rstrip()` only remove white space from left or right.
- **`split(sep=None, maxsplit=- 1)`** return a list of words in str, split by sep, when sep is None, runs of consecutive whitespace are regarded as a single separator
- **`replace(old, new[, count])`** return str after replacing old with new for count times.
- **`maketrans()`** make a map for translate() to substitute or remove a set of characters, you can pass 0, 2 or 3 argument. 
    - 1 argument: has to be dictionary, the keys (character or int ASCII number) will be replaced by value (character or int)
    - 2 arguments: both need to be string and need to have the same length, each char in string1 will be replace by the corresponding char in str2
    - 3 arguments: characters in the third string will be deleted, string 1&2 will do the same replacement
    - the map has to be passed into string.translate(transmap) to function
- **`upper(), lower(), capitalize(), title()`** return all upper, all lower, first letter cap, and first letter cap for each word.

### Class 

```python
class classname:
    def __init__(self, p1, p2, p3): #constructor
        self.property1 =  p1
        self.property2 =  p2
    def fuc1(self, arg1): #functions
    def fuc2(self, arg1, arg2, arg3, ...): 
```
### Miscellaneous

- List, set and  dictionary comprehension

    ```python
    squarelist = [x ** 2 for x in range(5) if x%2==0]
    squareset = {x**2 for x in range(5)}
    squaredic = {x : x ** 2 for x in range(5)}
    ```
    comprehension can be nested `[[expression for j in range(nj)] for i in range(ni)]`
- Map: `list(map(float,[1,2,3]))` same as `[float(i) for i in [1,2,3]]`.
- Lambda function: an anonymous inline function consisting of a single expression. `lambda [parameters]: expression`
- `*a_list` unpack a list into separate positional arguments. e.g. `func(*[1, 2])` same as `func(1, 2)`
- File I/O `open(file, mode='r', buffering=- 1, encoding=None, errors=None, newline=None, closefd=True, opener=None)`
    - read `data = [line.strip() for line in open('data.txt')]` or use `with open(file) as f:`, which closes the file after.
    - write to file `with open("data.txt",'a',newline='\n') as f: f.write(data)`
    - pickle and JSON

### Numpy & ndarray (N-dimensional array)

`np.info()` for help

- Init 
  - `np.array(a_list, dtype=datatype)`
  - `np.arange([start, ]stop, [step, ]dtype=None, *, like=None)`
  - `np.linspace(start, stop, num=50, endpoint=True, dtype=None, axis=0)`
  - `np.random.random(d0, d1, ..., dn)` Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
  - `np.zeros(shape)` shape is int or tuple like (nrow, ncol)
  - `np.ones(shape)`
  - `numpy.full(shape, fill_value, dtype=None, order='C', *, like=None)`
  - `np.eye(ndim)` or `np.identity(ndim)` n-dimension identity matrix
  - `np.empty(shape)`
- Attributes
  - `arr.shape` shape as tuple
  - `arr.ndim` n-dimension (like 1d, 2d or 3d arrays)
  - `arr.size` number of elements
  - `arr.dtype` and `arr.dtype.name`
- Slicing and searching
  - `arr[start:stop:step]` applies to muliti dimenions `matrix[start:stop:step, start:stop:step]`, `:` selects all 
  - `arr[::-1]` is the same as `np.flip(arr, axis=None)`, reverse the array
  - `arr[[0, 1, 3]]` return array with elements at index 0, 1 and 3.
  - `arr[boolean list]` return array with elements at `True` positions. `arr[arr<1]` selects all elements smaller than 1
  - `numpy.where(condition, [x, y, ]/)` 
  - `numpy.extract(condition, arr)[source]`
  - `np.ndindex(nrow, ncol)` return zipped iterables row_index, col_index
- Operations
  - `arr.reshape(shape)` return without changing arr
  - `arr.resize(shape)` change arr directly, no return
  - `arr.ravel()`, `arr.flatten()` and `arr.reshape((-1,))`, flatten returns a copy, reshape returns a view, ravel returns a view of the original array whenever possible. 
  - `np.concatenate((a1, a2, ...), axis=0)`, `np.vstack((a1, a2))` or `np.r_[a1, a2]` stack row wise, `np.hstack` or `np.c_[a1, a2]` stack column wise, `np.dstack` are special cases, stacking arrays along axis = 0, 1, 2
  - `np.vsplit(arr, obj), hsplit()` Split the array vertically or horizontally. obj : index or slice 
  - `np.append(arr, values, axis=None)` return a copy of appended array
  - `np.insert(arr, obj, values, axis=None)` obj : index, slice or sequence of ints
  - `np.delete(arr, obj, axis=None)` return new array
  - `arr.copy()` deep copy
  - `arr.T` or `np.transpose(arr)` return a copy of the transposed array.
- Math functions
  - `+ - * /` array arithmetic
  - `np.mod(), exp(), log(), sqrt(), mean(), maximum(), minimum(), round(), floor(), ceil(), sin(), cos(), ...,` [full list](https://numpy.org/doc/stable/reference/routines.math.html)
  - `A @ B` matrix multiplication 
  - `np.dot(), vdot(), cross()` doc product and cross product
  - `np.linalg.qr()， linalg.svd()` matrix decomposition
  - `np.linalg.eig()`  return eigenvalues and right eigenvectors
- [Random Generator](https://numpy.org/doc/stable/reference/random/index.html) `rng = np.random.default_rng()`
  - `rng.random(size=None)` uniform random number from half-open interval [0.0, 1.0), customize range: `rng.uniform(low=0.0, high=1.0, size=None)`
  - `rng.normal(mean=0.0, std=1.0, size=None)` normal (Gaussian) distribution, `rng.standard_normal(size)` is a special case where mean=0 and std = 1
  - `rng.binomial(n, p, size=None)`; `random.Generator.poisson(lam=1.0, size=None)`
- Sorting and counting
  - `arr.sort(axis=- 1, kind=None, order=None)` sort array in place. `np.sort(arr)` return a copy
  - `arr.argsort()` or `np.argsort(arr)` return indices; `argmin()`, `argmax()`, `np.argwhere()`
  - `numpy.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None, *, equal_nan=True)` return array of sorted unique elements, if index or counts are true, return zipped items array, index/counts
- Numpy [dtypes](https://numpy.org/doc/stable/user/basics.types.html) and converting
  - `np.int_` == `np.int64` (code 'l') and `np.unit`  == `np.uint64` ('L') unsigned; Other size `np.(u)int8/16/32/64` 
  - `np.float_` == `np.double` == `np.float64`('d');  Other size `np.float16/32/64/128`
  - `np.complex_``np.cdouble` == `np.complex128` ('D') Complex number contain 2 64-bit-precision floating-point numbers.
  - `np.bool_` == `np.bool8` (code '?')
  - `np.str_` == `np.unicode_` ('U)
  - `np.datetime64` ('M') and `timedelta64` ('m')
  - `np.object_` ('O')
  - `arr.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)`
  - `arr.tolist()`; `arr.tostring()`
- [File I/O](https://numpy.org/doc/stable/reference/routines.io.html)
  - Input `numpy.genfromtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, skip_header=0, skip_footer=0, converters=None, missing_values=None, filling_values=None, usecols=None, names=None, excludelist=None, deletechars=" !#$%&'()*+, -./:;<=>?@[\\]^{|}~", replace_space='_', autostrip=False, case_sensitive=True, defaultfmt='f%i', unpack=None, usemask=False, loose=True, invalid_raise=True, max_rows=None, encoding='bytes', *, ndmin=0, like=None)`
  - Output `numpy.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)`
  - `arr.tofile(fid, sep='', format='%s')` fid: filename or an open file object
  - `np.savez(file, *args, **kwds)` *args: arr1, arr2, ...; Save several arrays into a single file in uncompressed .npz format. access : `np.load(npzfile)`
  - `numpy.save(file, arr, allow_pickle=True, fix_imports=True)` save one arr to a .npy file.

### Pandas & DataFrame

Pandas DataFrame is essentially a spreadsheet / 2D array. Each column is a pandas Series.

- Init
    - `pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)` data can be ndarray, dict, iterable or DataFrame (copy=False for df).
    - `pd.read_csv(file)` options:
        - `sep = ','` same as delimiter, `header` int, list of int, None; `names` column names; `index_col`; `usecols` specify cols (by list of name or index) to keep; `skiprows`; `skipfooter`; `nrows` num of rows to read; `delim_whitespace=False` and [other options](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html#pandas.read_csv)
    - `pd.read_excel(file)`, `pd.read_table(filename)`, `pd.read_sql(query, connection_object)`, `pd.read_json(json_string)`, `pd.read_html(url)`
- Attributes
    - `df.index` return row index names as padas RangeIndex object, `df.index.to_numpy()` or `to_list()` convert to ndarray or list
    - `df.columns` return column name as Index object `df.keys()`
    - `df.dtypes`
    - `df.values` return ndarray of the spread sheet
    - `df.shape` return tuple of (nrow, ncol)
    - `df.size` return number of elements/cells
    - `df.axes` return a list [row index obj, columns]
    - `df.ndim`
    - `df.empty` return bool of whether df is empty
- Indexing and slicing
    - `df.at([row_label, col_label])` row labels are int by default, col labels are often strings; `df.iat([row_index,col_index])`
    - `df.loc[]`access a group of rows and columns by label(s) or a boolean array. Int numbers will be interpreted as a _label_ and **never** the index position. Bool array needs to be the same length as the axis being sliced.
    -  `df.iloc[]` similar to loc[], but arguments are indices instead of labels. `df.iloc[0]` slice first row, `df.iloc[:, 0]` first column, `df.iloc[0,0]` first cell. Note that `df.iloc[0]` return series and `df.iloc[[0]]` return dataframe.
- Data Inspection
    - `df.describe(percentiles=[.25,.5,.75])` for each column, display counts, min, max, std and percentiles
    - `df.info()` show index, dtype and memory
    - `df.head(n)` and `df.tail(n)` return the first and last n rows.
    - `s.value_counts(dropna=Fase)` unique values and counts for series `df.value_count()` counts unique rows
    - `df.count(axis=0, level=None, numeric_only=False)` count non-NA cells for each col or row, `s.count()` non-NA elements in series
- Operations
    - `df[new_col]=values` add a new column
    - `df.insert(loc, col_name, value, allow_duplicates=False)` insert a column at index loc.
    - `df.pop(col_name)` return column and drop from frame. Raise KeyError if not found.
    - `df.drop(labels, axis=0, index, columns, level, inplace=False)` drop specified labels from rows or columns.`axis=0` drop index, `axis=1` drop columns
    - `pd.concat(objs, axis=0, join='outer', ignore_index=False, sort=False, copy=True)` concatenate pandas objects along a particular axis.
    - `df.pivot(index=None, columns=None, values=None)` return reshaped df organized by given index / column values.
    - `pd.pivot_table(data, values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False, sort=True)` create a spreadsheet-style pivot table as df.
    - `df.groupby(by, axis=0, as_index=True, sort=True, dropna=True)` group df with columns or mapping function
    - `df.apply(func, axis=0, raw=False)` . e.g. `df.apply(pd.Series.value_counts)` return unique value and counts for all columns
    - `df.copy()` deep copy
    - `df.dropna()`  `df.fillna(val)`
- Iterations
    - `for i in obj` produce values if obj is Series, and column labels if obj is df
    - `for index, Series in df.iterrows()` Iterating through pandas objects is generally slow, and you should never modify the df, because depending on the data types, the iterator may return a copy and not a view.
    - `df.items()` act like `dict.items()` iterates through key-value (label-series) pairs.
- Math functions
    - `df.max(), min(), mean(), median(), std(), corr()` std calculates variance, corr calculates correlation
- File I/O 
    - pandas has a set of `reader` and `writer` functions such as `pd.read_csv()` and `df.to_csv()`, the official guide is [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html?highlight=file)