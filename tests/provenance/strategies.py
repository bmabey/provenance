import numpy as np
import hypothesis.strategies as st

primitive_data = (
    st.floats(allow_nan=False)
    | st.booleans()
    | st.text()
    | st.none()
    | st.fractions()
    | st.integers()
    | st.characters()
)
# | st.complex_numbers() \ nanj is annoying to deal with
# | st.decimals() can add back in once a new version of joblib is released with bug fix

hashable_data = primitive_data | st.tuples(primitive_data)
sets = st.sets(hashable_data)
builtin_data = st.recursive(
    primitive_data | sets,
    lambda children: st.lists(children)
    | st.dictionaries(st.text(), children)
    | st.tuples(children),
)


def rand_nparray(seed, w=3, h=3):
    rnd = np.random.RandomState(seed)
    return rnd.random_sample((w, h))


np_random_states = st.integers(0, 4294967295).map(np.random.RandomState)
fixed_numpy_arrays = st.integers(0, 4294967295).map(rand_nparray)
numpy_data = fixed_numpy_arrays
data = st.recursive(
    primitive_data | sets | fixed_numpy_arrays,
    lambda children: st.lists(children)
    | st.dictionaries(st.text(), children)
    | st.tuples(children),
)
