import numpy as np

def dirac_basis(sz, mask=None):
    if mask is None:
        mask = np.ones(sz, dtype=bool)

    basis = {
        'type': 0,
        'sz': sz,
        'count': np.sum(mask),
        'mask': mask
    }

    basis['evaluate'] = lambda v: dirac_evaluate(v, basis)
    basis['expand'] = lambda x: dirac_expand(x, basis)
    basis['evaluate_t'] = basis['expand']

    d = len(sz)

    basis['mat_evaluate'] = lambda V: mdim_mat_fun_conj(V, 1, d, basis['evaluate'])
    basis['mat_expand'] = lambda X: mdim_mat_fun_conj(X, d, 1, basis['expand'])
    basis['mat_evaluate_t'] = lambda X: mdim_mat_fun_conj(X, d, 1, basis['evaluate_t'])

    return basis

def dirac_evaluate(v, basis):
    if v.shape[0] != basis['count']:
        raise ValueError('First dimension of v must be of size basis.count')

    v, sz_roll = unroll_dim(v, axis=1)

    x = np.zeros((np.prod(basis['sz']), v.shape[1]), dtype=v.dtype)
    x[basis['mask'].flatten(order='F'), :] = v

    x = x.reshape((*basis['sz'], x.shape[1]), order='F')
    x = roll_dim(x, sz_roll)

    return x

def dirac_expand(x, basis):
    sz_x = x.shape
    
    if len(sz_x) < len(basis['sz']) or not all(sz_x[i] == basis['sz'][i] for i in range(len(basis['sz']))):
        raise ValueError("First dimensions of x must match basis.sz")

    x, sz_roll = unroll_dim(x, axis=len(basis['sz']))

    x = x.reshape((np.prod(basis['sz']), x.shape[-1]), order='F')
    v = x[basis['mask'].flatten(order='F'), :]

    v = roll_dim(v, sz_roll)
    return v

def mdim_mat_fun_conj(X, d1, d2, fun):
    X, sz_roll = unroll_dim(X, axis=2 * d1)

    X = fun(X)

    perm1 = list(range(d2, d2 + d1)) + list(range(d2)) + [d1 + d2]
    X = np.conj(np.transpose(X, perm1))

    X = fun(X)

    perm2 = list(range(d2, 2 * d2)) + list(range(d2)) + [2 * d2]
    X = np.conj(np.transpose(X, perm2))

    X = roll_dim(X, sz_roll)

    return X

def unroll_dim(X, axis):
    shape = list(X.shape)
    if axis >= len(shape):
        shape.extend([1] * (axis - len(shape) + 1))
    roll_sz = shape[axis:]
    new_shape = shape[:axis] + [np.prod(roll_sz)]
    X = X.reshape(new_shape, order='F')
    return X, roll_sz

def roll_dim(X, roll_sz):
    if len(roll_sz) > 1:
        sz = list(X.shape)
        if len(sz) > 1 and sz[1] == 1:
            sz = [sz[0]]
        X = X.reshape(sz[:-1] + roll_sz, order='F')
    return X
