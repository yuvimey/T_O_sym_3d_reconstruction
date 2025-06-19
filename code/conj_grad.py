import numpy as np

def conj_grad(Afun, b, cg_opt=None, init=None):
    if cg_opt is None:
        cg_opt = {}
    if init is None:
        init = {}

    x = init.get('x', np.zeros_like(b))

    cg_opt.setdefault('max_iter', 50)
    cg_opt.setdefault('verbose', 0)
    cg_opt.setdefault('iter_callback', None)
    cg_opt.setdefault('preconditioner', lambda x: x)
    cg_opt.setdefault('rel_tolerance', 1e-15)
    cg_opt.setdefault('store_iterates', False)

    b_norm = np.sqrt(np.sum(np.abs(b)**2, axis=0))

    r = b
    s = cg_opt['preconditioner'](r)
    
    if np.any(x != 0):
        if cg_opt['verbose']:
            print('[CG] Calculating initial residual...', end='')
        Ax = Afun(x)
        r -= Ax
        s = cg_opt['preconditioner'](r)
        if cg_opt['verbose']:
            print('OK')
    else:
        Ax = np.zeros_like(x)

    obj = np.real(np.sum(np.conj(x) * Ax, axis=0)) - 2 * np.real(np.sum(np.conj(b) * x, axis=0))

    p = init.get('p', s)

    info = [{}]
    info[0]['iter'] = 0
    if cg_opt['store_iterates']:
        info[0]['x'] = x
        info[0]['r'] = r
        info[0]['p'] = p
    info[0]['res'] = np.sqrt(np.sum(np.abs(r)**2, axis=0))
    info[0]['obj'] = obj

    if cg_opt['verbose']:
        print(f"[CG] Initialized. Residual: {np.linalg.norm(info[0]['res'])}. Objective: {np.sum(obj)}")

    if np.all(b_norm == 0):
        return x, obj, info

    iter = 1
    while iter < cg_opt['max_iter']:
        if cg_opt['verbose']:
            print('[CG] Applying matrix & preconditioner...')
        Ap = Afun(p)

        old_gamma = np.real(np.sum(np.conj(s) * r, axis=0))
        alpha = old_gamma / np.real(np.sum(np.conj(p) * Ap, axis=0))
        x += alpha * p
        Ax += alpha * Ap

        r -= alpha * Ap
        s = cg_opt['preconditioner'](r)
        new_gamma = np.real(np.sum(np.conj(r) * s, axis=0))
        beta = new_gamma / old_gamma
        p = s + beta * p

        if cg_opt['verbose']:
            print('OK')

        obj = np.real(np.sum(np.conj(x) * Ax, axis=0)) - 2 * np.real(np.sum(np.conj(b) * x, axis=0))

        info.append({'iter': iter})
        if cg_opt['store_iterates']:
            info[-1]['x'] = x
            info[-1]['r'] = r
            info[-1]['p'] = p
        info[-1]['res'] = np.sqrt(np.sum(np.abs(r)**2, axis=0))
        info[-1]['obj'] = obj

        if cg_opt['verbose']:
            print(f"[CG] Iteration {iter}. Residual: {np.linalg.norm(info[-1]['res'])}. Objective: {np.sum(obj)} | max_res - b_norm*rel_tol: {np.max(info[-1]['res'])} - {b_norm * cg_opt['rel_tolerance']}")

        if cg_opt['iter_callback'] is not None:
            cg_opt['iter_callback'](info)

        if np.all(info[-1]['res'] < b_norm * cg_opt['rel_tolerance']):
            break

        iter += 1

    if iter == cg_opt['max_iter']:
        print('Warning: Conjugate gradient reached maximum number of iterations!')

    return x, obj, info
