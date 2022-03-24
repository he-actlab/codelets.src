def range_from_cfg(cfg, as_int=True):
    if cfg['signed']:
        upper_val = (1 << (cfg['n_word'] - 1)) - 1
        lower_val = -upper_val - 1
    else:
        upper_val = (1 << cfg['n_word']) - 1
        lower_val = 0

    if not as_int:
        upper_val = upper_val / 2.0 ** cfg['n_frac']
        lower_val = lower_val / 2.0 ** cfg['n_frac']
    # precision = 1 / 2.0 ** cfg['n_frac']

    return (lower_val, upper_val)