

if type(get_ipython()).__module__.startswith('ipykernel.'):
    from tqdm import tqdm_notebook as tqdm
else:
    import tqdm
