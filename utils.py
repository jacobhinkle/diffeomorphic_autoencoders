# from https://github.com/slundberg/shap/issues/121
IN_IPYNB = None

def in_ipynb():
  if IN_IPYNB is not None:
    return IN_IPYNB

  try:
    cfg = get_ipython().config
    if str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>":
    # if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
      # print ('Running in ipython notebook env.')
      return True
    else:
      return False
  except NameError:
    # print ('NOT Running in ipython notebook env.')
    return False

if IN_IPYNB:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
