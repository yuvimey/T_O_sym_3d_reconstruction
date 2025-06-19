# Ab initio reconstruction for volumes with tetrahedral and octahedral symmetries.

Please make sure all the necessary requirements are installed:

    pip install -r requirements.txt

Run the code using the following command line:

    python reconstruct.py --sym {sym} --i {stack.mrcs}  --o {vol.mrc}

Flags:

--sym SYM   Symmetry indicator, either T or O
--i I       Name of the 2d projections mrc file (e.g. stack.mrc)
--o O       Name of the output volume mrc file (e.g. vol.mrc)