import argparse
from cryo_abinitio_TO import cryo_abinitio_TO

def main():
    parser = argparse.ArgumentParser(description="Process a character and file names.")
    parser.add_argument('--sym', type=str, required=True, help='Symmetry indicator, either T or O')
    parser.add_argument('--i', type=str, required=True, help='Name of the 2d projections mrc file (e.g. stack.mrc)')
    parser.add_argument('--o', type=str, required=True, help='Name of the output volume mrc file (e.g. vol.mrc)')

    args = parser.parse_args()

    symmetry = args.sym
    projs_fname = args.i
    vol_fname = args.o

    if str.upper(symmetry) != 'T' and str.upper(symmetry) != 'O':
        raise ValueError("Symmetry must be either T or O.")
    
    if str.upper(symmetry) == 'T':
        mat_frame = 'T_symmetry_3987_candidates_cache.mat.npz'
    else:
        mat_frame = 'O_symmetry_2027_candidates_cache.mat.npz'

    cryo_abinitio_TO(symmetry, projs_fname, vol_fname, mat_frame)

if __name__ == "__main__":
    main()