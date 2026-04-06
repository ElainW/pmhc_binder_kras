"""
util.py
Minimal replacement for Baker lab util.py.
Provides aa2num and aa2long for the 14-atom heavy atom representation
used in pmhc_fold.py's parse_pdb function.
"""

# 3-letter AA code → integer index (20 standard AAs + unknown)
aa2num = {
    'ALA':  0, 'ARG':  1, 'ASN':  2, 'ASP':  3, 'CYS':  4,
    'GLN':  5, 'GLU':  6, 'GLY':  7, 'HIS':  8, 'ILE':  9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
    'UNK': 20,
}

# Canonical 14 heavy atoms per residue, in RoseTTAFold order.
# Positions that don't exist for a given AA are None.
# Order: N, CA, C, O, CB, then up to 9 sidechain atoms.
aa2long = [
    # 0 ALA
    (' N  ',' CA ',' C  ',' O  ',' CB ', None, None, None, None, None, None, None, None, None),
    # 1 ARG
    (' N  ',' CA ',' C  ',' O  ',' CB ',' CG ',' CD ',' NE ',' CZ ',' NH1',' NH2', None, None, None),
    # 2 ASN
    (' N  ',' CA ',' C  ',' O  ',' CB ',' CG ',' OD1',' ND2', None, None, None, None, None, None),
    # 3 ASP
    (' N  ',' CA ',' C  ',' O  ',' CB ',' CG ',' OD1',' OD2', None, None, None, None, None, None),
    # 4 CYS
    (' N  ',' CA ',' C  ',' O  ',' CB ',' SG ', None, None, None, None, None, None, None, None),
    # 5 GLN
    (' N  ',' CA ',' C  ',' O  ',' CB ',' CG ',' CD ',' OE1',' NE2', None, None, None, None, None),
    # 6 GLU
    (' N  ',' CA ',' C  ',' O  ',' CB ',' CG ',' CD ',' OE1',' OE2', None, None, None, None, None),
    # 7 GLY
    (' N  ',' CA ',' C  ',' O  ', None, None, None, None, None, None, None, None, None, None),
    # 8 HIS
    (' N  ',' CA ',' C  ',' O  ',' CB ',' CG ',' ND1',' CD2',' CE1',' NE2', None, None, None, None),
    # 9 ILE
    (' N  ',' CA ',' C  ',' O  ',' CB ',' CG1',' CG2',' CD1', None, None, None, None, None, None),
    # 10 LEU
    (' N  ',' CA ',' C  ',' O  ',' CB ',' CG ',' CD1',' CD2', None, None, None, None, None, None),
    # 11 LYS
    (' N  ',' CA ',' C  ',' O  ',' CB ',' CG ',' CD ',' CE ',' NZ ', None, None, None, None, None),
    # 12 MET
    (' N  ',' CA ',' C  ',' O  ',' CB ',' CG ',' SD ',' CE ', None, None, None, None, None, None),
    # 13 PHE
    (' N  ',' CA ',' C  ',' O  ',' CB ',' CG ',' CD1',' CD2',' CE1',' CE2',' CZ ', None, None, None),
    # 14 PRO
    (' N  ',' CA ',' C  ',' O  ',' CB ',' CG ',' CD ', None, None, None, None, None, None, None),
    # 15 SER
    (' N  ',' CA ',' C  ',' O  ',' CB ',' OG ', None, None, None, None, None, None, None, None),
    # 16 THR
    (' N  ',' CA ',' C  ',' O  ',' CB ',' OG1',' CG2', None, None, None, None, None, None, None),
    # 17 TRP
    (' N  ',' CA ',' C  ',' O  ',' CB ',' CG ',' CD1',' CD2',' NE1',' CE2',' CE3',' CZ2',' CZ3',' CH2'),
    # 18 TYR
    (' N  ',' CA ',' C  ',' O  ',' CB ',' CG ',' CD1',' CD2',' CE1',' CE2',' CZ ',' OH ', None, None),
    # 19 VAL
    (' N  ',' CA ',' C  ',' O  ',' CB ',' CG1',' CG2', None, None, None, None, None, None, None),
    # 20 UNK
    (' N  ',' CA ',' C  ',' O  ', None, None, None, None, None, None, None, None, None, None),
]