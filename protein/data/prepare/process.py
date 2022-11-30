import logging
import os
import torch
import tarfile
from torch.nn.utils.rnn import pad_sequence

#charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9} #Commented this out as seemed unnecessary


import math

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def sigmoid(x):
      if x < 0:
            return 1 - 1/(1 + math.exp(x))
      else:
            return 1/(1 + math.exp(-x))

residue_types=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','X']
secondary_structures=['H','E','C']

def getOneHotRepresentation(residue):
    arr=np.zeros(21, dtype=int)
    res_index=residue_types.index(residue)
    arr[res_index]=1
    #one_hot=''
    #for bit in arr:
    #    one_hot=one_hot+str(bit)+' '
    #one_hot=''.join(one_hot.rstrip())
    one_hot=[]
    for bit in arr:
        one_hot.append(bit)

    return one_hot

def getResidueTypeIndex(residue):
  res_index=residue_types.index(residue)
  return res_index

def aaGroup(aa):
    aa = aa.capitalize().strip()
    if(aa == 'A' or aa == 'V' or aa == 'L' or aa == 'I' or aa == 'P' or aa == 'F' or aa == 'M' or aa == 'W'): #non-polar
        label = 1
    elif(aa == 'G' or aa == 'S' or aa == 'T' or aa == 'C' or aa == 'Y' or aa == 'N' or aa == 'Q'): #polar
        label = 2
    elif(aa == 'D' or aa == 'E'): #acidic
        label = 3
    elif(aa == 'K' or aa == 'R' or aa == 'H'): #basic
        label = 4
    else:
        label = 5

    return label

def get8to3ss(ss_parm):
    eTo3=""
    if (ss_parm == "H" or ss_parm == "G" or ss_parm == "I"):
            eTo3="H"
    elif(ss_parm == "E" or ss_parm == "B"):
            eTo3="E"
    else:
            eTo3="C"
    return eTo3

def getSSIndex(ss):
  ss_index=secondary_structures.index(ss)
  return ss_index


def split_dataset(data, split_idxs):
    """
    Splits a dataset according to the indices given.

    Parameters
    ----------
    data : dict
        Dictionary to split.
    split_idxs :  dict
        Dictionary defining the split.  Keys are the name of the split, and
        values are the keys for the items in data that go into the split.

    Returns
    -------
    split_dataset : dict
        The split dataset.
    """
    split_data = {}
    for set, split in split_idxs.items():
        split_data[set] = {key: val[split] for key, val in data.items()}

    return split_data

# def save_database()


def process_xyz_files(data, process_file_fn, file_ext=None, file_idx_list=None, stack=True, protein_file=None, protein_list=None, protein_numbers=None, split_type=None): #added the protein parameters
    """
    Take a set of datafiles and apply a predefined data processing script to each
    one. Data can be stored in a directory, tarfile, or zipfile. An optional
    file extension can be added.

    Parameters
    ----------
    data : str
        Complete path to datafiles. Files must be in a directory, tarball, or zip archive.
    process_file_fn : callable
        Function to process files. Can be defined externally.
        Must input a file, and output a dictionary of properties, each of which
        is a torch.tensor. Dictionary must contain at least three properties:
        {'num_elements', 'charges', 'positions'}
    file_ext : str, optional
        Optionally add a file extension if multiple types of files exist.
    file_idx_list : ?????, optional
        Optionally add a file filter to check a file index is in a
        predefined list, for example, when constructing a train/valid/test split.
    stack : bool, optional
        ?????
    """
    logging.info('Processing data file: {}'.format(data))
    if tarfile.is_tarfile(data):
        tardata = tarfile.open(data, 'r')
        files = tardata.getmembers()

        readfile = lambda data_pt: tardata.extractfile(data_pt)

    elif os.is_dir(data):
        files = os.listdir(data)
        files = [os.path.join(data, file) for file in files]

        readfile = lambda data_pt: open(data_pt, 'r')

    else:
        raise ValueError('Can only read from directory or tarball archive!')

    # Use only files that end with specified extension.
    if file_ext is not None:
        files = [file for file in files if file.endswith(file_ext)]

    # Use only files that match desired filter.
    if file_idx_list is not None:
        files = [file for idx, file in enumerate(files) if idx in file_idx_list]

    # Now loop over files using readfile function defined above
    # Process each file accordingly using process_file_fn

    molecules = []

    #for file in files:
    #    with readfile(file) as openfile:
    #        molecules.append(process_file_fn(openfile, protein_file, 'abc', 1)) #added the protein_file parameter

    #The following logic is written for processing the protein files
    for protein in protein_list:
      for num in protein_numbers:
        molecules.append(process_file_fn(protein_file, protein, num, split_type))

    # Check that all molecules have the same set of items in their dictionary:
    props = molecules[0].keys()
    assert all(props == mol.keys() for mol in molecules), 'All molecules must have same set of properties/keys!'

    # Convert list-of-dicts to dict-of-lists
    molecules = {prop: [mol[prop] for mol in molecules] for prop in props}

    # If stacking is desireable, pad and then stack.
    if stack:
        molecules = {key: pad_sequence(val, batch_first=True) if val[0].dim() > 0 else torch.stack(val) for key, val in molecules.items()}

    return molecules


def process_xyz_md17(datafile):
    """
    Read xyz file and return a molecular dict with number of atoms, energy, forces, coordinates and atom-type for the MD-17 dataset.

    Parameters
    ----------
    datafile : python file object
        File object containing the molecular data in the MD17 dataset.

    Returns
    -------
    molecule : dict
        Dictionary containing the molecular properties of the associated file object.
    """
    xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]

    line_counter = 0
    atom_positions = []
    atom_types = []
    for line in xyz_lines:
        if line[0] is '#':
            continue
        if line_counter is 0:
            num_atoms = int(line)
        elif line_counter is 1:
            split = line.split(';')
            assert (len(split) == 1 or len(split) == 2), 'Improperly formatted energy/force line.'
            if (len(split) == 1):
                e = split[0]
                f = None
            elif (len(split) == 2):
                e, f = split
                f = f.split('],[')
                atom_energy = float(e)
                atom_forces = [[float(x.strip('[]\n')) for x in force.split(',')] for force in f]
        else:
            split = line.split()
            if len(split) is 4:
                type, x, y, z = split
                atom_types.append(split[0])
                atom_positions.append([float(x) for x in split[1:]])
            else:
                logging.debug(line)
        line_counter += 1

    atom_charges = [charge_dict[type] for type in atom_types]

    molecule = {'num_atoms': num_atoms, 'energy': atom_energy, 'charges': atom_charges,
                'forces': atom_forces, 'positions': atom_positions}

    molecule = {key: torch.tensor(val) for key, val in molecule.items()}

    return molecule


def process_xyz_protein(protein_file, protein, number, split_type): #First parameter was datafile and has been removed
    """
    Read xyz file and return a molecular dict with number of atoms, energy, forces, coordinates and atom-type for the gdb9 dataset.

    Parameters
    ----------
    datafile : python file object
        File object containing the molecular data in the MD17 dataset.

    Returns
    -------
    molecule : dict
        Dictionary containing the molecular properties of the associated file object.

    Notes
    -----
    TODO : Replace breakpoint with a more informative failure?
    """

    #Protein
    positions_list=[]
    features=[]
    sa_list=[]
    protein_name=protein
    i=number
    split_type=split_type
    #i=protein_file
    #protein_dssp_path='/home/common/proj/EDM_Protein/DSSPFiles/'+protein_name+'_'+str(i)+'.dssp'
    protein_dssp_path='/home/common/proj/EDM_Protein/DSSPFiles/'+split_type+'/'+protein_name+'.dssp'
    f1=open(protein_dssp_path)
    proteinDssp = []
    line1Count = 0
    for line in f1:
        if (line1Count<1):
            if (line[2:(2+1)] == '#'):
                line1Count += 1
                continue
        if(line1Count > 0):
            if(len(line) > 0):
                if line[13:14].strip()=='!':
                    continue
                proteinDssp.append(line)
                #residue_numbers_protein.append(int(line[6:(6+4)].strip()))

    f1.close()

    protein_position=[]
    protein_feature=[]
    #residue_index=[]
    residue_code=[]
    ss_code=[]
    #phi=[]
    #psi=[]
    dihedrals=[]
    sa=[]
    relative_pos=[]

    phi_value=0.0
    psi_value=0.0
    sa_value=0.0 #Using solvent accessibility in place of charges
    x1=0.0
    y1=0.0
    z1=0.0

    #print('Number of Ca')
    #print(len(proteinDssp))
    for ds1 in range(len(proteinDssp)): #Was len(proteinDssp) instead of 9
        #(protein_name=='1hkfA' and (ds1>=28 and ds1<=47)) or (protein_name=='1hlqA' and (ds1>=3 and ds1<=17)) or (protein_name=='1bl0A' and (ds1>=1 and ds1<=18)) or (protein_name=='1qysA' and (ds1>=14 and ds1<=34))
        #(protein_name=='1hkfA' and (ds1>=28 and ds1<=35)) or (protein_name=='1hlqA' and (ds1>=9 and ds1<=22)) or (protein_name=='1bl0A' and (ds1>=1 and ds1<=9)) or (protein_name=='1qysA' and (ds1>=25 and ds1<=34))
        #(protein_name=='1hkfA' and (ds1>=3 and ds1<=7)) or (protein_name=='1hlqA' and (ds1>=9 and ds1<=12))
        '''if (protein_name=='1hkfA' and (ds1>=28 and ds1<=47)) or (protein_name=='1hlqA' and (ds1>=3 and ds1<=17)) or (protein_name=='1bl0A' and (ds1>=1 and ds1<=18)) or (protein_name=='1qysA' and (ds1>=14 and ds1<=34)):
            coord=[]
            phipsi=[]
            res1=proteinDssp[ds1][13:14].strip()
            res_index=getResidueTypeIndex(res1)
            #residue_index.append(res_index+1) #Numbers are going to be between 1 and 21 instead of 0 and 20
            #res_code=aaGroup(res1)
            #residue_code.append(res_code)
            residue_code.append(res_index+1)
            ss1 = get8to3ss(proteinDssp[ds1][16:(16+1)])
            ss_index=getSSIndex(ss1)
            ss_code.append(ss_index+1)
            #one_hot=getOneHotRepresentation(res1)
            #protein_feature.append(one_hot)
            #calculate the residue type and return the info with the dictionary molecule ########## THINK ABOUT THIS IDEA

            if(isfloat(proteinDssp[ds1][103:(103+6)].strip())):
                phi_value = float(proteinDssp[ds1][103:(103+6)].strip())
            if(isfloat(proteinDssp[ds1][109:(109+6)].strip())):
                psi_value = float(proteinDssp[ds1][109:(109+6)].strip())
            #phi.append(phi_value)
            #psi.append(psi_value)
            phipsi.append(math.sin(phi_value))
            phipsi.append(math.cos(phi_value))
            phipsi.append(math.sin(psi_value))
            phipsi.append(math.cos(psi_value))
            dihedrals.append(phipsi)

            if(isfloat(proteinDssp[ds1][35:(35+3)])):
                sa_value = float(proteinDssp[ds1][35:(35+3)])
                sa_value = sigmoid(sa_value)
            sa.append(sa_value)

            x1=float(proteinDssp[ds1][117:(117+5)])
            y1=float(proteinDssp[ds1][124:(124+5)])
            z1=float(proteinDssp[ds1][131:(131+5)])
            coord.append(x1)
            coord.append(y1)
            coord.append(z1)
            protein_position.append(coord)'''
    for ds1 in range(len(proteinDssp)): #Was len(proteinDssp) instead of 9
        #if (protein_name=='1hkfA' and (ds1>=3 and ds1<=7)) or (protein_name=='1hlqA' and (ds1>=9 and ds1<=12)):
        coord=[]
        phipsi=[]
        res1=proteinDssp[ds1][13:14].strip()
        res_index=getResidueTypeIndex(res1)
        #residue_index.append(res_index+1) #Numbers are going to be between 1 and 21 instead of 0 and 20
        #res_code=aaGroup(res1)
        #residue_code.append(res_code)
        residue_code.append(res_index+1)
        ss1 = get8to3ss(proteinDssp[ds1][16:(16+1)])
        ss_index=getSSIndex(ss1)
        ss_code.append(ss_index+1)
        #one_hot=getOneHotRepresentation(res1)
        #protein_feature.append(one_hot)
        #calculate the residue type and return the info with the dictionary molecule ########## THINK ABOUT THIS IDEA

        if(isfloat(proteinDssp[ds1][103:(103+6)].strip())):
            phi_value = float(proteinDssp[ds1][103:(103+6)].strip())
        if(isfloat(proteinDssp[ds1][109:(109+6)].strip())):
            psi_value = float(proteinDssp[ds1][109:(109+6)].strip())
        #phi.append(phi_value)
        #psi.append(psi_value)
        #phipsi.append(math.sin(phi_value))
        #phipsi.append(math.cos(phi_value))
        #phipsi.append(math.sin(psi_value))
        #phipsi.append(math.cos(psi_value))
        phipsi.append(math.radians(phi_value))
        phipsi.append(math.radians(psi_value))
        dihedrals.append(phipsi)

        if(isfloat(proteinDssp[ds1][35:(35+3)])):
            sa_value = float(proteinDssp[ds1][35:(35+3)])
            sa_value = sigmoid(sa_value)
        sa.append(sa_value)

        rel_pos=float((ds1+1)/len(proteinDssp))
        relative_pos.append(rel_pos)

        x1=float(proteinDssp[ds1][117:(117+5)])
        y1=float(proteinDssp[ds1][124:(124+5)])
        z1=float(proteinDssp[ds1][131:(131+5)])
        coord.append(x1)
        coord.append(y1)
        coord.append(z1)
        protein_position.append(coord)
    #positions_list.append(protein_position)
    #sa_list.append(sa)
    #features.append(protein_feature)
    #i=i+1
    
    #xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]

    #num_atoms = int(xyz_lines[0])
    num_atoms=len(proteinDssp) #Used 9 earlier
    #mol_props = xyz_lines[1].split()
    #mol_xyz = xyz_lines[2:num_atoms+2]
    #mol_freq = xyz_lines[num_atoms+2]
    #print('Num atoms')
    #print(num_atoms)
    #print('Mol props')
    #print(mol_props)
    #print('Mol xyz')
    #print(mol_xyz)

    atom_charges, atom_positions = [], []
    #for line in mol_xyz:
    #    atom, posx, posy, posz, _ = line.replace('*^', 'e').split()
    #    atom_charges.append(charge_dict[atom])
    #    atom_positions.append([float(posx), float(posy), float(posz)])
    atom_charges=sa #Using solvent accessibility values instead of charges
    atom_positions=protein_position

    #prop_strings = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    #prop_strings = prop_strings[1:]
    #mol_props = [int(mol_props[1])] + [float(x) for x in mol_props[2:]]
    #mol_props = dict(zip(prop_strings, mol_props))
    #mol_props['omega1'] = 0.0 #was max(float(omega) for omega in mol_freq.split())

    molecule = {'num_atoms': num_atoms, 'charges': atom_charges, 'positions': atom_positions, 'residue_code': residue_code, 'ss_code': ss_code, 'dihedrals': dihedrals, 'relative_pos': relative_pos}
    #molecule.update(mol_props)
    molecule = {key: torch.tensor(val) for key, val in molecule.items()}
    #print('Printing molecule from process file')
    #print(molecule)

    return molecule
