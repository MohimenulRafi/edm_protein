import torch
import numpy as np

import logging
import os

from torch.utils.data import DataLoader
from protein.data.dataset_class import ProcessedDataset
from protein.data.prepare import prepare_dataset

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
#secondary_structures=['H','E','C']

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

def initialize_datasets(args, datadir, dataset, subset=None, splits=None,
                        force_download=False, subtract_thermo=False,
                        remove_h=False):
    """
    Initialize datasets.

    Parameters
    ----------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datadir : str
        Path to the directory where the data and calculations and is, or will be, stored.
    dataset : str
        String specification of the dataset.  If it is not already downloaded, must currently by "qm9" or "md17".
    subset : str, optional
        Which subset of a dataset to use.  Action is dependent on the dataset given.
        Must be specified if the dataset has subsets (i.e. MD17).  Otherwise ignored (i.e. GDB9).
    splits : str, optional
        TODO: DELETE THIS ENTRY
    force_download : bool, optional
        If true, forces a fresh download of the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.
    remove_h: bool, optional
        If True, remove hydrogens from the dataset
    Returns
    -------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datasets : dict
        Dictionary of processed dataset objects (see ????? for more information).
        Valid keys are "train", "test", and "valid"[ate].  Each associated value
    num_species : int
        Number of unique atomic species in the dataset.
    max_charge : pytorch.Tensor
        Largest atomic number for the dataset.

    Notes
    -----
    TODO: Delete the splits argument.
    """
    # Set the number of points based upon the arguments
    num_pts = {'train': args.num_train,
               'test': args.num_test, 'valid': args.num_valid}
    #print(num_pts)

    # Download and process dataset. Returns datafiles.
    datafiles = prepare_dataset(
        datadir, 'protein', subset, splits, force_download=True) #force_download was equal to force_download instead of True

    #Read protein data
    positions_list=[]
    features=[]
    sa_list=[]
    i=1
    while(i<=6):
        protein_dssp_path='/home/common/proj/EDM/DSSPFiles/T1104_'+str(i)+'.dssp'
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
        sa=[]
        
        sa_value=0.0 #Using solvent accessibility in place of charges
        x1=0.0
        y1=0.0
        z1=0.0

        #print('Number of Ca')
        #print(len(proteinDssp))
        for ds1 in range(9): #Was len(proteinDssp) instead of 9
            coord=[]
            res1=proteinDssp[ds1][13:14].strip()
            one_hot=getOneHotRepresentation(res1)
            protein_feature.append(one_hot)

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
            protein_position.append(coord)
        positions_list.append(protein_position)
        sa_list.append(sa)
        features.append(protein_feature)
        i=i+1

    #print('sa length')
    #print(len(sa_list))
    positions=np.array(positions_list)
    num_atoms=np.full(6, 9, dtype=int) #117 is the number of Ca in T1104 pdbs; Using 6 proteins for training; using 9 (num_atoms) now to match the original work
    charges=np.array(sa_list) #Using sa values in place of charges
    index=np.full(6, 0, dtype=int)
    A=np.full(6, 0, dtype=int)
    B=np.full(6, 0, dtype=int)
    C=np.full(6, 0, dtype=int)
    mu=np.full(6, 0, dtype=int)
    alpha=np.full(6, 0, dtype=int)
    homo=np.full(6, 0, dtype=int)
    lumo=np.full(6, 0, dtype=int)
    gap=np.full(6, 0, dtype=int)
    r2=np.full(6, 0, dtype=int)
    zpve=np.full(6, 0, dtype=int)
    U0=np.full(6, 0, dtype=int)
    U=np.full(6, 0, dtype=int)
    H=np.full(6, 0, dtype=int)
    G=np.full(6, 0, dtype=int)
    Cv=np.full(6, 0, dtype=int)
    omega1=np.full(6, 0, dtype=int)
    zpve_thermo=np.full(6, 0, dtype=float)
    U0_thermo=np.full(6, 0, dtype=float)
    U_thermo=np.full(6, 0, dtype=float)
    H_thermo=np.full(6, 0, dtype=float)
    G_thermo=np.full(6, 0, dtype=float)
    Cv_thermo=np.full(6, 0, dtype=float)

    train_dictionary={'num_atoms': num_atoms, 'charges': charges, 'positions': positions, 'index': index, 'A': A, 'B': B, 'C': C, 'mu': mu, 'alpha': alpha, 'homo': homo, 'lumo': lumo, 'gap': gap, 'r2': r2, 'zpve': zpve, 'U0': U0, 'U': U, 'H': H, 'G': G, 'Cv': Cv, 'omega1': omega1, 'zpve_thermo': zpve_thermo, 'U0_thermo': U0_thermo, 'U_thermo': U_thermo, 'H_thermo': H_thermo, 'G_thermo': G_thermo, 'Cv_thermo': Cv_thermo}
    #print('Positions type')
    #print(type(positions))
    #print(positions)
    #print(features)

    # Load downloaded/processed datasets
    datasets = {}
    count=0 #My test variable
    for split, datafile in datafiles.items():
        #datasets[split] = {key: torch.from_numpy(
        #        val) for key, val in train_dictionary.items()}
        with np.load(datafile) as f:
            datasets[split] = {key: torch.from_numpy(
                val) for key, val in f.items()}
            #count=count+1
        if count==0: #My check condition
            if count==0:
              #print(split)
              #print((datasets[split]))
              pass
            count=count+1
    #print(count)

    if dataset != 'protein':
        np.random.seed(42)
        fixed_perm = np.random.permutation(len(datasets['train']['num_atoms']))
        if dataset == 'qm9_second_half':
            sliced_perm = fixed_perm[len(datasets['train']['num_atoms'])//2:]
        elif dataset == 'qm9_first_half':
            sliced_perm = fixed_perm[0:len(datasets['train']['num_atoms']) // 2]
        else:
            raise Exception('Wrong dataset name')
        for key in datasets['train']:
            datasets['train'][key] = datasets['train'][key][sliced_perm]

    # Basic error checking: Check the training/test/validation splits have the same set of keys.
    keys = [list(data.keys()) for data in datasets.values()]
    assert all([key == keys[0] for key in keys]
               ), 'Datasets must have same set of keys!'

    # TODO: remove hydrogens here if needed
    if remove_h:
        for key, dataset in datasets.items():
            pos = dataset['positions']
            charges = dataset['charges']
            num_atoms = dataset['num_atoms']

            # Check that charges corresponds to real atoms
            assert torch.sum(num_atoms != torch.sum(charges > 0, dim=1)) == 0

            mask = dataset['charges'] > 1
            new_positions = torch.zeros_like(pos)
            new_charges = torch.zeros_like(charges)
            for i in range(new_positions.shape[0]):
                m = mask[i]
                p = pos[i][m]   # positions to keep
                p = p - torch.mean(p, dim=0)    # Center the new positions
                c = charges[i][m]   # Charges to keep
                n = torch.sum(m)
                new_positions[i, :n, :] = p
                new_charges[i, :n] = c

            dataset['positions'] = new_positions
            dataset['charges'] = new_charges
            dataset['num_atoms'] = torch.sum(dataset['charges'] > 0, dim=1)

    # Get a list of all species across the entire dataset
    #all_species = _get_species(datasets, ignore_check=False)
    #print('Printing all_species from qm9->data->utils')
    #print(all_species)
    #all_species = torch.tensor([1,6,7,8,9]) #Commented out the all_species calculation. Passing the atom types manually.
    all_species = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])
    #all_species = torch.tensor([1,2,3,4,5])

    # Now initialize MolecularDataset based upon loaded data
    datasets = {split: ProcessedDataset(data, num_pts=num_pts.get(
        split, -1), included_species=all_species, subtract_thermo=subtract_thermo) for split, data in datasets.items()}

    # Now initialize MolecularDataset based upon loaded data

    # Check that all datasets have the same included species:
    #assert(len(set(tuple(data.included_species.tolist()) for data in datasets.values())) ==
    #       1), 'All datasets must have same included_species! {}'.format({key: data.included_species for key, data in datasets.items()})

    # These parameters are necessary to initialize the network
    num_species = datasets['train'].num_species
    max_charge = datasets['train'].max_charge

    # Now, update the number of training/test/validation sets in args
    args.num_train = datasets['train'].num_pts
    args.num_valid = datasets['valid'].num_pts
    args.num_test = datasets['test'].num_pts
    #print('Args---Number of train')
    #print(args.num_train)

    return args, datasets, num_species, max_charge


def _get_species(datasets, ignore_check=False):
    """
    Generate a list of all species.

    Includes a check that each split contains examples of every species in the
    entire dataset.

    Parameters
    ----------
    datasets : dict
        Dictionary of datasets.  Each dataset is a dict of arrays containing molecular properties.
    ignore_check : bool
        Ignores/overrides checks to make sure every split includes every species included in the entire dataset

    Returns
    -------
    all_species : Pytorch tensor
        List of all species present in the data.  Species labels shoguld be integers.

    """
    # Get a list of all species in the dataset across all splits
    all_species = torch.cat([dataset['charges'].unique()
                             for dataset in datasets.values()]).unique(sorted=True)

    # Find the unique list of species in each dataset.
    split_species = {split: species['charges'].unique(
        sorted=True) for split, species in datasets.items()}

    # If zero charges (padded, non-existent atoms) are included, remove them
    if all_species[0] == 0:
        all_species = all_species[1:]

    # Remove zeros if zero-padded charges exst for each split
    split_species = {split: species[1:] if species[0] ==
                     0 else species for split, species in split_species.items()}

    # Now check that each split has at least one example of every atomic spcies from the entire dataset.
    '''if not all([split.tolist() == all_species.tolist() for split in split_species.values()]):
        # Allows one to override this check if they really want to. Not recommended as the answers become non-sensical.
        if ignore_check:
            logging.error(
                'The number of species is not the same in all datasets!')
        else:
            raise ValueError(
                'Not all datasets have the same number of species!')'''

    # Finally, return a list of all species
    return all_species
