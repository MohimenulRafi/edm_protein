import torch
from torch.utils.data import Dataset

import os
from itertools import islice
from math import inf

import logging

class ProcessedDataset(Dataset):
    """
    Data structure for a pre-processed cormorant dataset.  Extends PyTorch Dataset.

    Parameters
    ----------
    data : dict
        Dictionary of arrays containing molecular properties.
    included_species : tensor of scalars, optional
        Atomic species to include in ?????.  If None, uses all species.
    num_pts : int, optional
        Desired number of points to include in the dataset.
        Default value, -1, uses all of the datapoints.
    normalize : bool, optional
        ????? IS THIS USED?
    shuffle : bool, optional
        If true, shuffle the points in the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.

    """
    def __init__(self, data, included_species=None, num_pts=-1, normalize=True, shuffle=True, subtract_thermo=True):

        self.data = data

        if num_pts < 0:
            self.num_pts = len(data['charges'])
        else:
            if num_pts > len(data['charges']):
                logging.warning('Desired number of points ({}) is greater than the number of data points ({}) available in the dataset!'.format(num_pts, len(data['charges'])))
                self.num_pts = len(data['charges'])
            else:
                self.num_pts = num_pts

        # If included species is not specified
        if included_species is None:
            included_species = torch.unique(self.data['charges'], sorted=True)
            if included_species[0] == 0:
                included_species = included_species[1:]

        if subtract_thermo:
            thermo_targets = [key.split('_')[0] for key in data.keys() if key.endswith('_thermo')]
            if len(thermo_targets) == 0:
                logging.warning('No thermochemical targets included! Try reprocessing dataset with --force-download!')
            else:
                logging.info('Removing thermochemical energy from targets {}'.format(' '.join(thermo_targets)))
            for key in thermo_targets:
                data[key] -= data[key + '_thermo'].to(data[key].dtype)

        self.included_species = included_species
        ss=torch.tensor([1,2,3])
        #residue_type=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
        #residue_type_tensor=torch.tensor(residue_type, dtype=torch.int)
        #self.included_species = residue_type
        #print('Printing included_species from qm9 data dataset_class')
        #print(included_species)
        #print('Printing included species from qm9->data->dataset_class')
        #print(self.included_species)
        #print('Printing charges form qm9->data->dataset_class')
        #print(self.data['charges'].unsqueeze(-1))

        #print('Printing charges')
        #print(data['charges'].shape)
        #residue_index_proteins=data['residue_index'].tolist()
        number_of_proteins=len(data['charges'])
        atom_type_protein=[]
        #residue_index=[]
        number_of_atoms_list=data['num_atoms'].tolist()
        max_num=max(number_of_atoms_list)
        '''for i in range(number_of_proteins):
            #residue_index=residue_index_proteins[i]
            number_of_atoms=number_of_atoms_list[i]
            num_blank_atoms=max_num-number_of_atoms #used 117 instad of max_num earlier
            carbon_atom_list=[]
            for j in range(number_of_atoms):
                carbon_atom_list.append(6) #was passing 6 for carbon type earlier
            for j in range(num_blank_atoms):
                carbon_atom_list.append(0)
            atom_type_protein.append(carbon_atom_list)
        atoms=torch.tensor(atom_type_protein)'''

        #self.data['one_hot'] = self.data['charges'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
        #self.data['one_hot'] = atoms.unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
        self.data['one_hot'] = self.data['residue_code'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
        self.data['one_hot_2'] = self.data['ss_code'].unsqueeze(-1) == ss.unsqueeze(0).unsqueeze(0)
        #print('dataclass')
        #print(self.data['charges'].shape)
        #print(self.data['phi'].shape)
        #self.data['one_hot'] = self.data['residue_index'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
        #print(self.data['one_hot'].shape)
        #print('Printing one_hot from qm9->data->dataset_class')
        #print(self.data['one_hot'])

        self.num_species = len(included_species)
        self.max_charge = max(included_species)
        #self.num_species = len(residue_type)
        #self.max_charge = max(residue_type)

        self.parameters = {'num_species': self.num_species, 'max_charge': self.max_charge}

        # Get a dictionary of statistics for all properties that are one-dimensional tensors.
        self.calc_stats()

        if shuffle:
            self.perm = torch.randperm(len(data['charges']))[:self.num_pts]
        else:
            self.perm = None

    def calc_stats(self):
        self.stats = {key: (val.mean(), val.std()) for key, val in self.data.items() if type(val) is torch.Tensor and val.dim() == 1 and val.is_floating_point()}

    def convert_units(self, units_dict):
        for key in self.data.keys():
            if key in units_dict:
                self.data[key] *= units_dict[key]

        self.calc_stats()

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]
        return {key: val[idx] for key, val in self.data.items()}
