from torch.utils.data import DataLoader
from protein.data.args import init_argparse
from protein.data.collate import PreprocessProtein
from protein.data.utils import initialize_datasets
import os
import numpy as np


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

def retrieve_dataloaders(cfg):
    if 'protein' in cfg.dataset:
        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        filter_n_atoms = cfg.filter_n_atoms
        # Initialize dataloader
        args = init_argparse('protein')
        # data_dir = cfg.data_root_dir
        args, datasets, num_species, charge_scale = initialize_datasets(args, cfg.datadir, cfg.dataset,
                                                                        subtract_thermo=args.subtract_thermo,
                                                                        force_download=args.force_download,
                                                                        remove_h=cfg.remove_h)
        #qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114,
        #             'lumo': 27.2114}

        #####Modifications#####
        '''positions=[]
        features=[]
        i=1
        while(i<=10):
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
            #one_hot=''
            protein_position=[]
            protein_feature=[]
            x1=0.0
            y1=0.0
            z1=0.0

            #print('Number of Ca')
            #print(len(proteinDssp))
            for ds1 in range(len(proteinDssp)):
                coord=[]
                res1=proteinDssp[ds1][13:14].strip()
                one_hot=getOneHotRepresentation(res1)
                protein_feature.append(one_hot)

                x1=float(proteinDssp[ds1][117:(117+5)])
                y1=float(proteinDssp[ds1][124:(124+5)])
                z1=float(proteinDssp[ds1][131:(131+5)])
                coord.append(x1)
                coord.append(y1)
                coord.append(z1)
                protein_position.append(coord)
            positions.append(protein_position)
            features.append(protein_feature)
            i=i+1

        #print(positions)
        #print(features)'''
        #####End of the modification#####

        for dataset in datasets.values():
            #print('Printing dataset from qm9 dataset')
            #print(dataset)
            #dataset.convert_units(qm9_to_eV) #Commented this out (Mohimenul)
            pass

        #for x in datasets.items():
            #print(type(x[1]))
                
        if filter_n_atoms is not None:
            print("Retrieving molecules with only %d atoms" % filter_n_atoms)
            datasets = filter_atoms(datasets, filter_n_atoms)

        # Construct PyTorch dataloaders from datasets
        preprocess = PreprocessProtein(load_charges=cfg.include_charges)
        #print('After preprocess')
        #for k, v in datasets.items():
        #    if k=='train':
        #        print(v)
        dataloaders = {split: DataLoader(dataset,
                                         batch_size=1 if (split == 'test') else batch_size,
                                         shuffle=args.shuffle if (split == 'train') else False,
                                         num_workers=num_workers,
                                         collate_fn=preprocess.collate_fn)
                             for split, dataset in datasets.items()}
    elif 'geom' in cfg.dataset:
        import build_geom_dataset
        from configs.datasets_config import get_dataset_info
        data_file = './data/geom/geom_drugs_30.npy'
        dataset_info = get_dataset_info(cfg.dataset, cfg.remove_h)

        # Retrieve QM9 dataloaders
        split_data = build_geom_dataset.load_split_data(data_file,
                                                        val_proportion=0.1,
                                                        test_proportion=0.1,
                                                        filter_size=cfg.filter_molecule_size)
        transform = build_geom_dataset.GeomDrugsTransform(dataset_info,
                                                          cfg.include_charges,
                                                          cfg.device,
                                                          cfg.sequential)
        dataloaders = {}
        for key, data_list in zip(['train', 'val', 'test'], split_data):
            dataset = build_geom_dataset.GeomDrugsDataset(data_list,
                                                          transform=transform)
            shuffle = (key == 'train') and not cfg.sequential

            # Sequential dataloading disabled for now.
            dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
                sequential=cfg.sequential, dataset=dataset,
                batch_size=cfg.batch_size,
                shuffle=shuffle)
        del split_data
        charge_scale = None
    else:
        raise ValueError(f'Unknown dataset {cfg.dataset}')

    return dataloaders, charge_scale


def filter_atoms(datasets, n_nodes):
    for key in datasets:
        dataset = datasets[key]
        idxs = dataset.data['num_atoms'] == n_nodes
        for key2 in dataset.data:
            dataset.data[key2] = dataset.data[key2][idxs]

        datasets[key].num_pts = dataset.data['one_hot'].size(0)
        datasets[key].perm = None
    return datasets
