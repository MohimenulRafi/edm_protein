import wandb
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import numpy as np
import protein.visualizer as vis
from protein.analyze import analyze_stability_for_molecules
from protein.sampling import sample_chain, sample, sample_sweep_conditional
import utils
import protein.utils as proteinutils
from protein import losses
import time
import torch


def train_epoch(args, loader, epoch, model, model_dp, model_ema, ema, device, dtype, property_norms, optim,
                nodes_dist, gradnorm_queue, dataset_info, prop_dist):
    model_dp.train()
    model.train()
    nll_epoch = []
    #loss_array=[]
    f_loss_writer=open('/home/common/proj/EDM_Protein/Loss/losses_fulllength100norm_v3.txt', 'a')
    n_iterations = len(loader)
    for i, data in enumerate(loader):
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        one_hot_2 = data['one_hot_2'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
        dihedrals = data['dihedrals'].to(device, dtype)
        relative_pos = data['relative_pos'].to(device, dtype)
        #phi = (data['phi'] if args.include_charges else torch.zeros(0)).to(device, dtype)
        #psi = (data['psi'] if args.include_charges else torch.zeros(0)).to(device, dtype)

        #print('Printing node mask from train_test.py')
        #print(node_mask.shape)
        #print(node_mask)

        x = remove_mean_with_mask(x, node_mask)

        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise

        x = remove_mean_with_mask(x, node_mask)
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()

        check_mask_correct([x, one_hot, one_hot_2, dihedrals, charges, relative_pos], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'categorical2': one_hot_2, 'integer': charges, 'phipsi': dihedrals, 'relative_pos': relative_pos} #added 'categorical2': one_hot_2,

        if len(args.conditioning) > 0:
            context = proteinutils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        optim.zero_grad()

        # transform batch through flow
        nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, model_dp, nodes_dist,
                                                                x, h, node_mask, edge_mask, context, 'Train', epoch)
        # standard nll from forward KL
        loss = nll + args.ode_regularization * reg_term
        loss.backward()

        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.

        optim.step()

        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)

        if i % args.n_report_steps == 0:
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                  f"RegTerm: {reg_term.item():.1f}, "
                  f"GradNorm: {grad_norm:.1f}")
            if i==n_iterations-1:
                f_loss_writer.write(str(loss.item()))
                f_loss_writer.write('\n')
            #if loss.item()<=1100.0:
            #    break
        nll_epoch.append(nll.item())
        '''if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0):
            start = time.time()
            if len(args.conditioning) > 0:
                save_and_sample_conditional(args, device, model_ema, prop_dist, dataset_info, epoch=epoch)
            save_and_sample_chain(model_ema, args, device, dataset_info, prop_dist, epoch=epoch,
                                  batch_id=str(i))
            sample_different_sizes_and_save(model_ema, nodes_dist, args, device, dataset_info,
                                            prop_dist, epoch=epoch)
            print(f'Sampling took {time.time() - start:.2f} seconds')

            vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}", dataset_info=dataset_info, wandb=wandb)
            vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}_{i}/chain/", dataset_info, wandb=wandb)
            if len(args.conditioning) > 0:
                vis.visualize_chain("outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch), dataset_info,
                                    wandb=wandb, mode='conditional')'''
        wandb.log({"Batch NLL": nll.item()}, commit=True)
        if args.break_train_epoch:
            break

        #del x,node_mask,edge_mask,one_hot,one_hot_2,charges,phi,psi,h
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)
    f_loss_writer.close()


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            #print('Printing variable shape')
            #print(variable.shape)
            #print('Printing variable from train_test check_mask_correct function')
            #print(variable)
            assert_correctly_masked(variable, node_mask)


def allZero(seq):
    zero_count=seq.count(0.0)
    if zero_count == 3:
        return True
    else:
        return False

def test(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, dataset_info, partition='Test'):
    #print('This point 1 is reached in test')
    eval_model.eval()
    #print('This reached')
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0
        #print('Check 1 after no grad')

        n_iterations = len(loader)
        #print('Check 2 after no grad')

        protein_number=0
        for i, data in enumerate(loader):
            #print('Point 1 in loop')
            x = data['positions'].to(device, dtype)
            batch_size = x.size(0)
            #print('Printing batch size from train test py')
            #print(batch_size)
            #print(x)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            #print('Point 2 in loop')
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            one_hot_2 = data['one_hot_2'].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
            dihedrals = data['dihedrals'].to(device, dtype)
            relative_pos = data['relative_pos'].to(device, dtype)
            #phi = (data['phi'] if args.include_charges else torch.zeros(0)).to(device, dtype)
            #psi = (data['psi'] if args.include_charges else torch.zeros(0)).to(device, dtype)

            #print('This point 2 is reached')
            if args.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(x.size(),
                                                                    x.device,
                                                                    node_mask)
                x = x + eps * args.augment_noise

            #print('This point 3 is reached')
            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, one_hot_2, dihedrals, charges, relative_pos], node_mask) #added one_hot_2,
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'categorical2': one_hot_2, 'integer': charges, 'phipsi': dihedrals, 'relative_pos': relative_pos} #added 'categorical2': one_hot_2,

            if len(args.conditioning) > 0:
                context = proteinutils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            #Added this condition- Mohimenul
            if partition!='Test':
                # transform batch through flow
                nll, _, _ = losses.compute_loss_and_nll(args, eval_model, nodes_dist, x, h,
                                                        node_mask, edge_mask, context, partition, epoch)


            if partition=='Test':
                molecule_size=0
                cat_h = torch.cat([h['categorical'], h['categorical2'], h['phipsi'], h['integer'], h['relative_pos']], dim=2)
                if partition=='Test':
                    x_list=x.tolist()
                    for molecule in range(len(x_list)):
                        for line_no in range(len(x_list[molecule])):
                            line=x_list[molecule][line_no]
                            if(allZero(line)):
                                break
                            molecule_size=molecule_size+1
                #print(molecule_size)
                m_size=[]
                m_size.append(molecule_size)

                max_size=203 #116 #14 #21 #5 #108 #245 #203
                x_list=x.tolist()
                h_list=cat_h.tolist()
                if(len(x_list[0])<max_size):
                    diff=max_size-len(x_list[0])
                    for i in range(diff):
                        zero_list_x=[]
                        for j in range(3):
                            zero_list_x.append(0.0)
                        x_list[0].append(zero_list_x)
                        zero_list_h=[]
                        for j in range(28):
                            zero_list_h.append(0.0)
                        h_list[0].append(zero_list_h)

                #print(x_list)
                test_x=torch.tensor(x_list).to(device, dtype)
                test_h=torch.tensor(h_list).to(device, dtype)
                
                test_z = torch.cat([test_x, test_h], dim=2).to(device, dtype) #Preparing for denoising- Mohimenul
            if epoch==2000:
                batch_size = 1
                #protein_number=protein_number*batch_size+1
                for counter in range(1):
                    #nodesxsample = nodes_dist.sample(batch_size)
                    nodesxsample = torch.tensor(m_size)
                    #print('printing nodesxsample')
                    #print(nodesxsample)
                    one_hot, one_hot_2, charges, x, node_mask = sample(args, device, eval_model, nodesxsample=nodesxsample, dataset_info=dataset_info, test_z=test_z, protein_size=molecule_size) #added one_hot_2,
                    coord_list=x.tolist()
                    one_hot_list=one_hot.tolist()
                    one_hot_list_2=one_hot_2.tolist()
                    #print(one_hot_list)
                    #print(one_hot_list_2)
                    for p in range(len(coord_list)):
                        prtn_num=protein_number+p
                        #fp=open('/home/common/proj/EDM_Protein/PositionAndOnehot/protein_'+str(prtn_num)+'positions.txt', 'w')
                        fp=open('/home/common/proj/EDM_Protein/PositionAndOnehot/FullLength_100Norm_v3/protein_'+str(molecule_size)+'_0_positions.txt', 'w')
                        #fo=open('/home/common/proj/EDM_Protein/PositionAndOnehot/protein_'+str(prtn_num)+'one_hot.txt', 'w')
                        fo=open('/home/common/proj/EDM_Protein/PositionAndOnehot/FullLength_100Norm_v3/protein_'+str(molecule_size)+'_0_one_hot.txt', 'w')
                        #fo2=open('/home/common/proj/EDM_Protein/PositionAndOnehot/protein_'+str(prtn_num)+'one_hot_2.txt', 'w')
                        for c in range(len(coord_list[p])):
                            coord_line=coord_list[p][c]
                            for coord in coord_line:
                                fp.write(str(coord)+' ')
                            fp.write('\n')
                        for oh in range(len(one_hot_list[p])):
                            oh_line=one_hot_list[p][oh]
                            for b in oh_line:
                                fo.write(str(b)+' ')
                            fo.write('\n')
                        #for oh2 in range(len(one_hot_list_2[p])):
                        #    oh_line_2=one_hot_list_2[p][oh2]
                        #    for b in oh_line_2:
                        #        fo2.write(str(b)+' ')
                        #    fo2.write('\n')
                        fp.close()
                        fo.close()
                        #fo2.close()
                            
            # standard nll from forward KL

            #Added this condition- Mohimenul
            if partition!='Test':
                nll_epoch += nll.item() * batch_size
                n_samples += batch_size
                if i % args.n_report_steps == 0:
                    print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                          f"NLL: {nll_epoch/n_samples:.2f}")
    if partition=='Test':
        return 0
    return nll_epoch/n_samples


def save_and_sample_chain(model, args, device, dataset_info, prop_dist,
                          epoch=0, id_from=0, batch_id=''):
    one_hot, charges, x = sample_chain(args=args, device=device, flow=model,
                                       n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)

    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                      one_hot, charges, x, dataset_info, id_from, name='chain')

    return one_hot, charges, x


def sample_different_sizes_and_save(model, nodes_dist, args, device, dataset_info, prop_dist,
                                    n_samples=5, epoch=0, batch_size=100, batch_id=''):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(args, device, model, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample,
                                                dataset_info=dataset_info)
        print(f"Generated molecule: Positions {x[:-1, :, :]}")
        vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/', one_hot, charges, x, dataset_info,
                          batch_size * counter, name='molecule')


def analyze_and_save(epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=1000, batch_size=100):
    print(f'Analyzing molecule stability at epoch {epoch}...')
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    for i in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(args, device, model_sample, dataset_info, prop_dist,
                                                nodesxsample=nodesxsample)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)

    wandb.log(validity_dict)
    if rdkit_tuple is not None:
        wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
    return validity_dict


def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0):
    one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)

    vis.save_xyz_file(
        'outputs/%s/epoch_%d/conditional/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    return one_hot, charges, x
