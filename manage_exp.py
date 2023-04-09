import os
from os import mkdir, path
import sys
import filters
import torch
from typing import Callable
import numpy as np
from tqdm import tqdm

_v0 = None 
_x0 = None

# create initial tensor
def get_x0(
        b_size: int, # batch size
        x_dim: int,  # dimension of x
        sigma: float # noise level
        ) -> torch.Tensor:
    
    global _x0
    if _x0 is None:
        _x0 = 3*torch.ones(b_size, x_dim)\
             + sigma * torch.randn(b_size, x_dim)
    x0 = _x0
    return x0

# create initial tensor test
def get_x0_test(
        b_size: int, # batch size
        x_dim: int,  # dimension of x
        sigma: float # noise level
        ) -> torch.Tensor:
    
    _x0_test = 3*torch.ones(b_size, x_dim)\
               + sigma * torch.randn(b_size, x_dim)
    x0 = _x0_test
    return x0

# get initial hidden layers
def get_ha0(b_size, h_dim):
    global _v0
    if _v0 is None:
        _v0 = torch.zeros(1, h_dim)

    ha0 = torch.zeros(b_size, h_dim)

    for b in range(b_size):
        ha0[b, :] = _v0
        
    return ha0

# activate cuda or not
def set_tensor_type(
        tensor_type: str,
        cuda: bool) -> None:
    
    print('use gpu',cuda)
    print('use tensor_type',tensor_type)
    if (tensor_type == "double") and cuda:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    elif (tensor_type == "float") and cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    elif (tensor_type == "double") and (not cuda):
        torch.set_default_tensor_type(torch.DoubleTensor)
    elif (tensor_type == "float") and (not cuda):
        torch.set_default_tensor_type(torch.FloatTensor)
    else:
        raise NameError("Unknown tensor_type")

def pre_train_full(
        net: filters.DAN, # the DAN
        b_size: int, # batch size
        h_dim: int, # dimension of hidden layers
        x_dim: int, # dimension of x
        sigma0: float, # initial noise level
        optimizer_classname: str, # optimizer class name
        optimizer_kwargs: dict, # optimizer parameters
        verbose: bool = True # verbose
        ) -> None:
    
    """
    Pre-train c at t=0
    # learn the parameters in net.c using ha0 and x0
    # by minimizing the L_0(q_0^a) loss
    """
    
    print('Pre-train c at t=0 \n')
    
    # generate x0 of batch size b_size at t=0
    x0 = get_x0(b_size, x_dim, sigma0)
    print(f"Empirical mean of $x_0$: {torch.mean(x0)}")
    
    # create an optimizer optimizer0 for the paramerters in c
    optimizer0 = eval(optimizer_classname)(net.c.parameters(), **optimizer_kwargs)
    # a counter of number of evaluations
    ite = 0
    
    # Initialize h0
    ha0 = get_ha0(b_size, h_dim)
    
    # close for LBFGS : clear the gradients, compute loss and return it
    def closure0():
        optimizer0.zero_grad()
        logpdf_a0 = - torch.mean(net.c.forward(ha0).log_prob(x0))

        logpdf_a0.backward()
        
        nonlocal ite
        ite += 1
        
        return logpdf_a0
    
    # run the optimizer
    optimizer0.step(closure0)
    
    # print out the final mean and covariance of q_0^a
    pdf_a0 = net.c(ha0)
    if verbose:
        print('## INIT a0 mean', pdf_a0.mean[0, :])  # mean of first sample
        print('## INIT a0 var', pdf_a0.variance[0, :])  # var first sample
        print('## INIT a0 covar', pdf_a0.covariance_matrix[0, :, :]) # covar first sample

def train_full(
        net: filters.DAN, # the DAN
        b_size: int, # batch size
        h_dim: int, # dimension of hidden layers
        x_dim: int, # dimension of x
        T: int, # time horizon
        checkpoint: int, # checkpoint
        direxp: str, # directory of experiment
        prop: Callable, # propagation function
        obs: Callable, # observation function
        sigma0: float, # initial noise level
        optimizer_classname: str, # optimizer class name
        optimizer_kwargs: dict, # optimizer parameters
        full_observable: bool = True, # fully observable system,
        verbose: bool = True # verbose
        ) -> None:
    
    """
    Train over full time 0..T with BPTT
    # learn the parameters in net.a, net.b, net.c using t=0..T
    # by minimizing the total loss
    """
    
    if not path.exists(direxp):
        print("Directory", direxp, "does not exist. Creating it.")
        mkdir(direxp)          
    
    print('Train full method with BPTT')

    # generate training data seq for t=0..T
    # rewrite xt and yt
    x0 = get_x0(b_size, x_dim, sigma0)
    y0 = obs(x0).sample(sample_shape=torch.Size([1])).squeeze(0)

    xt = [x0]
    yt = [y0]

    x = x0
    y = y0

    print('Computing data in outside loop')
    for t in range(1, T):
        x = prop(x).sample(sample_shape=torch.Size([1])).squeeze(0)
        
        # if not full observable, one chance out of 2 to observe
        if full_observable or np.random.rand() <= 0.5:
            y = obs(x).sample(sample_shape=torch.Size([1])).squeeze(0)
            yt.append(y.detach())
        else:
            yt.append(None)

        xt.append(x.detach())

    print('Finished computing data in outside loop')
    print('Starting optimization')
    # Train net using xt and yt, t = 1 .. T and x0
    # minimize total loss, by constructing optimizer and rewriting closure    
    ite = 0
    optimizer = eval(optimizer_classname)(net.parameters(), **optimizer_kwargs)
    ha0 = get_ha0(b_size, h_dim)

    # close for LBFGS : clear the gradients, compute loss and return it
    def closure():        
        # first use optimizer to set all the gradients to zero
        ha = ha0
        optimizer.zero_grad()

        # initial loss
        forw = net.c.forward(ha)
        loss0 = -torch.mean(forw.log_prob(x0))

        # compute total loss
        loss_total = torch.zeros_like(loss0)

        for t in range(0, T):
            x_cur, y_cur = xt[t], yt[t]

            if y_cur is not None:
                loss, ha = net.forward(ha, x_cur, y_cur)
                
            else:
                hb = net.b.forward(ha)
                y_pred = net.c.forward(hb).mean

                loss, ha = net.forward(ha, x_cur, y_pred)
            
            loss_total = loss_total + loss

        loss_total = loss_total / T + loss0
        loss_total.backward()

        # Checkpoint
        nonlocal ite
        if verbose and (ite == 1 or (ite % checkpoint == 0)):
            print("## Train Ite " + str(ite)+" ##")
            print_scores(net.scores)
        
        ite += 1

        return loss_total
    
    # run optimizer
    optimizer.step(closure)

    print('Finished optimization')
    print('Computing predictions')

    ha = ha0

    list_qa = [net.c.forward(ha0)]

    for t in range(0, T):
        x_cur, y_cur = xt[t], yt[t]

        if y_cur is not None:
            _, ha = net.forward(ha, x_cur, y_cur)
            
        else:
            hb = net.b.forward(ha)
            y_pred = net.c.forward(hb).mean

            _, ha = net.forward(ha, x_cur, y_pred)
        
        list_qa.append(net.c.forward(ha))

    save_dict(
        direxp,
        net=net,
        list_qa=list_qa,
        x=xt,
        y=yt,
        scores=net.scores,
        optimizer=optimizer.state_dict())
    
def train_online(
        net: filters.DAN, 
        b_size: int, 
        h_dim: int, 
        x_dim: int,
        T: int, 
        checkpoint: int, 
        direxp: str,
        prop: Callable, 
        obs: Callable, 
        sigma0: float,
        optimizer_classname: str, 
        optimizer_kwargs: dict, 
        scheduler_classname: str,
        scheduler_kwargs: dict,
        full_observable: bool = True,
        verbose: bool = True
        ) -> None:
    
    """
    Train functions for the DAN, online and truckated BPTT
    """
    
    if not path.exists(direxp):
        print("Directory", direxp, "does not exist. Creating it.")
        mkdir(direxp)
        
    # construct optimizer and scheduler
    assert(optimizer_classname != "NONE")
    print(' optimizer_classname', optimizer_classname)

    optimizer = eval(optimizer_classname)(net.parameters(), **optimizer_kwargs)

    assert(scheduler_classname != "NONE")
    print(' scheduler_classname', scheduler_classname)
    
    scheduler = eval(scheduler_classname)(optimizer, **scheduler_kwargs)

    x0 = get_x0(b_size, x_dim, sigma0)
    y0 = obs(x0).sample(sample_shape=torch.Size([1])).squeeze(0)
    ha0 = get_ha0(b_size, h_dim)

    x = x0
    ha = ha0

    xt = [x0]
    yt = [y0]

    for t in (pbar := tqdm(range(1, T))):
        # on the fly data generation
        x = prop(x0)\
            .sample(sample_shape=torch.Size([1]))\
            .squeeze(0)
        
        if full_observable or np.random.rand() <= 0.5:
            observed = "true"
            y = obs(x).sample(sample_shape=torch.Size([1])).squeeze(0)
            yt.append(y.detach())

            # optimization
            optimizer.zero_grad()
            loss, ha = net.forward(ha, x, y)

        else:
            observed = "false"
            yt.append(None)

            hb = net.b.forward(ha)
            y_pred = net.c.forward(hb).mean

            # optimization
            optimizer.zero_grad()
            loss, ha = net.forward(ha, x, y_pred)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Truncated back propagation through time
        ha = ha.detach()
        x0 = x

        # save data
        xt.append(x0.detach())
        
        pbar.set_description_str(f"Time {t: >5}")
        pbar.set_postfix(
            {   
                "observed": f"{observed}",
                "train_loss": f"{loss.item():.2f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}",
            }
        )

        # Checkpoint
        if verbose and ((t % checkpoint == 0) or (t == T)):
            if ha is not None:
                print("## Train Cycle " + str(t)+" ##")
                print_scores(net.scores)

    save_dict(
            direxp,
            net=net,
            x=xt,
            y=yt,
            scores=net.scores,
            optimizer=optimizer.state_dict())

@torch.no_grad()
def test(
    net: filters.DAN, 
    b_size: int, 
    h_dim: int, 
    x_dim: int,
    T: int, 
    checkpoint: int, 
    direxp: str,  
    prop: Callable, 
    obs: Callable, 
    sigma0: float,
    full_observable: bool = True,
    verbose: bool = True
    ) -> None:

    x0 = get_x0_test(b_size, x_dim, sigma0)
    y0 = obs(x0).sample(sample_shape=torch.Size([1])).squeeze(0)
    ha0 = get_ha0(b_size, h_dim)

    x_test = [x0]
    y_test = [y0]
    list_qa_test = [net.c.forward(ha0)]

    x = x0
    y = y0
    ha = ha0

    net.eval()

    for t in range(1, T+1):
        # on the fly data generation
        x = prop(x).sample(sample_shape=torch.Size([1])).squeeze(0)
        
        # if not full observable, one chance out of 2 to observe
        if full_observable or np.random.rand() <= 0.5:
            y = obs(x).sample(sample_shape=torch.Size([1])).squeeze(0)
            y_test.append(y.detach())

            # Evaluates the loss
            _, ha = net(ha, x, y)
            list_qa_test.append(net.c.forward(ha))

        else:
            y_test.append(None)
            hb = net.b.forward(ha)
            y_pred = net.c.forward(hb).mean

            _, ha = net(ha, x, y_pred)
            list_qa_test.append(net.c.forward(ha))
        
        
        x_test.append(x.detach())
        

        # Checkpoint
        if verbose and ((t % checkpoint == 0) or (t == T)):
            print("## Test Cycle " + str(t)+" ##")
            print_scores(net.scores)
    
    save_dict(
        direxp,
        x_test=x_test,
        y_test=y_test,
        list_qa_test=list_qa_test,
        test_scores=net.scores)

# launch the experiment
def experiment(
    tensor_type: str, 
    seed: int,
    net_classname: str, 
    net_kwargs: dict,
    sigma0: float, 
    prop_kwargs: dict, 
    obs_kwargs: dict,
    train_kwargs: dict, 
    test_kwargs: dict,
    optimizer_classname: str, 
    optimizer_kwargs: dict,
    scheduler_classname: str, 
    scheduler_kwargs: dict,
    directory: str, 
    nameexp: str, 
    full_observable: bool = True,
    verbose: bool = True):

    # CPU or GPU tensor
    cuda = torch.cuda.is_available()
    
    set_tensor_type(tensor_type, cuda)

    # Reproducibility
    torch.manual_seed(seed)

    net = eval(net_classname)(**net_kwargs)
    prop = filters.Constructor(**prop_kwargs)
    obs = filters.Constructor(**obs_kwargs)
    b_size = train_kwargs['b_size']
    h_dim = train_kwargs['h_dim']
    x_dim = train_kwargs['x_dim']
    T = train_kwargs['T']
    checkpoint = train_kwargs['checkpoint']
    direxp = directory + nameexp
    
    if train_kwargs["mode"] == "full":
        print('Pretraining the network')
        pre_train_full(net,b_size,h_dim,x_dim,sigma0,
                       optimizer_classname,optimizer_kwargs,verbose)        
        print('Training the network')
        train_full(net, b_size, h_dim, x_dim,
                   T, checkpoint, direxp,
                   prop, obs, sigma0,
                   optimizer_classname, optimizer_kwargs, full_observable, verbose)
    else:
        print('Training the network')
        train_online(net, b_size, h_dim, x_dim,
                     T, checkpoint, direxp,
                     prop, obs, sigma0,
                     optimizer_classname, optimizer_kwargs, 
                     scheduler_classname, scheduler_kwargs, full_observable, verbose)
    
    # Clear scores
    net.clear_scores()

    # Testing
    b_size = test_kwargs['b_size']
    h_dim = test_kwargs['h_dim']
    x_dim = test_kwargs['x_dim']
    T = test_kwargs['T']
    checkpoint = test_kwargs['checkpoint']
    test(net, b_size, h_dim, x_dim,
         T, checkpoint, direxp,
         prop, obs, sigma0, full_observable, verbose)


def save_dict(prefix, **kwargs):
    """
    saves the arg dict val with name "prefix + key + .pt"
    """
    for key, val in kwargs.items():
        torch.save(val, prefix + key + ".pt")


def print_scores(scores):
    for key, val in scores.items():
        if len(val) > 0:
            print(key+"= "+str(val[-1]))


def update(k_default, k_update):
    """Update a default dict with another dict
    """
    for key, value in k_update.items():
        if isinstance(value, dict):
            k_default[key] = update(k_default[key], value)
        else:
            k_default[key] = value
    return k_default


def update_and_save(k_default, list_k_update, name_fun):
    """update and save a default dict for each dict in list_k_update,
    generates a name for the exp with name_fun: dict -> string
    returns the exp names on stdout
    """
    out, directory = "", k_default["directory"]
    for k_update in list_k_update:
        nameexp = name_fun(k_update)
        if not os.path.exists(nameexp):
            os.mkdir(nameexp)
        k_default["nameexp"] = nameexp + "/"
        torch.save(update(k_default, k_update), nameexp + "/kwargs.pt")
        out += directory + "," + nameexp

    # return the dir and nameexp
    sys.stdout.write(out)


if __name__ == "__main__":
    """
    the next argument is the experiment name
    - launch the exp
    """
    torch.autograd.set_detect_anomaly(True)
    cuda = torch.cuda.is_available()

    # device = torch.device("cuda" if cuda else "cpu")

    # Set device
    if (int(torch.__version__.split(".")[0]) >= 2) or (int(torch.__version__.split(".")[1]) >= 13) and torch.has_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    experiment(
        **torch.load(
            sys.argv[1] + "/kwargs.pt",
            map_location=torch.device(device))
        )
