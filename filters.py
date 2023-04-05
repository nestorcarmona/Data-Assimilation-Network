"""
This file contains the DAN and function to construct the neural networks
"""
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal as Mvn
from typing import Tuple, Union
import numpy as np

class DAN(nn.Module):
    """
    A Data Assimilation Network class
    """

    def __init__(
            self: nn.Module,
            a_kwargs: dict,
            b_kwargs: dict,
            c_kwargs: dict 
            ) -> None:

        nn.Module.__init__(self)
        self.a = Constructor(**a_kwargs)
        self.b = Constructor(**b_kwargs)
        self.c = Constructor(**c_kwargs)

        self.scores = {
            "RMSE_b": [],
            "RMSE_a": [],
            "LOGPDF_b": [],
            "LOGPDF_a": [],
            "LOSS": []}

    # forward pass in the DAN
    def forward(
            self: nn.Module, # class
            ha: torch.Tensor, # hidden state
            x: torch.Tensor, # prior state
            y: torch.Tensor # new observation
            ) -> Tuple[torch.Tensor, torch.Tensor]:


        hb = self.b.forward(ha)
        pdf_b = self.c.forward(hb)

        ha = self.a.forward(torch.concat([hb, y], dim=1))
        pdf_a = self.c.forward(ha)

        # Compute loss
        logpdf_a = -torch.mean(pdf_a.log_prob(x))
        logpdf_b = -torch.mean(pdf_b.log_prob(x))

        loss = logpdf_a + logpdf_b

        # Compute scores
        with torch.no_grad():
            if logpdf_a is not None:
                self.scores["RMSE_b"].append(
                    torch.mean(
                        torch.norm(pdf_b.mean - x, dim=1)*x.size(1)**-.5).item()
                )
                
                self.scores["RMSE_a"].append(
                    torch.mean(torch.norm(
                        pdf_a.mean - x, dim=1)*x.size(1)**-.5).item()
                )
                
                self.scores["LOGPDF_b"].append(
                    logpdf_b.item()
                )
                
                self.scores["LOGPDF_a"].append(
                    logpdf_a.item()
                )

                self.scores["LOSS"].append(
                    loss.item()
                )


        return loss, ha

    # clear the scores lists
    def clear_scores(
            self: nn.Module) -> None:

        for v in self.scores.values():
            v.clear()


class Id(nn.Module):
    """ A simple id function
    """

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        """ trivial
        """
        return x


class Cst(nn.Module):
    """ A constant scale_vec
    """

    def __init__(self, init, dim=None):
        nn.Module.__init__(self)
        if isinstance(init, torch.Tensor):
            self.c = init.unsqueeze(0)
        else:
            raise NameError("Cst init unknown")
        

    def forward(self, x):
        return self.c.expand(x.size(0), self.c.size(0))


class Lin2d(nn.Module):
    # rotation dynamics
    def __init__(
            self: nn.Module, # class
            x_dim: int, # dimension of input tensir to rotate
            N: int, # number of rotations
            dt: float, # time steps
            init: str, # ??
            window=None # window of the input tensor
            ) -> None:
        
        assert(x_dim == 2)
        nn.Module.__init__(self)
    
        # implement M, rotation matrix
        theta = np.pi/100
        self.M = torch.tensor(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)]
            ]
        )

        # time dependent ?
        self.Mt = self.M.t()

        self.x_dim = x_dim
        self.N = N
        self.dt = dt
        self.init = init
        self.window = window


    def forward(
            self: nn.Module, # class
            x: torch.Tensor # input tensor
            ) -> torch.Tensor:
        
        # input x: (mb,x_dim)
        # output Mx: (mb,x_dim)
        
        # apply rotation matrix N times
        for i in range(self.N):
            x = torch.matmul(x, self.Mt)

        return x 


class EDO(nn.Module):
    """ Integrates an EDO with RK4
    """

    def __init__(
            self: nn.Module, # class
            x_dim: int, # dimension of input tensor
            N: int, # number of times to rotate
            dt: float, # time steps
            init: str, # Lorenz initialization
            window=None # window of the input tensor
            ) -> None:

        nn.Module.__init__(self)
        self.x_dim = x_dim
        self.N = N
        self.dt = dt

        if init == "95":
            """ Lorenz95 (96) initialization
            """
            self.window = (-2, -1, 0, 1)
            self.diameter = 4
            self.A = torch.tensor([[[0., 0., 0., 0.],
                                  [-1., 0., 0., 0.],
                                  [0., 0., 0., 0.],
                                  [0., 1., 0., 0.]]])
            self.b = torch.tensor([[0., 0., -1., 0.]])
            self.c = torch.tensor([8.])
        else:
            raise NameError("EDO init not available")

    def edo(
            self: nn.Module, # class
            x: torch.Tensor # input
            ) -> torch.Tensor:
        
        # input x: (mb,x_dim)
        # output dx/dt: (mb,x_dim)
        # Hint: convert x into v (mb,x_dim,4), then reshape into (mb*x_dim,4)
        # and apply the matrix self.A using torch.nn.functional.bilinear, etc
        """v= x-2 x-1 x0 x1
        """

        # convert x into v
        v = torch.cat(
            [torch.roll(
                x.unsqueeze(1), -i, 2
            ) for i in self.window], 1
        )

        v = torch.transpose(v, 1, 2)
        v_reshaped = v.reshape(-1, self.diameter)

        # solve system
        dx = torch.nn.functional.bilinear(v_reshaped, v_reshaped, self.A)\
            + torch.nn.functional.linear(v_reshaped, self.b, self.c)
        
        return dx.view(x.size(0), x.size(1))

    def forward(self, x):
        for _ in range(self.N):
            k1 = self.edo(x)
            k2 = self.edo(x + 0.5*self.dt*k1)
            k3 = self.edo(x + 0.5*self.dt*k2)
            k4 = self.edo(x + self.dt*k3)
            x = x + (self.dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
        return x


class FullyConnected(nn.Module):
    """ Fully connected NN ending with a linear layer
    """

    def __init__(self, layers, activation_classname):
        nn.Module.__init__(self)
        n = len(layers)
        self.lins = nn.ModuleList(
            [nn.Linear(d0, d1) for
             d0, d1 in zip(layers[:-1], layers[1:])])
        self.acts = nn.ModuleList(
            [eval(activation_classname)() for _ in range(n-2)])

    def forward(self, h):
        for lin, act in zip(self.lins[:-1], self.acts):
            h = act(lin(h))
        return self.lins[-1](h)


class FcZero(nn.Module):
    """
    Fully connected neural network with ReZero trick
    """

    def __init__(
            self: nn.Module, 
            dim: int, 
            deep: int, 
            activation_classname: str
            ) -> None:
        
        """
        layers: the list of the layers dimensions
        """

        nn.Module.__init__(self)
        layers = (deep+1)*[dim]

        self.lins = nn.ModuleList([nn.Linear(d0, d1) for d0, d1 in zip(layers[:-1], layers[1:])])
        self.acts = nn.ModuleList([eval(activation_classname)() for _ in range(deep)])
        self.alphas = nn.Parameter(torch.zeros(deep))

    def forward(self, h):
        for lin, act, alpha in zip(self.lins, self.acts, self.alphas):
            h = h + alpha*act(lin(h))
        return h


class FcZeroLin(nn.Module):
    """
    FcZero network ending with linear layer
    """

    def __init__(
            self: nn.Module, 
            in_dim: int, 
            out_dim: int, 
            deep: int, 
            activation_classname: str
            ) -> None:
        
        """
        layers: the list of the layers dimensions
        """

        nn.Module.__init__(self)
        self.fcZero = FcZero(in_dim, deep, activation_classname)
        self.out_dim = out_dim
        assert(out_dim <= in_dim)
        self.lin = FullyConnected([in_dim, out_dim], activation_classname)

    def forward(self, h):
        h = self.fcZero(h)
        h = self.lin(h)
        return h


class Gaussian(Mvn):
    """
    Return a pytorch Gaussian pdf from args
    args is either a (loc, scale_tril) or a (x_dim, vec)
    """

    def __init__(
            self: nn.Module, # class
            *args: Union[torch.Tensor, int] # arguments
            ) -> None:
        
        self.minexp = torch.Tensor([-8.0])
        self.maxexp = torch.Tensor([8.0])

        if isinstance(args[0], int):
            """args is a (x_dim, vec)
            loc is the first x_dim coeff of vec
            if the rest is one coeff c then
                scale_tril = e^c*I
            else
                scale_tril is filled diagonal by diagonal
                starting by the main one
                (which is exponentiated to ensure strict positivity)
            """

            # initializations
            x_dim, vec = args
            vec_dim = vec.size(-1)

            if vec_dim == x_dim + 1:
                # print('Mvn: scale_tril = e^c*I')

                loc = vec[:, :x_dim]

                scale_tril = torch.eye(x_dim)\
                                  .unsqueeze(0)\
                                  .expand(vec.size(0), -1, -1)
                
                scale_tril = torch.exp(vec[:, x_dim])\
                                  .view(vec.size(0), 1, 1)*scale_tril
                
                # clamp scale_tril to avoid numerical issues
                scale_tril = torch.clamp(scale_tril, self.minexp, self.maxexp)
            
            else:
                # print('Mvn by mean and cov')
                # rewrite loc and scale_tril
                # hint: use vec_to_inds
                # vec.size(0) is the mini-batch size
                
                indexes = self.vec_to_inds(x_dim, vec_dim)
                loc = vec[:, :x_dim]

                diag = vec[:, x_dim:2*x_dim]

                # exponentiated diagonal
                diag_values = torch.cat((torch.exp(diag), vec[:, 2*x_dim:]), dim=1)

                scale_tril = torch.zeros(vec.size(0), x_dim, x_dim)

                # fill the diagonal
                scale_tril[:, indexes[0], indexes[1]] = diag_values
                
                scale_tril = torch.clamp(scale_tril, self.minexp, self.maxexp)

            Mvn.__init__(self, loc=loc, scale_tril=scale_tril)

        else:
            """args is a loc, scale_tril
            """
            print('Init Mvn by full arg')
            Mvn.__init__(self, loc=args[0], scale_tril=args[1])

    def vec_to_inds(self, x_dim, vec_dim):
        """Computes the indices of scale_tril coeffs,
        scale_tril is filled main diagonal first

        x_dim: dimension of the random variable
        vec_dim: dimension of the vector containing
                 the coeffs of loc and scale_tril
        """
        ldiag, d, c = x_dim, 0, 0  # diag length, diag index, column index
        inds = [[], []]  # list of line and column indexes
        for i in range(vec_dim - x_dim):  # loop over the non-mean coeff
            inds[0].append(c+d)  # line index
            inds[1].append(c)  # column index
            if c == ldiag-1:  # the current diag end is reached
                ldiag += -1  # the diag length is decremented
                c = 0  # the column index is reinitialized
                d += 1  # the diag index is incremented
            else:  # otherwise, only the column index is incremented
                c += 1
        return inds


class Constructor(nn.Module):
    """Construct functions and conditional Gaussians from strings and kwargs
    - scale_vec_class is not None: return a Gaussian made from a vector,
        this vector is made of the concatenation of loc and scale_vec
    - scale_vec_class is None:
        if gauss_dim is not None: return a Gaussian made from a vector,
        else: return a vector
    """

    def __init__(
            self, 
            loc_classname: str, 
            loc_kwargs: dict,
            gauss_dim=None,
            scale_vec_classname=None, 
            scale_vec_kwargs=None) -> None:
        
        nn.Module.__init__(self)
        self.gauss_dim = gauss_dim
        self.loc = eval(loc_classname)(**loc_kwargs)
        if scale_vec_classname is not None:
            self.scale_vec = eval(scale_vec_classname)(**scale_vec_kwargs)
        else:
            self.scale_vec = None

    def forward(
            self, 
            *args: Union[torch.Tensor, int]) -> Union[torch.Tensor, Gaussian]:
        
        lc = self.loc(*args)
        if self.gauss_dim is not None:
            if self.scale_vec is not None:
                sc = self.scale_vec(*args)
                return Gaussian(self.gauss_dim, torch.cat((lc, sc), dim=1))
            else:
                return Gaussian(self.gauss_dim, lc)
        else:
            return lc
