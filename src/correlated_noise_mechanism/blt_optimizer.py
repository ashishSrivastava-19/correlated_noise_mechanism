import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


class BLTOptimizer:
    """
    An optimizer that implements the Buffered Linear Transformation (BLT) mechanism for differentially
    private training. This optimizer provides an alternative implementation of the BLT mechanism
    that focuses on optimizing the noise correlation parameters using closed-form expressions
    and gradient-based optimization.

    Parameters
    ----------
    n : int
        Size of the matrix (number of steps)
    d : int
        Number of parameters
    b : int, default=5
        Minimum separation parameter
    k : int, default=10
        Number of columns to consider in sensitivity
    error_type : str, default='rmse'
        Type of error to minimize: 'rmse' or 'max'
    participation : str, default='minSep'
        Participation pattern: 'minSep', 'cyclic', or 'single'
    device : str, default='cuda' if available else 'cpu'
        Computation device
    """

    def __init__(
        self,
        n,
        d,
        b=5,
        k=10,
        error_type="rmse",
        participation="minSep",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.n = n
        self.d = d
        self.b = b
        self.k = k
        self.error_type = error_type
        self.participation = participation
        self.device = device

    def gamma_n(self, lambda_val, n):

        return (1 - lambda_val**n) / (1 - lambda_val)

    def calculate_S_i(self, lambda_hat_i, n):

        gamma_n_i = self.gamma_n(lambda_hat_i, n)
        return 1 - lambda_hat_i + (lambda_hat_i / (1 - lambda_hat_i)) * (n - gamma_n_i)

    def calculate_S_ij(self, lambda_hat_i, lambda_hat_j, n):

        lambda_prod = lambda_hat_i * lambda_hat_j
        gamma_n_ij = self.gamma_n(lambda_prod, n)
        return 1 - lambda_prod + (lambda_prod / (1 - lambda_prod)) * (n - gamma_n_ij)

    def calculate_Gamma_hat_i(self, lambda_hat_i, n):

        S_i = self.calculate_S_i(lambda_hat_i, n)
        return 1 + (1 / (1 - lambda_hat_i)) * ((n * (n - 1) / 2) - S_i)

    def calculate_Gamma_hat_ij(self, lambda_hat_i, lambda_hat_j, n):

        S_i = self.calculate_S_i(lambda_hat_i, n)
        S_j = self.calculate_S_i(lambda_hat_j, n)
        S_ij = self.calculate_S_ij(lambda_hat_i, lambda_hat_j, n)

        denominator = (1 - lambda_hat_i) * (1 - lambda_hat_j)
        return 1 + (1 / denominator) * ((n * (n - 1) / 2) - S_i - S_j + S_ij)

    def calculate_Gamma_i(self, lambda_hat_i, n):

        gamma_n_i = self.gamma_n(lambda_hat_i, n)
        return (1 / (1 - lambda_hat_i)) * (n - gamma_n_i)

    def calculate_Gamma_ij(self, lambda_hat_i, lambda_hat_j, n):

        gamma_n_i = self.gamma_n(lambda_hat_i, n)
        gamma_n_j = self.gamma_n(lambda_hat_j, n)
        gamma_n_ij = self.gamma_n(lambda_hat_i * lambda_hat_j, n)

        denominator = (1 - lambda_hat_i) * (1 - lambda_hat_j)
        return (1 / denominator) * (n - gamma_n_i - gamma_n_j + gamma_n_ij)

    def calculate_C_squared_1_to_2(self, alpha, lambda_val):

        result = 1.0

        for i in range(self.d):
            for j in range(self.d):
                gamma_term = self.gamma_n(lambda_val[i] * lambda_val[j], self.n - 1)
                result += alpha[i] * alpha[j] * gamma_term

        return result

    def calculate_sensitivity_squared(self, alpha, lambda_val):
        """
        Calculate sensitivity squared based on participation schema
        """
        if self.participation == "single":
            # For single participation, use ||C||²_{1→2}
            return self.calculate_C_squared_1_to_2(alpha, lambda_val)

        # For minSep and cyclic, we need to calculate using Lemma 3.3
        # First, construct the first column of C
        first_col = torch.zeros(self.n, device=self.device)
        first_col[0] = 1

        for i in range(1, self.n):
            entry = 0
            for j in range(self.d):
                entry += alpha[j] * (lambda_val[j] ** (i - 1))
            first_col[i] = entry

        # Calculate C^T C elements needed
        if self.participation == "cyclic":
            # For cyclic, sum the first k diagonal elements of C^T C
            sens_squared = 0
            for i in range(min(self.k, self.n)):
                # Extract the relevant subcolumn
                col = first_col[0 : self.n - i]
                # Calculate the diagonal element (i,i) of C^T C
                sens_squared += torch.sum(col**2)

        else:  # minSep
            # For minSep, sum the L2 norms of columns with separation b
            sens_squared = 0
            for i in range(min(self.k, (self.n - 1) // self.b + 1)):
                col_idx = i * self.b
                # Extract the relevant column
                col = first_col[0 : self.n - col_idx]
                sens_squared += torch.sum(col**2)

        return sens_squared

    def calculate_B_norm_squared(self, alpha_hat, lambda_hat):
        """
        Calculate the squared norm of B = AC^(-1) using closed form expressions
        from Lemma A.6.1
        """
        if self.error_type == "rmse":
            # For RMSE, calculate Frobenius norm squared and divide by n
            # Using formula from Lemma A.6.1(c)
            B_norm_squared = (self.n + 1) / 2.0

            # Add the second term
            for i in range(self.d):
                gamma_hat_i = self.calculate_Gamma_hat_i(lambda_hat[i], self.n)
                B_norm_squared += (2 / self.n) * alpha_hat[i] * gamma_hat_i

            # Add the third term
            for i in range(self.d):
                for j in range(self.d):
                    gamma_hat_ij = self.calculate_Gamma_hat_ij(
                        lambda_hat[i], lambda_hat[j], self.n
                    )
                    B_norm_squared += (
                        (1 / self.n) * alpha_hat[i] * alpha_hat[j] * gamma_hat_ij
                    )

        else:  # max error
            # For max error, calculate the 2→∞ norm squared using Lemma A.6.1(b)
            B_norm_squared = self.n

            # Add the second term
            for i in range(self.d):
                gamma_i = self.calculate_Gamma_i(lambda_hat[i], self.n)
                B_norm_squared += 2 * alpha_hat[i] * gamma_i

            # Add the third term
            for i in range(self.d):
                for j in range(self.d):
                    gamma_ij = self.calculate_Gamma_ij(
                        lambda_hat[i], lambda_hat[j], self.n
                    )
                    B_norm_squared += alpha_hat[i] * alpha_hat[j] * gamma_ij

        return B_norm_squared

    def objective_function(self, params):

        # Split the parameters
        alpha = params[: self.d]
        lambda_val = params[self.d : 2 * self.d]
        alpha_hat = params[2 * self.d : 3 * self.d]
        lambda_hat = params[3 * self.d :]

        # Apply constraints: lambda values must be in [0,1)
        lambda_val = torch.clamp(lambda_val, 0, 0.999)
        lambda_hat = torch.clamp(lambda_hat, 0, 0.999)

        alpha = torch.abs(alpha)
        alpha_hat = -torch.abs(alpha_hat)

        # Calculate sensitivity squared
        sens_squared = self.calculate_sensitivity_squared(alpha, lambda_val)

        # Calculate B norm squared
        B_norm_squared = self.calculate_B_norm_squared(alpha_hat, lambda_hat)

        # Return the product
        return sens_squared * B_norm_squared

    def optimize(self, num_iterations=1000, lr=0.01, verbose=False):
        """
        Optimize the parameters using gradient descent
        """
        params = torch.rand(4 * self.d, requires_grad=True, device=self.device)

        optimizer = torch.optim.Adam([params], lr=lr)

        best_loss = float("inf")
        best_params = None

        for i in range(num_iterations):
            optimizer.zero_grad()

            loss = self.objective_function(params)

            loss.backward()

            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = params.detach().clone()

            if verbose and i % 100 == 0:
                logger.info("Iteration %d, Loss: %s", i, loss.item())

        alpha_opt = torch.abs(best_params[: self.d])
        lambda_opt = torch.clamp(best_params[self.d : 2 * self.d], 0, 0.999)
        alpha_hat_opt = -torch.abs(best_params[2 * self.d : 3 * self.d])
        lambda_hat_opt = torch.clamp(best_params[3 * self.d :], 0, 0.999)
        logger.info("Optimization Completed")
        return {
            "alpha": alpha_opt,
            "lambda": lambda_opt,
            "alpha_hat": alpha_hat_opt,
            "lambda_hat": lambda_hat_opt,
            "objective_value": best_loss,
        }
