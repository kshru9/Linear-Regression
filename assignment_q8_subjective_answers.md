# ES654-2020 Assignment 3

*Your Name* - *Your Roll Number*

------

> Write the answers for the subjective questions here

- As we increase the no. of samples, the time required for vectorized gradient descent increases very sharply.
 - But there is no significant change in the time required for normal equation

Theoretical time complexity for normal method is O(P^2 * N) + O(D^3)
Theoretical time complexity for gradient descent method is O((t+N) * D^2)

- For normal equation, for a constant N, as P increases, there is huge increase in time computation as seen in heat map

- The time complexity increases as a product of number of samples and number of dimensions while keeping number of iterations constant.

## Heatmaps

### For gradient descent 
![](./figures/fresh_grad.png)

### For Normal method
![](./figures/norm_hm.png)

## Using matplotlib

### Varying N
![](./figures/q8_n_plt.png)

### Varying P
![](./figures/q8_p_plt.png)