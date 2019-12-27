from OnePixelAttack.differential_evolution import differential_evolution

def square_fn(x,a):
    return a*x**2


def find_min(a,maxiter=75, popsize=400):

    def square(x):
        return square_fn(x,a)

    bounds = [(-100,100)]
    popmul = max(1, popsize // len(bounds))

    idk = differential_evolution(
            square, bounds, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=-1)

    return idk.x[0]


print(find_min(10))