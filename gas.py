import numpy as np


class GaussGas():
    
    def __init__( self, particles, evaluate ):
        self.particles = particles
        print('parts:', particles)
        self.evaluate = evaluate
        
    # def getBest( self ):
    #     return max( zip( (self.evaluate(p) for p in self.particles), self.particles ) )[1]

    def update( self ):
        self.particles[:] = self.sample( len( self.particles ) )
    
    def sample( self, num_samples ):
        # get fitness of every particle
        fitness = np.array( list( self.evaluate(p) for p in self.particles ) )
        print( 'average fitness:', np.mean( fitness ) )
        #print('fitness[0]:', fitness[0])
        # MAYBE this is unnecessary since we're messing with covariance anyway... it might help but it definitely complicates things
        # removing it could help by preventing premature convergence as errant particles coalesce around more optimal ones
        # but removing it could hurt by reducing the search around optimal points, and at that point the particles never interact
        # if we're trying to improve upon particle filtering, we need to keep it
        
        # sample the corresponding pdf
        samples = np.searchsorted( # we can simulate the inverse by mapping the co-domain to the domain with binary search
            np.cumsum( normalize( fitness ) ), # construct a cdf from the fitness array (necessarily sorted)
            np.random.random( num_samples ) # we'll use uniformly random samples from [0, 1) as a basis
        )
        
        # END MAYBE
        
        # add gaussian noise to each sample
        dimension = len( self.particles[0] )
        noisy_samples = np.array( list(
            np.random.multivariate_normal( # we'll sample a new gaussian for each sample
                mean=self.particles[i], # instead of explicitly adding the noise, we'll just sample from pdfs centered at each sample
                cov=np.identity( dimension ) * (1 / fitness[i]) # covariance is inversely proportional to fitness (this can probably be improved with consideration to other solutions we've evaluated)
            ) for i in samples )
        )
        
        return noisy_samples


def normalize( X ):
    X = np.array( list( X ) )
    return X / sum( X )


"""
def samplePDF( pdf ):
    cdf = np.cumsum( pdf )
"""

def closeness(X):
    return 1 / np.var(X)

def main():
    num_particles = 20
    seed = np.array([0, 0, 0])
    initial_variance = 50
    num_iterations = 40
    
    g = GaussGas(
        np.random.multivariate_normal(
            mean=seed,
            cov=np.identity( len( seed ) ) * initial_variance,
            size=num_particles
        ), closeness )
    for i in range( num_iterations ):
        g.update()
    print( g.particles )
    
if __name__ == '__main__':
    main()
