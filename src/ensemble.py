import numpy as np

class Ensemble( ):


    def __init__(self, numParticles, numDimensions):
        
        self.numParticles = numParticles
        self.numDimensions = numDimensions
        self.q = np.zeros((numDimensions, numParticles))
        self.p = np.zeros((numDimensions, numParticles))
        self.m = np.zeros(numParticles)
        self.weights = np.zeros(numParticles)


    def particle(self, particleNum):
# I'm not sure if brackets follow correct style here - please correct if needed        
        if not 0 <= particleNum < self.numParticles:

            raise IndexError(f'Index {particleNum} out of bounds. '\
                f'numParticles={self.numParticles}') 

        return self.q[:, particleNum], self.p[:, particleNum], \
        self.m[particleNum], self.weights[particleNum]

## MOVE TO TEST - I DONT GET VC
def main( ):
    numDimensions = 4
    numParticles = 100

    ensemble1 = Ensemble(numParticles, numDimensions)

    # expected output
    qExp = np.zeros( numDimensions )
    pExp = np.zeros( numDimensions )
    mExp = 0
    wExp = 0

    q1, p1, m1, w1 = ensemble1.particle( 10 )

    if (qExp == q1).all( ) and (pExp == p1).all( ) and (mExp == m1).all( ) and \
        (wExp == w1).all( ):
        print('Test 1 passed.')

    try:
        _ = ensemble1.particle(numParticles + 1)
        print('Test 2 Failed')
    except IndexError as error: 
        print(error)
        print('Test 2 Passed')

if __name__ == '__main__':
    main()