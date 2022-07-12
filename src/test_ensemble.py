from ensemble import Ensemble
import numpy as np


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
