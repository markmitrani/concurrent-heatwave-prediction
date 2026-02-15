import numpy as np
import h5py

def main():
    with h5py.File('data/pcha_results_8a.hdf5', 'r') as f: # run from mmi393 directory or gives error
        XC = f['/XC'][:]
        S_PCHA = f['/S_PCHA'][:]
        C = f['/C'][:]
    
    n_arch = XC.shape[1]
    n_samples = S_PCHA.shape[1]

    percentages = np.asarray([20, 25, 30, 35, 40, 50, 60, 70, 80, 85, 90, 95, 100])

    ratio_arr = np.zeros((n_arch, len(percentages)))

    for i in range(n_arch):
        for j in range(len(percentages)):
            ratio = np.sum(np.where(S_PCHA[i, :] >= (percentages[j]/100), 1, 0))/n_samples
            ratio_arr[i,j] = ratio
            print(f"A{i+1}, >= {percentages[j]}%: {ratio*100}%")
        print()

if __name__ == '__main__':
    main()