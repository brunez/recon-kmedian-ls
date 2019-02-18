import sys
import getopt
import numpy as np
import pickle
import datetime

def km_cost(facility_distances):
    all_dists = np.min(facility_distances, axis=0)
    total_cost = np.sum(all_dists)
    return float(total_cost)/facility_distances.shape[1]

def pd_cost(S, distance_matrix):
    total_cost = 0
    for i in range(len(S)):
        for j in range(i+1, len(S)):        
            total_cost += distance_matrix[S[i], S[j]]
    return float(total_cost)/(len(S)*(len(S)-1)/2.)

def compute_cost(S, facility_distances, pairwise_distance_matrix, reg_lambda):
    return km_cost(facility_distances) + reg_lambda*pd_cost(S, pairwise_distance_matrix)

def printUsage():
    print ("python local_search.py -o|--output=<filename> -f|--facility_dists=<filename> -d|--client_dists=<filename> -k <n. of clusters> -r|--reg=<regularization term>")

def main(argv):

    try:
        opts, args = getopt.getopt(argv,"o:f:d:k:r:",["output=", "facility_dists=", "k=", "reg=", "client_dists="])
    except getopt.GetoptError:
        printUsage()
        sys.exit(1)
    for opt, arg in opts:
        if opt in ("-o", "--output"):
            file_out = arg
        elif opt in ("-f", "--facility_dists"):
            facility_dist_matrix_file = arg
        elif opt in ("-d", "--client_dists"):
            ftc_dist_matrix_file = arg
        elif opt in ("-k", "--k"):
            K = int(arg)
        elif opt in ("-r", "--reg"):
            reg_lambda = float(arg)

    print ("#########################")
    print ("Local search for Reconciliation k-medians")
    print ('Output file: {}'.format(file_out))
    print ('Facility-facility distance matrix file: {}'.format(facility_dist_matrix_file))
    print ('Facility-client distance matrix file: {}'.format(ftc_dist_matrix_file))
    print ('k: {}'.format(K))
    print ('Regularization term: {}'.format(reg_lambda))
    print ("#########################")

    pairwise_distance_matrix = np.load(facility_dist_matrix_file) # Facility pairwise distances
    # Use option below if matrix is in whitespace-separated-values format
    #pairwise_distance_matrix = np.loadtxt(facility_dist_matrix_file)
    print 'PD matrix: {}'.format(pairwise_distance_matrix.shape)
    facility_distances = np.load(ftc_dist_matrix_file) # facility to client distances
    # Use option below if matrix is in whitespace-separated-values format
    #facility_distances = np.loadtxt(ftc_dist_matrix_file)
    print 'FD matrix: {}'.format(facility_distances.shape)

    facilities = range(pairwise_distance_matrix.shape[0])
    # Initialize facility set randomly
    S = np.random.choice(facilities, K, replace=False)
    converged = False
    iterations = 0
    times = []
    current_cost = compute_cost(S, facility_distances[S,:], pairwise_distance_matrix, reg_lambda)
    while not converged:        
        pre_cost = np.copy(current_cost)
        for pos in range(K):
            start = datetime.datetime.now()
            # Remove one for local-search swap
            reduced_set = [S[i] for i in range(K) if i != pos]
            current_S_distances = facility_distances[reduced_set,:]
            costs = []
            fS = np.copy(S)
            for f in facilities:
                if f in S:
                    costs.append(np.inf)
                    continue
                else:
                    facility_distances_submat = np.vstack([facility_distances[[f],:], current_S_distances])
                    fS[pos] = f                    
                    cost = compute_cost(fS, facility_distances_submat, pairwise_distance_matrix, reg_lambda)
                    costs.append(cost)

            # We only make a replacement if the cost improves
            if np.min(costs) < current_cost:
                S[pos] = np.argmin(costs)
                current_cost = np.min(costs)
               
            end = datetime.datetime.now()
            times.append((end-start).total_seconds())
        
        converged = pre_cost == current_cost 
        iterations += 1
    with open(file_out, 'w') as f:
        s = ''
        s += "#########################\n"
        s += "Local search\n"
        s += 'file_out: {}\n'.format(file_out)
        s += 'facility_dist_matrix_file: {}\n'.format(facility_dist_matrix_file)
        s += 'k: {}\n'.format(K)
        s += 'reg_lambda: {}\n'.format(reg_lambda)
        s += "#########################"
        s += '\n\nResults:\n'
        f.write(s)
        f.write('Facilities: {}\n'.format(','.join([str(x) for x in S])))        
        f.write('Cost (km/pd/total): {}/{}/{}\n'.format(km_cost(facility_distances[S,:]), reg_lambda*pd_cost(S, pairwise_distance_matrix), current_cost))
        f.write('Iterations: {}\n'.format(iterations))
        f.write('Running times: {}\n'.format(times))
        
if __name__ == "__main__":
   main(sys.argv[1:])
